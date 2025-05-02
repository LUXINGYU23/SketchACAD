#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
综合数据处理脚本：处理CAD数据集的JSON文件
1. 扫描指定目录下的所有JSON文件
2. 为每个文件分配ID并重命名
3. 保存文件到指定目录的/json文件夹
4. 将JSON文件转换为向量表示并保存到/vec文件夹
5. 创建STEP文件到/step文件夹
6. 创建网格文件到/stl文件夹
7. 创建点云文件到/ply文件夹
8. 按照8:2比例划分训练和测试集
9. 支持基于JSON内容哈希的去重功能
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
import platform

# 添加项目根目录到Python路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))

from cad_sequence import CADSequence
from utility.macro import N_BIT, MAX_CAD_SEQUENCE_LENGTH
from utility.utils import ensure_dir, get_files_scan
from utility.logger import CLGLogger
from utility.decorator import measure_performance
from src.CadSeqProc.utility.macro import *
from src.CadSeqProc.utility.utils import (
    generate_attention_mask,
    ensure_dir,
    hash_map,
    get_files_scan,
)
# 根据平台选择合适的多进程启动方法
if platform.system() != "Windows":
    # Unix/Linux/MacOS使用forkserver
    multiprocessing.set_start_method("forkserver", force=True)
else:
    # Windows使用spawn
    multiprocessing.set_start_method("spawn", force=True)

# 配置日志
clglogger = CLGLogger().configure_logger().logger

class DataProcessor:
    """
    CAD数据集处理器
    处理JSON文件，分配ID，转换为不同格式，划分训练和测试集
    """
    def __init__(
        self,
        input_dir,
        output_dir,
        bit=N_BIT,
        max_workers=8,
        train_ratio=0.8,
        prefix_digits=8,
        deduplicate=False
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.bit = bit
        self.max_workers = max_workers
        self.train_ratio = train_ratio
        self.prefix_digits = prefix_digits
        self.file_counter = 0
        self.id_mapping = {}
        self.processed_files = []
        self.deduplicate = deduplicate
        self.unique_model_hashes = set()  # 存储唯一模型的哈希值
        self.duplicate_count = 0  # 跟踪重复文件数量
        
        # 创建输出目录结构
        self._create_directories()
    
    def _create_directories(self):
        """创建输出目录结构"""
        clglogger.info(f"创建输出目录结构: {self.output_dir}")
        
        # 主文件夹
        ensure_dir(self.output_dir)
        
        # 子文件夹
        for folder in ["json", "vec", "step", "stl", "ply"]:
            ensure_dir(os.path.join(self.output_dir, folder))
    
    def _generate_id(self):
        """生成唯一ID"""
        self.file_counter += 1
        return str(self.file_counter).zfill(self.prefix_digits)
    
    def scan_and_process_files(self):
        """扫描并处理所有JSON文件"""
        clglogger.info(f"扫描目录: {self.input_dir}")
        
        # 获取所有JSON文件
        try:
            # 使用简单的方法查找所有JSON文件
            json_files = []
            for root, _, files in os.walk(self.input_dir):
                for file in files:
                    if file.lower().endswith('.json'):
                        json_files.append(os.path.join(root, file))
            
            if not json_files:
                clglogger.error(f"在目录 {self.input_dir} 中未找到JSON文件")
                return
            
            clglogger.info(f"找到 {len(json_files)} 个JSON文件")
            
            # 处理所有JSON文件
            self.process_all_files(json_files)
            
            # 生成训练和测试集
            self.split_train_test()
            
            # 输出去重统计信息
            if self.deduplicate:
                clglogger.info(f"检测到 {self.duplicate_count} 个重复文件")
                clglogger.info(f"成功处理 {len(self.processed_files)} 个唯一文件")
                
                # 保存重复文件信息
                duplicate_info = {
                    "total_files": len(json_files),
                    "unique_files": len(self.processed_files),
                    "duplicate_files": self.duplicate_count,
                    "duplicate_percentage": self.duplicate_count / len(json_files) if len(json_files) > 0 else 0
                }
                
                with open(os.path.join(self.output_dir, "duplicate_info.json"), 'w') as f:
                    json.dump(duplicate_info, f, indent=2)
            
        except Exception as e:
            clglogger.error(f"扫描目录失败: {e}")
            import traceback
            traceback.print_exc()
    
    def process_all_files(self, json_files):
        """处理所有JSON文件"""
        clglogger.info(f"开始处理 {len(json_files)} 个JSON文件")
        
        # 使用进程池加速处理
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交处理任务
            futures = []
            for json_path in json_files:
                file_id = self._generate_id()
                self.id_mapping[json_path] = file_id
                futures.append(executor.submit(self.process_single_file, json_path, file_id))
            
            # 等待并处理结果
            for future in tqdm(as_completed(futures), desc="处理文件", total=len(futures)):
                result, is_duplicate = future.result()
                if result and not is_duplicate:
                    self.processed_files.append(result)
                elif is_duplicate:
                    self.duplicate_count += 1
        
        clglogger.info(f"成功处理 {len(self.processed_files)} 个文件")
    
    def is_duplicate(self, json_data):
        """检查文件是否重复"""
        if not self.deduplicate:
            return False
            
        try:
            # 转换为向量表示
            _, cad_vec, _, _ = CADSequence.json_to_vec(
                data=json_data,
                bit=self.bit,
                padding=True,
                max_cad_seq_len=MAX_CAD_SEQUENCE_LENGTH,
            )
            
            # 提取参数并计算哈希
            param = cad_vec[torch.where(cad_vec >= len(END_TOKEN))[0]].tolist()
            hash_val = hash_map(param)
            
            # 检查是否已存在
            if hash_val in self.unique_model_hashes:
                return True
                
            # 添加到哈希集合
            self.unique_model_hashes.add(hash_val)
            return False
            
        except Exception as e:
            clglogger.error(f"计算哈希值失败: {e}")
            return False
    
    def process_single_file(self, json_path, file_id):
        """处理单个JSON文件"""
        try:
            clglogger.info(f"处理文件: {json_path}, 分配ID: {file_id}")
            
            # 1. 读取JSON文件
            with open(json_path, 'r') as f:
                json_data = json.load(f)
                
            # 检查是否重复
            if self.is_duplicate(json_data):
                clglogger.info(f"文件 {json_path} 被识别为重复，跳过处理")
                return None, True
            
            # 2. 复制到/json目录
            json_output_path = os.path.join(self.output_dir, "json", f"{file_id}.json")
            with open(json_output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            # 3. 转换为向量表示并保存到/vec目录
            self.json_to_vec(json_data, file_id)
            
            # 4. 创建STEP文件并保存到/step目录
            self.json_to_step(json_data, file_id)
            
            # # 5. 创建STL (网格) 文件并保存到/stl目录
            # # 6. 创建PLY (点云) 文件并保存到/ply目录
            # self.create_mesh_and_pointcloud(json_data, file_id)
            
            return file_id, False
            
        except Exception as e:
            clglogger.error(f"处理文件 {json_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            return None, False
    
    def json_to_vec(self, json_data, file_id):
        """将JSON转换为向量表示并保存"""
        try:
            # 转换为向量表示
            cad_obj, cad_vec, flag_vec, index_vec = CADSequence.json_to_vec(
                data=json_data,
                bit=self.bit,
                padding=True,
                max_cad_seq_len=MAX_CAD_SEQUENCE_LENGTH,
            )
            
            # 创建掩码
            attn_mask = generate_attention_mask(cad_vec.shape[0] - 1)
            key_padding_mask = cad_vec == END_TOKEN.index("PADDING")
            
            # 保存向量表示
            cad_seq_dict = {
                "vec": {
                    "cad_vec": cad_vec,
                    "flag_vec": flag_vec,
                    "index_vec": index_vec,
                },
                "mask_cad_dict": {
                    "attn_mask": attn_mask,
                    "key_padding_mask": key_padding_mask,
                },
            }
            
            # 保存到文件
            vec_output_path = os.path.join(self.output_dir, "vec", f"{file_id}.pth")
            torch.save(cad_seq_dict, vec_output_path)
            
            clglogger.info(f"向量表示已保存到: {vec_output_path}")
            return True
            
        except Exception as e:
            clglogger.error(f"转换JSON到向量失败, ID {file_id}: {e}")
            return False
    
    
    def json_to_step(self, json_data, file_id):
        """将JSON转换为STEP文件并保存"""
        try:
            # 创建CADSequence对象
            cad_seq = CADSequence.from_dict(json_data)
            cad_seq = CADSequence.json_to_NormalizedCAD(data=json_data, bit=self.bit)
            
            # 创建CAD模型
            try:
                cad_seq.create_cad_model()
            except Exception as e:
                clglogger.warning(f"创建CAD模型失败, ID {file_id}: {e}")
            
            # 保存STEP文件
            step_output_path = os.path.join(self.output_dir, "step", f"{file_id}.step")
            output_dir = os.path.dirname(step_output_path)
            filename = os.path.basename(step_output_path).split('.')[0]
            
            cad_seq.save_stp(
                filename=filename,
                output_folder=output_dir,
                type="step"
            )
            
            clglogger.info(f"STEP文件已保存到: {step_output_path}")
            return True
            
        except Exception as e:
            clglogger.error(f"转换JSON到STEP失败, ID {file_id}: {e}")
            return False
    
    def create_mesh_and_pointcloud(self, json_data, file_id, n_points=10000):
        """创建网格和点云文件"""
        try:
            # 创建CADSequence对象
            cad_seq = CADSequence.json_to_NormalizedCAD(data=json_data, bit=self.bit)
            
            # 创建网格
            try:
                cad_seq.create_mesh()
            except Exception as e:
                clglogger.warning(f"创建网格失败, ID {file_id}: {e}")
                return False
            
            # 保存STL文件
            stl_output_path = os.path.join(self.output_dir, "stl", f"{file_id}.stl")
            cad_seq.mesh.export(stl_output_path, file_type="stl")
            clglogger.info(f"STL文件已保存到: {stl_output_path}")
            
            # 采样点云并着色
            try:
                cad_seq.sample_sketch_points3D(n_points=n_points, color=True)
            except Exception as e:
                clglogger.warning(f"采样点云失败, ID {file_id}: {e}")
                # 继续尝试保存点云
            
            # 保存PLY文件
            ply_output_path = os.path.join(self.output_dir, "ply", f"{file_id}.ply")
            
            # 使用Open3D保存点云
            try:
                import open3d as o3d
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(cad_seq.sketch_points3D)
                
                # 设置点云颜色
                if hasattr(cad_seq, 'sketch_points3D_color') and cad_seq.sketch_points3D_color.size > 0:
                    colors = o3d.utility.Vector3dVector(cad_seq.sketch_points3D_color)
                    point_cloud.colors = colors
                
                o3d.io.write_point_cloud(ply_output_path, point_cloud)
                clglogger.info(f"PLY文件已保存到: {ply_output_path}")
            except Exception as e:
                clglogger.error(f"保存点云失败, ID {file_id}: {e}")
            
            return True
            
        except Exception as e:
            clglogger.error(f"创建网格和点云失败, ID {file_id}: {e}")
            return False
    
    def split_train_test(self):
        """划分训练集和测试集"""
        if not self.processed_files:
            clglogger.warning("没有处理成功的文件，无法划分训练集和测试集")
            return
        
        clglogger.info("划分训练集和测试集")
        
        # 随机打乱处理过的文件ID
        random.shuffle(self.processed_files)
        
        # 计算训练集大小
        train_size = int(len(self.processed_files) * self.train_ratio)
        
        # 划分训练集和测试集
        train_ids = self.processed_files[:train_size]
        test_ids = self.processed_files[train_size:]
        
        # 保存训练集和测试集信息
        train_data = {
            "total": len(train_ids),
            "prefix": train_ids[0][:2] if train_ids else "",  # 取前两位作为前缀示例
            "files": train_ids
        }
        
        test_data = {
            "total": len(test_ids),
            "prefix": test_ids[0][:2] if test_ids else "",  # 取前两位作为前缀示例
            "files": test_ids
        }
        
        # 保存到JSON文件
        with open(os.path.join(self.output_dir, "train.json"), 'w') as f:
            json.dump(train_data, f, indent=2)
        
        with open(os.path.join(self.output_dir, "test.json"), 'w') as f:
            json.dump(test_data, f, indent=2)
        
        clglogger.info(f"训练集: {len(train_ids)} 个文件, 测试集: {len(test_ids)} 个文件")
        clglogger.info(f"训练集和测试集信息已保存到: {self.output_dir}/train.json 和 {self.output_dir}/test.json")


@measure_performance
def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="CAD数据处理脚本: 扫描JSON文件，分配ID，转换为不同格式，划分训练和测试集"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="输入目录，包含JSON文件"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="输出目录"
    )
    parser.add_argument(
        "--bit", type=int, default=N_BIT,
        help=f"量化位数，默认为{N_BIT}"
    )
    parser.add_argument(
        "--max_workers", type=int, default=8,
        help="最大工作进程数，默认为8"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="训练集比例，默认为0.8"
    )
    parser.add_argument(
        "--prefix_digits", type=int, default=8,
        help="ID前缀位数，默认为8"
    )
    parser.add_argument(
        "--deduplicate", action="store_true",
        help="启用去重功能，基于JSON内容哈希"
    )
    
    args = parser.parse_args()
    
    # 配置日志文件
    log_file = os.path.join(args.output, "processing_log.txt")
    clglogger.add(log_file, rotation="100 MB")
    
    clglogger.info("启动CAD数据处理脚本")
    clglogger.info(f"输入目录: {args.input}")
    clglogger.info(f"输出目录: {args.output}")
    
    # 创建处理器并执行处理
    processor = DataProcessor(
        input_dir=args.input,
        output_dir=args.output,
        bit=args.bit,
        max_workers=args.max_workers,
        train_ratio=args.train_ratio,
        prefix_digits=args.prefix_digits,
        deduplicate=args.deduplicate
    )
    
    processor.scan_and_process_files()
    
    clglogger.info("处理完成")


if __name__ == "__main__":
    main()