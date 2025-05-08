#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JSON到向量转换工具

该脚本用于将已处理的JSON文件批量转换为向量表示（.pth文件）
特点:
1. 多线程处理
2. 自动跳过已生成的向量文件
3. 批处理模式避免内存溢出
4. 详细的日志输出

使用方法:
python json2vec_converter.py -i /path/to/json/dir -o /path/to/output/dir --max_workers 8
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import concurrent
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import multiprocessing
import platform
import gc

# 添加项目根目录到Python路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("../..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))

from cad_sequence import CADSequence
from utility.macro import N_BIT, MAX_CAD_SEQUENCE_LENGTH, END_TOKEN
from utility.utils import ensure_dir, generate_attention_mask
from utility.logger import CLGLogger

# 根据平台选择合适的多进程启动方法
if platform.system() != "Windows":
    # Unix/Linux/MacOS使用forkserver
    multiprocessing.set_start_method("forkserver", force=True)
else:
    # Windows使用spawn
    multiprocessing.set_start_method("spawn", force=True)

# 配置日志
clglogger = CLGLogger().configure_logger().logger

class VectorConverter:
    """JSON到向量转换器"""
    def __init__(
        self,
        json_dir,
        output_dir,
        bit=N_BIT,
        max_workers=8,
        batch_size=1000,
        timeout=300
    ):
        self.json_dir = json_dir
        self.output_dir = output_dir
        self.bit = bit
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.timeout = timeout
        self.processed_count = 0
        self.skipped_count = 0
        self.error_count = 0
        
        # 创建输出目录
        self._create_directories()
        
    def _create_directories(self):
        """创建输出目录结构"""
        clglogger.info(f"创建输出目录结构: {self.output_dir}")
        ensure_dir(self.output_dir)
        ensure_dir(os.path.join(self.output_dir, "vec"))
        
    def scan_files(self):
        """扫描JSON文件"""
        clglogger.info(f"扫描目录: {self.json_dir}")
        
        json_files = []
        for root, _, files in os.walk(self.json_dir):
            for file in files:
                if file.lower().endswith('.json'):
                    json_files.append(os.path.join(root, file))
        
        if not json_files:
            clglogger.error(f"在目录 {self.json_dir} 中未找到JSON文件")
            return []
        
        clglogger.info(f"找到 {len(json_files)} 个JSON文件")
        return json_files
    
    def process_all_files(self):
        """处理所有JSON文件"""
        json_files = self.scan_files()
        if not json_files:
            return
        
        # 按批次处理
        total_batches = (len(json_files) + self.batch_size - 1) // self.batch_size
        
        start_time = time.time()
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(json_files))
            batch_files = json_files[start_idx:end_idx]
            
            clglogger.info(f"处理批次 {batch_idx+1}/{total_batches}，文件数量: {len(batch_files)}")
            
            self.process_batch(batch_files)
            
            # 强制进行垃圾回收
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 输出进度
            elapsed_time = time.time() - start_time
            files_processed = (batch_idx + 1) * self.batch_size
            total_files = len(json_files)
            files_per_second = files_processed / elapsed_time if elapsed_time > 0 else 0
            estimated_time_left = (total_files - files_processed) / files_per_second if files_per_second > 0 else 0
            
            clglogger.info(f"已处理: {files_processed}/{total_files} | "
                          f"速度: {files_per_second:.2f} 文件/秒 | "
                          f"剩余时间: {estimated_time_left/60:.2f} 分钟")
        
        # 输出最终统计
        clglogger.info(f"处理完成: 总计 {len(json_files)} 文件")
        clglogger.info(f"成功: {self.processed_count} | 跳过: {self.skipped_count} | 错误: {self.error_count}")
    
    def process_batch(self, json_files):
        """处理一批JSON文件"""
        # 使用进程池加速处理
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交处理任务
            futures = []
            for json_path in json_files:
                file_id = os.path.basename(json_path).split('.')[0]  # 从文件名获取ID
                futures.append(executor.submit(self.process_file, json_path, file_id))
            
            # 等待并处理结果
            batch_processed = 0
            batch_skipped = 0
            batch_errors = 0
            
            for future in tqdm(as_completed(futures), desc="处理文件", total=len(futures)):
                try:
                    result, status = future.result(timeout=self.timeout)
                    if status == "processed":
                        batch_processed += 1
                        self.processed_count += 1
                    elif status == "skipped":
                        batch_skipped += 1
                        self.skipped_count += 1
                    else:  # error
                        batch_errors += 1
                        self.error_count += 1
                except concurrent.futures.TimeoutError:
                    batch_errors += 1
                    self.error_count += 1
                    clglogger.error(f"处理超时")
                except Exception as e:
                    batch_errors += 1
                    self.error_count += 1
                    clglogger.error(f"处理出错: {e}")
            
            clglogger.info(f"批次完成: 处理 {batch_processed} | 跳过 {batch_skipped} | 错误 {batch_errors}")
    
    def process_file(self, json_path, file_id):
        """处理单个JSON文件转换为向量"""
        try:
            # 检查是否已存在向量文件
            vec_path = os.path.join(self.output_dir, "vec", f"{file_id}.pth")
            if os.path.exists(vec_path):
                return file_id, "skipped"
            
            # 读取JSON文件
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # 转换为向量表示
            cad_obj, cad_vec, flag_vec, index_vec = CADSequence.json_to_vec(
                data=json_data,
                bit=self.bit,
                padding=True,
                max_cad_seq_len=MAX_CAD_SEQUENCE_LENGTH,
            )
            
            # 创建掩码
            attn_mask = generate_attention_mask(cad_vec.shape[0])
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
            torch.save(cad_seq_dict, vec_path)
            
            return file_id, "processed"
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            clglogger.error(f"处理文件 {json_path} 失败: {e}\n{error_msg}")
            return file_id, "error"


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="JSON到向量转换工具: 将JSON文件批量转换为向量表示"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="输入目录，包含JSON文件"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="输出目录，将保存vec子文件夹"
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
        "--batch_size", type=int, default=1000,
        help="批处理大小，默认为1000"
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="单个文件处理超时时间（秒），默认为300"
    )
    args = parser.parse_args()
    
    # 配置日志文件
    log_file = os.path.join(args.output, "vec_conversion_log.txt")
    clglogger.add(log_file, rotation="100 MB")
    
    clglogger.info("启动JSON到向量转换工具")
    clglogger.info(f"输入目录: {args.input}")
    clglogger.info(f"输出目录: {args.output}")
    clglogger.info(f"处理参数: 位数={args.bit}, 工作进程={args.max_workers}, "
                 f"批大小={args.batch_size}, 超时={args.timeout}秒")
    
    # 创建转换器并执行处理
    converter = VectorConverter(
        json_dir=args.input,
        output_dir=args.output,
        bit=args.bit,
        max_workers=args.max_workers,
        batch_size=args.batch_size,
        timeout=args.timeout
    )
    
    converter.process_all_files()
    
    clglogger.info("转换完成")


if __name__ == "__main__":
    main()