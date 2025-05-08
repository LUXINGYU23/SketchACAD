#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据集划分工具

该脚本用于将处理后的数据集随机划分为训练集和测试集
特点:
1. 可自定义训练集和测试集的比例
2. 只划分已处理完成的文件(存在对应vec文件的json文件)
3. 可自定义输出文件名
4. 支持按模型复杂度分层抽样

使用方法:
python split_dataset.py -i /path/to/data/dir -o /path/to/output/dir --train_ratio 0.8
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")
sys.path.append("../..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))

from utility.logger import CLGLogger

# 配置日志
clglogger = CLGLogger().configure_logger().logger

class DatasetSplitter:
    """数据集划分工具"""
    def __init__(
        self,
        data_dir,
        output_dir,
        train_ratio=0.8,
        random_seed=42,
        stratified=False
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.random_seed = random_seed
        self.stratified = stratified
        
        # 设置随机种子，确保结果可重复
        random.seed(self.random_seed)
        
        # 创建输出目录
        self._create_output_dir()
    
    def _create_output_dir(self):
        """创建输出目录"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            clglogger.info(f"创建输出目录: {self.output_dir}")
    
    def scan_files(self):
        """扫描已处理完成的文件"""
        clglogger.info(f"扫描数据目录: {self.data_dir}")
        
        # 检查json和vec目录
        json_dir = self.data_dir / "json"
        vec_dir = self.data_dir / "vec"
        
        if not json_dir.exists() or not vec_dir.exists():
            clglogger.error(f"数据目录结构不完整，缺少json或vec子目录")
            return []
        
        # 获取所有json文件
        json_files = list(json_dir.glob("*.json"))
        clglogger.info(f"找到 {len(json_files)} 个JSON文件")
        
        # 筛选已处理完成的文件(存在对应vec文件)
        valid_files = []
        for json_file in tqdm(json_files, desc="检查文件"):
            file_id = json_file.stem
            vec_file = vec_dir / f"{file_id}.pth"
            if vec_file.exists():
                valid_files.append(file_id)
        
        clglogger.info(f"找到 {len(valid_files)} 个已处理完成的文件")
        return valid_files
    
    def split_dataset(self):
        """划分数据集"""
        # 扫描有效文件
        valid_files = self.scan_files()
        if not valid_files:
            clglogger.error("没有找到有效文件，无法划分数据集")
            return False
        
        # 随机打乱文件列表
        random.shuffle(valid_files)
        
        # 计算训练集大小
        train_size = int(len(valid_files) * self.train_ratio)
        
        # 划分训练集和测试集
        train_files = valid_files[:train_size]
        test_files = valid_files[train_size:]
        
        clglogger.info(f"训练集: {len(train_files)} 文件, 测试集: {len(test_files)} 文件")
        
        # 创建输出数据
        train_data = {
            "total": len(train_files),
            "prefix": train_files[0][:2] if train_files else "",
            "files": train_files
        }
        
        test_data = {
            "total": len(test_files),
            "prefix": test_files[0][:2] if test_files else "",
            "files": test_files
        }
        
        # 保存到文件
        train_path = self.output_dir / "train.json"
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        test_path = self.output_dir / "test.json"
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        clglogger.info(f"数据集划分完成，已保存到: {train_path} 和 {test_path}")
        return True
    
    def split_stratified(self):
        """按模型复杂度分层抽样划分数据集"""
        # 扫描有效文件
        valid_files = self.scan_files()
        if not valid_files:
            clglogger.error("没有找到有效文件，无法划分数据集")
            return False
        
        # 读取模型信息，判断复杂度
        clglogger.info("分析模型复杂度...")
        
        # 分组：简单、中等、复杂
        simple_models = []
        moderate_models = []
        complex_models = []
        
        for file_id in tqdm(valid_files, desc="分析模型"):
            try:
                json_file = self.data_dir / "json" / f"{file_id}.json"
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # 根据操作数量判断复杂度
                if "sequence" in data:
                    num_operations = len([op for op in data["sequence"] if op["type"] in ["ExtrudeFeature", "RevolveFeature"]])
                    
                    if num_operations <= 3:
                        simple_models.append(file_id)
                    elif num_operations <= 8:
                        moderate_models.append(file_id)
                    else:
                        complex_models.append(file_id)
                else:
                    # 如果无法判断复杂度，默认为中等
                    moderate_models.append(file_id)
                    
            except Exception as e:
                clglogger.warning(f"分析文件 {file_id} 时出错: {e}")
                # 出错时默认为中等复杂度
                moderate_models.append(file_id)
        
        # 输出各复杂度模型数量
        clglogger.info(f"简单模型: {len(simple_models)} 个")
        clglogger.info(f"中等模型: {len(moderate_models)} 个")
        clglogger.info(f"复杂模型: {len(complex_models)} 个")
        
        # 随机打乱各组文件
        random.shuffle(simple_models)
        random.shuffle(moderate_models)
        random.shuffle(complex_models)
        
        # 按比例划分各组
        train_simple = simple_models[:int(len(simple_models) * self.train_ratio)]
        test_simple = simple_models[int(len(simple_models) * self.train_ratio):]
        
        train_moderate = moderate_models[:int(len(moderate_models) * self.train_ratio)]
        test_moderate = moderate_models[int(len(moderate_models) * self.train_ratio):]
        
        train_complex = complex_models[:int(len(complex_models) * self.train_ratio)]
        test_complex = complex_models[int(len(complex_models) * self.train_ratio):]
        
        # 合并训练集和测试集
        train_files = train_simple + train_moderate + train_complex
        test_files = test_simple + test_moderate + test_complex
        
        # 再次随机打乱
        random.shuffle(train_files)
        random.shuffle(test_files)
        
        clglogger.info(f"训练集: {len(train_files)} 文件, 测试集: {len(test_files)} 文件")
        
        # 创建输出数据
        train_data = {
            "total": len(train_files),
            "prefix": train_files[0][:2] if train_files else "",
            "files": train_files,
            "distribution": {
                "simple": len(train_simple),
                "moderate": len(train_moderate),
                "complex": len(train_complex)
            }
        }
        
        test_data = {
            "total": len(test_files),
            "prefix": test_files[0][:2] if test_files else "",
            "files": test_files,
            "distribution": {
                "simple": len(test_simple),
                "moderate": len(test_moderate),
                "complex": len(test_complex)
            }
        }
        
        # 保存到文件
        train_path = self.output_dir / "train.json"
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        test_path = self.output_dir / "test.json"
        with open(test_path, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        clglogger.info(f"数据集分层抽样划分完成，已保存到: {train_path} 和 {test_path}")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="数据集划分工具: 将处理后的数据集随机划分为训练集和测试集"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="数据目录，包含json和vec子目录"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="输出目录，将保存train.json和test.json"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8,
        help="训练集比例，默认为0.8"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机数种子，默认为42"
    )
    parser.add_argument(
        "--stratified", action="store_true",
        help="是否按模型复杂度进行分层抽样划分"
    )
    args = parser.parse_args()
    
    # 配置日志文件
    log_file = os.path.join(args.output, "dataset_split_log.txt")
    clglogger.add(log_file, rotation="100 MB")
    
    clglogger.info("启动数据集划分工具")
    clglogger.info(f"数据目录: {args.input}")
    clglogger.info(f"输出目录: {args.output}")
    clglogger.info(f"训练集比例: {args.train_ratio}")
    clglogger.info(f"随机数种子: {args.seed}")
    clglogger.info(f"分层抽样: {'是' if args.stratified else '否'}")
    
    # 创建划分器并执行划分
    splitter = DatasetSplitter(
        data_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        stratified=args.stratified
    )
    
    if args.stratified:
        success = splitter.split_stratified()
    else:
        success = splitter.split_dataset()
    
    if success:
        clglogger.info("数据集划分成功")
    else:
        clglogger.error("数据集划分失败")


if __name__ == "__main__":
    main()