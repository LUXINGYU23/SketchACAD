import os
import sys

# 获取当前脚本路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))  # 指向SketchACAD根目录
sys.path.append(PROJECT_ROOT)

# Adding Python Path
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

from tqdm import tqdm
from src.dataset.CadSeqProc.utility.decorator import measure_performance
from src.dataset.CadSeqProc.utility.logger import CLGLogger
from src.dataset.CadSeqProc.utility.macro import *
from src.dataset.CadSeqProc.utility.utils import get_files_scan
from src.dataset.CadSeqProc.cad_sequence import CADSequence
import argparse
import multiprocessing
import json
import platform
import warnings
import gc

warnings.filterwarnings("ignore")

# 根据平台选择合适的多进程启动方法
if platform.system() != "Windows":
    # Unix/Linux/MacOS使用forkserver
    multiprocessing.set_start_method("forkserver", force=True)
else:
    # Windows使用spawn
    multiprocessing.set_start_method("spawn", force=True)

clglogger = CLGLogger().configure_logger().logger
# ---------------------------------------------------------------------------- #
#                           DeepCAD Json to Brep/Mesh                          #
# ---------------------------------------------------------------------------- #


@measure_performance
def main():
    """
    Parse Json into sketch and extrusion sequence tokens
    """
    parser = argparse.ArgumentParser(
        description="Creating Sketch and Extrusion Sequence"
    )
    parser.add_argument(
        "-p", "--input_dir", help="Input Directory for DeepCAD Json dataset", type=str
    )
    parser.add_argument(
        "--split_json", help="Input Directory for DeepCAD split json", type=str
    )
    parser.add_argument(
        "--single_json", help="Path to a single JSON file to process", type=str
    )
    parser.add_argument("-o", "--output_dir", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepcad",
        choices=["deepcad", "fusion360", "cad_parser"],
    )
    parser.add_argument("--bit", type=int, default=N_BIT)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--save_type", type=str, default="step")
    args = parser.parse_args()

    clglogger.info(f"Running Task with {args.max_workers} workers.")

    if args.single_json:
        process_single_json(args.single_json, args, clglogger)
    elif args.dataset == "deepcad":
        process_deepcad(args, clglogger)
    elif args.dataset == "fusion360":
        process_fusion360(args, clglogger)


def process_single_json(json_path, args, clglogger):
    """
    Processes a single JSON file.

    Args:
        json_path (str): The path to the JSON file.
    """
    clglogger.info(f"Processing single JSON file: {json_path}")
    if not os.path.exists(json_path):
        clglogger.error(f"JSON file does not exist: {json_path}")
        return

    try:
        process_json(json_path, args)
        clglogger.success(f"Successfully processed single JSON file: {json_path}")
    except Exception as e:
        clglogger.error(f"Error processing single JSON file: {json_path}. Error: {e}")


def process_fusion360(args, clglogger):
    """
    Processes the Fusion360 dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    all_files = get_files_scan(args.input_dir, max_workers=args.max_workers)
    all_json_files = [
        file
        for file in all_files
        if file.endswith(".json") and file.split("/")[-2] == "json"
    ]
    clglogger.info(
        f"Preprocessing {len(all_json_files)} Fusion360 dataset using Method 1."
    )
    process_all_jsons(all_json_files, args, clglogger)
    clglogger.success(f"Task Complete")


def process_deepcad(args, clglogger):
    """
    Processes the DeepCAD dataset

    Args:
        args (argparse.Namespace): The arguments.
    """
    if args.split_json is None:
        clglogger.warning("No split_json provided. Processing all JSON files in input directory.")
        
        # 规范化输入目录路径
        input_dir = os.path.normpath(args.input_dir)
        
        # 检查目录是否存在
        if not os.path.exists(input_dir):
            clglogger.error(f"Input directory does not exist: {input_dir}")
            return
        
        # 直接使用os.walk来扫描JSON文件
        all_json_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json'):
                    # 使用os.path.join确保路径分隔符正确
                    all_json_files.append(os.path.join(root, file))
        
        if not all_json_files:
            clglogger.error(f"No JSON files found in directory: {input_dir}")
            return
            
        clglogger.info(f"Found {len(all_json_files)} JSON files.")
        # --------------------------------- Method 1 --------------------------------- #
        process_all_jsons(all_json_files, args, clglogger)
        return
        
    # 如果提供了split_json，按原来的方式处理
    with open(args.split_json, "r") as f:
        data = json.load(f)

    all_json_files = (
        data["train"][:82000]
        + data["train"][84000:]
        + data["test"]
        + data["validation"]
    )

    # --------------------------------- Method 1 --------------------------------- #
    process_all_jsons(all_json_files, args, clglogger)

    extra_json_files = [
        os.path.join(args.input_dir, uid + ".json")
        for uid in data["train"][82000:84000]
    ]

    # --------------------------------- Method 2 --------------------------------- #
    clglogger.info(f"Preprocessing {len(extra_json_files)} using Method 2")
    for json_path in tqdm(all_json_files):
        try:
            process_json(json_path, args)
        except:
            pass

    clglogger.success(f"Task Complete")


def process_all_jsons(all_json_files, args, clglogger):
    """
    Processes all the JSON files in the list and saves the CAD models

    Args:
        all_json_files (list): A list of JSON files.
    """
    # Create a ProcessPoolExecutor
    executor = ThreadPoolExecutor(max_workers=args.max_workers)

    # Submit tasks to the executor
    futures = [
        executor.submit(process_json, json_path, args)
        for json_path in tqdm(all_json_files, desc="Submitting Tasks")
    ]

    # Wait for the tasks to complete
    for future in tqdm(as_completed(futures), desc="Processing Files"):
        future.result()
    
    clglogger.success(f"Method 1 Complete")


def process_json(json_path, args):
    """
    Processes a JSON file and saves the whole CAD model as well as intermediate ones

    Args:
        json_path (str): The path to the JSON file.
    """
    try:
        # 规范化JSON文件路径
        json_path = os.path.normpath(json_path)
        
        if args.dataset == "deepcad":
            # 使用os.path.basename和os.path.dirname提取目录和文件名
            file_name = os.path.basename(json_path)
            parent_dir = os.path.basename(os.path.dirname(json_path))
            uid = os.path.join(parent_dir, file_name.replace('.json', ''))
        elif args.dataset == "fusion360":
            parts = json_path.split(os.sep)
            uid = os.path.join(parts[-4], parts[-3])
            
        name = os.path.basename(uid)  # 00003121

        # Open the JSON file.
        with open(json_path, "r") as f:
            data = json.load(f)

        # cad_seq = CADSequence.json_to_NormalizedCAD(data=data, bit=args.bit)
        cad_seq = CADSequence.from_dict(all_stat=data)

        # ------------------------- Save the final cad Model ------------------------- #
        cad_seq.save_stp(
            filename=name + "_final",
            output_folder=os.path.join(args.output_dir, uid, args.save_type),
            type=args.save_type,
        )
        #TODO： 还没修复顺序问题这里
        # # ------------------------ Save the intermediate models ----------------------- #
        # num_intermediate_model = len(cad_seq.sketch_seq)
        # if num_intermediate_model > 1:
        #     # 确定每个草图对应的操作类型(拉伸或旋转)
        #     extrude_count = len(cad_seq.extrude_seq)
        #     revolve_count = len(cad_seq.revolve_seq)
            
        #     # 处理拉伸操作对应的中间模型
        #     for i in range(min(extrude_count, num_intermediate_model)):
        #         new_cad_seq = CADSequence(
        #             sketch_seq=[cad_seq.sketch_seq[i]],
        #             extrude_seq=[cad_seq.extrude_seq[i]],
        #         )

        #         # Make the operation as NewBodyOperation to create a solid body
        #         new_cad_seq.extrude_seq[0].metadata["boolean"] = 0
        #         new_cad_seq.save_stp(
        #             filename=name + f"_intermediate_extrude_{i+1}",
        #             output_folder=os.path.join(args.output_dir, uid, args.save_type),
        #             type=args.save_type,
        #         )
        #         del new_cad_seq
            
        #     # 处理旋转操作对应的中间模型
        #     for i in range(min(revolve_count, num_intermediate_model - extrude_count)):
        #         sketch_idx = extrude_count + i  # 旋转操作对应的草图索引
        #         if sketch_idx >= len(cad_seq.sketch_seq):
        #             # 如果草图索引超出范围，跳过
        #             #warning
        #             clglogger.warning(f"Skipped sketch index {sketch_idx} for json_path: {json_path}")
        #             continue
                    
        #         new_cad_seq = CADSequence(
        #             sketch_seq=[cad_seq.sketch_seq[sketch_idx]],
        #             revolve_seq=[cad_seq.revolve_seq[i]],
        #         )
        #         # 设置为新建体操作
        #         new_cad_seq.revolve_seq[0].metadata["boolean"] = 0
        #         new_cad_seq.save_stp(
        #             filename=name + f"_intermediate_revolve_{i+1}",
        #             output_folder=os.path.join(args.output_dir, uid, args.save_type),
        #             type=args.save_type,
        #         )
        #         del new_cad_seq
        
        gc.collect()
       
    except Exception as e:
        pass
        clglogger.error(f"Problem processing {json_path}. Error: {e}")


if __name__ == "__main__":
    main()
