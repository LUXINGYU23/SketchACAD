#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试脚本：检查Json -> Vec -> Json -> Step的转换过程
特别用于验证旋转操作(Revolve)的向量表示是否正确
"""

import os
import sys
import json
import numpy as np
import torch
import logging
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# 获取当前脚本路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 添加项目根目录到Python路径
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../.."))  # 指向SketchACAD根目录
sys.path.append(PROJECT_ROOT)

from cad_sequence import CADSequence
from utility.macro import N_BIT, EXTRUDE_FLAG_START, EXTRUDE_FLAG_RANGE, REVOLVE_FLAG_START, REVOLVE_FLAG_RANGE, PADDING_FLAG
from utility.utils import ensure_dir, get_files_scan
from utility.decorator import measure_performance
from utility.logger import CLGLogger

# 导入OpenCascade相关库，用于渲染
from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Ax1, gp_Pnt, gp_Dir, gp_Trsf
from OCC.Core.V3d import V3d_XposYposZpos, V3d_XnegYposZpos, V3d_XposYnegZpos, V3d_XposYposZneg
from OCC.Core.Quantity import Quantity_NOC_WHITE, Quantity_NOC_BLACK
from OCC.Core.AIS import AIS_Shape
import PIL.Image as Image
import io

clglogger = CLGLogger().configure_logger().logger

def render_model_views(shape_or_step_path, output_path, size=(400, 400)):
    """Render 8 views of a model from different angles and save as a single image
    
    Args:
        shape_or_step_path: TopoDS_Shape object or STEP file path
        output_path: Output image path
        size: Image size
    """
    try:
        # Check if input is a STEP file path or shape object
        from OCC.Core.TopoDS import TopoDS_Shape
        from OCC.Extend.DataExchange import read_step_file
        
        if isinstance(shape_or_step_path, str) and os.path.exists(shape_or_step_path) and shape_or_step_path.endswith(('.step', '.stp')):
            # Read shape from STEP file
            clglogger.info(f"Reading shape from STEP file: {shape_or_step_path}")
            shape = read_step_file(shape_or_step_path)
        elif isinstance(shape_or_step_path, TopoDS_Shape):
            # Already a shape object
            shape = shape_or_step_path
        else:
            clglogger.error(f"Invalid input: {shape_or_step_path}")
            return False
        
        # Create output directory
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Set up offscreen rendering
        os.environ["PYTHONOCC_OFFSCREEN_RENDERER"] = "1"
        
        # Initialize display
        display, start_display, add_menu, add_function_to_menu = init_display(
            size=size, 
        )
        
        # Configure rendering parameters
        try:
            # Get current rendering parameters
            viewer = display.Context.CurrentViewer()
            rendering_params = viewer.DefaultRenderingParams()
            
            # Try to set high quality rendering parameters
            from OCC.Core.Graphic3d import Graphic3d_RM_RAYTRACING
            rendering_params.Method = Graphic3d_RM_RAYTRACING
            rendering_params.RaytracingDepth = 3
            rendering_params.IsAntialiasingEnabled = True
            
            if hasattr(viewer, "SetRenderingParams"):
                viewer.SetRenderingParams(rendering_params)
            elif hasattr(viewer, "SetDefaultRenderingParams"):
                viewer.SetDefaultRenderingParams(rendering_params)
        except Exception as e:
            clglogger.warning(f"Could not set advanced rendering parameters: {e}, using default rendering")
        
        # Create AIS_Shape display object and set properties
        ais_shape = AIS_Shape(shape)
        
        # Set color and material
        from OCC.Core.Quantity import Quantity_Color
        from OCC.Core.Graphic3d import Graphic3d_NOM_ALUMINIUM, Graphic3d_MaterialAspect
        display.Context.SetColor(ais_shape, Quantity_Color(0.5, 0.7, 0.9, 0), False)
        material = Graphic3d_MaterialAspect(Graphic3d_NOM_ALUMINIUM)
        display.Context.SetMaterial(ais_shape, material, False)
        
        # Display shape
        display.Context.Display(ais_shape, True)
        display.FitAll()
        
        # Create composite image
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        view_configs = [
            {"name": "Front View", "view": "Front", "row": 0, "col": 0},
            {"name": "Right View", "view": "Right", "row": 0, "col": 1},
            {"name": "Back View", "view": "Back", "row": 0, "col": 2},
            {"name": "Left View", "view": "Left", "row": 1, "col": 0},
            {"name": "Top View", "view": "Top", "row": 1, "col": 1},
            {"name": "Bottom View", "view": "Bottom", "row": 1, "col": 2},
            {"name": "Isometric View", "view": "Iso", "row": 2, "col": 0},
            {"name": "Rear Isometric", "view": "RearIso", "row": 2, "col": 1}
        ]
        
        for view_config in view_configs:
            # Set view angle
            if view_config["view"] == "Front":
                display.View_Front()
            elif view_config["view"] == "Right":
                display.View_Right()
            elif view_config["view"] == "Back":
                display.View_Rear()
            elif view_config["view"] == "Left":
                display.View_Left()
            elif view_config["view"] == "Top":
                display.View_Top()
            elif view_config["view"] == "Bottom":
                display.View_Bottom()
            elif view_config["view"] == "Iso":
                display.View_Iso()
            elif view_config["view"] == "RearIso":
                # Create a custom view for rear isometric
                display.View.SetProj(V3d_XnegYposZpos)
            
            # Fit view
            display.FitAll()
            
            # Export image
            temp_img_path = os.path.join(output_dir, f"temp_view_{view_config['view']}.png")
            display.View.Dump(temp_img_path)
            
            # Read and add to matplotlib chart
            if os.path.exists(temp_img_path):
                img = plt.imread(temp_img_path)
                ax = fig.add_subplot(gs[view_config["row"], view_config["col"]])
                ax.imshow(img)
                ax.set_title(view_config["name"])
                ax.axis('off')
                # Delete temporary file
                try:
                    os.remove(temp_img_path)
                except:
                    pass
            else:
                clglogger.warning(f"Could not create view: {view_config['name']}")
        
        # Add empty plot in the last position to make it look balanced (3x3 grid with 8 views)
        ax = fig.add_subplot(gs[2, 2])
        ax.axis('off')
        
        # Save composite image
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        
        clglogger.info(f"Multi-view render saved to: {output_path}")
        
        # Clean up display
        display.Context.Remove(ais_shape, True)
        
        return True
    except Exception as e:
        clglogger.error(f"Failed to render views: {e}")
        import traceback
        traceback.print_exc()
        return False

@measure_performance
def process_json_file(json_path, output_dir, bit=N_BIT, save_step=True, save_vec=True, render_views=True):
    """处理单个JSON文件，执行完整的转换流程"""
    clglogger.info(f"处理文件: {json_path}")
    
    # 创建输出目录
    file_name = os.path.splitext(os.path.basename(json_path))[0]
    model_output_dir = os.path.join(output_dir, file_name)
    ensure_dir(model_output_dir)
    
    # 读取JSON文件
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)
    except Exception as e:
        clglogger.error(f"无法读取JSON文件 {json_path}: {e}")
        return False
    
    try:
        # 步骤1：JSON -> CADSequence
        clglogger.info("步骤1: JSON -> CADSequence")
        cad_seq = CADSequence.from_dict(json_data)
        #保存原始step文件
        original_step_path = os.path.join(model_output_dir, f"{file_name}_original.step")
        stp_cad_seq = CADSequence.json_to_NormalizedCAD(data=json_data, bit=N_BIT)
        try:
            stp_cad_seq.create_cad_model()
            stp_cad_seq.save_stp(output_folder=model_output_dir, filename=f"{file_name}_original", type="step")
            clglogger.info(f"原始CAD模型STEP文件已保存到: {original_step_path}")
            
            # 如果需要渲染视图，渲染原始模型
            if render_views:
                original_views_path = os.path.join(model_output_dir, f"{file_name}_original_views.png")
                if render_model_views(original_step_path, original_views_path):
                    clglogger.info(f"原始模型四视图已保存到: {original_views_path}")
        except Exception as e:
            clglogger.error(f"创建原始STEP文件失败: {e}")
            return False
        # 步骤2：正规化
        clglogger.info("步骤2: 正规化")
        cad_seq.normalize(bit=bit)
        # 步骤3：数值化
        clglogger.info("步骤3: 数值化")
        cad_seq.numericalize(bit=bit)
        
        # 步骤4：转换为向量
        clglogger.info("步骤4: 转换为向量")
        cad_seq.to_vec(padding=True)
        
        # 保存向量表示（可选）
        if save_vec:
            vec_output_path = os.path.join(model_output_dir, f"{file_name}_vec.npz")
            np.savez(
                vec_output_path, 
                cad_vec=cad_seq.cad_vec.numpy(), 
                flag_vec=cad_seq.flag_vec.numpy(), 
                index_vec=cad_seq.index_vec.numpy()
            )
            clglogger.info(f"向量已保存到: {vec_output_path}")
            
            # 记录向量中的旋转操作标志统计
            flags = cad_seq.flag_vec.numpy()
            revolve_flags = flags[flags == REVOLVE_FLAG_START]
            revolve_params = sum(1 for f in flags if f in REVOLVE_FLAG_RANGE)
            clglogger.info(f"向量中包含 {len(revolve_flags)} 个旋转操作标志和 {revolve_params} 个旋转参数标志")
        
        # 步骤5：从向量重建CADSequence
        clglogger.info("步骤5: 从向量重建CADSequence")
        restored_cad_seq = CADSequence.from_vec(
            cad_seq.cad_vec,
            bit=bit,
            post_processing=True,
            denumericalize=True
        )
        
        # 步骤6：生成STEP文件（可选）
        if save_step:
            clglogger.info("步骤6: 生成CAD模型和STEP文件")
            
            # 创建从向量重建的模型的STEP文件
            restored_step_path = os.path.join(model_output_dir, f"{file_name}_restored.step")
            try:
                restored_cad_seq.create_cad_model()
                restored_cad_seq.save_stp(output_folder=model_output_dir, filename=f"{file_name}_restored", type="step")
                clglogger.info(f"重建的CAD模型STEP文件已保存到: {restored_step_path}")
                
                # 如果需要渲染视图，渲染重建模型
                if render_views:
                    restored_views_path = os.path.join(model_output_dir, f"{file_name}_restored_views.png")
                    if render_model_views(restored_step_path, restored_views_path):
                        clglogger.info(f"重建模型视图已保存到: {restored_views_path}")
                    
                    # 创建原始模型和重建模型的视图对比
                    if os.path.exists(restored_views_path) and os.path.exists(os.path.join(model_output_dir, f"{file_name}_original_views.png")):
                        # 创建对比图
                        fig, axs = plt.subplots(2, 1, figsize=(10, 16))
                        
                        img1 = plt.imread(os.path.join(model_output_dir, f"{file_name}_original_views.png"))
                        img2 = plt.imread(restored_views_path)
                        
                        axs[0].imshow(img1)
                        axs[0].set_title("Original Model Views")
                        axs[0].axis('off')
                        
                        axs[1].imshow(img2)
                        axs[1].set_title("Reconstructed Model Views")
                        axs[1].axis('off')
                        
                        plt.tight_layout()
                        comparison_path = os.path.join(model_output_dir, f"{file_name}_comparison.png")
                        plt.savefig(comparison_path)
                        plt.close(fig)
                        clglogger.info(f"原始与重建模型对比图已保存到: {comparison_path}")
            except Exception as e:
                clglogger.error(f"创建重建STEP文件失败: {e}")
        
        return True
    
    except Exception as e:
        clglogger.error(f"处理文件 {json_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(input_dir, output_dir, bit=N_BIT, save_step=True, save_vec=True, render_views=True):
    """处理目录中的所有JSON文件"""
    clglogger.info(f"处理目录: {input_dir}")
    
    # 获取所有文件，然后过滤出JSON文件
    try:
        all_files = get_files_scan(input_dir)
        json_files = [f for f in all_files if f.lower().endswith('.json')]
        
        if not json_files:
            clglogger.error(f"在 {input_dir} 中未找到JSON文件")
            return
        
        clglogger.info(f"找到 {len(json_files)} 个JSON文件")
        
        # 处理每个文件
        success_count = 0
        for json_path in json_files:
            if process_json_file(json_path, output_dir, bit, save_step, save_vec, render_views):
                success_count += 1
        
        clglogger.info(f"总共处理 {len(json_files)} 个文件，成功 {success_count} 个，失败 {len(json_files) - success_count} 个")
    except Exception as e:
        clglogger.error(f"扫描目录失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    # 配置命令行参数
    parser = argparse.ArgumentParser(description='测试JSON到向量再到STEP的转换流程')
    parser.add_argument('--input', type=str, required=True, 
                        help='输入JSON文件路径或包含JSON文件的目录')
    parser.add_argument('--output', type=str, default='./test_output', 
                        help='输出目录')
    parser.add_argument('--bit', type=int, default=N_BIT, 
                        help='量化位数')
    parser.add_argument('--no-step', action='store_true', 
                        help='不生成STEP文件')
    parser.add_argument('--no-vec', action='store_true', 
                        help='不保存向量文件')
    parser.add_argument('--no-render', action='store_true',
                        help='不渲染模型视图')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_dir(args.output)
    
    # 配置日志
    log_file = os.path.join(args.output, "conversion_log.txt")
    clglogger.add(log_file, rotation="100 MB")
    
    # 处理输入
    if os.path.isfile(args.input) and args.input.endswith('.json'):
        process_json_file(args.input, args.output, args.bit, not args.no_step, not args.no_vec, not args.no_render)
    elif os.path.isdir(args.input):
        process_directory(args.input, args.output, args.bit, not args.no_step, not args.no_vec, not args.no_render)
    else:
        clglogger.error(f"无效的输入路径: {args.input}")


if __name__ == "__main__":
    main()