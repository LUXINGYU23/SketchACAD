import os, sys
from typing import Any

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from utility.logger import CLGLogger
from utility.macro import *
from utility.utils import dequantize_verts, int_round, quantize, float_round
from loguru import logger
from sequence.sketch.coord_system import CoordinateSystem
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
clglogger = CLGLogger().configure_logger().logger


class RevolveSequence(object):
    """
    旋转操作序列类，表示CAD模型中的旋转操作

    该类用于表示绕轴旋转草图以生成3D模型的操作
    """

    def __init__(self, metadata: dict, coordsystem: CoordinateSystem = None):
        """
        初始化旋转序列对象

        Args:
            metadata (dict): 包含旋转信息的元数据字典
            coordsystem (CoordinateSystem, optional): 坐标系统。默认为None
        """
        self.metadata = metadata
        self.quantized_metadata = {}
        self.is_numerical = False
        self.coordsystem = coordsystem


    @property
    def token_index(self):
        """返回旋转操作的结束标记索引"""
        return END_TOKEN.index("END_REVOLVE")

    @staticmethod
    def from_dict(all_stat, uid):
        """
        从字典创建旋转序列对象

        Args:
            all_stat (dict): 包含所有CAD数据的字典
            uid (str): 旋转实体的唯一标识符

        Returns:
            RevolveSequence: 从字典创建的旋转序列对象
        """
        metadata = {}
        revolve_entity = all_stat["entities"][uid]  # 只获取旋转实体

        # 验证旋转实体类型
        assert revolve_entity["type"] == "RevolveFeature", clglogger.critical(
            f"uid {uid} 不是旋转操作"
        )

        # 保存配置文件的uids
        metadata["profile_uids"] = [
            [profile["sketch"], profile["profile"]]
            for profile in revolve_entity["profiles"]
        ]
        # 提取旋转角度
        metadata["angle"] = revolve_entity.get("extent_one", {}).get("angle", {}).get("value", 360.0)

        # 提取旋转轴
        axis = revolve_entity.get("axis_line", {})
        metadata["axis_start"] = np.array([
            axis.get("start_point", {}).get("x", 0.0),
            axis.get("start_point", {}).get("y", 0.0),
            axis.get("start_point", {}).get("z", 0.0)
        ])
        metadata["axis_end"] = np.array([
            axis.get("end_point", {}).get("x", 0.0),
            axis.get("end_point", {}).get("y", 0.0),
            axis.get("end_point", {}).get("z", 0.0)
        ])

        # 提取布尔操作类型
        metadata["boolean"] = REVOLVE_OPERATIONS.index(revolve_entity["operation"])
        
        # 标记是否为对称旋转
        metadata["is_symmetric"] = revolve_entity.get("is_symmetric", False)

        return RevolveSequence(metadata)

    @staticmethod
    def from_minimal_json(revolve_entity):
        """从最小JSON格式创建旋转序列对象"""
        metadata = {
            "angle": revolve_entity['angle'],
            "axis_start": np.array(revolve_entity['axis_start']),
            "axis_end": np.array(revolve_entity['axis_end']),
            "sketch_size": revolve_entity['sketch_scale'],
            "boolean": REVOLVE_OPERATIONS.index(revolve_entity["operation"]),
            "is_symmetric": revolve_entity.get("is_symmetric", False)
        }
    
        return RevolveSequence(metadata)
    
    def add_info(self, key, val):
        """添加额外信息到元数据中"""
        self.metadata[key] = val
    
    def transform(self, coortranslate, translate, scale, merge_extent=False):
        """转换旋转操作的参数"""
        # 确保translate向量是3D的
        if not isinstance(translate, int) and not isinstance(translate, float):
            if translate.shape[0] != 3:
                translate = np.concatenate([translate, np.zeros(3 - len(translate))])

        # # 平移缩放旋转轴起点和终点
        self.metadata["axis_start"] = (self.metadata["axis_start"] + translate )* scale
        self.metadata["axis_end"] = (self.metadata["axis_end"] + translate) * scale
        # 平移缩放旋转轴起点和终点
        # #打印原始值
        # self.metadata["axis_start"] = (self.metadata["axis_start"])* scale
        # self.metadata["axis_end"] = (self.metadata["axis_end"]) * scale       
        # # # 再应用平移
        # if translate is not None and not isinstance(translate, (int, float)):
        #     self.metadata["axis_start"] = self.metadata["axis_start"] + translate
        #     self.metadata["axis_end"] = self.metadata["axis_end"] + translate

        # 确保旋转轴方向一致性
        axis_vec = self.metadata["axis_end"] - self.metadata["axis_start"]
        axis_len = np.linalg.norm(axis_vec)
        if axis_len > 0:
            # 保持单位向量方向
            axis_vec = axis_vec / axis_len

        # 坐标系变换
        self.coordsystem.transform(coortranslate, scale)
            
        # 草图尺寸
        if "sketch_size" in self.metadata:
            self.metadata["sketch_size"] *= scale

    def __repr__(self) -> str:
        """返回旋转序列的字符串表示"""
        metadata_str = ", ".join(
            f"{key}: {value}" for key, value in self.metadata.items()
        )

        repr_str = f'{self.__class__.__name__}: ({metadata_str}) Euler Angles {self.coordsystem.metadata["euler_angles"]}'

        return repr_str

    def get_profile_uids(self):
        """获取配置文件的唯一标识符列表"""
        return self.metadata.get("profile_uids", [])

    def get_boolean(self):
        """获取布尔操作类型"""
        return self.metadata["boolean"]

    def numericalize(self, bit):
        """
        将旋转操作参数量化为离散值
        
        Args:
            bit: 量化位数
        """
        self.is_numerical = True
        size = 2**bit - 1
        # clglogger.debug(
        #     f"旋转操作原始值: {self.metadata}, 坐标系: {self.coordsystem.metadata}"
        # )
        # 量化角度 (通常范围是0-360度)
        angle_value = self.metadata["angle"] / 360.0 * size
        self.metadata["angle"] = int_round(
            [np.clip(angle_value, 0, size)]
        )[0]
        
        # 量化旋转轴起点和终点
        self.metadata["axis_start"] = int_round(
            ((self.metadata["axis_start"] + 1.0) / 2 * size).clip(min=0, max=size)
        )
        self.metadata["axis_end"] = int_round(
            ((self.metadata["axis_end"] + 1.0) / 2 * size).clip(min=0, max=size)
        )
        
        # 量化布尔操作类型和对称标志
        self.metadata["boolean"] = int(self.metadata["boolean"])
        self.metadata["is_symmetric"] = int(self.metadata.get("is_symmetric", False))
        
        # 量化草图尺寸
        if "sketch_size" in self.metadata:
            self.metadata["sketch_size"] = int_round(
                [(self.metadata["sketch_size"] / 2 * (size + 1)).clip(min=0, max=size)]
            )[0]
            
            # 防止草图尺寸被量化为0
            if self.metadata["sketch_size"] == 0:
                self.metadata["sketch_size"] = 1
        
        # 量化坐标系
        self.coordsystem.numericalize(bit)
        # clglogger.debug(
        #     f"旋转操作量化值: {self.metadata}, 坐标系: {self.coordsystem.metadata}"
        # )
        return self

    def denumericalize(self, bit, post_processing=True):
        """
        将量化的旋转操作参数还原为连续值
        
        Args:
            bit: 量化位数
            post_processing: 是否进行后处理
        """
        self.is_numerical = False
        size = 2**bit
        # clglogger.debug(
        #     f"旋转操作量化值: {self.metadata}"
        # )
        # 还原角度
        self.metadata["angle"] = (self.metadata["angle"] / (2**bit - 1)) * 360.0
        
        # 还原旋转轴点
        self.metadata["axis_start"] = self.metadata["axis_start"] / size * 2 - 1.0
        self.metadata["axis_end"] = self.metadata["axis_end"] / size * 2 - 1.0
        
        # 还原坐标系
        self.coordsystem.denumericalize(bit)
        
        # 还原草图尺寸
        if "sketch_size" in self.metadata:
            self.metadata["sketch_size"] = self.metadata["sketch_size"] / size * 2
        # clglogger.debug(
        #     f"旋转操作还原值: {self.metadata}, 坐标系: {self.coordsystem.metadata}"
        # )
        return self

    def to_vec(self):
        """
        生成旋转操作的向量表示，遵循与拉伸操作相同的编码规则
        
        Returns:
            list: 旋转操作的向量表示，包含以下元素:
                - 角度
                - 轴起点 (x,y,z)
                - 轴终点 (x,y,z) 
                - 坐标系原点 (x,y,z)
                - 欧拉角 (theta,phi,gamma)
                - 布尔操作类型
                - 对称标志
                - 草图尺寸
                - END_REVOLVE标记
        """
        assert self.is_numerical is True, clglogger.error("值尚未量化")
        # #打印原始值
        # clglogger.debug(f"旋转操作原始值: {self.metadata}")
        # 旋转角度
        angle_vec = [self.metadata["angle"] + END_PAD + BOOLEAN_PAD, 0]
        
        # 轴起点和终点
        axis_start = [[i, 0] for i in self.metadata["axis_start"] + END_PAD + BOOLEAN_PAD]
        axis_end = [[i, 0] for i in self.metadata["axis_end"] + END_PAD + BOOLEAN_PAD]
        
        # 坐标系参数
        origin = [[i, 0] for i in self.coordsystem.metadata["origin"] + END_PAD + BOOLEAN_PAD]
        euler_angles = [[i, 0] for i in self.coordsystem.metadata["euler_angles"] + END_PAD + BOOLEAN_PAD]
        
        # 布尔操作类型和对称标志
        boolean = [self.metadata["boolean"] + END_PAD, 0]
        is_symmetric = [self.metadata.get("is_symmetric", 0) + END_PAD + BOOLEAN_PAD, 0]
        
        # 草图尺寸
        sketch_size = [0 + END_PAD + BOOLEAN_PAD, 0]
        if "sketch_size" in self.metadata:
            sketch_size = [self.metadata["sketch_size"] + END_PAD + BOOLEAN_PAD, 0]
        
        # 结束标记
        token = [self.token_index, 0]
        
        # 组合所有参数为向量
        vec = (
            [angle_vec]
            + axis_start
            + axis_end
            + origin
            + euler_angles
            + [boolean]
            + [is_symmetric]
            + [sketch_size]
            + [token]
        )
        # (angle,axis_start_xyz,axis_end_xyz,ox,oy,oz,theta,phi,gamma,b,is_sym,s,END_REVOLVE) -> 17
        # clglogger.debug(
        #     f"旋转操作向量表示: {vec}, 长度: {len(vec)}"
        # )
        return vec

    @staticmethod
    def from_vec(vec, bit, post_processing):
        """
        从向量表示创建旋转序列对象
        
        Args:
            vec: 包含旋转操作数据的向量
            bit: 量化位数
            post_processing: 是否进行后处理
            
        Returns:
            RevolveSequence: 从向量创建的旋转序列对象
        """
        # 验证向量长度
        expected_len = ONE_REV_SEQ_LENGTH
        if len(vec) < expected_len:
            clglogger.error(f"旋转操作向量长度不足: {len(vec)}/{expected_len}")
            # 填充默认值到正确长度
            padding = [[0, 0]] * (expected_len - len(vec))
            vec = np.concatenate([vec, padding])
        elif len(vec) > expected_len and vec[-1][0] == END_TOKEN.index("END_REVOLVE"):
            # 确保结束标记存在且处于正确位置
            vec = vec[-(expected_len):]
        
        # 移除结束标记（如果存在）
        if vec[-1][0] == END_TOKEN.index("END_REVOLVE"):
            vec = vec[:-1]
        
        metadata = {}
        
        # 解析各个参数
        # 1. 角度
        metadata["angle"] = vec[0][0] - (END_PAD + BOOLEAN_PAD)
        
        # 2. 旋转轴起点 (3个向量元素)
        axis_start = vec[1:4]
        axis_start = np.array([item[0] for item in axis_start]) - (END_PAD + BOOLEAN_PAD)
        metadata["axis_start"] = axis_start
        
        # 3. 旋转轴终点 (3个向量元素)
        axis_end = vec[4:7]
        axis_end = np.array([item[0] for item in axis_end]) - (END_PAD + BOOLEAN_PAD)
        metadata["axis_end"] = axis_end
        
        # 4. 从向量中提取坐标系 (6个向量元素: 原点xyz + 欧拉角xyz)
        coord_data = []
        for i in range(7, 13):
            coord_data.append(vec[i][0] - (END_PAD + BOOLEAN_PAD))
        coordsystem = CoordinateSystem.from_vec(
            np.array(coord_data), bit, post_processing
        )
        
        # 5. 布尔操作类型
        metadata["boolean"] = vec[13][0] - END_PAD
        
        # 6. 是否对称标志
        metadata["is_symmetric"] = vec[14][0] - (END_PAD + BOOLEAN_PAD)
        
        # 7. 草图尺寸
        metadata["sketch_size"] = vec[15][0] - (END_PAD + BOOLEAN_PAD)
        
        # 创建旋转序列对象并保存量化后的元数据
        rev = RevolveSequence(metadata=metadata, coordsystem=coordsystem)
        rev.quantized_metadata = metadata.copy()
        return rev

    def _json(self):
        """
        生成旋转操作的JSON表示
        
        Returns:
            dict: 包含旋转操作数据的字典
        """
        revolve_json = {
            "angle": float(float_round(self.metadata["angle"])),
            "axis_start": [float(float_round(x)) for x in self.metadata["axis_start"]],
            "axis_end": [float(float_round(x)) for x in self.metadata["axis_end"]],
            "sketch_scale": float(float_round(self.metadata.get("sketch_size", 1.0))),
            "operation": REVOLVE_OPERATIONS[self.metadata["boolean"]],
            "is_symmetric": bool(self.metadata.get("is_symmetric", False)),
        }

        return revolve_json
    def visualize_revolve_setup(self, sketch_face, axis_start_array, axis_end_array, angle):
        """
        可视化旋转操作的草图和旋转轴，显示详细参数信息
        
        Args:
            sketch_face: 要旋转的草图面
            axis_start_array: 旋转轴起点坐标
            axis_end_array: 旋转轴终点坐标
            angle: 旋转角度
        """
        try:
            from OCC.Display.SimpleGui import init_display
            from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1
            from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
            from OCC.Core.AIS import AIS_Line, AIS_Shape
            from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
            from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
            from utility.utils import create_point_from_array
            import math
            
            # 初始化显示
            display, start_display, add_menu, add_function_to_menu = init_display()
            
            # 显示草图面
            ais_shape = AIS_Shape(sketch_face)
            ais_shape.SetColor(Quantity_Color(0.8, 0.8, 0.8, Quantity_TOC_RGB))  # 浅灰色
            ais_shape.SetTransparency(0.3)  # 半透明
            display.Context.Display(ais_shape, True)
            
            # 创建并显示旋转轴
            axis_start_pnt = create_point_from_array(axis_start_array)
            axis_end_pnt = create_point_from_array(axis_end_array)
            
            # 延长轴以便更好地可视化
            axis_vec = gp_Vec(axis_start_pnt, axis_end_pnt)
            axis_len = axis_vec.Magnitude()
            if axis_len < 0.001:  # 轴太短，使用默认轴
                axis_end_pnt = gp_Pnt(axis_start_pnt.X(), axis_start_pnt.Y(), axis_start_pnt.Z() + 1.0)
                axis_vec = gp_Vec(axis_start_pnt, axis_end_pnt)
                axis_len = 1.0
                
            # 延长轴的两端
            extended_factor = max(5.0, 10.0 / axis_len) if axis_len > 0 else 5.0
            extended_start = gp_Pnt(
                axis_start_pnt.X() - axis_vec.X() * extended_factor,
                axis_start_pnt.Y() - axis_vec.Y() * extended_factor,
                axis_start_pnt.Z() - axis_vec.Z() * extended_factor
            )
            extended_end = gp_Pnt(
                axis_end_pnt.X() + axis_vec.X() * extended_factor,
                axis_end_pnt.Y() + axis_vec.Y() * extended_factor,
                axis_end_pnt.Z() + axis_vec.Z() * extended_factor
            )
            
            # 创建轴线
            axis_edge = BRepBuilderAPI_MakeEdge(extended_start, extended_end).Edge()
            ais_axis = AIS_Shape(axis_edge)
            ais_axis.SetColor(Quantity_Color(1, 0, 0, Quantity_TOC_RGB))  # 红色
            ais_axis.SetWidth(3)  # 较粗的线
            display.Context.Display(ais_axis, True)
            
            # 显示轴起点和终点
            display.DisplayShape(axis_start_pnt, color="green", update=True)
            display.DisplayShape(axis_end_pnt, color="blue", update=True)
            
            # 获取并显示草图面信息
            surf = BRepAdaptor_Surface(sketch_face).Plane()
            face_location = surf.Location()
            face_normal = surf.Axis().Direction()
            
            # 显示草图面法向量
            normal_end = gp_Pnt(
                face_location.X() + face_normal.X(),
                face_location.Y() + face_normal.Y(),
                face_location.Z() + face_normal.Z()
            )
            normal_edge = BRepBuilderAPI_MakeEdge(face_location, normal_end).Edge()
            ais_normal = AIS_Shape(normal_edge)
            ais_normal.SetColor(Quantity_Color(0, 0.7, 0, Quantity_TOC_RGB))  # 绿色
            ais_normal.SetWidth(2)
            display.Context.Display(ais_normal, True)
            
            # 显示坐标系（可选）
            display.DisplayMessage(gp_Pnt(0, 0, 0), "Origin", update=True)
            
            # 添加图例说明
            display.DisplayMessage(gp_Pnt(-1, -1, -1), f"旋转角度: {angle}°", update=True)
            
            # 设置视图
            display.View_Iso()
            display.FitAll()
            
            # 显示窗口
            clglogger.info("显示旋转操作调试视图，关闭窗口继续执行程序")
            start_display()
            
        except Exception as e:
            clglogger.error(f"可视化旋转操作失败: {e}")
    def build_body(self, sketch_face, normal):
        """创建旋转体，带有增强的异常处理"""
        from OCC.Core.gp import gp_Ax1, gp_Pnt, gp_Dir, gp_Vec
        from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeRevol
        from OCC.Core.BRepCheck import BRepCheck_Analyzer
        from OCC.Core.ShapeFix import ShapeFix_Shape
        from utility.utils import create_point_from_array
        import math

        try:
            # 检查输入数据
            if sketch_face is None:
                clglogger.error("无法创建旋转体：面为空")
                raise ValueError("面为空")
            # 获取旋转轴点并转换为草图的坐标系
            axis_start_array = self.metadata["axis_start"] 
            axis_end_array = self.metadata["axis_end"]
            # 使用coordsystem将轴点转换到草图实际的坐标系
            if self.coordsystem:
                axis_start_array = self.coordsystem.rotate_vec(axis_start_array)
                axis_end_array = self.coordsystem.rotate_vec(axis_end_array)
            # 创建OCC点对象
            axis_start_pnt = create_point_from_array(axis_start_array)
            axis_end_pnt = create_point_from_array(axis_end_array)
            axis = gp_Ax1(axis_start_pnt, gp_Dir(gp_Vec(axis_start_pnt, axis_end_pnt)))

            # 处理角度
            angle = float(self.metadata["angle"])
            if angle <= 0 or angle > 360:
                angle = 360.0 if angle <= 0 else min(angle, 360.0)
                self.metadata["angle"] = angle
            angle_rad = angle * math.pi / 180.0
            axis_dir = gp_Dir(gp_Vec(axis_start_pnt, axis_end_pnt))
            surf = BRepAdaptor_Surface(sketch_face).Plane()
            face_normal = surf.Axis().Direction()
            # 检查法向量是否与旋转轴垂直
            if abs(face_normal.X() * axis_dir.X() + face_normal.Y() * axis_dir.Y() + face_normal.Z() * axis_dir.Z()) > 1e-5:
                clglogger.error("旋转轴与草图法向量不垂直")
            # self.visualize_revolve_setup(sketch_face, axis_start_array, axis_end_array, angle)
            revolve_maker = BRepPrimAPI_MakeRevol(sketch_face, axis, angle_rad)
            revolve_maker.Build()

            if not revolve_maker.IsDone():
                raise Exception("旋转操作构建失败")

            return revolve_maker.Shape()

        except Exception as e:
            clglogger.error(f"无法创建旋转体: {e}")
            raise Exception(f"旋转体创建失败: {e}")
        