import os
import sys
import uuid
import json
import math
import argparse
import numpy as np
import time  # 引入 time 模块用于 ID 生成
import gc    # 引入 gc 模块用于内存管理
from concurrent.futures import ProcessPoolExecutor, as_completed # 改用 ProcessPoolExecutor
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append("..")

from utility.decorator import measure_performance
from utility.logger import CLGLogger
from utility.utils import ensure_dir

# 配置日志
clglogger = CLGLogger().configure_logger().logger

# ---------------------------------------------------------------------------- #
#                     CADParser JSON to DeepCAD JSON                           #
# ---------------------------------------------------------------------------- #

class CadParserToDeepCadConverter:
    """将CADParser格式的JSON转换为DeepCAD格式"""
    
    def __init__(self):
        """初始化转换器"""
        self.id_counter = 0
        self.sketch_mappings = {}  # CADParser ID到DeepCAD ID的映射
        self.profile_mappings = {}  # 每个草图的profile ID映射
    
    def generate_id(self, prefix: str) -> str:
        """更高效的ID生成"""
        self.id_counter += 1
        # 使用简单的计数器和时间哈希组合，避免大量UUID生成
        # 使用时间戳哈希增加随机性，取模减少长度
        rand_part = str(hash(str(time.time()) + str(self.id_counter)))[-6:]
        return f"{prefix}{rand_part}_{self.id_counter}"
        
    def _find_profile_mapping(self, sketch_id, profile_id):
        """更高效的轮廓映射查找"""
        # 1. 尝试直接匹配
        if sketch_id in self.profile_mappings and profile_id in self.profile_mappings[sketch_id]:
            return self.profile_mappings[sketch_id][profile_id], self.sketch_mappings[sketch_id]
        
        # 2. 如果没有直接匹配，并且该草图存在任何轮廓映射，使用第一个作为备选
        # 避免了成本较高的字符串比较
        if sketch_id in self.profile_mappings and self.profile_mappings[sketch_id]:
            first_deepcad_profile_id = next(iter(self.profile_mappings[sketch_id].values()))
            ##clglogger.debug(f"未找到直接映射 {profile_id}，使用第一个可用轮廓 {first_deepcad_profile_id}")
            return first_deepcad_profile_id, self.sketch_mappings[sketch_id]
            
        # 3. 如果草图本身都未映射或没有任何轮廓，则返回 None
        clglogger.warning(f"无法找到轮廓映射: sketch={sketch_id}, profile={profile_id}")
        return None, None
    
    def convert_point(self, point) -> dict:
        """转换点的表示形式
        
        Args:
            point: CADParser格式的点
            
        Returns:
            DeepCAD格式的点
        """
        if isinstance(point, dict) and "type" in point and point["type"] == "Point3D":
            return {
                "x": point.get("x", 0.0),
                "y": point.get("y", 0.0),
                "z": point.get("z", 0.0)
            }
        return point  # 如果已经是兼容格式则直接返回
    
    def _convert_transform(self, transform: dict) -> dict:
        """转换坐标系信息
        
        Args:
            transform: CADParser格式的坐标系
            
        Returns:
            DeepCAD格式的坐标系
        """
        result = {
            "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
            "x_axis": {"x": 1.0, "y": 0.0, "z": 0.0},
            "y_axis": {"x": 0.0, "y": 1.0, "z": 0.0},
            "z_axis": {"x": 0.0, "y": 0.0, "z": 1.0}
        }
        
        # 处理旋转矩阵
        if "Rotation" in transform and len(transform["Rotation"]) >= 3:
            rot = transform["Rotation"]
            result["x_axis"] = {"x": rot[0][0], "y": rot[0][1], "z": rot[0][2]}
            result["y_axis"] = {"x": rot[1][0], "y": rot[1][1], "z": rot[1][2]}
            result["z_axis"] = {"x": rot[2][0], "y": rot[2][1], "z": rot[2][2]}
        
        # 处理原点
        if "origin" in transform and len(transform["origin"]) >= 3:
            result["origin"] = {
                "x": transform["origin"][0], 
                "y": transform["origin"][1], 
                "z": transform["origin"][2]
            }
            
        return result
    
    def convert_sketch(self, cadparser_sketch: dict, entity_uuid: str) -> tuple:
        """转换草图实体
        
        Args:
            cadparser_sketch: CADParser格式的草图
            entity_uuid: 原始的UUID键
            
        Returns:
            (deepcad_sketch, sketch_id): 转换后的草图和ID
        """
        # 生成新的DeepCAD格式的ID
        sketch_id = self.generate_id("F")
        
        # 保存ID映射关系
        sketch_entity_id = cadparser_sketch.get("id", entity_uuid)
        if sketch_entity_id is None:
            # 如果没有显式的ID字段，使用其他标识符
            sketch_entity_id = cadparser_sketch.get("name", str(len(self.sketch_mappings)))
        
        self.sketch_mappings[sketch_entity_id] = sketch_id
        
        # 创建基本草图结构
        deepcad_sketch = {
            "transform": self._convert_transform(cadparser_sketch.get("transform", {})),
            "type": "Sketch",
            "name": cadparser_sketch.get("name", "Sketch"),
            "profiles": {},
            "reference_plane": {}
        }
        
        # 墛强调试输出
        ##clglogger.debug(f"处理草图: {sketch_entity_id}")
        
        # 收集所有可能的轮廓来源
        contours = cadparser_sketch.get("contours", {})
        default_profiles = cadparser_sketch.get("defaultProfiles", {})
        profiles = cadparser_sketch.get("profiles", {})
        
        ##clglogger.debug(f"发现轮廓: contours={list(contours.keys())}, defaultProfiles={list(default_profiles.keys())}, profiles={list(profiles.keys())}")
        
        # 创建映射数据结构，确保所有ID都被正确处理
        profile_map = {}
        all_processed_ids = set()
        
        # 1. 首先处理contours - 通常包含主要几何信息
        for contour_id, contour in contours.items():
            # 检测是否包含圆弧曲线来决定ID前缀
            has_arc = False
            for loop in contour.get("loops", []):
                for curve in loop.get("curves", []):
                    curve_type = curve.get("type", "")
                    if "Arc" in curve_type or "Circle" in curve_type:
                        has_arc = True
                        break
            
            profile_id = self.generate_id("K" if has_arc else "J")
            profile_map[contour_id] = profile_id
            all_processed_ids.add(contour_id)
            #打印count字典
            #clglogger.debug(f"contours字典: {contours}")
            deepcad_sketch["profiles"][profile_id] = self._convert_contour(contour)
            #clglogger.debug(f"映射contour轮廓: {contour_id} -> {profile_id}")
        
        # 2. 处理defaultProfiles - 通常包含草图的默认轮廓
        for profile_id, profile in default_profiles.items():
            if profile_id not in profile_map and profile_id not in all_processed_ids:
                # 检测是否包含圆弧曲线
                has_arc = False
                for loop in profile.get("loops", []):
                    for curve in loop.get("curves", []):
                        curve_type = curve.get("type", "")
                        if "Arc" in curve_type or "Circle" in curve_type:
                            has_arc = True
                            break
                
                new_profile_id = self.generate_id("K" if has_arc else "J")
                profile_map[profile_id] = new_profile_id
                all_processed_ids.add(profile_id)
                deepcad_sketch["profiles"][new_profile_id] = self._convert_contour(profile)
                #clglogger.debug(f"映射defaultProfile轮廓: {profile_id} -> {new_profile_id}")
        
        # 3. 最后处理profiles - 可能包含更多细节轮廓
        for profile_id, profile in profiles.items():
            if profile_id not in profile_map and profile_id not in all_processed_ids:
                # 检测是否包含圆弧曲线
                has_arc = False
                for loop in profile.get("loops", []):
                    for curve in loop.get("curves", []):
                        curve_type = curve.get("type", "")
                        if "Arc" in curve_type or "Circle" in curve_type:
                            has_arc = True
                            break
                
                new_profile_id = self.generate_id("K" if has_arc else "J")
                profile_map[profile_id] = new_profile_id
                all_processed_ids.add(profile_id)
                deepcad_sketch["profiles"][new_profile_id] = self._convert_contour(profile)
                #clglogger.debug(f"映射profile轮廓: {profile_id} -> {new_profile_id}")
        
        # 保存profile映射关系
        self.profile_mappings[sketch_entity_id] = profile_map
        #clglogger.debug(f"草图 {sketch_entity_id} 共生成 {len(profile_map)} 个轮廓映射")
        
        return deepcad_sketch, sketch_id

    def _convert_contour(self, contour: dict) -> dict:
        """转换轮廓
        
        Args:
            contour: CADParser格式的轮廓
            
        Returns:
            DeepCAD格式的轮廓
        """
        result = {"loops": []}
        for loop in contour.get("loops", []):
            # 打印loop字典
            # clglogger.debug(f"loop字典: {loop}")
            deepcad_loop = self._ensure_profile_data(loop)
            
            # 先复制原始曲线列表，避免循环中修改正在遍历的列表
            original_curves = deepcad_loop.get("profile_curves", [])[:]
            # 清空原列表
            deepcad_loop["profile_curves"] = []
            
            # 转换所有曲线
            converted_curves = []
            for curve in original_curves:
                converted_curves.append(self._convert_curve(curve))
            
            # 对曲线进行连接顺序排序
            ordered_curves = self._sort_curves_by_connectivity(converted_curves)
            
            # 使用排序后的曲线
            deepcad_loop["profile_curves"] = ordered_curves
            result["loops"].append(deepcad_loop)
            
        return result
    def _sort_curves_by_connectivity(self, curves):
        """对曲线进行排序，确保它们首尾相连
    
        Args:
            curves: 未排序的曲线列表
        
        Returns:
            排序后的曲线列表
        """
        if len(curves) <= 1:
            return curves
    
        # 创建一个新的排序列表
        sorted_curves = [curves[0]]  # 从第一条曲线开始
        remaining_curves = curves[1:]
    
        # 跟踪已处理的曲线以防止无限循环
        processed_count = 1
    
        # 不断寻找与当前末尾曲线的终点连接的下一条曲线
        while remaining_curves and processed_count < len(curves):
            last_curve = sorted_curves[-1]
            last_end_point = last_curve.get("end_point", {})
        
            # 在剩余曲线中查找起点与last_end_point匹配的曲线
            found = False
            for i, curve in enumerate(remaining_curves):
                if self._points_are_equal(curve.get("start_point", {}), last_end_point):
                    # 找到匹配的曲线
                    sorted_curves.append(curve)
                    remaining_curves.pop(i)
                    found = True
                    processed_count += 1
                    break
        
            # 如果没找到匹配的曲线，尝试检查是否需要反转某条曲线
            if not found:
                for i, curve in enumerate(remaining_curves):
                    if self._points_are_equal(curve.get("end_point", {}), last_end_point):
                        # 需要反转这条曲线
                        reversed_curve = self._reverse_curve(curve)
                        sorted_curves.append(reversed_curve)
                        remaining_curves.pop(i)
                        found = True
                        processed_count += 1
                        break
        
            # 如果仍然没有找到匹配的曲线，添加剩余曲线中的第一条
            # 这可能会导致几何不连续，但至少保证了处理所有曲线
            if not found and remaining_curves:
                clglogger.warning(f"发现不连续的曲线，可能导致几何重建错误")
                sorted_curves.append(remaining_curves[0])
                remaining_curves.pop(0)
                processed_count += 1
    
        # 检查环是否闭合（最后一条曲线的终点应该连接到第一条曲线的起点）
        if len(sorted_curves) > 1:
            last_curve = sorted_curves[-1]
            first_curve = sorted_curves[0]
            if not self._points_are_equal(last_curve.get("end_point", {}), first_curve.get("start_point", {})):
                clglogger.debug("环不闭合，尝试反转最后一条曲线")
                # 尝试反转最后一条曲线使环闭合
                if self._points_are_equal(last_curve.get("start_point", {}), first_curve.get("start_point", {})):
                    sorted_curves[-1] = self._reverse_curve(last_curve)
    
        return sorted_curves
    def _points_are_equal(self, point1, point2, tolerance=1e-9):
        """检查两个点是否在允许的公差范围内相等

        Args:
            point1: 第一个点坐标
            point2: 第二个点坐标
            tolerance: 浮点比较的容差

        Returns:
            bool: 如果点在容差范围内相等则为True
        """
        if not point1 or not point2:
            return False

        dx = point1.get("x", 0) - point2.get("x", 0)
        dy = point1.get("y", 0) - point2.get("y", 0)
        dz = point1.get("z", 0) - point2.get("z", 0)

        distance_squared = dx * dx + dy * dy + dz * dz
        return distance_squared < tolerance * tolerance
    def _reverse_curve(self, curve):
        """反转曲线的方向
        
        Args:
            curve: 原始曲线
            
        Returns:
            反转后的曲线
        """
        # 深复制曲线对象
        reversed_curve = {}
        for key, value in curve.items():
            reversed_curve[key] = value
        
        # 交换起点和终点
        reversed_curve["start_point"] = curve.get("end_point", {})
        reversed_curve["end_point"] = curve.get("start_point", {})
        
        # 对于圆弧，需要特殊处理
        if curve.get("type") == "Arc3D":
            # 交换开始角度和结束角度
            if "start_angle" in reversed_curve and "end_angle" in reversed_curve:
                reversed_curve["start_angle"], reversed_curve["end_angle"] = \
                    reversed_curve["end_angle"], reversed_curve["start_angle"]
            
            # 可能还需要调整法向量，根据具体需求决定
            if "normal" in reversed_curve and reversed_curve["normal"] is not None:
                normal = reversed_curve["normal"]
                # 反转法向量（对于2D圆弧可能不需要）
                reversed_curve["normal"] = {
                    "x": -normal.get("x", 0),
                    "y": -normal.get("y", 0),
                    "z": -normal.get("z", 1)
                }
        
        return reversed_curve
    def _convert_curve(self, curve: dict) -> dict:
        """转换曲线实体
        
        Args:
            curve: CADParser格式的曲线
            
        Returns:
            DeepCAD格式的曲线
        """
        curve_type = curve.get("type")
        curve_id = self.generate_id("K" if "Arc" in curve_type or "Circle" in curve_type else "J")
        
        result = {}
        
        if curve_type == "Line3D":
            result = {
                "type": "Line3D",
                "start_point": self.convert_point(curve.get("start_point", {})),
                "curve": curve_id,
                "end_point": self.convert_point(curve.get("end_point", {}))
            }
            
        elif curve_type == "Arc3D":
            normal_vec = curve.get("normal_vec", {})
            length = normal_vec.get("length", 1.0)
            normalized_normal = {
                "x": normal_vec.get("x", 0.0) / length,
                "y": normal_vec.get("y", 0.0) / length,
                "z": normal_vec.get("z", 1.0) / length
            }
            # Calculate vectors from center to start and end points
            center_point = curve.get("center_point", {})
            start_point = curve.get("start_point", {})
            end_point = curve.get("end_point", {})
            
            start_vec = {
                "x": start_point.get("x", 0) - center_point.get("x", 0),
                "y": start_point.get("y", 0) - center_point.get("y", 0),
                "z": start_point.get("z", 0) - center_point.get("z", 0)
            }
            end_vec = {
                "x": end_point.get("x", 0) - center_point.get("x", 0),
                "y": end_point.get("y", 0) - center_point.get("y", 0),
                "z": end_point.get("z", 0) - center_point.get("z", 0)
            }
            
            # Calculate angles in standard position (counterclockwise from positive x-axis)
            start_angle = math.atan2(start_vec["y"], start_vec["x"])
            end_angle = math.atan2(end_vec["y"], end_vec["x"])

            # Normalize angles to [0, 2π) range
            start_angle = start_angle % (2 * math.pi)
            end_angle = end_angle % (2 * math.pi)
            original_start_point = start_point
            original_end_point = end_point              
            # Ensure end_angle > start_angle for counterclockwise arc, and swap the point
            if end_angle < start_angle:
                tmp_angle = end_angle
                end_angle = start_angle
                start_angle = tmp_angle
                start_point = original_end_point
                end_point = original_start_point

            result = {
                "type": "Arc3D",
                "start_point": self.convert_point(curve.get("start_point", {})),
                "curve": curve_id,
                "end_point": self.convert_point(curve.get("end_point", {})),
                "center_point": self.convert_point(curve.get("center_point", {})),
                "radius": curve.get("radius", 0.0),
                "normal": normalized_normal,
                "start_angle": start_angle,
                "end_angle": end_angle,
                "reference_vector": {"x": 1.0, "y": 0.0, "z": 0.0}
            }
            
            
        elif curve_type == "Circle3D":
            result = {
                "type": "Circle3D",
                "center_point": self.convert_point(curve.get("center_point", {})),
                "curve": curve_id,
                "radius": curve.get("radius", 0.0),
                "normal": {
                    "x": 0.0, 
                    "y": 0.0, 
                    "z": 1.0
                }
            }
            
            # 处理法向量
            if "normal_vec" in curve:
                normal = curve["normal_vec"]
                result["normal"] = {
                    "x": normal.get("x", 0.0),
                    "y": normal.get("y", 0.0),
                    "z": normal.get("z", 1.0)
                }
            
        return result

    def convert_extrusion(self, cadparser_extrusion: dict, cadparser_data: dict, entity_uuid: str) -> tuple:
        """转换拉伸特征
        
        Args:
            cadparser_extrusion: CADParser格式的拉伸特征
            cadparser_data: CADParser格式的数据
            entity_uuid: 原始的UUID键
            
        Returns:
            (deepcad_extrusion, extrusion_id): 转换后的拉伸特征和ID
        """
        extrusion_id = self.generate_id("F")
        
        # 创建基本拉伸结构
        deepcad_extrusion = {
            "name": cadparser_extrusion.get("name", "Extrude"),
            "type": "ExtrudeFeature",
            "profiles": [],
            "extent_two": {
                "distance": {
                    "type": "ModelParameter",
                    "role": "AgainstDistance",
                    "name": "none",
                    "value": 0.0
                },
                "type": "DistanceExtentDefinition",
                "taper_angle": {
                    "type": "ModelParameter",
                    "role": "Side2TaperAngle",
                    "name": "none",
                    "value": 0.0
                }
            },
            "extent_one": {
                "distance": {
                    "type": "ModelParameter",
                    "role": "AlongDistance",
                    "name": "none",
                    "value": 0.0
                },
                "type": "DistanceExtentDefinition",
                "taper_angle": {
                    "type": "ModelParameter",
                    "role": "TaperAngle",
                    "name": "none",
                    "value": 0.0
                }
            },
            "operation": "NewBodyFeatureOperation",
            "start_extent": {
                "type": "ProfilePlaneStartDefinition"
            },
            "extent_type": "OneSideFeatureExtentType"
        }
        
        # 设置距离参数
        if "extent_one" in cadparser_extrusion and "distance" in cadparser_extrusion["extent_one"]:
            distance_info = cadparser_extrusion["extent_one"]["distance"]
            
            # 更灵活地获取前向距离值（兼容带空格和不带空格的键名）
            forward_dist = distance_info.get("forward_distance", 
                   distance_info.get("forward distance", 0.0))
            # 兼容带空格和不带空格的键名
            reverse_dist = distance_info.get("reverse_distance", 
                       distance_info.get("reverse distance", 0.0))
            # clglogger.debug(f"前向距离: {forward_dist}")
            # 更灵活地获取是否反向的标志
            is_reversed = distance_info.get("IsReversed", False)
            
            # 处理ThroughNext条件和其他特殊情况 - 兼容两种键名形式
            forward_condition = distance_info.get("forward condition")
            # clglogger.debug(f"前向条件: {forward_condition}")
            if forward_condition in ["ThroughNext"]:
                # 根据模型尺寸确定穿透距离
                bbox = cadparser_data.get("properties", {}).get("bounding_box", {})
                if bbox:
                    # 计算边界框对角线长度
                    min_pt = self.convert_point(bbox.get("min_point", {"x": 0, "y": 0, "z": 0}))
                    max_pt = self.convert_point(bbox.get("max_point", {"x": 1, "y": 1, "z": 1}))
                    
                    dx = max_pt.get("x", 1) - min_pt.get("x", 0) 
                    dy = max_pt.get("y", 1) - min_pt.get("y", 0)
                    dz = max_pt.get("z", 1) - min_pt.get("z", 0)
                    diagonal = math.sqrt(dx**2 + dy**2 + dz**2)
                    
                    # 使用对角线的2倍作为穿透距离
                    forward_dist = diagonal
                else:
                    # 如果没有边界框信息，使用一个足够大的默认值
                    forward_dist = 1000
             
            # 如果是反向拉伸
            if is_reversed and (isinstance(is_reversed, bool) or (isinstance(is_reversed, str) and is_reversed.lower() == "true")):
                # 将反向距离放在extent_two中
                deepcad_extrusion["extent_two"]["distance"]["value"] = -reverse_dist
                deepcad_extrusion["extent_one"]["distance"]["value"] = 0  # 正向为0
                
                # 设置为双向拉伸类型
                if reverse_dist > 0:
                    deepcad_extrusion["extent_type"] = "TwoSidesFeatureExtentType"
            else:
                # 正向拉伸
                deepcad_extrusion["extent_one"]["distance"]["value"] = forward_dist
    
                # 检查是否同时有反向距离
                reverse_dist = distance_info.get("reverse_distance", 
                    distance_info.get("reverse distance", 0.0))
                if reverse_dist > 0:
                    deepcad_extrusion["extent_two"]["distance"]["value"] = -reverse_dist
                    deepcad_extrusion["extent_type"] = "TwoSidesFeatureExtentType"
        
        # 设置操作类型
        if cadparser_extrusion.get("type") == "ExtrusionCut":
            deepcad_extrusion["operation"] = "CutFeatureOperation"
        elif cadparser_extrusion.get("operation") == "CutFeatureOperation":
            deepcad_extrusion["operation"] = "CutFeatureOperation"
        
        # 处理profiles引用 - 支持多种可能的格式
        profiles_list = []
        
        # 尝试从不同可能的字段获取profile引用
        for field in ["Profiles", "profiles"]:
            if field in cadparser_extrusion:
                profiles_list = cadparser_extrusion[field]
                break
        
        # 如果仍然找不到，尝试从其他字段获取
        if not profiles_list and "defaultProfiles" in cadparser_extrusion:
            # 创建模拟的profiles引用
            profiles_list = [
                {
                    "sketch": cadparser_extrusion.get("id", ""), 
                    "profile": profile_id
                } 
                for profile_id in cadparser_extrusion["defaultProfiles"].keys()
            ]
            ##clglogger.debug(f"从defaultProfiles创建了{len(profiles_list)}个轮廓引用")
        
        # 如果依然找不到，检查是否有草图字段可以用作兜底方案
        if not profiles_list and "sketch" in cadparser_extrusion:
            sketch_id = cadparser_extrusion["sketch"]
            # 通常sketch ID可以用来查找默认轮廓
            if sketch_id in self.profile_mappings and self.profile_mappings[sketch_id]:
                # 使用该草图下所有已知的轮廓
                profiles_list = [
                    {"sketch": sketch_id, "profile": profile_id}
                    for profile_id in cadparser_extrusion.get("defaultProfiles", {}).keys() or [list(self.profile_mappings[sketch_id].keys())[0]]
                ]
                ##clglogger.debug(f"从sketch字段创建了{len(profiles_list)}个轮廓引用")
        
        # 如果没有找到profile引用列表，记录警告
        if not profiles_list:
            clglogger.warning(f"未在拉伸特征中找到profiles引用: {cadparser_extrusion.get('name')}")
        
        # 处理每个profile引用
        for profile_ref in profiles_list:
            sketch_id = profile_ref.get("sketch")
            profile_id = profile_ref.get("profile")
            
            # 记录正在处理的轮廓
            ##clglogger.debug(f"处理轮廓引用: sketch={sketch_id}, profile={profile_id}")
            
            # 正常情况: 检查是否有直接映射
            deepcad_profile_id, deepcad_sketch_id = self._find_profile_mapping(sketch_id, profile_id)
            if deepcad_profile_id and deepcad_sketch_id:
                deepcad_extrusion["profiles"].append({
                    "profile": deepcad_profile_id,
                    "sketch": deepcad_sketch_id
                })
                ##clglogger.debug(f"找到轮廓映射: {profile_id} -> {deepcad_profile_id}")
                continue
            
            # 尝试在默认配置文件中查找
            found = False
            if sketch_id in self.sketch_mappings:
                # 获取已创建的deepcad草图ID
                deepcad_sketch_id = self.sketch_mappings[sketch_id]
                
                # 检查是否存在任何轮廓映射
                if sketch_id in self.profile_mappings:
                    # 只使用第一个可用的轮廓（若找不到确切匹配）
                    profile_map = self.profile_mappings[sketch_id]
                    if profile_map:
                        first_profile = next(iter(profile_map.values()))
                        deepcad_extrusion["profiles"].append({
                            "profile": first_profile,
                            "sketch": deepcad_sketch_id
                        })
                        ##clglogger.debug(f"使用可用轮廓替代: {profile_id} -> {first_profile}")
                        found = True
            
            if not found:
                clglogger.warning(f"无法找到对应的草图或轮廓: sketch={sketch_id}, profile={profile_id}")
        
        return deepcad_extrusion, extrusion_id

    def convert_revolve(self, cadparser_revolve: dict, entity_uuid: str) -> tuple:
        """转换旋转特征
        
        Args:
            cadparser_revolve: CADParser格式的旋转特征
            entity_uuid: 原始的UUID键
            
        Returns:
            (deepcad_revolve, revolve_id): 转换后的旋转特征和ID
        """
        revolve_id = self.generate_id("F")
        
        # 创建基本旋转结构
        deepcad_revolve = {
            "name": cadparser_revolve.get("name", "Revolve"),
            "type": "RevolveFeature",
            "profiles": [],
            "extent_two": {
                "angle": {
                    "type": "ModelParameter",
                    "role": "AgainstAngle", 
                    "name": "none",
                    "value": 0.0
                },
                "type": "AngleExtentDefinition"
            },
            "extent_one": {
                "angle": {
                    "type": "ModelParameter",
                    "role": "AlongAngle",
                    "name": "none",
                    "value": 0.0
                },
                "type": "AngleExtentDefinition"
            },
            "operation": "NewBodyFeatureOperation",
            "start_extent": {
                "type": "ProfilePlaneStartDefinition"
            },
            "axis_line": self._convert_axis(cadparser_revolve.get("Axis", {})),
            "extent_type": "OneSideFeatureExtentType"
        }
        
        # 设置角度参数 - CADParser中角度可能是弧度制
        # 检查Angle值并转换为角度制(如果需要)
        angle_value = cadparser_revolve.get("Angle", 6.28318)  # 默认约360度
        if angle_value > 6.30 or angle_value < 0:  # 超出正常弧度范围
            # 可能已经是角度值
            deepcad_revolve["extent_one"]["angle"]["value"] = angle_value
        else:
            # 转换弧度制为角度制
            deepcad_revolve["extent_one"]["angle"]["value"] = angle_value * 180.0 / 3.14159
        
        # 设置操作类型
        if cadparser_revolve.get("type") == "RevCut":
            deepcad_revolve["operation"] = "CutFeatureOperation"
        elif "Cut" in cadparser_revolve.get("name", ""):
            # 如果名称中包含"Cut"，可能也是一个切除操作
            deepcad_revolve["operation"] = "CutFeatureOperation"
        
        # 处理profiles引用 - 支持多种可能的格式
        profiles_list = []
        
        # 尝试从不同可能的字段获取profile引用
        for field in ["Profiles", "profiles"]:
            if field in cadparser_revolve:
                profiles_list = cadparser_revolve[field]
                break
        
        # 如果没有找到profile引用列表，记录警告
        if not profiles_list:
            clglogger.warning(f"未在旋转特征中找到profiles引用: {cadparser_revolve.get('name')}")
        
        # 处理每个profile引用
        for profile_ref in profiles_list:
            sketch_id = profile_ref.get("sketch")
            profile_id = profile_ref.get("profile")
            
            # 正常情况: 检查是否有直接映射
            deepcad_profile_id, deepcad_sketch_id = self._find_profile_mapping(sketch_id, profile_id)
            if deepcad_profile_id and deepcad_sketch_id:
                deepcad_revolve["profiles"].append({
                    "profile": deepcad_profile_id,
                    "sketch": deepcad_sketch_id
                })
                continue
            
            # 尝试在默认配置文件中查找
            found = False
            if sketch_id in self.sketch_mappings:
                # 获取已创建的deepcad草图ID
                deepcad_sketch_id = self.sketch_mappings[sketch_id]
                
                # 检查是否存在任何轮廓映射
                if sketch_id in self.profile_mappings and self.profile_mappings[sketch_id]:
                    # 只使用第一个可用的轮廓（若找不到确切匹配）
                    for cadparser_profile_id, deepcad_profile_id in self.profile_mappings[sketch_id].items():
                        deepcad_revolve["profiles"].append({
                            "profile": deepcad_profile_id,
                            "sketch": deepcad_sketch_id
                        })
                        found = True
                        break
            
            if not found:
                clglogger.warning(f"无法找到对应的草图或轮廓: sketch={sketch_id}, profile={profile_id}")
        
        return deepcad_revolve, revolve_id

    def _convert_axis(self, axis: dict) -> dict:
        """转换旋转轴定义，提供健壮的处理方式
        
        Args:
            axis: CADParser格式的旋转轴
            
        Returns:
            DeepCAD格式的旋转轴
        """
        # 默认值提供有效的旋转轴
        default_axis = {
            "start_point": {"x": 0.0, "y": 0.0, "z": 0.0},
            "end_point": {"x": 0.0, "y": 0.0, "z": 1.0}  # 默认z轴
        }
        
        if not axis:
            clglogger.warning("旋转特征缺少轴定义，使用默认Z轴")
            return default_axis
        
        if axis.get("type") == "SketchSegment" and "start_point" in axis and "end_point" in axis:
            return {
                "start_point": self.convert_point(axis["start_point"]),
                "end_point": self.convert_point(axis["end_point"])
            }
            
        # 尝试从向量形式解析轴信息
        if "vector" in axis:
            vector = axis["vector"]
            return {
                "start_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                "end_point": self.convert_point(vector)
            }
            
        # 尝试从方向信息解析轴
        if "direction" in axis:
            dir_info = axis["direction"]
            if isinstance(dir_info, dict):
                return {
                    "start_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "end_point": self.convert_point(dir_info)
                }
            elif isinstance(dir_info, list) and len(dir_info) >= 3:
                return {
                    "start_point": {"x": 0.0, "y": 0.0, "z": 0.0},
                    "end_point": {"x": dir_info[0], "y": dir_info[1], "z": dir_info[2]}
                }
                
        clglogger.warning(f"无法解析旋转轴信息，使用默认Z轴: {axis}")
        return default_axis

    def _convert_bounding_box(self, bb: dict) -> dict:
        """转换边界框
        
        Args:
            bb: CADParser格式的边界框
            
        Returns:
            DeepCAD格式的边界框
        """
        return {
            "max_point": self.convert_point(bb.get("max_point", {"x": 0, "y": 0, "z": 0})),
            "type": "BoundingBox3D",
            "min_point": self.convert_point(bb.get("min_point", {"x": 0, "y": 0, "z": 0}))
        }

    def _ensure_profile_data(self, loop_data):
        """确保轮廓数据完整，修复可能的缺失值或格式问题 (优化：避免复制)
        
        Args:
            loop_data: 输入的环数据 (会被直接修改)
            
        Returns:
            修复后的环数据 (原始对象的引用)
        """
        
        if "curves" in loop_data and "profile_curves" not in loop_data:
            loop_data["profile_curves"] = loop_data.pop("curves")
            
        return loop_data # 返回修改后的原始对象引用

    def convert(self, cadparser_data: dict) -> dict:
        """执行完整的转换过程
        
        Args:
            cadparser_data: CADParser格式的JSON数据
            
        Returns:
            DeepCAD格式的JSON数据
        """
        # 重置内部状态
        self.id_counter = 0
        self.sketch_mappings = {}
        self.profile_mappings = {}
        # gc.collect() # 可选：强制垃圾回收
        
        deepcad_json = {
            "entities": {},
            "properties": {
                "bounding_box": self._convert_bounding_box(
                    cadparser_data.get("properties", {}).get("bounding_box", {})
                )
            },
            "sequence": []
        }
        
        # 处理实体和序列
        entities_map = cadparser_data.get("entities", {})
        for item in cadparser_data.get("sequence", []):
            entity_uuid = item.get("entity") # 这是原始的 UUID key
            entity_type = item.get("type")
            entity_index = item.get("index", 0)
            
            if entity_uuid not in entities_map:
                clglogger.warning(f"序列项引用的实体 {entity_uuid} 在 entities 中未找到，跳过")
                continue
                
            entity_data = entities_map[entity_uuid]
            
            converted_entity = None
            new_entity_id = None
            deepcad_type = None
            
            try:
                if entity_type == "Sketch":
                    # 传递 entity_uuid 以确保使用原始UUID作为映射键
                    converted_entity, new_entity_id = self.convert_sketch(entity_data, entity_uuid)
                    deepcad_type = "Sketch"
                    
                elif entity_type in ["Extrusion", "ExtrusionCut"]:
                    # 传递 entity_uuid
                    converted_entity, new_entity_id = self.convert_extrusion(entity_data, cadparser_data, entity_uuid)
                    deepcad_type = "ExtrudeFeature"
                    
                elif entity_type in ["Revolution", "Revolve", "RevCut"]:
                    # 传递 entity_uuid
                    converted_entity, new_entity_id = self.convert_revolve(entity_data, entity_uuid)
                    deepcad_type = "RevolveFeature"
                else:
                    clglogger.warning(f"未知的实体类型: {entity_type}，跳过实体 {entity_uuid}")
                    continue
                    
                if converted_entity and new_entity_id and deepcad_type:
                    deepcad_json["entities"][new_entity_id] = converted_entity
                    deepcad_json["sequence"].append({
                        "index": entity_index,
                        "type": deepcad_type,
                        "entity": new_entity_id
                    })
                else:
                     clglogger.warning(f"转换实体 {entity_uuid} (类型: {entity_type}) 失败或未返回有效结果")

            except Exception as e:
                import traceback
                clglogger.error(f"处理实体 {entity_uuid} (类型: {entity_type}) 时出错: {e}")
                ##clglogger.debug(traceback.format_exc())
        
        # 清理可能不再需要的引用
        del entities_map
        # gc.collect() # 可选：强制垃圾回收
        
        return deepcad_json

def process_file(input_file: str, output_file: str) -> bool:
    """处理单个文件，将CADParser JSON转换为DeepCAD JSON
    
    Args:
        input_file: 输入的CADParser文件路径
        output_file: 输出的DeepCAD文件路径
        
    Returns:
        成功返回True，失败返回False
    """
    try:
        # 尝试多种编码模式读取文件内容
        content = None
        encodings = ['utf-8', 'gbk', 'latin1']
        
        for encoding in encodings:
            try:
                with open(input_file, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            clglogger.error(f"无法使用支持的编码读取文件: {input_file}")
            return False
        
        # 替换中文词汇
        cn_to_en_mapping = {
            "草图": "Sketch",
            "拉伸": "Extrude",
            "旋转": "Revolve",
            "切除": "Cut",
            "圆": "Circle",
            "直线": "Line",
            "曲线": "Curve",
            "弧": "Arc",
            "轮廓": "Profile",
            "特征": "Feature",
            "穿透": "Through",
            "平面": "Plane",
            "反向": "Reverse"
        }
        
        for cn, en in cn_to_en_mapping.items():
            content = content.replace(f'"{cn}"', f'"{en}"')
            content = content.replace(f'"{cn}:', f'"{en}:')
            content = content.replace(f':{cn}"', f':{en}"')
        
        # 解析JSON
        try:
            cadparser_data = json.loads(content)
        except json.JSONDecodeError as je:
            clglogger.error(f"JSON解析错误: {str(je)}")
            return False
        
        # 转换数据
        converter = CadParserToDeepCadConverter()
        deepcad_data = converter.convert(cadparser_data)
        
        # 保存转换后的数据
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deepcad_data, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        import traceback
        clglogger.error(f"转换文件 {input_file} 失败: {str(e)}")
        #clglogger.debug(traceback.format_exc())
        return False


@measure_performance
def main():
    """主函数，处理命令行参数并执行转换"""
    parser = argparse.ArgumentParser(description="将CADParser JSON转换为DeepCAD JSON格式")
    parser.add_argument("--input", required=True, help="输入CADParser JSON文件或目录")
    parser.add_argument("--output", required=True, help="DeepCAD JSON文件的输出目录")
    parser.add_argument("--max_workers", type=int, default=8, help="并行处理的最大工作线程数")
    args = parser.parse_args()
    
    success_count = 0
    failure_count = 0
    
    if os.path.isfile(args.input):
        # 处理单个文件
        output_file = os.path.join(args.output, os.path.basename(args.input))
        result = process_file(args.input, output_file)
        if result:
            success_count += 1
            clglogger.info(f"成功转换: {args.input} -> {output_file}")
        else:
            failure_count += 1
            
    elif os.path.isdir(args.input):
        # 处理目录中的所有JSON文件
        all_files = []
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.endswith('.json'):
                    input_file = os.path.join(root, file)
                    rel_path = os.path.relpath(input_file, args.input)
                    output_file = os.path.join(args.output, rel_path)
                    all_files.append((input_file, output_file))
        
        clglogger.info(f"发现 {len(all_files)} 个JSON文件需要转换")
        
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_file, input_file, output_file): input_file 
                      for input_file, output_file in all_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="转换进度"):
                input_file = futures[future]
                if future.result():
                    success_count += 1
                else:
                    failure_count += 1
    else:
        clglogger.error(f"输入路径 {args.input} 无效")
        return
    
    clglogger.info(f"转换完成: 成功 {success_count} 个文件, 失败 {failure_count} 个文件")


if __name__ == "__main__":
    main()
