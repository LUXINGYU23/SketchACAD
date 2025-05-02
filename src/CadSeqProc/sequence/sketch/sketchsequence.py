import os, sys

sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))


import numpy as np
from src.CadSeqProc.utility.logger import CLGLogger
from src.CadSeqProc.utility.utils import (
    create_point_from_array,
    perform_op,
    random_sample_points,
    split_array,
    write_ply,
    create_matched_pair,
    create_colored_wire,
)
from src.CadSeqProc.utility.macro import *
from rich import print
from .face import FaceSequence, LoopSequence
from loguru import logger
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.gp import gp_Vec, gp_Pln, gp_Dir, gp_Ax3
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism
from OCC.Extend.DataExchange import write_step_file
from .coord_system import CoordinateSystem
import copy
from scipy.optimize import linear_sum_assignment
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

clglogger = CLGLogger().configure_logger().logger


class SketchSequence(object):

    def __init__(
        self,
        facedata: List[FaceSequence],
        coordsystem: CoordinateSystem = None,
        reorder: bool = True,
    ):

        self.facedata = facedata
        self.quantized_facedata = {}
        self.coordsystem = coordsystem

        if reorder:
            # Reorder Faces
            self.reorder()

    @property
    def token_index(self):
        return SKETCH_TOKEN.index("END_SKETCH")

    def reorder(self):
        if len(self.facedata) <= 1:
            return
        all_faces_bbox_min = np.stack(
            [face.bbox[0] for face in self.facedata], axis=0
        ).round(6)
        ind = np.lexsort(all_faces_bbox_min.transpose()[[1, 0]])
        self.facedata = [self.facedata[i] for i in ind]

    @staticmethod
    def from_dict(all_stat, profile_uid_list):
        facedata = []
        coordsystem = CoordinateSystem.from_dict(
            all_stat["entities"][profile_uid_list[0][0]]["transform"]
        )

        for i in range(len(profile_uid_list)):
            sketch_entity = all_stat["entities"][profile_uid_list[i][0]]
            assert sketch_entity["type"] == "Sketch", clglogger.critical(
                f"Uid Mismatch for {profile_uid_list[i]}"
            )
            facedata.append(
                FaceSequence.from_dict(sketch_entity, profile_uid_list[i][1])
            )

        return SketchSequence(facedata=facedata, coordsystem=coordsystem, reorder=True)

    @staticmethod
    def from_minimal_json(sketch_stat, coord_system):
        face_seq = []
        for _, val in sketch_stat.items():
            face_seq.append(FaceSequence.from_minimal_json(val))
        
        coord_system=CoordinateSystem.from_minimal_json(coord_system)
        return SketchSequence(face_seq, coord_system, True)
    
    def sample_points(self, n_points, point_dimension=3):
        all_points = []

        for fc in self.facedata:
            all_points.append(
                fc.sample_points(n_points=n_points)
            )
        all_points = np.vstack(all_points)

        random_points = random_sample_points(all_points, n_points)[0]
        if random_points.shape[-1] == 2 and point_dimension == 3:
            random_points = self.coordsystem.rotate_vec(random_points)
        return random_points

    def __repr__(self):
        s = "Sketch:"
        s += f"\n       - {self.coordsystem.__repr__()}"
        for face in self.facedata:
            s += f"\n       - {face.__repr__()}"
        return s

    def to_vec(self):
        """
        Vector Representation of One Sketch sequence
        """
        coord_token = []
        for fc in self.facedata:
            vec = fc.to_vec()
            coord_token += vec
        coord_token.append([self.token_index, 0])
        return coord_token

    @staticmethod
    def from_vec(vec, bit, post_processing, fix_collinearity):
        """
        Vec is the list of faces
        """
        fc = []
        merged_vec = split_array(vec, val=SKETCH_TOKEN.index("END_FACE"))
        for fc_tokens in merged_vec:
            fc.append(
                FaceSequence.from_vec(
                    vec=fc_tokens,
                    bit=bit,
                    post_processing=post_processing,
                    fix_collinearity=fix_collinearity,
                )
            )
        if len(fc) == 0:
            raise Exception(f"No Loops Added for vec {vec}")
        return SketchSequence(facedata=fc, reorder=False)

    @property
    def bbox(self):
        all_min_box = []
        all_max_box = []
        for fc in self.facedata:
            bbox = fc.bbox
            all_min_box.append(bbox[0])
            all_max_box.append(bbox[1])
        return np.array([np.min(all_min_box, axis=0), np.max(all_max_box, axis=0)])

    @property
    def length(self):
        bbox_min=self.bbox[0]
        bbox_max=self.bbox[1]
        return abs(bbox_max[0]-bbox_min[0])
    
    @property
    def width(self):
        bbox_min=self.bbox[0]
        bbox_max=self.bbox[1]
        return abs(bbox_max[1]-bbox_min[1])

    @property
    def dimension(self):
        return self.length, self.width
    
    @property
    def all_loops(self):
        all_loops = []
        for fc in self.facedata:
            all_loops += fc.all_loops

        return all_loops

    @property
    def bbox_size(self):
        """compute bounding box size (max of height and width)"""
        bbox_min, bbox_max = self.bbox[0], self.bbox[1]
        bbox_size = np.max(
            np.abs(
                np.concatenate(
                    [bbox_max - self.start_point, bbox_min - self.start_point]
                )
            )
        )
        # clglogger.debug(f"{self.bbox} {bbox_size} {bbox_max-self.start_point}")
        return bbox_size

    def add_info(self, key: str, val: FaceSequence):
        self.facedata[key] = val

    def transform(self, translate=None, scale=1):
        for fc in self.facedata:
            fc.transform(translate=translate, scale=scale)

    @property
    def all_curves(self):
        curves = []

        for fc in self.facedata:
            curves += fc.all_curves

        return curves

    @property
    def start_point(self):
        # return self.facedata[0].start_point
        return self.bbox[0]

    @property
    def sketch_position(self):
        return (
            self.start_point[0] * self.coordsystem.get_property("x_axis")
            + self.start_point[1] * self.coordsystem.get_property("y_axis")
            + self.coordsystem.get_property("origin")
        )

    def sketch_plane(self):

        origin = create_point_from_array(self.sketch_position)
        return gp_Pln(origin, gp_Dir(*self.coordsystem.metadata["z_axis"]))

    def build_body(self, extrude_params: dict):
        """
        extrude params must contain {"extrude_values": [float,float]}
        """
        all_faces = []
        for fc in self.facedata:
            ref_face, face = fc.build_body(
                plane=self.sketch_plane(),
                normal=self.coordsystem.normal,
                coordsystem=self.coordsystem,
            )
            all_faces.append(face)
            # clglogger.debug("Success for a face")

        # Merge all faces in the same plane
        plane_face = all_faces[0]

        for face in all_faces[1:]:
            # self.debug_face(face, "PlaneFace")
            plane_face = perform_op(plane_face, face, "fuse")
        # Extrude face to 3d shape
        solid = self.extrude_face(ref_face, plane_face, extrude_params)

        return solid
    def build_face(self):
        """
        创建草图面，用于旋转操作
    
        Returns:
            TopoDS_Face: 创建的草图面
        """
        all_faces = []
        for fc in self.facedata:
            ref_face, face = fc.build_body(
                plane=self.sketch_plane(),
                normal=self.coordsystem.normal,
                coordsystem=self.coordsystem,
            )
            all_faces.append(face)
    
        # 合并同一平面中的所有面
        if len(all_faces) == 0:
            raise Exception("没有创建任何面")
    
        plane_face = all_faces[0]
        # self.debug_face(plane_face, "PlaneFace")
        for face in all_faces[1:]:
            # self.debug_face(face, "PlaneFace")
            plane_face = perform_op(plane_face, face, "fuse")
        
        return plane_face
    def get_normal(self):
        """
        获取草图平面的法向量，用于旋转操作
    
        Returns:
            gp_Dir: 草图平面的法向量
        """
        return self.coordsystem.normal
    def extrude_face(self, ref_face, face, extrude_params):
        distance = extrude_params["extrude_values"]
        surf = BRepAdaptor_Surface(ref_face).Plane()
        normal = surf.Axis().Direction()
        extruded_shape = self.extrudeBasedOnType(face, normal, distance)
        return extruded_shape

    def extrudeBasedOnType(self, face, normal, distance):
        # Extrude based on the two bound values
        # if not (distance[0] < distance[1]):
        #     sorted(distance)
        # large_value = max(distance)
        # small_value = min(distance)
        if distance[0] == 0:
            ext_vec = gp_Vec(normal.Reversed()).Multiplied(distance[1])
            body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
        else:
            ext_vec = gp_Vec(normal).Multiplied(distance[0])
            body = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
            if distance[1] > 0:
                ext_vec = gp_Vec(normal.Reversed()).Multiplied(distance[1])
                body_two = BRepPrimAPI_MakePrism(face, ext_vec).Shape()
                body = perform_op(body, body_two, "fuse")
        return body

        # if large_value == 0:
        #     return self.build_prism(face, -normal, -small_value)
        # elif small_value == 0:
        #     return self.build_prism(face, normal, large_value)
        # elif np.sign(large_value) == np.sign(small_value):
        #     if large_value < 0:
        #         body1 = self.build_prism(face, -normal, -small_value)
        #         body2 = self.build_prism(face, -normal, -large_value)
        #         return perform_op(body1, body2, 'cut')
        #     else:
        #         assert large_value > 0
        #         body1 = self.build_prism(face, normal, small_value)
        #         body2 = self.build_prism(face, normal, large_value)
        #         return perform_op(body2, body1, 'cut')
        # else:
        #     assert np.sign(large_value) != np.sign(small_value)
        #     body1 = self.build_prism(face, normal, large_value)
        #     body2 = self.build_prism(face, -normal, -small_value)
        #     return perform_op(body1, body2, 'fuse')

    def build_prism(self, face, normal, value):
        extrusion_vec = gp_Vec(normal).Multiplied(value)
        make_prism = BRepPrimAPI_MakePrism(face, extrusion_vec)
        make_prism.Build()
        prism = make_prism.Prism()
        return prism.Shape()

    def normalize(self, translate=None, bit=N_BIT):
        """
        Normalize the sketch and shift the sketch to the start point.
        Only used for 2d representation
        """
        size = 2**bit
        cur_size = self.bbox_size
        # scale = (size / 2 * NORM_FACTOR - 1) / cur_size # prevent potential overflow if data augmentation applied
        scale = (size - 1) / self.bbox_size
        if translate is None:
            self.transform(-self.start_point, scale)
        else:
            self.transform(translate, scale)

    def denormalize(self, bbox_size=None, translate=0.0, bit=N_BIT):
        """
        Inverse operation of normalize. Only used for 2d representation.
        """
        size = 2**bit
        # if bbox_size is None:
        #     bbox_size=self.bbox_size
        # scale = bbox_size / (size / 2 * NORM_FACTOR - 1)
        scale = bbox_size / (size - 1)
        if translate is None:
            translate = -np.array((size / 2, size / 2))
        self.transform(translate, scale)

    def numericalize(self, bit):
        """
        Quantization
        """
        for fc in self.facedata:
            fc.numericalize(bit=bit)

    def denumericalize(self, bit):
        """
        Dequantization
        """
        for fc in self.facedata:
            fc.denumericalize(bit=bit)

    def create_skt3d_edge(self):
        """Creates TopoDS shape for 3d sketch visualization"""
        solid = self.build_body(2, {"extrude_values": [0.001, 0]})
        return solid

    @staticmethod
    def loop_match(gt_sketch, pred_sketch, scale: float, multiplier: int = 2):
        """
        Match Loops according to the bounding box.

        Args:
            gt_sketch (object): The current object. (self must be ground truth)
            pred_sketch (object): The pred sketch object. (pred is prediction)
            scale (float): The scaling factor.
            multiplier (int): cost of distance with None

        Returns:
            list: List of matched loop pairs.
        """

        if pred_sketch is None:
            pred_loops = [None]
        else:
            pred_loops = copy.deepcopy(pred_sketch.all_loops)

        if gt_sketch is None:
            gt_loops = [None]
        else:
            gt_loops = copy.deepcopy(gt_sketch.all_loops)

        num_gt_loops = len(gt_loops)
        num_pred_loops = len(pred_loops)

        n_max = max(num_gt_loops, num_pred_loops)

        # Pad lists with None if needed
        if len(gt_loops) < n_max:
            gt_loops += [None] * (n_max - len(gt_loops))
        if len(pred_loops) < n_max:
            pred_loops += [None] * (n_max - len(pred_loops))

        cost_matrix = (
            np.ones((n_max, n_max)) * multiplier
        )  # Fixed the shape of the cost matrix

        # Calculate Cost by calculating the distance between loops
        for ind_self in range(num_gt_loops):
            for ind_pred in range(num_pred_loops):
                if gt_loops[ind_self] is not None and pred_loops[ind_pred] is not None:
                    cost_matrix[ind_self, ind_pred] = gt_loops[ind_self].loop_distance(
                        pred_loops[ind_pred], scale
                    )

        # Use Hungarian matching to find the best matching
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Create matched loop pairs
        matched_loop_pair = create_matched_pair(
            list1=gt_loops,
            list2=pred_loops,
            row_indices=row_indices,
            col_indices=col_indices,
        )

        # After loops are matched, match primitives
        # (This will change the object from a pair of LoopSequences to a pair of list of CurveSequences)
        matched_curve_pair = []
        for i, pair in enumerate(matched_loop_pair):
            matched_curve_pair += LoopSequence.match_primitives(
                pair[0], pair[1], scale, multiplier
            )

        return matched_curve_pair, matched_loop_pair

    def draw(self, ax=None, colors=None):
        if ax is None:
            fig, ax = plt.subplots()
        if colors is None:
            colors = [
                "red",
                "blue",
                "green",
                "brown",
                "pink",
                "yellow",
                "purple",
                "black",
            ] * 10
        else:
            colors = [colors] * 100
        for i, face in enumerate(self.facedata):
            face.draw(ax, colors[i])

    def _json(self):
        sketch_json = {}
        for i, face in enumerate(self.facedata):
            sketch_json[f"face_{i+1}"] = face._json()

        # sketch_json["coordinate_system"]=self.coordsystem._json()
        return sketch_json

    def debug_face(self, face, label="", save_visualization=True, output_dir="debug_output"):
        """
        调试函数：打印面的详细信息
    
        Args:
            face: 要检查的OCCT面对象
            label: 面的标识标签
            save_visualization: 是否保存可视化图像
            output_dir: 保存图像的目录
        """
        from OCC.Core.BRepTools import breptools
        from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.BRepCheck import BRepCheck_Face
        from OCC.Core.TopoDS import TopoDS_Face, topods
        from OCC.Display.SimpleGui import init_display
        from OCC.Core.Graphic3d import Graphic3d_NOM_ALUMINIUM, Graphic3d_NOM_COPPER
        from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
        from OCC.Extend.DataExchange import write_step_file, write_stl_file
        if face is None:
            clglogger.error(f"{label} 面对象为空")
            return
         # 创建rgb_color辅助函数
        def rgb_color(r, g, b):
            return Quantity_Color(r, g, b, Quantity_TOC_RGB)
        try:
            # 基本信息
            clglogger.info(f"=== {label} 面分析开始 ===")
            
            # 转换为面
            try:
                face_shape = topods.Face(face)
                clglogger.info("成功转换为TopoDS_Face")
            except Exception as e:
                clglogger.warning(f"不是有效的TopoDS_Face: {e}")
            
            # 检查拓扑元素
            edges_explorer = TopExp_Explorer(face, TopAbs_EDGE)
            edge_count = 0
            while edges_explorer.More():
                edge_count += 1
                edges_explorer.Next()
            
            vertices_explorer = TopExp_Explorer(face, TopAbs_VERTEX)
            vertex_count = 0
            while vertices_explorer.More():
                vertex_count += 1
                vertices_explorer.Next()
            
            clglogger.info(f"面包含 {edge_count} 条边和 {vertex_count} 个顶点")
            #输出详细参数
            clglogger.info(f"面参数: {breptools.UVBounds(face_shape)}")
            #输出面类型
            surf = BRepAdaptor_Surface(face)
            surf_type = surf.GetType()
            plane = surf.Plane()
            normal_dir = plane.Axis().Direction()
            normal_vec = [normal_dir.X(), normal_dir.Y(), normal_dir.Z()]
            clglogger.info(f"法向量: {normal_vec}")
            clglogger.info(f"面类型: {surf_type}")
            # 检查面的有效性
            checker = BRepCheck_Face(face_shape)
            if checker.IntersectWires():
                clglogger.warning("面内部线条相交")
                        # 使用OpenCASCADE的显示功能创建图像
            try:
                os.makedirs(output_dir, exist_ok=True)
                safe_label = label.replace(" ", "_").replace("/", "_").replace("\\", "_")
                if not safe_label:
                    safe_label = "unnamed_face"
                # 初始化显示
                display, start_display, add_menu, add_function_to_menu = init_display(size=(800, 600))
                
                # 显示面
                display.DisplayShape(face_shape, color=rgb_color(0.1, 0.8, 0.1), 
                                    material=Graphic3d_NOM_ALUMINIUM, 
                                    transparency=0.2)
                
                # 显示边
                edges_explorer = TopExp_Explorer(face, TopAbs_EDGE)
                while edges_explorer.More():
                    edge = topods.Edge(edges_explorer.Current())
                    display.DisplayShape(edge, color=rgb_color(0, 0, 0.8), 
                                        material=Graphic3d_NOM_COPPER)
                    edges_explorer.Next()
                
                # 截图并保存
                display.View_Iso()
                display.FitAll()
                img_filename = os.path.join(output_dir, f"{safe_label}.png")
                display.View.Dump(img_filename)
                clglogger.info(f"已保存预览图像: {img_filename}")
                
                # 关闭显示
                display.Repaint()
                # 如果需要交互查看，可以取消下面这行注释
                start_display()
            except Exception as vis_error:
                clglogger.error(f"可视化失败: {vis_error}")
            clglogger.info(f"=== {label} 面分析结束 ===")
            
        except Exception as e:
            clglogger.error(f"面分析失败: {e}")




if __name__ == "__main__":
    import json

    json_path = "/data/3d_cluster/Brep2Seq/deepcad_data/cad_json/0043/00430950.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    lst = [["FcWd1Kjyasi3dQe_0", "JGC"], ["FcWd1Kjyasi3dQe_0", "JGG"]]
    skt = SketchSequence.from_dict(data, lst)

    # print(skt.start_point)
    # print(skt)
    skt.transform(translate=-skt.start_point)
    # print(skt)
    skt.transform3D()
    points = skt.sample_points(num_points=1000)
    # print("Points")
    # print(points)

    # print(points.shape)

    # points=np.hstack([points,np.zeros((len(points),1))])

    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Save PointCloud object as PLY file
    o3d.io.write_point_cloud("/home/mkhan/Codes/point2cad/output/output.ply", pcd)
