# ---------------------------------------------------------------------------- #
#                               Global Variables                               #
# ---------------------------------------------------------------------------- #


N_BIT=8

END_TOKEN=["PADDING", "START", "END_SKETCH",
                "END_FACE", "END_LOOP", "END_CURVE", "END_EXTRUSION", "END_REVOLVE"]

END_PAD=8  # 原来是7，加入了END_REVOLVE后变为8
BOOLEAN_PAD=4

MAX_CAD_SEQUENCE_LENGTH=289

SKETCH_TOKEN = ["PADDING", "START", "END_SKETCH",
                "END_FACE", "END_LOOP", "END_CURVE", "CURVE"]
EXTRUSION_TOKEN = ["PADDING", "START", "END_EXTRUDE_SKETCH"]

CURVE_TYPE=["Line","Arc","Circle"]

EXTRUDE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"]

# 添加旋转操作的布尔操作类型
REVOLVE_OPERATIONS = ["NewBodyFeatureOperation", "JoinFeatureOperation",
                      "CutFeatureOperation", "IntersectFeatureOperation"]

NORM_FACTOR=0.75
EXTRUDE_R=1
SKETCH_R=1

PRECISION = 1e-5
eps = 1e-7


MAX_SKETCH_SEQ_LENGTH = 150
MAX_EXTRUSION = 10
MAX_REVOLVE = 10
ONE_EXT_SEQ_LENGTH = 10  # Without including start/stop and pad token ((e1,e2),ox,oy,oz,theta,phi,gamma,b,s,END_EXTRUSION) -> 10
# 旋转操作序列长度(angle,axis_start_xyz,axis_end_xyz,ox,oy,oz,theta,phi,gamma,b,is_sym,s,END_REVOLVE) -> 17
ONE_REV_SEQ_LENGTH = 17  # 修正为17，角度1 + 轴起点3 + 轴终点3 + 原点3 + 欧拉角3 + 布尔操作1 + 对称标志1 + 草图尺寸1 + 结束标记1
VEC_TYPE=2 # Different types of vector representation (Keep only 2)

# 拉伸操作标志：1 + 范围1-10，占用值1-10
EXTRUDE_FLAG_START = 1
EXTRUDE_FLAG_RANGE = list(range(1, 11))

# 旋转操作标志：11 + 范围11-27，占用值11-27
REVOLVE_FLAG_START = 11
REVOLVE_FLAG_RANGE = list(range(11, 28))

# 填充标记：28
PADDING_FLAG = 28
# 草图标记 0
SKETCH_FLAG = 0
CAD_CLASS_INFO = {
    'one_hot_size': END_PAD+BOOLEAN_PAD+2**N_BIT,  #8+4+256=268
    'index_size': MAX_EXTRUSION+MAX_REVOLVE+1, # +1 for padding  10+10+1=21
    'flag_size': ONE_EXT_SEQ_LENGTH+ONE_REV_SEQ_LENGTH+1, # +1 for padding 10+17+1=28
}

