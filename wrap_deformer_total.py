import numpy as np
import trimesh
import math

eps = 1e-12

# ======================================================
# 0) 核心函数区
# ======================================================


def compute_barycentric_coordinates(p, tri_vertices):
    """
    计算点 p 在三角形 tri_vertices 中的重心坐标 (u, v)。
    返回 (u, v)，其中 w = 1 - u - v。
    对退化三角形做一下防护。
    """
    v0 = tri_vertices[1] - tri_vertices[0]
    v1 = tri_vertices[2] - tri_vertices[0]
    v2 = p - tri_vertices[0]

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        # 退化三角形，返回 0,0 或直接返回 NaN
        return 0.0, 0.0

    u = (d00 * d21 - d01 * d20) / denom
    v = (d11 * d20 - d01 * d21) / denom
    return (u, v)

def polar_decomposition(m):
    """
    极分解，提取正交旋转矩阵 R。
    做一下行列式过小或者为负的防护。
    """
    U, _, Vt = np.linalg.svd(m)
    R = np.dot(U, Vt)
    # 若行列式<0，则说明翻转了某个轴，需要修正
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = np.dot(U, Vt)
    return R
def make_transform(z_vec, y_vec):
    """
    构造一个 3x3 旋转矩阵 (X, Y, Z) = (cross(Y, Z), Y, Z)。
    在构造过程中做归一化，防止数值爆炸。
    """
    # 1. 规范化输入向量
    z_norm = z_vec / (np.linalg.norm(z_vec) + eps)
    y_norm = y_vec / (np.linalg.norm(y_vec) + eps)
    
    # 2. 计算 X 轴 = cross(Y, Z)
    x_vec = np.cross(y_norm, z_norm)
    x_norm = x_vec / (np.linalg.norm(x_vec) + eps)
    
    # 3. 重新计算正交的 Y 轴 = cross(Z, X)
    y_orth = np.cross(z_norm, x_norm)
    y_norm_orth = y_orth / (np.linalg.norm(y_orth) + eps)
    
    # 4. 构建旋转矩阵
    M = np.array([
        [x_norm[0], y_norm_orth[0], z_norm[0]],
        [x_norm[1], y_norm_orth[1], z_norm[1]],
        [x_norm[2], y_norm_orth[2], z_norm[2]]
    ], dtype=np.float64)
                    
    return M

def compute_rest_xform(mesh, pt_id, neighbors):
    """
    针对单个点 pt_id，基于邻居构造多组参考坐标系 (ors_list) 与其权重 (weights_list)，
    并计算邻居的平均距离作为缩放因子 scale。
    """
    P = mesh.vertices[pt_id]
    ors_list = []
    weights_list = []

    # 若无邻居，则直接返回空
    if len(neighbors) == 0:
        return [], [], 1.0

    # “上一个邻居”向量
    prev_pt = neighbors[-1]
    toprevious = mesh.vertices[prev_pt] - P
    
    for npt in neighbors:
        toneighbour = mesh.vertices[npt] - P
        up = np.cross(toneighbour, toprevious)
        # 权重：用 “两个邻居叉乘长度” 作为权重
        thisweight = np.linalg.norm(up)
        this_orient = make_transform(toneighbour, toprevious)

        ors_list.append(this_orient)
        weights_list.append(thisweight)

        toprevious = toneighbour
    
    # 计算平均邻居距离作为 scale
    dists = np.linalg.norm(mesh.vertices[neighbors] - P, axis=1)
    scale_val = dists.mean() if len(dists) > 0 else 1.0

    return ors_list, weights_list, scale_val

def compute_delta(rest_P, driver_P,
                  rest_ors, driver_ors,
                  rest_weights, driver_weights,
                  rest_scale, driver_scale,
                  eps=1e-12):
    """
    计算单点的位移差 dp、旋转差 dor、缩放差 ds。
    rest_ors / driver_ors: list of 3x3
    rest_weights / driver_weights: list of float
    """
    # 1) 位移差
    dp = driver_P - rest_P
    
    # 2) 旋转差：加权平均
    total_mat = np.zeros((3, 3), dtype=np.float64)
    total_weight = 0.0

    for r_or, a_or, r_w, a_w in zip(rest_ors, driver_ors, rest_weights, driver_weights):
        if r_w < eps or a_w < eps:
            continue
        thisdor = a_or @ r_or.T
        w = math.sqrt(r_w * a_w)
        total_mat += thisdor * w
        total_weight += w

    if total_weight < eps:
        avg_rot = np.eye(3)
    else:
        avg_rot = total_mat / total_weight
        # 做极分解保证是纯旋转
        avg_rot = polar_decomposition(avg_rot)
    
    # 3) 缩放差
    ds = driver_scale / rest_scale if rest_scale > eps else 1.0

    return avg_rot, ds, dp


# -----------------------------
# 辅助插值函数
# -----------------------------
def barycentric_interpolate_vector(values, wuv):
    """
    对三角形 3 个顶点的向量做线性插值:
    values.shape = (3, 3)
    wuv = (w, u, v)
    """
    w, u, v = wuv
    return values[0]*w + values[1]*u + values[2]*v

def barycentric_interpolate_float(values, wuv):
    """
    对三角形 3 个顶点的标量做线性插值:
    values.shape = (3,)
    wuv = (w, u, v)
    """
    w, u, v = wuv
    return values[0]*w + values[1]*u + values[2]*v

def barycentric_interpolate_matrix3(values, wuv):
    """
    对三角形 3 个顶点的 3×3 矩阵做线性插值，再做极分解保证结果是纯旋转。
    这里插值可能导致矩阵略偏离正交，需要在外部再做 polar_decomposition。
    """
    w, u, v = wuv
    mat = values[0]*w + values[1]*u + values[2]*v
    return mat


# -----------------------------
# 四元数 <-> 旋转矩阵 <-> 欧拉角
# -----------------------------
def matrix_to_quaternion(m):
    """
    从 3×3 旋转矩阵提取四元数 (w, x, y, z)。
    """
    trace = m[0,0] + m[1,1] + m[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2,1] - m[1,2]) * s
        y = (m[0,2] - m[2,0]) * s
        z = (m[1,0] - m[0,1]) * s
    else:
        # 下面三个分支是按对角线最大值来选取
        if (m[0,0] > m[1,1]) and (m[0,0] > m[2,2]):
            s = 2.0*np.sqrt(1.0 + m[0,0] - m[1,1] - m[2,2])
            w = (m[2,1] - m[1,2]) / s
            x = 0.25*s
            y = (m[0,1] + m[1,0]) / s
            z = (m[0,2] + m[2,0]) / s
        elif m[1,1] > m[2,2]:
            s = 2.0*np.sqrt(1.0 + m[1,1] - m[0,0] - m[2,2])
            w = (m[0,2] - m[2,0]) / s
            x = (m[0,1] + m[1,0]) / s
            y = 0.25*s
            z = (m[1,2] + m[2,1]) / s
        else:
            s = 2.0*np.sqrt(1.0 + m[2,2] - m[0,0] - m[1,1])
            w = (m[1,0] - m[0,1]) / s
            x = (m[0,2] + m[2,0]) / s
            y = (m[1,2] + m[2,1]) / s
            z = 0.25*s
    return np.array([w, x, y, z])

def quaternion_to_eulerXYZ(q, degrees=True):
    """
    将四元数转换为 XYZ 欧拉角（默认返回度数）。
    """
    
    q = q / (np.linalg.norm(q) + eps)
    w, x, y, z = q

    # 参考 Tait-Bryan angle (XYZ) 转换
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    rx = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w*y - z*x)
    if abs(sinp) >= 1:
        ry = math.copysign(math.pi/2, sinp)  # 90 或 -90 度
    else:
        ry = math.asin(sinp)

    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    rz = math.atan2(siny_cosp, cosy_cosp)

    if degrees:
        return np.degrees([rx, ry, rz])
    else:
        return np.array([rx, ry, rz])

def make_4x4_transform(translate, rotate_euler_deg, scale, pivot):
    """
    构造一个 4x4 变换矩阵，按 S->R->T 顺序作用于点，
    旋转顺序为 X->Y->Z，且绕 pivot 为中心进行缩放/旋转。
    """
    if isinstance(scale, (int, float)):
        sx = sy = sz = scale
    else:
        sx, sy, sz = scale

    rx = math.radians(rotate_euler_deg[0])
    ry = math.radians(rotate_euler_deg[1])
    rz = math.radians(rotate_euler_deg[2])

    def T(tx, ty, tz):
        mat = np.eye(4, dtype=float)
        mat[0, 3] = tx
        mat[1, 3] = ty
        mat[2, 3] = tz
        return mat

    T_trans = T(*translate)
    T_pivot = T(*pivot)
    T_negpivot = T(-pivot[0], -pivot[1], -pivot[2])

    # Scale
    S_mat = np.eye(4, dtype=float)
    S_mat[0, 0] = sx
    S_mat[1, 1] = sy
    S_mat[2, 2] = sz

    # Rotation: Rz * Ry * Rx
    def Rx_(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([
            [1, 0,  0, 0],
            [0, c, -s, 0],
            [0, s,  c, 0],
            [0, 0,  0, 1]
        ], dtype=float)

    def Ry_(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ], dtype=float)

    def Rz_(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([
            [c, -s, 0, 0],
            [s,  c, 0, 0],
            [0,  0, 1, 0],
            [0,  0, 0, 1]
        ], dtype=float)

    R_mat = Rz_(rz) @ Ry_(ry) @ Rx_(rx)

    # 最终组合
    M = T_trans @ T_pivot @ R_mat @ S_mat @ T_negpivot
    return M


# ------------------------------------------------------
# 主流程示例
# ------------------------------------------------------
if __name__ == "__main__":
    # 0) 加载网格
    mesh_rest   = trimesh.load("wrap_rest.obj",   process=False)
    mesh_driver = trimesh.load("wrap_driver.obj", process=False)
    mesh_high   = trimesh.load("wrap_high.obj",   process=False)

    # --------------------------------------------------
    # 1) 最近点查询 (wrap_rest 上)
    #    获取 wrapcloseprim, wrapcloseuv, wrappivot
    # --------------------------------------------------
    query_points = mesh_high.vertices
    closest_points, distances, triangle_ids = mesh_rest.nearest.on_surface(query_points)

    wrapcloseprim = triangle_ids
    wrapcloseuv   = np.empty((len(query_points), 2), dtype=np.float64)
    wrappivot     = np.empty((len(query_points), 3), dtype=np.float64)

    triangles_of_rest = mesh_rest.triangles  # (n_faces, 3, 3)

    for i in range(len(query_points)):
        tri_idx = triangle_ids[i]
        tri_vertices = triangles_of_rest[tri_idx]
        p = closest_points[i]
        # 计算 (u, v)
        u, v = compute_barycentric_coordinates(p, tri_vertices)
        wrapcloseuv[i] = [u, v]

        w = 1.0 - u - v
        wrappivot[i] = (tri_vertices[0]*w +
                        tri_vertices[1]*v +
                        tri_vertices[2]*u)

    # --------------------------------------------------
    # 2) 在 wrap_rest & wrap_driver 上，计算每个顶点对应的 dp, dor, ds
    #    （此过程可以并行/批量化处理）
    # --------------------------------------------------
    adjacency_rest   = mesh_rest.vertex_neighbors
    adjacency_driver = mesh_driver.vertex_neighbors

    num_points = len(mesh_rest.vertices)
    assert num_points == len(mesh_driver.vertices), \
        "wrap_rest 和 wrap_driver 顶点数量必须相同！"

    # 先缓存所有邻居对应的参考矩阵/权重/scale
    all_rest_ors      = []
    all_rest_weights  = []
    all_rest_scales   = []
    all_driver_ors    = []
    all_driver_weights= []
    all_driver_scales = []

    vertices_rest   = mesh_rest.vertices
    vertices_driver = mesh_driver.vertices

    for pt_id in range(num_points):
        neighbors_rest   = sorted(list(adjacency_rest[pt_id]))
        neighbors_driver = sorted(list(adjacency_driver[pt_id]))
        
        ors_r, weights_r, scale_r = compute_rest_xform(mesh_rest, pt_id, neighbors_rest)
        ors_d, weights_d, scale_d = compute_rest_xform(mesh_driver, pt_id, neighbors_driver)
        
        all_rest_ors.append(ors_r)
        all_rest_weights.append(weights_r)
        all_rest_scales.append(scale_r)
        all_driver_ors.append(ors_d)
        all_driver_weights.append(weights_d)
        all_driver_scales.append(scale_d)

    # 计算 delta (dp, dor, ds)
    all_dp  = np.zeros((num_points, 3),     dtype=np.float64)
    all_dor = np.zeros((num_points, 3, 3),  dtype=np.float64)  # 旋转矩阵
    all_ds  = np.ones(num_points,           dtype=np.float64)

    for pt_id in range(num_points):
        rest_P      = vertices_rest[pt_id]
        driver_P    = vertices_driver[pt_id]
        rest_ors    = all_rest_ors[pt_id]
        driver_ors  = all_driver_ors[pt_id]
        rest_weights   = all_rest_weights[pt_id]
        driver_weights = all_driver_weights[pt_id]
        rest_scale     = all_rest_scales[pt_id]
        driver_scale   = all_driver_scales[pt_id]

        dorient, ds, dp = compute_delta(
            rest_P, driver_P,
            rest_ors, driver_ors,
            rest_weights, driver_weights,
            rest_scale, driver_scale
        )
        all_dor[pt_id] = dorient
        all_dp[pt_id]  = dp
        all_ds[pt_id]  = ds

    # --------------------------------------------------
    # 3) 模拟 VEX 里的 primuv + maketransform
    #    计算 wrapdeltam
    # --------------------------------------------------
    wrapdeltam = np.zeros((len(mesh_high.vertices), 4, 4), dtype=np.float64)

    faces_of_rest = mesh_rest.faces  # (n_faces, 3)

    for i in range(len(mesh_high.vertices)):
        prim_idx = wrapcloseprim[i]
        u, v = wrapcloseuv[i]
        w = 1.0 - u - v
        pivot = wrappivot[i]

        face_vids = faces_of_rest[prim_idx]  # 长度=3

        # ----------------
        # dp
        dp_verts = all_dp[face_vids]  # shape (3, 3)
        dp_final = barycentric_interpolate_vector(dp_verts, (w, v, u))

        # ----------------
        # dor
        dor_verts = all_dor[face_vids]  # shape (3, 3, 3)
        dor_final = barycentric_interpolate_matrix3(dor_verts, (w, v, u))

        # ----------------
        # ds
        ds_verts = all_ds[face_vids]  # shape (3,)
        ds_final = barycentric_interpolate_float(ds_verts, (w, v, u))


        # 构建 4×4 矩阵
        # dor_final => 四元数 => 欧拉角(度)
        q = matrix_to_quaternion(dor_final)
        drot = quaternion_to_eulerXYZ(q, degrees=True)  # (rx_deg, ry_deg, rz_deg)

        dm = make_4x4_transform(dp_final, drot, ds_final, pivot)
        wrapdeltam[i] = dm


    # --------------------------------------------------
    # 4) 形成形变后的高精度网格
    # --------------------------------------------------
    original_vertices = mesh_high.vertices
    original_faces    = mesh_high.faces
    num_high_points   = len(original_vertices)
    assert num_high_points == wrapdeltam.shape[0], \
        "wrapdeltam 个数与 high mesh 顶点数不匹配"

    # 对每个顶点做齐次变换
    # 若有需要，可将这一步进一步向量化处理
    deformed_vertices = np.zeros_like(original_vertices)

    for i in range(num_high_points):
        old_pos_homo = np.array([original_vertices[i,0],
                                 original_vertices[i,1],
                                 original_vertices[i,2],
                                 1.0], dtype=np.float64)
        new_pos_homo = wrapdeltam[i].dot(old_pos_homo)
        deformed_vertices[i] = new_pos_homo[:3]

    # 构造新的网格并导出
    deformed_mesh = trimesh.Trimesh(vertices=deformed_vertices,
                                    faces=original_faces,
                                    process=False)
    output_path = "wrapped_high_deformed.obj"
    deformed_mesh.export(output_path)
    print(f"已将驱动后的高精度网格保存到 {output_path}")
    