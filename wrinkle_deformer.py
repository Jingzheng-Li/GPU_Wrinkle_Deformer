import numpy as np
import trimesh

np.set_printoptions(precision=8, threshold=np.inf, linewidth=200)

class MeshData:
    def __init__(self, rest_pos, curr_pos, vertex_normals, constraints, stretchstiffness, compressstiffness, lagrange_multipliers, point_mass, faces):
        self.rest_pos = rest_pos
        self.curr_pos = curr_pos
        self.vertex_normals = vertex_normals
        self.constraints = constraints
        self.stretchstiffness = stretchstiffness
        self.compressstiffness = compressstiffness
        self.lagrange_multipliers = lagrange_multipliers
        self.point_mass = point_mass
        self.faces = faces

def load_mesh(input_path):
    mesh = trimesh.load(input_path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"加载的文件不包含有效的Trimesh对象: {input_path}")
    return mesh

def check_topology(mesh1, mesh2):
    if mesh1.faces.shape != mesh2.faces.shape:
        raise ValueError("两个网格的面数量不同。")
    if not np.array_equal(mesh1.faces, mesh2.faces):
        raise ValueError("两个网格的面索引不相同。")
    if mesh1.vertices.shape != mesh2.vertices.shape:
        raise ValueError("两个网格的顶点数量不同。")

def find_bend_pairs(mesh):
    adjacency = mesh.face_adjacency
    shared_edges = mesh.face_adjacency_edges
    faces = mesh.faces
    vertices = mesh.vertices

    C_bend = adjacency.shape[0]
    v1 = np.empty(C_bend, dtype=np.int32)
    v2 = np.empty(C_bend, dtype=np.int32)
    distance = np.empty(C_bend, dtype=np.float32)

    for idx, (face_a, face_b) in enumerate(adjacency):
        shared = shared_edges[idx]
        unique_a = np.setdiff1d(faces[face_a], shared)
        unique_b = np.setdiff1d(faces[face_b], shared)
        if len(unique_a) != 1 or len(unique_b) != 1:
            v1[idx] = -1
            v2[idx] = -1
            distance[idx] = 0.0
            continue
        v1[idx] = unique_a[0]
        v2[idx] = unique_b[0]
        distance[idx] = np.linalg.norm(vertices[v1[idx]] - vertices[v2[idx]])

    valid = v1 >= 0
    bend_constraints = np.vstack((
        v1[valid],
        v2[valid],
        distance[valid],
        np.ones_like(distance[valid])  # type=1 表示 bend constraint
    )).T.astype(np.float32)

    return bend_constraints

def find_unique_edges_constraints(mesh):
    unique_edges = mesh.edges_unique
    vertices = mesh.vertices
    pos1 = vertices[unique_edges[:, 0]]
    pos2 = vertices[unique_edges[:, 1]]
    distance = np.linalg.norm(pos1 - pos2, axis=1)

    edge_constraints = np.hstack((
        unique_edges,
        distance.reshape(-1, 1),
        np.zeros((unique_edges.shape[0], 1))  # type=0 表示 edge constraint
    )).astype(np.float32)

    return edge_constraints

def merge_constraints(bend_constraints, edge_constraints):
    merged_constraints = np.vstack((bend_constraints, edge_constraints))
    return merged_constraints

def compute_vertex_normals(mesh):
    if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
        mesh.compute_vertex_normals()
    return mesh.vertex_normals

def prepare_mesh_data(
    rest_mesh_path, 
    bend_mesh_path, 
    mass, 
    stretchstiffness, 
    compressstiffness, 
):
    rest_mesh = load_mesh(rest_mesh_path)
    bend_mesh = load_mesh(bend_mesh_path)
    check_topology(rest_mesh, bend_mesh)

    rest_pos = rest_mesh.vertices.copy()
    curr_pos = bend_mesh.vertices.copy()
    vertex_normals = compute_vertex_normals(bend_mesh)

    bend_constraints = find_bend_pairs(rest_mesh)
    edge_constraints = find_unique_edges_constraints(rest_mesh)
    merged_constraints = merge_constraints(bend_constraints, edge_constraints)

    C_total = merged_constraints.shape[0]

    stretchstiffness_array = np.full(C_total, stretchstiffness, dtype=np.float32)
    compressstiffness_array = np.full(C_total, compressstiffness, dtype=np.float32)

    lagrange_multipliers = np.zeros(C_total, dtype=np.float32)

    if isinstance(mass, (int, float)):
        point_mass = np.full(rest_pos.shape[0], mass, dtype=np.float32)
    else:
        point_mass = np.array(mass, dtype=np.float32)
        if point_mass.shape[0] != rest_pos.shape[0]:
            raise ValueError("mass 数组的长度必须与顶点数量相同。")

    mesh_data = MeshData(
        rest_pos=rest_pos,
        curr_pos=curr_pos,
        vertex_normals=vertex_normals,
        constraints=merged_constraints,
        stretchstiffness=stretchstiffness_array,
        compressstiffness=compressstiffness_array,
        lagrange_multipliers=lagrange_multipliers,
        point_mass=point_mass,
        faces=rest_mesh.faces.copy()
    )

    return mesh_data

def xpbd_iteration_loop(mesh_data, time_step):
    rest_pos = mesh_data.rest_pos
    curr_pos = mesh_data.curr_pos.copy()  # 复制当前位置信息
    vertex_normals = mesh_data.vertex_normals
    constraints = mesh_data.constraints
    stretchstiffness = mesh_data.stretchstiffness
    compressstiffness = mesh_data.compressstiffness
    lagrange_multipliers = mesh_data.lagrange_multipliers.copy()  # 复制拉格朗日乘子
    point_mass = mesh_data.point_mass

    inv_mass = 1.0 / point_mass

    C_total = constraints.shape[0]

    # 初始化位移累积数组
    dP = np.zeros_like(curr_pos, dtype=np.float32)  # 每个顶点的位移累积
    dPw = np.zeros(curr_pos.shape[0], dtype=np.float32)  # 每个顶点的权重累积

    for i in range(C_total):
        v1 = int(constraints[i, 0])
        v2 = int(constraints[i, 1])
        rest_length = constraints[i, 2]
        p1 = curr_pos[v1]
        p2 = curr_pos[v2]
        n = p2 - p1
        d = np.linalg.norm(n)
        if d < 1e-6:
            continue
        n /= d  # 归一化向量

        # 确定拉伸或压缩刚度
        if d < rest_length:
            k = compressstiffness[i]
        else:
            k = stretchstiffness[i]

        if k < 1e-6:
            continue

        alpha = 1.0 / k
        alpha /= (time_step ** 2)

        C = d - rest_length

        wsum = inv_mass[v1] + inv_mass[v2]

        denominator = wsum + alpha
        if denominator == 0.0:
            denominator = 1e-8  # 防止除零错误

        delta_lambda = (-C - alpha * lagrange_multipliers[i]) / denominator

        delta_p1 = inv_mass[v1] * n * (-delta_lambda)
        delta_p2 = inv_mass[v2] * n * (delta_lambda)

        # 累积位移
        dP[v1] += delta_p1
        dP[v2] += delta_p2

        # 累积权重
        dPw[v1] += 1.0
        dPw[v2] += 1.0

        # 更新拉格朗日乘子
        lagrange_multipliers[i] += delta_lambda

    # 更新所有顶点的位置
    valid = dPw > 0.0
    # 为了确保维度匹配，将dPw的维度扩展以进行广播
    curr_pos[valid] += dP[valid] / dPw[valid, np.newaxis]

    # 更新MeshData中的当前位置和拉格朗日乘子
    mesh_data.curr_pos = curr_pos
    mesh_data.lagrange_multipliers = lagrange_multipliers

    return mesh_data

def perform_xpbd(mesh_data, iterations, time_step):
    for it in range(iterations):
        mesh_data = xpbd_iteration_loop(mesh_data, time_step)
        if it % 10 == 0 or it == iterations - 1:
            print(f"Iteration {it+1}/{iterations} completed.")
    return mesh_data

if __name__ == "__main__":
    rest_mesh_path = 'Assets/tubemesh.obj'
    bend_mesh_path = 'Assets/tubemesh_bend.obj'
    output_deformed_path = 'Assets/tubemesh_deformed.obj'

    try:
        mesh_data = prepare_mesh_data(
            rest_mesh_path,
            bend_mesh_path,
            mass=2.22505e-05,
            stretchstiffness=10.0,
            compressstiffness=10.0
        )
    except Exception as e:
        print(f"处理网格数据时出错: {e}")
        exit(1)

    # 设定迭代次数和时间步长
    iterations = 200  # 与OpenCL代码中运行200次一致
    time_step = 0.033333  # 例如每帧的时间步长

    mesh_data = perform_xpbd(mesh_data, iterations=iterations, time_step=time_step)

    try:
        updated_mesh = trimesh.Trimesh(
            vertices=mesh_data.curr_pos,
            faces=mesh_data.faces,
            process=False
        )
        updated_mesh.export(output_deformed_path)
        print(f"变形后的网格已导出到 {output_deformed_path}")
    except Exception as e:
        print(f"导出网格时出错: {e}")
