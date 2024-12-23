import numpy as np
import trimesh

def delta_mush_smooth(mesh, iterations=10, step_size=0.5):
    """
    对 mesh 的顶点进行加权平滑（基于边长权重），类似 delta mush 的一部分功能。
    
    参数：
        mesh: trimesh.Trimesh 对象
        iterations: int, 平滑迭代次数
        step_size: float, 每次迭代的步长（介于0~1间为宜）
        
    返回：
        平滑后的新顶点坐标数组 (V, 3)
    """
    vertices = mesh.vertices.copy()
    edges = mesh.edges_unique

    # 构建邻接关系：对每个点列出邻居及相应的边长权重
    num_vertices = len(vertices)
    adjacency_list = [[] for _ in range(num_vertices)]
    edge_lengths = [[] for _ in range(num_vertices)]

    # 遍历每条边，记录两端的邻居及边长
    for e in edges:
        v1, v2 = e
        p1, p2 = vertices[v1], vertices[v2]
        length = np.linalg.norm(p2 - p1)
        adjacency_list[v1].append(v2)
        adjacency_list[v2].append(v1)
        edge_lengths[v1].append(length)
        edge_lengths[v2].append(length)

    # 根据边长计算权重：使用 1 / length 来给较短的边更大权重
    # 并对每个点的权重归一化，使得所有邻居权重和为 1
    weights = []
    for i in range(num_vertices):
        if len(edge_lengths[i]) == 0:
            weights.append(np.array([]))
            continue
        inv_lengths = np.array([1.0 / l for l in edge_lengths[i]])
        w = inv_lengths / inv_lengths.sum()
        weights.append(w)
    weights = np.array(weights, dtype=object)  # object类型数组，每个元素是不等长的权重列表

    # 开始迭代平滑
    for _ in range(iterations):
        new_positions = vertices.copy()
        for i in range(num_vertices):
            # 获得该点的邻居坐标与权重
            nbrs = adjacency_list[i]
            if len(nbrs) == 0:
                # 孤立点，不处理
                continue
            w = weights[i]
            if w.size == 0:
                continue
            neighbor_positions = vertices[nbrs]
            # 加权平均
            avg_pos = (neighbor_positions * w[:, None]).sum(axis=0)
            # 用step_size控制更新幅度
            new_positions[i] = vertices[i] + step_size * (avg_pos - vertices[i])

        vertices = new_positions

    return vertices

# 示例用法
if __name__ == "__main__":
    # 加载原始网格
    input_path = 'tempwrinkle_0delta.obj'  # 请确保路径正确
    test_mesh = trimesh.load(input_path)

    # 应用 Delta Mush 平滑
    smoothed_verts = delta_mush_smooth(test_mesh, iterations=5, step_size=0.5)

    # 将平滑后的顶点赋值回网格
    test_mesh.vertices = smoothed_verts

    # 保存平滑后的网格为 OBJ 文件
    output_path = 'smoothed_tempwrinkle_0delta.obj'  # 设定输出路径
    test_mesh.export(output_path)
    print(f"平滑后的网格已保存至 {output_path}")
