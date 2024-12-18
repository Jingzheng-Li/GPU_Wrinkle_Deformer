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
    # 首先建立点->邻居列表
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
        inv_lengths = np.array([1.0/l for l in edge_lengths[i]])
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
            neighbor_positions = vertices[nbrs]
            # 加权平均
            avg_pos = (neighbor_positions * w[:, None]).sum(axis=0)
            # 用step_size控制更新幅度
            new_positions[i] = vertices[i] + step_size * (avg_pos - vertices[i])

        vertices = new_positions

    return vertices

# 示例用法
if __name__ == "__main__":
    # 创建一个简单的三角形网格（或加载自己的网格）
    # 这里用一个三角面片组成的简单网格作为示例
    test_mesh = trimesh.load('../Assets/toomuchwrinkle.obj')
    
    smoothed_verts = delta_mush_smooth(test_mesh, iterations=2, step_size=0.5)
    