
import trimesh
import numpy as np
from collections import defaultdict


# 步骤3: 加载OBJ文件
loaded_mesh = trimesh.load('../Assets/gridmesh.obj')

# 步骤4: 提取唯一边并计算长度
unique_edges = loaded_mesh.edges_unique
vertices = loaded_mesh.vertices
edge_vectors = vertices[unique_edges[:, 1]] - vertices[unique_edges[:, 0]]
edge_lengths = np.linalg.norm(edge_vectors, axis=1)

# 步骤5: 保存约束信息
constraints = []
for i, (v0, v1) in enumerate(unique_edges):
    length = edge_lengths[i]
    constraints.append({
        'constraint_id': i + 1,
        'vertex1': int(v0),
        'vertex2': int(v1),
        'length': round(float(length), 5)
    })

# 打印约束信息
for constraint in constraints:
    print(f"约束{constraint['constraint_id']}: {constraint['vertex1']}, {constraint['vertex2']}, 长度={constraint['length']}")

# 步骤6: 验证约束数量和长度
length_counts = defaultdict(int)
for constraint in constraints:
    length = constraint['length']
    length_counts[length] += 1

print("\n约束长度统计：")
for length, count in length_counts.items():
    print(f"长度 {length}: {count} 条")