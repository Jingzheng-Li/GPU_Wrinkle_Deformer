// Deformer.cpp

#include "Deformer.hpp"
#include "CUDAUtils.hpp"
#include "LoadMesh.hpp"

#include <unordered_set>
#include <cmath>      // for std::sqrt
#include <iostream>

// ========== 原有 Deformer 构造与主要函数 ==========

Deformer::Deformer(SimulationContext& context, Simulator& simulator)
    : ctx(context), sim(simulator) {}

void Deformer::DeformerPipeline() {
    getHostMesh(ctx.instance);
}

void checkTopology(const CGAL_Mesh& mesh1, const CGAL_Mesh& mesh2)
{
    if (num_faces(mesh1) != num_faces(mesh2)) {
        throw std::runtime_error("两个网格的面数量不同");
    }
    if (num_vertices(mesh1) != num_vertices(mesh2)) {
        throw std::runtime_error("两个网格的顶点数量不同");
    }
    // 如果还需非常严格地面索引一致，可以在这里加更多检查
}

std::vector<std::array<Scalar, 4>> Deformer::findBendPairs(const CGAL_Mesh& mesh)
{
    std::vector<std::array<Scalar, 4>> bend_constraints;
    std::unordered_map<size_t, bool> visited; // 仅示例，不一定最优

    for (CGAL_Halfedge_index h : mesh.halfedges()) {
        CGAL_Face_index fA = face(h, mesh);
        CGAL_Face_index fB = face(opposite(h, mesh), mesh);

        if (fA == CGAL_Mesh::null_face() || fB == CGAL_Mesh::null_face()) {
            // 表示此半边在网格边界，只有一个面
            continue;
        }

        // 让 fA < fB，以免重复
        size_t a = fA;
        size_t b = fB;
        if (a > b) std::swap(a, b);

        size_t hashVal = (a << 32) ^ (b & 0xffffffff);
        if (visited.find(hashVal) != visited.end()) {
            continue; // 已处理过
        }
        visited[hashVal] = true;

        // faceA_indices
        std::vector<CGAL_Vertex_index> faceA_indices;
        {
            CGAL_Halfedge_index ha = mesh.halfedge(fA);
            for (int i = 0; i < 3; ++i) {
                faceA_indices.push_back(target(ha, mesh));
                ha = mesh.next(ha);
            }
        }
        // faceB_indices
        std::vector<CGAL_Vertex_index> faceB_indices;
        {
            CGAL_Halfedge_index hb = mesh.halfedge(fB);
            for (int i = 0; i < 3; ++i) {
                faceB_indices.push_back(target(hb, mesh));
                hb = mesh.next(hb);
            }
        }

        // 找共享点
        std::vector<CGAL_Vertex_index> shared;
        for (auto va : faceA_indices) {
            for (auto vb : faceB_indices) {
                if (va == vb) {
                    shared.push_back(va);
                }
            }
        }

        // 如果共享边恰好是 2 个顶点，则剩下各有一个“对顶点”
        if (shared.size() == 2) {
            CGAL_Vertex_index uniqueA, uniqueB;
            for (auto va : faceA_indices) {
                if (va != shared[0] && va != shared[1]) {
                    uniqueA = va;
                    break;
                }
            }
            for (auto vb : faceB_indices) {
                if (vb != shared[0] && vb != shared[1]) {
                    uniqueB = vb;
                    break;
                }
            }
            auto pA = mesh.point(uniqueA);
            auto pB = mesh.point(uniqueB);
            Scalar dist = std::sqrt(CGAL::squared_distance(pA, pB));

            bend_constraints.push_back({
                (Scalar)uniqueA, (Scalar)uniqueB, dist, 1.0
            });
        }
    }
    return bend_constraints;
}

std::vector<std::array<Scalar, 4>> Deformer::findUniqueEdgesConstraints(const CGAL_Mesh& mesh)
{
    std::vector<std::array<Scalar, 4>> edge_constraints;
    edge_constraints.reserve(num_edges(mesh));

    for (auto e : mesh.edges()) {
        auto h = halfedge(e, mesh);
        CGAL_Vertex_index v1 = target(h, mesh);
        CGAL_Vertex_index v2 = target(opposite(h, mesh), mesh);

        auto p1 = mesh.point(v1);
        auto p2 = mesh.point(v2);
        Scalar dist = std::sqrt(CGAL::squared_distance(p1, p2));

        edge_constraints.push_back({
            (Scalar)v1, (Scalar)v2, dist, 0.0
        });
    }
    return edge_constraints;
}

std::vector<std::array<Scalar, 4>> Deformer::mergeConstraints(
    const std::vector<std::array<Scalar, 4>>& bend_constraints,
    const std::vector<std::array<Scalar, 4>>& edge_constraints
)
{
    std::vector<std::array<Scalar, 4>> merged;
    merged.reserve(bend_constraints.size() + edge_constraints.size());
    merged.insert(merged.end(), bend_constraints.begin(), bend_constraints.end());
    merged.insert(merged.end(), edge_constraints.begin(), edge_constraints.end());
    return merged;
}

std::vector<CGAL_Vector_3> Deformer::computeVertexNormals(const CGAL_Mesh& mesh)
{
    std::vector<CGAL_Vector_3> normals(num_vertices(mesh));

    auto vnorm_map = CGAL::make_property_map(normals);
    CGAL::Polygon_mesh_processing::compute_vertex_normals(
        mesh,
        vnorm_map,
        CGAL::Polygon_mesh_processing::parameters::vertex_point_map(mesh.points())
    );

    return normals;
}

void Deformer::xpbdIterationLoop(MeshData& mesh_data, Scalar time_step)
{
    auto& curr_pos = mesh_data.curr_pos;
    auto& constraints = mesh_data.constraints;
    auto& stretch_stiff  = mesh_data.stretch_stiffness;
    auto& compress_stiff = mesh_data.compress_stiffness;
    auto& lambdas        = mesh_data.lagrange_multipliers;
    auto& masses         = mesh_data.point_mass;

    size_t nv = curr_pos.size();
    size_t nc = constraints.size();
    if (nv == 0 || nc == 0) return;

    // 1 / mass
    std::vector<Scalar> inv_mass(nv);
    for (size_t i = 0; i < nv; ++i) {
        inv_mass[i] = (masses[i] == 0.0) ? 0.0 : (1.0 / masses[i]);
    }

    // displacement
    std::vector<CGAL_Vector_3> dP(nv, CGAL_Vector_3(0,0,0));
    std::vector<Scalar>        dPw(nv, 0.0);

    for (size_t i = 0; i < nc; ++i) {
        int v1 = (int)constraints[i][0];
        int v2 = (int)constraints[i][1];
        Scalar rest_length = constraints[i][2];
        // ctype = constraints[i][3]; // 1 => bend, 0 => edge

        CGAL_Vector_3 dir = curr_pos[v2] - curr_pos[v1];
        Scalar d = std::sqrt(dir.squared_length());
        if (d < 1e-12) {
            continue;
        }
        dir = dir / d;

        Scalar stiffness = (d < rest_length) ? compress_stiff[i] : stretch_stiff[i];
        if (stiffness < 1e-12) {
            continue;
        }

        Scalar alpha = (1.0 / stiffness) / (time_step * time_step);
        Scalar C     = (d - rest_length);
        Scalar wsum  = inv_mass[v1] + inv_mass[v2];
        Scalar denom = wsum + alpha;
        if (std::fabs(denom) < 1e-12) {
            denom = 1e-12;
        }

        Scalar delta_lambda = (-C - alpha * lambdas[i]) / denom;
        CGAL_Vector_3 dp1   = -delta_lambda * inv_mass[v1] * dir;
        CGAL_Vector_3 dp2   =  delta_lambda * inv_mass[v2] * dir;

        dP[v1]  = dP[v1]  + dp1;
        dP[v2]  = dP[v2]  + dp2;
        dPw[v1] = dPw[v1] + 1.0;
        dPw[v2] = dPw[v2] + 1.0;

        lambdas[i] += delta_lambda;
    }

    // update positions
    for (size_t i = 0; i < nv; ++i) {
        if (dPw[i] > 1e-12) {
            Scalar w = 1.0 / dPw[i];
            CGAL_Vector_3 shift = dP[i] * w;
            curr_pos[i] = curr_pos[i] + shift;
        }
    }
}

void Deformer::performXPBD(MeshData& mesh_data, int iterations, Scalar time_step)
{
    for (int it = 0; it < iterations; ++it) {
        xpbdIterationLoop(mesh_data, time_step);
        if (it % 10 == 0 || it == iterations - 1) {
            std::cout << "[INFO] Iteration " << it+1 << "/" << iterations << " done.\n";
        }
    }
}

// 与 Python delta_mush_smooth 对应的 C++ 版本
// -------------------------------------------------------------
void Deformer::deltaMushSmooth(MeshData& mesh_data, int iterations, Scalar step_size)
{
    // 1) 把当前顶点位置拷贝到临时数组 positions 中
    size_t nV = mesh_data.curr_pos.size();
    if (nV == 0) {
        std::cerr << "[deltaMush] no vertices found.\n";
        return;
    }
    std::vector<CGAL_Point_3> positions = mesh_data.curr_pos;

    // 2) 构建邻接表 adjacency[i] 存储顶点 i 的邻居
    //    同时记录与邻居的边长 adj_lengths[i]
    std::vector<std::vector<int>>    adjacency(nV);
    std::vector<std::vector<Scalar>> adj_lengths(nV);

    // 利用 mesh_data.faces 中的三角面来找边
    // 为防止重复，存储到一个 set
    // (min, max) 形式表示一条边
    struct EdgeKey {
        int v1, v2;
        EdgeKey(int a, int b) {
            if(a<b){v1=a;v2=b;}else{v1=b;v2=a;}
        }
    };
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& e) const {
            // 64bit: (v1 << 32) ^ v2
            // 也可以用 std::hash<int> 组合
            return (std::size_t)e.v1 ^ ((std::size_t)e.v2<<16);
        }
    };
    struct EdgeKeyEq {
        bool operator()(const EdgeKey& e1, const EdgeKey& e2) const {
            return (e1.v1 == e2.v1 && e1.v2 == e2.v2);
        }
    };
    std::unordered_set<EdgeKey, EdgeKeyHash, EdgeKeyEq> edges_set;
    edges_set.reserve(mesh_data.faces.size() * 3);

    for (auto& f : mesh_data.faces) {
        // f 是三角面 [v0,v1,v2]
        int v0 = f[0], v1 = f[1], v2 = f[2];
        edges_set.insert(EdgeKey(v0,v1));
        edges_set.insert(EdgeKey(v1,v2));
        edges_set.insert(EdgeKey(v2,v0));
    }

    // 遍历 edges_set，填充 adjacency
    for (auto& eKey : edges_set) {
        int v1 = eKey.v1;
        int v2 = eKey.v2;

        CGAL_Point_3 p1 = positions[v1];
        CGAL_Point_3 p2 = positions[v2];
        Scalar length = std::sqrt(CGAL::squared_distance(p1, p2));

        adjacency[v1].push_back(v2);
        adjacency[v2].push_back(v1);

        adj_lengths[v1].push_back(length);
        adj_lengths[v2].push_back(length);
    }

    // 3) 计算加权： weight = (1 / length) / sum(1/length)
    std::vector<std::vector<Scalar>> weights(nV);
    for (size_t i = 0; i < nV; ++i) {
        auto& lens = adj_lengths[i];
        auto& nbrs = adjacency[i];
        if (lens.empty() || nbrs.empty()) {
            continue; // 孤立点或无邻居
        }
        Scalar inv_sum = 0.0;
        weights[i].resize(lens.size());
        for (size_t k = 0; k < lens.size(); ++k) {
            Scalar inv_l = (lens[k] > 1e-15) ? (1.0 / lens[k]) : 0.0;
            weights[i][k] = inv_l;
            inv_sum       += inv_l;
        }
        if (inv_sum > 1e-15) {
            for (size_t k = 0; k < lens.size(); ++k) {
                weights[i][k] /= inv_sum;
            }
        }
    }

    // 4) 开始迭代平滑
    for (int it = 0; it < iterations; ++it) {
        std::vector<CGAL_Point_3> new_positions = positions;

        for (size_t i = 0; i < nV; ++i) {
            auto& nbrs = adjacency[i];
            auto& w    = weights[i];
            if (nbrs.empty() || w.empty() || nbrs.size() != w.size()) {
                continue;
            }
            // 计算邻居加权平均
            CGAL_Vector_3 sum_vec(0,0,0);
            for (size_t k = 0; k < nbrs.size(); ++k) {
                CGAL_Vector_3 v = positions[nbrs[k]] - CGAL::ORIGIN;
                sum_vec = sum_vec + (v * w[k]);
            }
            CGAL_Point_3 avg_pos = CGAL::ORIGIN + sum_vec;

            // diff
            CGAL_Vector_3 diff = avg_pos - positions[i];
            // 用 step_size 控制更新幅度
            new_positions[i] = positions[i] + diff * step_size;
        }

        positions = std::move(new_positions);
    }

    // 5) 将结果写回 mesh_data_.curr_pos
    for (size_t i = 0; i < nV; ++i) {
        mesh_data.curr_pos[i] = positions[i];
    }
    std::cout << "[deltaMushSmooth] Done " << iterations
              << " iterations, step=" << step_size << std::endl;
}
// -------------------------------------------------------------


void exportMesh(const std::string& output_path, const MeshData& mesh_data)
{
    CGAL_Mesh out_mesh;
    std::vector<CGAL_Vertex_index> idxMap;
    idxMap.reserve(mesh_data.curr_pos.size());
    for (auto& p : mesh_data.curr_pos) {
        idxMap.push_back(out_mesh.add_vertex(p));
    }

    for (auto& f : mesh_data.faces) {
        out_mesh.add_face(idxMap[f[0]], idxMap[f[1]], idxMap[f[2]]);
    }

    if (!CGAL::IO::write_polygon_mesh(output_path, out_mesh)) {
        std::cerr << "[ERROR] Failed to export mesh to " << output_path << std::endl;
    } else {
        std::cout << "[INFO] Mesh exported to " << output_path << std::endl;
    }
}

// -------------- 准备网格数据 --------------
void Deformer::prepareMeshData(
    const std::string& rest_mesh_path,
    const std::string& bend_mesh_path,
    Scalar mass,
    Scalar stretchStiffness,
    Scalar compressStiffness
)
{
    CGAL_Mesh rest_mesh, bend_mesh;
    if (!LOADMESH::CGAL_readObj(rest_mesh_path, rest_mesh) 
     || !LOADMESH::CGAL_readObj(bend_mesh_path, bend_mesh)) {
        std::cerr << "[ERROR] Failed to load .obj files.\n";
        return;
    }

    // 拓扑检查
    checkTopology(rest_mesh, bend_mesh);

    // 顶点
    mesh_data_.rest_pos.reserve(num_vertices(rest_mesh));
    mesh_data_.curr_pos.reserve(num_vertices(rest_mesh));
    {
        for (CGAL_Vertex_index v : rest_mesh.vertices()) {
            auto p = rest_mesh.point(v);
            mesh_data_.rest_pos.push_back(p);
        }
        for (CGAL_Vertex_index v : bend_mesh.vertices()) {
            auto p = bend_mesh.point(v);
            mesh_data_.curr_pos.push_back(p);
        }
    }

    // 顶点法线
    mesh_data_.vertex_normals = computeVertexNormals(bend_mesh);

    // 约束
    auto bend_constraints = findBendPairs(rest_mesh);
    auto edge_constraints = findUniqueEdgesConstraints(rest_mesh);
    auto merged_constraints = mergeConstraints(bend_constraints, edge_constraints);
    mesh_data_.constraints = merged_constraints;

    mesh_data_.stretch_stiffness.resize(merged_constraints.size(), stretchStiffness);
    mesh_data_.compress_stiffness.resize(merged_constraints.size(), compressStiffness);
    mesh_data_.lagrange_multipliers.resize(merged_constraints.size(), 0.0);

    // 统一的点质量
    mesh_data_.point_mass.resize(num_vertices(rest_mesh), mass);

    // faces
    for (CGAL_Face_index f : rest_mesh.faces()) {
        std::vector<int> indices;
        CGAL_Halfedge_index h = rest_mesh.halfedge(f);
        for (int i=0; i<3; ++i) {
            auto v = rest_mesh.target(h);
            indices.push_back((int)v);
            h = rest_mesh.next(h);
        }
        if (indices.size()==3) {
            mesh_data_.faces.push_back({ indices[0], indices[1], indices[2] });
        }
    }
}

// -------------- 在 getHostMesh 中调用 --------------
void Deformer::getHostMesh(std::unique_ptr<GeometryManager>& instance) {

    std::string rest_mesh_path  = "../../Assets/tubemesh.obj";
    std::string bend_mesh_path  = "../../Assets/tubemesh_bend.obj";
    std::string output_deformed = "../../Assets/tubemesh_deformed.obj";

    prepareMeshData(rest_mesh_path, bend_mesh_path,
                    /*mass*/ 2.22505e-5, 
                    /*stretchStiffness*/ 10.0,
                    /*compressStiffness*/ 10.0);

    // 先做 XPBD
    int    iterations = 200;
    Scalar time_step  = 0.033333;
    performXPBD(mesh_data_, iterations, time_step);

    // 然后可选：delta mush 平滑
    // 例如：进行 5 次迭代，每次 step_size=0.5
    deltaMushSmooth(mesh_data_, 2, 0.5);

    // 导出
    exportMesh(output_deformed, mesh_data_);
    std::cout << "[INFO] Deformation completed and exported.\n";
}
