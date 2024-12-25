// Deformer.cpp

#include "Deformer.hpp"
#include "WrinkleDeformer.cuh"
#include "WrapDeformer.cuh"

#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <iostream>
#include <vector>

// 导出网格函数（供查看结果）
static void exportMesh(const std::string& output_path, const MeshData& mesh_data) {
    CGAL_Mesh out_mesh;
    out_mesh.reserve(mesh_data.curr_pos.size(),
                     mesh_data.faces.size() * 3,
                     mesh_data.faces.size());
    std::vector<CGAL_Vertex_index> idx_map;
    idx_map.reserve(mesh_data.curr_pos.size());
    for (const auto& p : mesh_data.curr_pos) {
        idx_map.push_back(out_mesh.add_vertex(p));
    }
    for (const auto& f : mesh_data.faces) {
        out_mesh.add_face(idx_map[f.x], idx_map[f.y], idx_map[f.z]);
    }
    if (!CGAL::IO::write_polygon_mesh(output_path, out_mesh)) {
        std::cerr << "Export failed " << output_path << std::endl;
    }
}

Deformer::Deformer(SimulationContext& context, Simulator& simulator)
    : ctx(context), sim(simulator) { }

// 查找弯曲约束对
static std::vector<Scalar3> findBendPairs(const CGAL_Mesh& mesh) {
    std::vector<Scalar3> bend_constraints;
    std::unordered_map<size_t, bool> visited;
    bend_constraints.reserve(num_halfedges(mesh) / 2);

    for (const CGAL_Halfedge_index& h : mesh.halfedges()) {
        CGAL_Face_index fA = face(h, mesh);
        CGAL_Face_index fB = face(opposite(h, mesh), mesh);
        if (fA == CGAL_Mesh::null_face() || fB == CGAL_Mesh::null_face()) 
            continue;

        size_t a = static_cast<size_t>(fA);
        size_t b = static_cast<size_t>(fB);
        if (a > b) std::swap(a, b);
        size_t hash_val = (a << 32) ^ (b & 0xFFFFFFFF);
        if (visited.find(hash_val) != visited.end()) 
            continue;
        visited[hash_val] = true;

        std::vector<CGAL_Vertex_index> fa_idx, fb_idx;
        {
            CGAL_Halfedge_index ha = mesh.halfedge(fA);
            for (int i = 0; i < 3; ++i) {
                fa_idx.push_back(target(ha, mesh));
                ha = mesh.next(ha);
            }
        }
        {
            CGAL_Halfedge_index hb = mesh.halfedge(fB);
            for (int i = 0; i < 3; ++i) {
                fb_idx.push_back(target(hb, mesh));
                hb = mesh.next(hb);
            }
        }
        std::vector<CGAL_Vertex_index> shared;
        for (const auto& va : fa_idx) {
            for (const auto& vb : fb_idx) {
                if (va == vb) shared.push_back(va);
            }
        }
        if (shared.size() == 2) {
            CGAL_Vertex_index uniqueA, uniqueB;
            for (const auto& va : fa_idx) {
                if (va != shared[0] && va != shared[1]) {
                    uniqueA = va;
                    break;
                }
            }
            for (const auto& vb : fb_idx) {
                if (vb != shared[0] && vb != shared[1]) {
                    uniqueB = vb;
                    break;
                }
            }
            auto pA = mesh.point(uniqueA);
            auto pB = mesh.point(uniqueB);
            Scalar dist = std::sqrt(CGAL::squared_distance(pA, pB));
            bend_constraints.emplace_back(Scalar3{
                static_cast<Scalar>(uniqueA),
                static_cast<Scalar>(uniqueB),
                dist
            });
        }
    }
    return bend_constraints;
}

// 查找独特边约束
static std::vector<Scalar3> findUniqueEdgesConstraints(const CGAL_Mesh& mesh) {
    std::vector<Scalar3> edge_constraints;
    edge_constraints.reserve(num_edges(mesh));
    for (const auto& e : mesh.edges()) {
        auto h = halfedge(e, mesh);
        CGAL_Vertex_index v1 = target(h, mesh);
        CGAL_Vertex_index v2 = target(opposite(h, mesh), mesh);
        auto p1 = mesh.point(v1);
        auto p2 = mesh.point(v2);
        Scalar dist = std::sqrt(CGAL::squared_distance(p1, p2));
        edge_constraints.emplace_back(Scalar3{
            static_cast<Scalar>(v1),
            static_cast<Scalar>(v2),
            dist
        });
    }
    return edge_constraints;
}

// 合并约束
static std::vector<Scalar3> mergeConstraints(
    const std::vector<Scalar3>& bend_constraints,
    const std::vector<Scalar3>& edge_constraints
) {
    std::vector<Scalar3> merged;
    merged.reserve(bend_constraints.size() + edge_constraints.size());
    merged.insert(merged.end(), bend_constraints.begin(), bend_constraints.end());
    merged.insert(merged.end(), edge_constraints.begin(), edge_constraints.end());
    return merged;
}

void Deformer::prepareMeshData(const std::string& rest_mesh_path,
                               const std::string& bend_mesh_path,
                               Scalar mass,
                               Scalar stretchStiffness,
                               Scalar compressStiffness)
{
    CGAL_Mesh rest_mesh, bend_mesh;
    if (!LOADMESH::CGAL_readObj(rest_mesh_path, rest_mesh) ||
        !LOADMESH::CGAL_readObj(bend_mesh_path, bend_mesh)) {
        std::cerr << "Load obj failed\n";
        return;
    }
    mesh_data_.rest_pos.reserve(num_vertices(rest_mesh));
    mesh_data_.curr_pos.reserve(num_vertices(rest_mesh));

    for (const CGAL_Vertex_index& v : rest_mesh.vertices()) {
        mesh_data_.rest_pos.emplace_back(rest_mesh.point(v));
    }
    for (const CGAL_Vertex_index& v : bend_mesh.vertices()) {
        mesh_data_.curr_pos.emplace_back(bend_mesh.point(v));
    }

    // 构造约束
    auto bend_constraints  = findBendPairs(rest_mesh);
    auto edge_constraints  = findUniqueEdgesConstraints(rest_mesh);
    auto merged_constraints = mergeConstraints(bend_constraints, edge_constraints);
    mesh_data_.constraints = merged_constraints;

    mesh_data_.stretch_stiffness  = stretchStiffness;
    mesh_data_.compress_stiffness = compressStiffness;
    mesh_data_.lagrange_multipliers.resize(merged_constraints.size(), 0.0);
    mesh_data_.point_mass.resize(num_vertices(rest_mesh), mass);

    // faces
    for (const CGAL_Face_index& f : rest_mesh.faces()) {
        std::vector<int> indices;
        indices.reserve(3);
        CGAL_Halfedge_index h = rest_mesh.halfedge(f);
        for (int i = 0; i < 3; ++i) {
            indices.push_back(static_cast<int>(rest_mesh.target(h)));
            h = rest_mesh.next(h);
        }
        mesh_data_.faces.emplace_back(uint3{
            static_cast<unsigned>(indices[0]),
            static_cast<unsigned>(indices[1]),
            static_cast<unsigned>(indices[2])
        });
    }
}

void Deformer::getHostMesh_CUDA(std::unique_ptr<GeometryManager>& instance) {
    std::string rest_mesh_path  = "../Assets/tubemesh.obj";
    std::string bend_mesh_path  = "../Assets/tubemesh_bend.obj";
    // 这里演示，给定一些参数
    prepareMeshData(rest_mesh_path, bend_mesh_path, 
                    2.22505e-5, /* mass */
                    10.0,       /* stretchStiffness */
                    10.0        /* compressStiffness */
    );

    int nv = static_cast<int>(mesh_data_.curr_pos.size());
    int nc = static_cast<int>(mesh_data_.constraints.size());

    // 构建邻接列表
    std::vector<std::unordered_set<int>> adjacency_lists(nv);
    for (const auto& face : mesh_data_.faces) {
        int v0 = face.x, v1 = face.y, v2 = face.z;
        adjacency_lists[v0].insert(v1);
        adjacency_lists[v0].insert(v2);
        adjacency_lists[v1].insert(v0);
        adjacency_lists[v1].insert(v2);
        adjacency_lists[v2].insert(v0);
        adjacency_lists[v2].insert(v1);
    }

    // 构建邻接数组
    std::vector<int> adjacency_start(nv), adjacency_count(nv);
    std::vector<int> adjacency_indices;  
    std::vector<int> adjacency_owners;  
    size_t total_neighbors = 0;
    for (int i = 0; i < nv; i++) {
        total_neighbors += adjacency_lists[i].size();
    }
    adjacency_indices.reserve(total_neighbors);
    adjacency_owners.reserve(total_neighbors);

    {
        int offset = 0;
        for (int i = 0; i < nv; i++) {
            adjacency_start[i] = offset;
            adjacency_count[i] = static_cast<int>(adjacency_lists[i].size());
            for (const auto& nb : adjacency_lists[i]) {
                adjacency_indices.push_back(nb);
                adjacency_owners.push_back(i);
            }
            offset += static_cast<int>(adjacency_lists[i].size());
        }
    }

    // 设备指针
    Scalar3* d_curr_pos     = nullptr;
    Scalar3* d_constraints  = nullptr;
    Scalar*  d_lambda       = nullptr;
    Scalar*  d_masses       = nullptr;
    Scalar3* d_dP           = nullptr;
    Scalar*  d_dPw          = nullptr;

    cudaMalloc(&d_curr_pos,     nv * sizeof(Scalar3));
    cudaMalloc(&d_constraints,  nc * sizeof(Scalar3));
    cudaMalloc(&d_lambda,       nc * sizeof(Scalar));
    cudaMalloc(&d_masses,       nv * sizeof(Scalar));
    cudaMalloc(&d_dP,           nv * sizeof(Scalar3));
    cudaMalloc(&d_dPw,          nv * sizeof(Scalar));

    // 拷贝数据
    std::vector<Scalar3> h_curr_pos(nv);
    for (int i = 0; i < nv; i++) {
        h_curr_pos[i].x = mesh_data_.curr_pos[i].x();
        h_curr_pos[i].y = mesh_data_.curr_pos[i].y();
        h_curr_pos[i].z = mesh_data_.curr_pos[i].z();
    }
    cudaMemcpy(d_curr_pos, h_curr_pos.data(), nv * sizeof(Scalar3), cudaMemcpyHostToDevice);

    std::vector<Scalar3> h_constraints(nc);
    for (int i = 0; i < nc; i++) {
        h_constraints[i] = mesh_data_.constraints[i];
    }
    cudaMemcpy(d_constraints, h_constraints.data(), nc * sizeof(Scalar3), cudaMemcpyHostToDevice);

    cudaMemcpy(d_lambda, mesh_data_.lagrange_multipliers.data(),
               nc * sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, mesh_data_.point_mass.data(),
               nv * sizeof(Scalar), cudaMemcpyHostToDevice);

    cudaMemset(d_dP,  0, nv * sizeof(Scalar3));
    cudaMemset(d_dPw, 0, nv * sizeof(Scalar));

    // 邻接数组分配
    int* d_adjacency       = nullptr;
    int* d_adjacencyOwners = nullptr;
    int* d_adjStart        = nullptr;
    int* d_adjCount        = nullptr;
    cudaMalloc(&d_adjacency,       adjacency_indices.size() * sizeof(int));
    cudaMalloc(&d_adjacencyOwners, adjacency_owners.size()  * sizeof(int));
    cudaMalloc(&d_adjStart,        nv * sizeof(int));
    cudaMalloc(&d_adjCount,        nv * sizeof(int));

    cudaMemcpy(d_adjacency, adjacency_indices.data(),
               adjacency_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjacencyOwners, adjacency_owners.data(),
               adjacency_owners.size()  * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjStart, adjacency_start.data(),
               nv * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjCount, adjacency_count.data(),
               nv * sizeof(int), cudaMemcpyHostToDevice);

    size_t total_edges = adjacency_indices.size();
    Scalar* d_rawWeights = nullptr;
    Scalar* d_weights    = nullptr;
    Scalar* d_sumW       = nullptr;
    cudaMalloc(&d_rawWeights, total_edges * sizeof(Scalar));
    cudaMalloc(&d_weights,    total_edges * sizeof(Scalar));
    cudaMalloc(&d_sumW,       nv * sizeof(Scalar));
    cudaMemset(d_rawWeights, 0, total_edges * sizeof(Scalar));
    cudaMemset(d_weights,    0, total_edges * sizeof(Scalar));
    cudaMemset(d_sumW,       0, nv * sizeof(Scalar));

    Scalar3* d_newPositions = nullptr;
    cudaMalloc(&d_newPositions, nv * sizeof(Scalar3));
    cudaMemset(d_newPositions, 0, nv * sizeof(Scalar3));

    // 迭代参数
    int xpbd_iters = 200;
    Scalar dt = 0.033333;

    // CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --------- XPBD 使用 cooperative 优化 ----------
    cudaEventRecord(start);

    // 使用新的 cooperative 版本，一次kernel内部循环 xpbd_iters 次
    __DEFORMER__::xpbdIterationAllInOneGPU_Cooperative(
        d_curr_pos,
        d_constraints,
        mesh_data_.stretch_stiffness,
        mesh_data_.compress_stiffness,
        d_lambda,
        d_masses,
        d_dP,
        d_dPw,
        nc,
        nv,
        dt,
        xpbd_iters
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float wrinkleTime;
    cudaEventElapsedTime(&wrinkleTime, start, stop);
    std::cout << "[Cooperative] xpbdIterationAllInOneGPU 时间: "
              << wrinkleTime << " ms" << std::endl;

    // --------- deltaMush 综合函数 ----------
    cudaEventRecord(start);
    __DEFORMER__::deltaMushAllInOneGPU(
        d_curr_pos, d_newPositions,
        d_adjacencyOwners, d_adjacency,
        d_adjStart, d_adjCount,
        d_rawWeights, d_weights, d_sumW,
        static_cast<int>(total_edges), nv,
        0.5, 2
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float deltaTime;
    cudaEventElapsedTime(&deltaTime, start, stop);
    std::cout << "deltaMushAllInOneGPU 时间: " << deltaTime << " ms" << std::endl;

    // 拷回主机
    cudaMemcpy(h_curr_pos.data(), d_curr_pos, 
               nv * sizeof(Scalar3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nv; i++) {
        mesh_data_.curr_pos[i] = CGAL_Point_3(
            h_curr_pos[i].x,
            h_curr_pos[i].y,
            h_curr_pos[i].z
        );
    }

    // 导出结果
    exportMesh("../Assets/tubemesh_deformed_gpuNoCopy.obj", mesh_data_);

    // 释放
    cudaFree(d_newPositions);
    cudaFree(d_rawWeights);
    cudaFree(d_weights);
    cudaFree(d_sumW);
    cudaFree(d_adjacency);
    cudaFree(d_adjacencyOwners);
    cudaFree(d_adjStart);
    cudaFree(d_adjCount);
    cudaFree(d_curr_pos);
    cudaFree(d_constraints);
    cudaFree(d_lambda);
    cudaFree(d_masses);
    cudaFree(d_dP);
    cudaFree(d_dPw);
}

void Deformer::DeformerPipeline() {
    getHostMesh_CUDA(ctx.instance);
}
