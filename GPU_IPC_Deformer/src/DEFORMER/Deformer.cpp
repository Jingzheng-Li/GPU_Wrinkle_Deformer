#include "Deformer.hpp"
#include "WrinkleDeformer.cuh"   // 里头声明了 xpbdIterationLoopCUDA, deltaMushSmoothGPU, 等
#include "WrapDeformer.cuh"

#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <iostream>
#include <vector>

// ========== 辅助函数：导出网格 ==========

static void exportMesh(const std::string& output_path, const MeshData& mesh_data) {
    CGAL_Mesh out_mesh;
    out_mesh.reserve(mesh_data.curr_pos.size(), mesh_data.faces.size()*3, mesh_data.faces.size());
    std::vector<CGAL_Vertex_index> idxMap;
    idxMap.reserve(mesh_data.curr_pos.size());
    for (auto& p : mesh_data.curr_pos) {
        idxMap.push_back(out_mesh.add_vertex(p));
    }
    for (auto& f : mesh_data.faces) {
        out_mesh.add_face(idxMap[f.x], idxMap[f.y], idxMap[f.z]);
    }
    if (!CGAL::IO::write_polygon_mesh(output_path, out_mesh)) {
        std::cerr << "Export failed " << output_path << std::endl;
    }
}

// ========== Deformer 成员函数实现 ==========

Deformer::Deformer(SimulationContext& context, Simulator& simulator)
    : ctx(context), sim(simulator) {}


void Deformer::DeformerPipeline() {
    // 直接调用GPU版本
    getHostMesh_CUDA(ctx.instance);
}

// 找到折叠(弯曲)对约束的辅助函数
static std::vector<Constraint> findBendPairs(const CGAL_Mesh& mesh) {
    std::vector<Constraint> bend_constraints;
    std::unordered_map<size_t, bool> visited;
    bend_constraints.reserve(num_halfedges(mesh) / 2);

    for (CGAL_Halfedge_index h : mesh.halfedges()) {
        CGAL_Face_index fA = face(h, mesh);
        CGAL_Face_index fB = face(opposite(h, mesh), mesh);
        if (fA == CGAL_Mesh::null_face() || fB == CGAL_Mesh::null_face()) {
            continue;
        }
        size_t a = (size_t)fA;
        size_t b = (size_t)fB;
        if (a > b) std::swap(a, b);
        size_t hashVal = (a << 32) ^ (b & 0xffffffff);
        if (visited.find(hashVal) != visited.end()) {
            continue;
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
        // 找 faceA, faceB 的公共顶点
        std::vector<CGAL_Vertex_index> shared;
        for (auto va : faceA_indices) {
            for (auto vb : faceB_indices) {
                if (va == vb) {
                    shared.push_back(va);
                }
            }
        }
        // 如果共享 2 顶点，则是一条公共边
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
            Constraint c;
            c.v1 = (Scalar)uniqueA; // 顶点索引
            c.v2 = (Scalar)uniqueB; 
            c.restLength = dist;
            c.ctype = 1.0; // 弯曲
            bend_constraints.push_back(c);
        }
    }
    return bend_constraints;
}

// 找到所有边约束
static std::vector<Constraint> findUniqueEdgesConstraints(const CGAL_Mesh& mesh) {
    std::vector<Constraint> edge_constraints;
    edge_constraints.reserve(num_edges(mesh));
    for (auto e : mesh.edges()) {
        auto h = halfedge(e, mesh);
        CGAL_Vertex_index v1 = target(h, mesh);
        CGAL_Vertex_index v2 = target(opposite(h, mesh), mesh);
        auto p1 = mesh.point(v1);
        auto p2 = mesh.point(v2);
        Scalar dist = std::sqrt(CGAL::squared_distance(p1, p2));
        Constraint c;
        c.v1 = (Scalar)v1;
        c.v2 = (Scalar)v2;
        c.restLength = dist;
        c.ctype = 0.0; // 拉伸
        edge_constraints.push_back(c);
    }
    return edge_constraints;
}

static std::vector<Constraint> mergeConstraints(
    const std::vector<Constraint>& bend_constraints,
    const std::vector<Constraint>& edge_constraints)
{
    std::vector<Constraint> merged;
    merged.reserve(bend_constraints.size() + edge_constraints.size());
    merged.insert(merged.end(), bend_constraints.begin(), bend_constraints.end());
    merged.insert(merged.end(), edge_constraints.begin(), edge_constraints.end());
    return merged;
}


void Deformer::prepareMeshData(const std::string& rest_mesh_path,
                               const std::string& bend_mesh_path,
                               Scalar mass,
                               Scalar stretchStiffness,
                               Scalar compressStiffness) {
    // 读 obj
    CGAL_Mesh rest_mesh, bend_mesh;
    if (!LOADMESH::CGAL_readObj(rest_mesh_path, rest_mesh) ||
        !LOADMESH::CGAL_readObj(bend_mesh_path, bend_mesh)) {
        std::cerr << "Load obj failed\n";
        return;
    }

    // 顶点坐标
    mesh_data_.rest_pos.reserve(num_vertices(rest_mesh));
    mesh_data_.curr_pos.reserve(num_vertices(rest_mesh));
    for (CGAL_Vertex_index v : rest_mesh.vertices()) {
        mesh_data_.rest_pos.push_back(rest_mesh.point(v));
    }
    for (CGAL_Vertex_index v : bend_mesh.vertices()) {
        mesh_data_.curr_pos.push_back(bend_mesh.point(v));
    }

    // 约束
    auto bend_constraints = /* findBendPairs(rest_mesh) */  std::vector<Constraint>();
    auto edge_constraints = /* findUniqueEdgesConstraints(rest_mesh) */ std::vector<Constraint>();
    // 你自己的函数 ...
    bend_constraints = findBendPairs(rest_mesh);
    edge_constraints = findUniqueEdgesConstraints(rest_mesh);

    auto merged_constraints = mergeConstraints(bend_constraints, edge_constraints);
    mesh_data_.constraints = merged_constraints;

    // stiffness, lambdas, mass
    mesh_data_.stretch_stiffness.resize(merged_constraints.size(), stretchStiffness);
    mesh_data_.compress_stiffness.resize(merged_constraints.size(), compressStiffness);
    mesh_data_.lagrange_multipliers.resize(merged_constraints.size(), 0.0);
    mesh_data_.point_mass.resize(num_vertices(rest_mesh), mass);

    // faces
    for (CGAL_Face_index f : rest_mesh.faces()) {
        std::vector<int> indices;
        indices.reserve(3);
        CGAL_Halfedge_index h = rest_mesh.halfedge(f);
        for (int i = 0; i < 3; ++i) {
            indices.push_back((int)rest_mesh.target(h));
            h = rest_mesh.next(h);
        }
        mesh_data_.faces.push_back(make_uint3(indices[0], indices[1], indices[2]));
    }
}


void Deformer::getHostMesh_CUDA(std::unique_ptr<GeometryManager>& instance) 
{
    // 1) 读并准备 mesh_data_
    std::string rest_mesh_path  = "../Assets/tubemesh.obj";
    std::string bend_mesh_path  = "../Assets/tubemesh_bend.obj";
    prepareMeshData(rest_mesh_path, bend_mesh_path,
                    2.22505e-5f, 10.0f, 10.0f);

    const int nv = (int)mesh_data_.curr_pos.size();
    const int nc = (int)mesh_data_.constraints.size();

    // 2) 在 CPU 上构建拓扑邻接(不含权重计算)
    //    adjacencySet[i] = { 所有与 i 相连的顶点 }
    std::vector<std::unordered_set<int>> adjacencySet(nv);
    for (auto &face : mesh_data_.faces) {
        int v0 = face.x, v1 = face.y, v2 = face.z;
        adjacencySet[v0].insert(v1); adjacencySet[v0].insert(v2);
        adjacencySet[v1].insert(v0); adjacencySet[v1].insert(v2);
        adjacencySet[v2].insert(v0); adjacencySet[v2].insert(v1);
    }

    // flatten:
    //  - adjacencyStart[i], adjacencyCount[i]
    //  - adjacencyIndices[]: size = totalNeighbors
    //  - adjacencyOwner[]:  同样大小，每个元素 = 中心顶点 i
    // 这样就可以对“每一条邻接 edge”并行处理
    std::vector<int> adjacencyStart(nv), adjacencyCount(nv);
    std::vector<int> adjacencyIndices;   // nbr
    std::vector<int> adjacencyOwner;     // i

    size_t totalNeighbors = 0;
    for(int i=0; i<nv; i++){
        totalNeighbors += adjacencySet[i].size();
    }
    adjacencyIndices.reserve(totalNeighbors);
    adjacencyOwner.  reserve(totalNeighbors);

    // 填充
    {
        int offset = 0;
        for(int i=0; i<nv; i++){
            adjacencyStart[i] = offset;
            adjacencyCount[i] = (int) adjacencySet[i].size();
            for(auto &nbr : adjacencySet[i]) {
                adjacencyIndices.push_back(nbr);
                adjacencyOwner.push_back(i); // 记录: adjacencyIndices[x] 为 i 的邻居
            }
            offset += (int) adjacencySet[i].size();
        }
    }

    // 3) 分配 GPU 数据: 顶点位置, 约束, stiffness, ...
    Scalar3* d_curr_pos          = nullptr; 
    __DEFORMER__::ConstraintGPU* d_constr    = nullptr;
    Scalar *d_stretch_stiff      = nullptr;
    Scalar *d_compress_stiff     = nullptr;
    Scalar *d_lambda             = nullptr;
    Scalar *d_masses             = nullptr;
    Scalar *d_inv_mass           = nullptr;
    Scalar3* d_dP                = nullptr;
    Scalar *d_dPw                = nullptr;

    cudaMalloc(&d_curr_pos,       nv*sizeof(Scalar3));
    cudaMalloc(&d_constr,         nc*sizeof(__DEFORMER__::ConstraintGPU));
    cudaMalloc(&d_stretch_stiff,  nc*sizeof(Scalar));
    cudaMalloc(&d_compress_stiff, nc*sizeof(Scalar));
    cudaMalloc(&d_lambda,         nc*sizeof(Scalar));
    cudaMalloc(&d_masses,         nv*sizeof(Scalar));
    cudaMalloc(&d_inv_mass,       nv*sizeof(Scalar));
    cudaMalloc(&d_dP,             nv*sizeof(Scalar3));
    cudaMalloc(&d_dPw,            nv*sizeof(Scalar));

    // 4) 拷贝 CPU -> GPU
    // 4.1) curr_pos
    std::vector<Scalar3> h_curr_pos(nv);
    for(int i=0; i<nv; i++){
        h_curr_pos[i].x = (Scalar)mesh_data_.curr_pos[i].x();
        h_curr_pos[i].y = (Scalar)mesh_data_.curr_pos[i].y();
        h_curr_pos[i].z = (Scalar)mesh_data_.curr_pos[i].z();
    }
    cudaMemcpy(d_curr_pos, h_curr_pos.data(), nv*sizeof(Scalar3), cudaMemcpyHostToDevice);

    // 4.2) constraints
    std::vector<__DEFORMER__::ConstraintGPU> h_constr(nc);
    for(int i=0; i<nc; i++){
        h_constr[i].v1         = (int)mesh_data_.constraints[i].v1;
        h_constr[i].v2         = (int)mesh_data_.constraints[i].v2;
        h_constr[i].restLength = mesh_data_.constraints[i].restLength;
        h_constr[i].ctype      = mesh_data_.constraints[i].ctype;
    }
    cudaMemcpy(d_constr, h_constr.data(), nc*sizeof(__DEFORMER__::ConstraintGPU), cudaMemcpyHostToDevice);

    // 4.3) stiffness, lambdas, masses
    cudaMemcpy(d_stretch_stiff,  mesh_data_.stretch_stiffness.data(),  nc*sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_compress_stiff, mesh_data_.compress_stiffness.data(), nc*sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda,         mesh_data_.lagrange_multipliers.data(), nc*sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses,         mesh_data_.point_mass.data(), nv*sizeof(Scalar), cudaMemcpyHostToDevice);

    // 4.4) d_dP, d_dPw 清零
    cudaMemset(d_dP,  0, nv*sizeof(Scalar3));
    cudaMemset(d_dPw, 0, nv*sizeof(Scalar));

    // 邻接相关
    int* d_adjacency      = nullptr;
    int* d_adjacencyOwner = nullptr;
    int* d_adjStart       = nullptr;
    int* d_adjCount       = nullptr;
    cudaMalloc(&d_adjacency,      adjacencyIndices.size()*sizeof(int));
    cudaMalloc(&d_adjacencyOwner, adjacencyOwner.size()*sizeof(int));
    cudaMalloc(&d_adjStart,       nv*sizeof(int));
    cudaMalloc(&d_adjCount,       nv*sizeof(int));

    cudaMemcpy(d_adjacency,      adjacencyIndices.data(), adjacencyIndices.size()*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjacencyOwner, adjacencyOwner.data(),   adjacencyOwner.size()*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjStart,       adjacencyStart.data(),   nv*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_adjCount,       adjacencyCount.data(),   nv*sizeof(int),   cudaMemcpyHostToDevice);

    // ========== 5) 先计算 invMass ==========
    __DEFORMER__::launchComputeInvMassKernel(d_masses, d_inv_mass, nv);

    // ========== 6) 进行 XPBD 迭代 ==========
    {
        int xpbdIters = 200;
        Scalar dt = 0.033333f;
        for(int it=0; it<xpbdIters; ++it){
            __DEFORMER__::xpbdIterationLoopCUDA(
                d_curr_pos, d_constr,
                d_stretch_stiff, d_compress_stiff,
                d_lambda, d_inv_mass,
                d_dP, d_dPw,
                nc, nv, dt
            );
        }
    }

    // ========== 7) 在 GPU 上计算 DeltaMush 权重(不回传 CPU) ==========

    // 分配临时数组： rawWeights (存 1/dist), sumW (原子加累积)
    // 以及最终要给 deltaMushSmoothGPU 用的 d_weights。
    const size_t totalEdges = adjacencyIndices.size();

    Scalar* d_rawWeights = nullptr;
    Scalar* d_weights    = nullptr;  // 归一化后
    Scalar* d_sumW       = nullptr;  // 每个顶点 i 的权重和
    cudaMalloc(&d_rawWeights, totalEdges*sizeof(Scalar));
    cudaMalloc(&d_weights,    totalEdges*sizeof(Scalar));
    cudaMalloc(&d_sumW,       nv*sizeof(Scalar));
    cudaMemset(d_rawWeights, 0, totalEdges*sizeof(Scalar));
    cudaMemset(d_weights,    0, totalEdges*sizeof(Scalar));
    cudaMemset(d_sumW,       0, nv*sizeof(Scalar));

    // 调用封装函数(内部会调用 2 个 kernel)
    __DEFORMER__::computeDeltaMushWeights(
        d_curr_pos,
        d_adjacencyOwner,
        d_adjacency,
        d_rawWeights,
        d_weights,
        d_sumW,
        (int)totalEdges,
        nv
    );

    // 这里 d_weights 已经是归一化的权重（和 CPU 计算一致），可直接给 deltaMushSmoothGPU 用

    // ========== 8) 调用 deltaMushSmoothGPU ==========
    // 需要一个 d_newPositions
    Scalar3* d_newPositions = nullptr;
    cudaMalloc(&d_newPositions, nv*sizeof(Scalar3));
    cudaMemset(d_newPositions, 0, nv*sizeof(Scalar3));

    {
        int mushIters = 2;
        float mushStep = 0.5f;
        // 注意，这里我们将 "d_adjacency, d_weights, d_adjStart, d_adjCount"
        // 传给 deltaMushSmoothGPU
        // d_weights 就是已经算好的“归一化权重”。
        __DEFORMER__::deltaMushSmoothGPU(
            d_curr_pos, d_newPositions,
            d_adjacency,
            d_weights,
            d_adjStart,
            d_adjCount,
            nv,
            mushStep,
            mushIters
        );
    }

    // ========== 9) 最终拷贝 d_curr_pos 回 CPU 并导出 .obj ==========
    //    (如果你还要在 GPU 上做别的，也可以不拷回)
    cudaMemcpy(h_curr_pos.data(), d_curr_pos, nv*sizeof(Scalar3), cudaMemcpyDeviceToHost);

    for(int i=0; i<nv; i++){
        mesh_data_.curr_pos[i] = CGAL_Point_3(
            h_curr_pos[i].x, h_curr_pos[i].y, h_curr_pos[i].z
        );
    }
    exportMesh("../Assets/tubemesh_deformed_gpuNoCopy.obj", mesh_data_);

    // ========== 10) 释放 GPU 内存 ==========
    cudaFree(d_newPositions);
    cudaFree(d_rawWeights);
    cudaFree(d_weights);
    cudaFree(d_sumW);
    cudaFree(d_adjacency);
    cudaFree(d_adjacencyOwner);
    cudaFree(d_adjStart);
    cudaFree(d_adjCount);

    cudaFree(d_curr_pos);
    cudaFree(d_constr);
    cudaFree(d_stretch_stiff);
    cudaFree(d_compress_stiff);
    cudaFree(d_lambda);
    cudaFree(d_masses);
    cudaFree(d_inv_mass);
    cudaFree(d_dP);
    cudaFree(d_dPw);
}
