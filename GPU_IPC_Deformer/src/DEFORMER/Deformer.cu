#include "Deformer.cuh"
#include "CUDAUtils.hpp"
#include "LoadMesh.hpp"
#include <iostream>
#include <unordered_set>
#include <cmath>
#include <omp.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>



Deformer::Deformer(SimulationContext& context, Simulator& simulator)
    : ctx(context), sim(simulator) {}

void Deformer::DeformerPipeline() {
    getHostMesh(ctx.instance);
    getHostMesh_CUDA(ctx.instance);
}


std::vector<Constraint> Deformer::findBendPairs(const CGAL_Mesh& mesh) {
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
        std::vector<CGAL_Vertex_index> faceA_indices;
        {
            CGAL_Halfedge_index ha = mesh.halfedge(fA);
            for (int i = 0; i < 3; ++i) {
                faceA_indices.push_back(target(ha, mesh));
                ha = mesh.next(ha);
            }
        }
        std::vector<CGAL_Vertex_index> faceB_indices;
        {
            CGAL_Halfedge_index hb = mesh.halfedge(fB);
            for (int i = 0; i < 3; ++i) {
                faceB_indices.push_back(target(hb, mesh));
                hb = mesh.next(hb);
            }
        }
        std::vector<CGAL_Vertex_index> shared;
        for (auto va : faceA_indices) {
            for (auto vb : faceB_indices) {
                if (va == vb) {
                    shared.push_back(va);
                }
            }
        }
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
            c.v1 = (Scalar)uniqueA;
            c.v2 = (Scalar)uniqueB;
            c.restLength = dist;
            c.ctype = 1.0;
            bend_constraints.push_back(c);
        }
    }
    return bend_constraints;
}

std::vector<Constraint> Deformer::findUniqueEdgesConstraints(const CGAL_Mesh& mesh) {
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
        c.ctype = 0.0;
        edge_constraints.push_back(c);
    }
    return edge_constraints;
}

std::vector<Constraint> Deformer::mergeConstraints(const std::vector<Constraint>& bend_constraints,
                                                   const std::vector<Constraint>& edge_constraints) {
    std::vector<Constraint> merged;
    merged.reserve(bend_constraints.size() + edge_constraints.size());
    merged.insert(merged.end(), bend_constraints.begin(), bend_constraints.end());
    merged.insert(merged.end(), edge_constraints.begin(), edge_constraints.end());
    return merged;
}


void xpbdIterationLoop(MeshData& mesh_data, Scalar time_step) {
    auto& curr_pos = mesh_data.curr_pos;
    auto& constraints = mesh_data.constraints;
    auto& stretch_stiff  = mesh_data.stretch_stiffness;
    auto& compress_stiff = mesh_data.compress_stiffness;
    auto& lambdas        = mesh_data.lagrange_multipliers;
    auto& masses         = mesh_data.point_mass;
    size_t nv = curr_pos.size();
    size_t nc = constraints.size();
    if (nv == 0 || nc == 0) return;
    std::vector<Scalar> inv_mass(nv);
    #pragma omp parallel for
    for (int i = 0; i < (int)nv; ++i) {
        inv_mass[i] = (masses[i] == 0.0) ? 0.0 : (1.0 / masses[i]);
    }
    std::vector<CGAL_Vector_3> dP(nv, CGAL_Vector_3(0,0,0));
    std::vector<Scalar> dPw(nv, 0.0);
    #pragma omp parallel
    {
        std::vector<CGAL_Vector_3> dP_local(nv, CGAL_Vector_3(0,0,0));
        std::vector<Scalar> dPw_local(nv, 0.0);
        #pragma omp for nowait
        for (int i = 0; i < (int)nc; ++i) {
            int v1 = (int)constraints[i].v1;
            int v2 = (int)constraints[i].v2;
            Scalar rest_length = constraints[i].restLength;
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
            Scalar C = (d - rest_length);
            Scalar wsum = inv_mass[v1] + inv_mass[v2];
            Scalar denom = wsum + alpha;
            if (std::fabs(denom) < 1e-12) {
                denom = 1e-12;
            }
            Scalar delta_lambda = (-C - alpha * lambdas[i]) / denom;
            CGAL_Vector_3 dp1 = -delta_lambda * inv_mass[v1] * dir;
            CGAL_Vector_3 dp2 =  delta_lambda * inv_mass[v2] * dir;
            dP_local[v1] = dP_local[v1] + dp1;
            dP_local[v2] = dP_local[v2] + dp2;
            dPw_local[v1] += 1.0;
            dPw_local[v2] += 1.0;
            #pragma omp atomic
            lambdas[i] += delta_lambda;
        }
        #pragma omp critical
        {
            for (size_t i = 0; i < nv; i++) {
                dP[i]  = dP[i] + dP_local[i];
                dPw[i] = dPw[i] + dPw_local[i];
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < (int)nv; ++i) {
        if (dPw[i] > 1e-12) {
            Scalar w = 1.0 / dPw[i];
            CGAL_Vector_3 shift = dP[i] * w;
            curr_pos[i] = curr_pos[i] + shift;
        }
    }
}

void Deformer::performXPBD(MeshData& mesh_data, int iterations, Scalar time_step) {
    for (int it = 0; it < iterations; ++it) {
        xpbdIterationLoop(mesh_data, time_step);
    }
}

void Deformer::deltaMushSmooth(MeshData& mesh_data, int iterations, Scalar step_size) {
    size_t nV = mesh_data.curr_pos.size();
    if (nV == 0) return;
    std::vector<CGAL_Point_3> positions = mesh_data.curr_pos;
    std::vector<std::vector<int>> adjacency(nV);
    std::vector<std::vector<Scalar>> adj_lengths(nV);
    struct EdgeKey {
        int v1, v2;
        EdgeKey(int a, int b) {
            if(a < b) { v1=a; v2=b; } else { v1=b; v2=a; }
        }
    };
    struct EdgeKeyHash {
        size_t operator()(const EdgeKey& e) const {
            return (std::size_t)e.v1 ^ ((std::size_t)e.v2 << 16);
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
        int v0 = f.v0, v1 = f.v1, v2 = f.v2;
        edges_set.insert(EdgeKey(v0, v1));
        edges_set.insert(EdgeKey(v1, v2));
        edges_set.insert(EdgeKey(v2, v0));
    }
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
    std::vector<std::vector<Scalar>> weights(nV);
    for (size_t i = 0; i < nV; ++i) {
        auto& lens = adj_lengths[i];
        auto& nbrs = adjacency[i];
        if (lens.empty() || nbrs.empty()) {
            continue;
        }
        Scalar inv_sum = 0.0;
        weights[i].resize(lens.size());
        for (size_t k = 0; k < lens.size(); ++k) {
            Scalar inv_l = (lens[k] > 1e-15) ? (1.0 / lens[k]) : 0.0;
            weights[i][k] = inv_l;
            inv_sum += inv_l;
        }
        if (inv_sum > 1e-15) {
            for (size_t k = 0; k < lens.size(); ++k) {
                weights[i][k] /= inv_sum;
            }
        }
    }
    for (int it = 0; it < iterations; ++it) {
        std::vector<CGAL_Point_3> new_positions = positions;
        #pragma omp parallel for
        for (int i = 0; i < (int)nV; ++i) {
            auto& nbrs = adjacency[i];
            auto& w = weights[i];
            if (nbrs.empty() || w.empty() || nbrs.size() != w.size()) {
                continue;
            }
            CGAL_Vector_3 sum_vec(0,0,0);
            for (size_t k = 0; k < nbrs.size(); ++k) {
                CGAL_Vector_3 v = positions[nbrs[k]] - CGAL::ORIGIN;
                sum_vec = sum_vec + (v * w[k]);
            }
            CGAL_Point_3 avg_pos = CGAL::ORIGIN + sum_vec;
            CGAL_Vector_3 diff = avg_pos - positions[i];
            new_positions[i] = positions[i] + diff * step_size;
        }
        positions = std::move(new_positions);
    }
    for (size_t i = 0; i < nV; ++i) {
        mesh_data.curr_pos[i] = positions[i];
    }
}


void Deformer::prepareMeshData(const std::string& rest_mesh_path,
                               const std::string& bend_mesh_path,
                               Scalar mass,
                               Scalar stretchStiffness,
                               Scalar compressStiffness) {
    CGAL_Mesh rest_mesh, bend_mesh;
    if (!LOADMESH::CGAL_readObj(rest_mesh_path, rest_mesh) || !LOADMESH::CGAL_readObj(bend_mesh_path, bend_mesh)) {
        std::cerr << "Load obj failed\n";
        return;
    }

    mesh_data_.rest_pos.reserve(num_vertices(rest_mesh));
    mesh_data_.curr_pos.reserve(num_vertices(rest_mesh));
    for (CGAL_Vertex_index v : rest_mesh.vertices()) {
        mesh_data_.rest_pos.push_back(rest_mesh.point(v));
    }
    for (CGAL_Vertex_index v : bend_mesh.vertices()) {
        mesh_data_.curr_pos.push_back(bend_mesh.point(v));
    }

    auto bend_constraints = findBendPairs(rest_mesh);
    auto edge_constraints = findUniqueEdgesConstraints(rest_mesh);
    auto merged_constraints = mergeConstraints(bend_constraints, edge_constraints);
    mesh_data_.constraints = merged_constraints;
    mesh_data_.stretch_stiffness.resize(merged_constraints.size(), stretchStiffness);
    mesh_data_.compress_stiffness.resize(merged_constraints.size(), compressStiffness);
    mesh_data_.lagrange_multipliers.resize(merged_constraints.size(), 0.0);
    mesh_data_.point_mass.resize(num_vertices(rest_mesh), mass);
    for (CGAL_Face_index f : rest_mesh.faces()) {
        std::vector<int> indices;
        CGAL_Halfedge_index h = rest_mesh.halfedge(f);
        for (int i = 0; i < 3; ++i) {
            indices.push_back((int)rest_mesh.target(h));
            h = rest_mesh.next(h);
        }
        Face face;
        face.v0 = indices[0];
        face.v1 = indices[1];
        face.v2 = indices[2];
        mesh_data_.faces.push_back(face);
    }
}

static void exportMesh(const std::string& output_path, const MeshData& mesh_data) {
    CGAL_Mesh out_mesh;
    std::vector<CGAL_Vertex_index> idxMap;
    idxMap.reserve(mesh_data.curr_pos.size());
    for (auto& p : mesh_data.curr_pos) {
        idxMap.push_back(out_mesh.add_vertex(p));
    }
    for (auto& f : mesh_data.faces) {
        out_mesh.add_face(idxMap[f.v0], idxMap[f.v1], idxMap[f.v2]);
    }
    if (!CGAL::IO::write_polygon_mesh(output_path, out_mesh)) {
        std::cerr << "Export failed " << output_path << std::endl;
    }
}


void Deformer::getHostMesh(std::unique_ptr<GeometryManager>& instance) {
    std::string rest_mesh_path  = "../../Assets/tubemesh.obj";
    std::string bend_mesh_path  = "../../Assets/tubemesh_bend.obj";
    std::string output_deformed = "../../Assets/tubemesh_deformed.obj";
    


    // CUDAMemcpyDToDSafe(instance->getCudaTriVerts(), instance->getCudaSurfVertPos(), instance->getHostNumClothVerts());
    // CUDAMemcpyDToDSafe(instance->getCudaTriEdges(), instance->getCudaSurfEdgeIds(), instance->getHostNumTriEdges());
    // instance->getHostNumTriEdges();
    // instance->getHostNumTriBendEdges();
    // instance->getHostNumClothVerts();
    // instance->getHostNumClothFaces();
    // instance->getCudaTriVerts();
    // instance->getCudaTriEdges();
    // instance->getCudaTriElement();
    // instance->getCudaTriBendVerts();
    // instance->getCudaTriBendEdges();


    prepareMeshData(rest_mesh_path, bend_mesh_path, 2.22505e-5, 10.0, 10.0);

    std::cout << "constraints size: " << mesh_data_.constraints.size() << std::endl;



    int iterations = 200;
    Scalar time_step  = 0.033333;

    auto start_time = std::chrono::high_resolution_clock::now();
    performXPBD(mesh_data_, iterations, time_step);
    deltaMushSmooth(mesh_data_, 2, 0.5);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "[INFO] getHostMesh 运行时长: " << duration.count() << " 毫秒" << std::endl;


    exportMesh(output_deformed, mesh_data_);
}
















































__global__ void computeInvMassKernel(const Scalar* d_masses, Scalar* d_inv_mass, int nv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nv) return;

    Scalar mass = d_masses[idx];
    d_inv_mass[idx] = (mass == 0.0f) ? 0.0f : (1.0f / mass);
}


void Deformer::getHostMesh_CUDA(std::unique_ptr<GeometryManager>& instance) {

    int iterations = 200;

    std::string rest_mesh_path  = "../../Assets/tubemesh.obj";
    std::string bend_mesh_path  = "../../Assets/tubemesh_bend.obj";
    prepareMeshData(rest_mesh_path, bend_mesh_path, /*mass*/2.22505e-5f, /*stretch*/10.0f, /*compress*/10.0f);

    int nv = (int)mesh_data_.curr_pos.size();
    int nc = (int)mesh_data_.constraints.size();

    Scalar3* d_curr_pos       = nullptr; 
    __DEFORMER__::ConstraintGPU* d_constr  = nullptr;
    Scalar* d_stretch_stiff   = nullptr;
    Scalar* d_compress_stiff  = nullptr;
    Scalar* d_lambda          = nullptr;
    Scalar* d_masses          = nullptr;
    Scalar* d_inv_mass        = nullptr;
    Scalar3* d_dP             = nullptr;
    Scalar*  d_dPw            = nullptr;
    cudaMalloc(&d_curr_pos,       nv * sizeof(Scalar3));
    cudaMalloc(&d_constr,         nc * sizeof(__DEFORMER__::ConstraintGPU));
    cudaMalloc(&d_stretch_stiff,  nc * sizeof(Scalar));
    cudaMalloc(&d_compress_stiff, nc * sizeof(Scalar));
    cudaMalloc(&d_lambda,         nc * sizeof(Scalar));
    cudaMalloc(&d_masses,         nv * sizeof(Scalar));
    cudaMalloc(&d_inv_mass,       nv * sizeof(Scalar));
    cudaMalloc(&d_dP,             nv * sizeof(Scalar3));
    cudaMalloc(&d_dPw,            nv * sizeof(Scalar));
    std::vector<Scalar3> h_curr_pos(nv);
    for(int i=0; i<nv; i++){
        // 假设 mesh_data_ 用 double, 这里做一次 Scalar 转换
        h_curr_pos[i].x = (Scalar)mesh_data_.curr_pos[i].x();
        h_curr_pos[i].y = (Scalar)mesh_data_.curr_pos[i].y();
        h_curr_pos[i].z = (Scalar)mesh_data_.curr_pos[i].z();
    }
    cudaMemcpy(d_curr_pos, h_curr_pos.data(), nv*sizeof(Scalar3), cudaMemcpyHostToDevice);

    // 同理 constraints
    std::vector<__DEFORMER__::ConstraintGPU> h_constr(nc);
    for(int i=0; i<nc; i++){
        h_constr[i].v1         = (int)mesh_data_.constraints[i].v1;
        h_constr[i].v2         = (int)mesh_data_.constraints[i].v2;
        h_constr[i].restLength = (Scalar)mesh_data_.constraints[i].restLength;
        h_constr[i].ctype      = (Scalar)mesh_data_.constraints[i].ctype;
    }
    cudaMemcpy(d_constr, h_constr.data(), nc*sizeof(__DEFORMER__::ConstraintGPU), cudaMemcpyHostToDevice);

    // 其他类似 ...
    cudaMemcpy(d_stretch_stiff,  mesh_data_.stretch_stiffness.data(),  nc*sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_compress_stiff, mesh_data_.compress_stiffness.data(), nc*sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lambda,         mesh_data_.lagrange_multipliers.data(), nc*sizeof(Scalar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_masses, mesh_data_.point_mass.data(), nv*sizeof(Scalar), cudaMemcpyHostToDevice);

    // 初始化 d_dP, d_dPw 为 0
    cudaMemset(d_dP, 0, nv*sizeof(Scalar3));
    cudaMemset(d_dPw,0, nv*sizeof(Scalar));

    // 3) 先计算 inv_mass
    {
        int blockSize = 256;
        int grid = (nv + blockSize - 1) / blockSize;
        computeInvMassKernel<<<grid,blockSize>>>(d_masses, d_inv_mass, nv);
        cudaDeviceSynchronize();
    }

    // // 4) 进行 XPBD 迭代
    // int iterations = 200;
    // Scalar time_step  = 0.033333f;
    // for(int it=0; it<iterations; ++it){
    //     xpbdIterationLoopCUDA(
    //         d_curr_pos,
    //         d_constr,
    //         d_stretch_stiff,
    //         d_compress_stiff,
    //         d_lambda,
    //         d_inv_mass,
    //         d_dP,
    //         d_dPw,
    //         nc, nv,
    //         time_step
    //     );
    //     cudaDeviceSynchronize();
    // }

    // // 如果需要再做 deltaMushSmoothGPU(...)，也可以类似分配 / 调用

    // // 5) 拷贝回 CPU 并导出
    // cudaMemcpy(h_curr_pos.data(), d_curr_pos, nv*sizeof(Scalar3), cudaMemcpyDeviceToHost);
    // for(int i=0; i<nv; i++){
    //     mesh_data_.curr_pos[i] = CGAL_Point_3(h_curr_pos[i].x, h_curr_pos[i].y, h_curr_pos[i].z);
    // }
    // // 写出 obj
    // std::string output_deformed = "../../Assets/tubemesh_deformed.obj";
    // exportMesh(output_deformed, mesh_data_);

    // 6) 释放 GPU 内存
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
