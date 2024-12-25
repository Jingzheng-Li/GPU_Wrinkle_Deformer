#pragma once

#include "WrinkleDeformer.cuh"
#include "WrapDeformer.cuh"
#include "SimulationContext.hpp"
#include "Simulator.hpp"
#include "CUDAUtils.hpp"
#include "LoadMesh.hpp"

#include <string>
#include <vector>
#include <memory>

// 使用双精度
using Scalar = double;
using Scalar2 = double2;
using Scalar3 = double3;

// MeshData 结构
struct MeshData {
    std::vector<CGAL_Point_3> rest_pos;
    std::vector<CGAL_Point_3> curr_pos;
    std::vector<Scalar3> constraints;
    Scalar stretch_stiffness;      // 单一的 Stretch 刚度
    Scalar compress_stiffness;     // 单一的 Compress 刚度
    std::vector<Scalar> lagrange_multipliers;
    std::vector<Scalar> point_mass;
    std::vector<uint3> faces;
};

// Deformer 类
class Deformer {
public:
    Deformer(SimulationContext& context, Simulator& simulator);
    void prepareMeshData(const std::string& rest_mesh_path,
                         const std::string& bend_mesh_path,
                         Scalar mass,
                         Scalar stretchStiffness,
                         Scalar compressStiffness);
    void getHostMesh_CUDA(std::unique_ptr<GeometryManager>& instance);
    void DeformerPipeline();
private:
    SimulationContext& ctx;
    Simulator& sim;
    MeshData mesh_data_;
};
