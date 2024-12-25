#pragma once

#include "WrinkleDeformer.cuh"   // 声明 GPU 函数
#include "WrapDeformer.cuh"
#include "SimulationContext.hpp"
#include "Simulator.hpp"
#include "CUDAUtils.hpp"
#include "LoadMesh.hpp"

// CGAL 相关
#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>

#include <string>
#include <vector>
#include <memory>
#include <unordered_set>
#include <stdexcept>
#include <iostream>
#include <cmath>

// 你的 Scalar, Scalar3, CGAL_Point_3 等类型假定已在这些头文件中定义
// 或者你自己定义 using Scalar = float; 之类

using Scalarf = float;  // 仅示例

// 约束结构
struct Constraint {
    Scalar v1;
    Scalar v2;
    Scalar restLength;
    Scalar ctype;
};

// 这里用 uint3 代表三角面的顶点索引 (x,y,z)
#include <cuda_runtime.h> // 里头有 uint3 定义，或你也可以自己定义
// 如果没有，也可自定义:
//// struct uint3 { unsigned int x,y,z; };

struct MeshData {
    std::vector<CGAL_Point_3> rest_pos;
    std::vector<CGAL_Point_3> curr_pos;
    std::vector<CGAL_Vector_3> vertex_normals;
    std::vector<Constraint> constraints;
    std::vector<Scalar> stretch_stiffness;
    std::vector<Scalar> compress_stiffness;
    std::vector<Scalar> lagrange_multipliers;
    std::vector<Scalar> point_mass;
    std::vector<uint3> faces;  // 三角面
};

// Deformer 类
class Deformer {
public:
    Deformer(SimulationContext& context, Simulator& simulator);

    // 准备 mesh_data_: 读 mesh, 构建 constraints, ...
    void prepareMeshData(const std::string& rest_mesh_path,
                         const std::string& bend_mesh_path,
                         Scalar mass,
                         Scalar stretchStiffness,
                         Scalar compressStiffness);

    // 核心：先做 XPBD，再做 deltaMush
    void getHostMesh_CUDA(std::unique_ptr<GeometryManager>& instance);

    // 暂时不使用 CPU 版本的话，可留空
    void DeformerPipeline();

private:
    SimulationContext& ctx;
    Simulator& sim;
    MeshData mesh_data_;
};

