
// Deformer.hpp


#pragma once

#include "WrinkleDeformer.cuh"
#include "WrapDeformer.cuh"
#include "SimulationContext.hpp"
#include "Simulator.hpp"


///
/// 用于存储 XPBD 所需的全部网格信息
///
struct MeshData {
    // rest_pos 和 curr_pos 大小均为 vertices.size()
    // 每个元素对应一个顶点的空间坐标
    std::vector<CGAL_Point_3> rest_pos;
    std::vector<CGAL_Point_3> curr_pos;

    // 与每个顶点对应的法线向量（和顶点数目一致）
    std::vector<CGAL_Vector_3> vertex_normals;

    // 合并后的约束 (bend + edge)
    // 每条约束的结构： [v1, v2, rest_length, type]
    // 其中 type = 1 => bend constraint; type = 0 => edge constraint
    std::vector<std::array<Scalar, 4>> constraints;

    // 分别存储所有约束在拉伸和压缩时的刚度
    std::vector<Scalar> stretch_stiffness;
    std::vector<Scalar> compress_stiffness;

    // 拉格朗日乘子，与 constraints 数目对应
    std::vector<Scalar> lagrange_multipliers;

    // 每个顶点的质量，对应 vertices.size()
    std::vector<Scalar> point_mass;

    // 网格中所有三角面（或多边形面）顶点索引
    // 此处只示例三角面，可根据自己网格类型改为多边形
    std::vector<std::array<int, 3>> faces;
};


class Deformer {

public:

    Deformer(SimulationContext& context, Simulator& simulator);

    void prepareMeshData(
        const std::string& rest_mesh_path,
        const std::string& bend_mesh_path,
        Scalar mass,
        Scalar stretchStiffness,
        Scalar compressStiffness
    );


    // 查找所有 bend constraint
    std::vector<std::array<Scalar, 4>> findBendPairs(const CGAL_Mesh& mesh);

    // 查找所有 edge constraint
    std::vector<std::array<Scalar, 4>> findUniqueEdgesConstraints(const CGAL_Mesh& mesh);

    // 合并约束
    std::vector<std::array<Scalar, 4>> mergeConstraints(
        const std::vector<std::array<Scalar, 4>>& bend_constraints,
        const std::vector<std::array<Scalar, 4>>& edge_constraints
    );

    // 计算顶点法线
    std::vector<CGAL_Vector_3> computeVertexNormals(const CGAL_Mesh& mesh);

    // 进行一次 XPBD 迭代
    void xpbdIterationLoop(MeshData& mesh_data, Scalar time_step);

    // 多次迭代
    void performXPBD(MeshData& mesh_data, int iterations, Scalar time_step);


public:
    
    void getHostMesh(std::unique_ptr<GeometryManager>& instance);

    void DeformerPipeline();
    
public:
    // ...
    void deltaMushSmooth(MeshData& mesh_data, int iterations, Scalar step_size = 0.5);


private:
    SimulationContext& ctx;
    Simulator& sim;

    MeshData mesh_data_;

};


