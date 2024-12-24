#pragma once

#include "WrinkleDeformer.cuh"
#include "WrapDeformer.cuh"
#include "SimulationContext.hpp"
#include "Simulator.hpp"
#include <vector>
#include <memory>
#include <string>
#include <stdexcept>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

using Scalarf = float; 

struct Constraint {
    Scalar v1;
    Scalar v2;
    Scalar restLength;
    Scalar ctype;
};

struct Face {
    int v0;
    int v1;
    int v2;
};

struct MeshData {
    std::vector<CGAL_Point_3> rest_pos;
    std::vector<CGAL_Point_3> curr_pos;
    std::vector<CGAL_Vector_3> vertex_normals;
    std::vector<Constraint> constraints;
    std::vector<Scalar> stretch_stiffness;
    std::vector<Scalar> compress_stiffness;
    std::vector<Scalar> lagrange_multipliers;
    std::vector<Scalar> point_mass;
    std::vector<Face> faces;
};

class Deformer {
public:
    Deformer(SimulationContext& context, Simulator& simulator);
    void prepareMeshData(const std::string& rest_mesh_path,
                         const std::string& bend_mesh_path,
                         Scalar mass,
                         Scalar stretchStiffness,
                         Scalar compressStiffness);
    std::vector<Constraint> findBendPairs(const CGAL_Mesh& mesh);
    std::vector<Constraint> findUniqueEdgesConstraints(const CGAL_Mesh& mesh);
    std::vector<Constraint> mergeConstraints(const std::vector<Constraint>& bend_constraints,
                                             const std::vector<Constraint>& edge_constraints);

    // void xpbdIterationLoop(MeshData& mesh_data, Scalar time_step);
    void performXPBD(MeshData& mesh_data, int iterations, Scalar time_step);
    void deltaMushSmooth(MeshData& mesh_data, int iterations, Scalar step_size = 0.5);

    void getHostMesh(std::unique_ptr<GeometryManager>& instance);

    void getHostMesh_CUDA(std::unique_ptr<GeometryManager>& instance);


    void DeformerPipeline();

private:
    SimulationContext& ctx;
    Simulator& sim;
    MeshData mesh_data_;
};
