// Simulator.hpp

#ifndef SIMULATOR_HPP
#define SIMULATOR_HPP

#include "SimulationContext.hpp"

class Simulator {
public:
    Simulator(SimulationContext& context);

    void init_Geometry_ptr();
    void init_Geometry_mesh();

    // Display functions
    void display_without_opengl_animation();
    void display_without_opengl_drag();
    void display();

private:
    // Initialization functions
    void init_ParamSettings();
    void init_BodyClothMesh();
    void init_Scene();


private:
    // Initialization body&cloth mesh
    void MergeDuplicateMesh();
    void GetReducedMappedMesh();
    void AvoidBodyMeshIntersection();
    void LinearInterpolationMesh();
    void TransformCGALClothtoVector();
    void TransformCGALBodytoVector();
    void MergeCGALClothBodytoSurface();
    void GetSurfaceInfo();

private:
    void SetSurfaceMassVelTypeCons();
    void init_ClothFEM();
    void AddSoftTargets();
    void AddStitchPairs();
    void SortTotalMesh();
    void SortClothMesh();
    void SortSoftTargets();
    void SortStitchPairs();
    void SortMASPreconditioner();
    void SimulatorCUDAMalloc();
    void SimulatorCUDAMemHtoD();
    void BuildLBVH();

private:
    SimulationContext& ctx;

    // New member variables to track mesh loading status
    bool cloth_mesh_loaded;
    bool body_mesh_loaded;
    bool static_mesh_loaded;
};

#endif // SIMULATOR_HPP
