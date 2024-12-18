// SimulationContext.hpp

#ifndef SIMULATIONCONTEXT_HPP
#define SIMULATIONCONTEXT_HPP

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "FEMEnergy.cuh"
#include "GIPC.cuh"
#include "LBVH.cuh"
#include "SortMesh.cuh"
#include "ImplicitIntegrator.cuh"

#include "UTILS/LoadMesh.hpp"
#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "UTILS/RenderUtils.hpp"

class SimulationContext {
public:
    // Constructor and Destructor
    SimulationContext();
    ~SimulationContext();

    // Pointers to various components
    std::unique_ptr<LBVHCollisionDetector> LBVH_CD_ptr;
    std::unique_ptr<PCGSolver> PCG_ptr;
    std::unique_ptr<BlockHessian> BH_ptr;
    std::unique_ptr<SIMMesh> simMesh_ptr;
    std::unique_ptr<OpenGLRender> glRender_ptr;
    std::unique_ptr<GeometryManager> instance;

    // Simulation parameters
    int collision_detection_buff_scale;
    int interpolation_frames;
    Scalar animation_motion_rate;

    bool do_OpenGLRender;
    bool do_addSoftTargets;
    bool do_addStitchPairs;

    // Paths and filenames
    std::string assets_dir_clothmesh;
    std::string assets_dir_clothmesh_save;
    std::string assets_dir_bodymesh;
    std::string assets_dir_staticmesh;
    std::string assets_dir_simjson;

    std::string clothmeshname;
    std::string bodymeshname;
    std::string bodytposename;
    std::string staticmeshname;

    std::string assets_dir_input_simjson;
    std::string assets_dir_output_clothmesh_json;

    // Other global variables
    std::string getCurrentTime();
    void reset_ptr();
    void restart_program(int index);
    void init_Geometry_ptr();


private:
    // Any private members or helper functions

};

#endif // SIMULATIONCONTEXT_HPP
