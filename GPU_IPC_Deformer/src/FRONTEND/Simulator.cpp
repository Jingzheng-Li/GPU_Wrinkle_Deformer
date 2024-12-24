#include "Simulator.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <omp.h>

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

Simulator::Simulator(SimulationContext& context)
    : ctx(context), cloth_mesh_loaded(false), body_mesh_loaded(false) {}

void Simulator::init_Geometry_mesh() {
    init_ParamSettings();
    init_BodyClothMesh();
    init_Scene();
}

// Initialization Functions
void Simulator::init_ParamSettings() {

    // Global settings
    ctx.instance->getHostNumSubsteps() = 1;
    ctx.instance->getHostIPCDt() = 1.0 / (30.0 * ctx.instance->getHostNumSubsteps());
    ctx.instance->getHostRelativeDHat() = 1e-3; // meter, dHat = relativeDHat^2 * bbx
    ctx.instance->getHostRelativeDHatThres() = 0.1 * ctx.instance->getHostRelativeDHat(); // dThreadhold for filter Hessian
    ctx.instance->getHostPCGThreshold() = 1e-3;
    ctx.instance->getHostNewtonSolverThreshold() = 5e-2; // NewtonIter error

    ctx.instance->getHostBoundMotionRate() = 1.0;
    ctx.instance->getHostSoftStiffness() = 0.01;
    ctx.instance->getHostStitchStiffness() = 0.01;
    ctx.instance->getHostFrictionRate() = 0.4;

    // Wind parameters
    ctx.instance->getHostWindDirection() = make_Scalar3(1, 0, 0);
    ctx.instance->getHostWindStrength() = 0.0; // Equivalent to some form of force
    ctx.instance->getHostAirResistance() = 0.0; // Equivalent to some form of force
    ctx.instance->getHostNoiseAmplitude() = 2.0;
    ctx.instance->getHostNoiseFrequency() = 1.0;

    // stitch parameter
    ctx.instance->getHostStitchBreak() = true;
    ctx.instance->getHostStitchBreakThreshold() = 2000.0 * ctx.instance->getHostStitchStiffness() * ctx.instance->getHostDHat();


    // animation subrate
    ctx.instance->getHostAnimationSubRate() = 1.0 / ctx.animation_motion_rate;

    // Preconditioner type
    ctx.instance->getHostPrecondType() = 0;
    // ctx.instance->getHostPrecondType() = 1;


    // Material properties
    #define Silk
    // #define Cotton
    // #define Wool
    // #define Denim
    // #define Polyester

    #if defined(Silk)
        ctx.instance->getHostClothDensity() = 1.3e3;  // 1,300 kg/m³
        ctx.instance->getHostClothYoungModulus() = 1e6;  // 1 MPa
        ctx.instance->getHostClothPoissonRate() = 0.35;
        ctx.instance->getHostClothThickness() = 5e-4;  // 0.5 mm
    #elif defined(Cotton)
        ctx.instance->getHostClothDensity() = 1.5e3;  // 1,500 kg/m³
        ctx.instance->getHostClothYoungModulus() = 1e6;  // 1 MPa
        ctx.instance->getHostClothPoissonRate() = 0.3;
        ctx.instance->getHostClothThickness() = 2e-3;  // 2 mm
    #elif defined(Wool)
        ctx.instance->getHostClothDensity() = 1.2e3;  // 1,200 kg/m³
        ctx.instance->getHostClothYoungModulus() = 1.5e6;  // 1.5 MPa
        ctx.instance->getHostClothPoissonRate() = 0.4;
        ctx.instance->getHostClothThickness() = 1.2e-3;  // 1.2 mm
    #elif defined(Denim) // Denim
        ctx.instance->getHostClothDensity() = 1.4e3;  // 1,400 kg/m³
        ctx.instance->getHostClothYoungModulus() = 3e6;  // 3 MPa
        ctx.instance->getHostClothPoissonRate() = 0.3;
        ctx.instance->getHostClothThickness() = 2e-3;  // 2 mm
    #elif defined(Polyester) // Polyester
        ctx.instance->getHostClothDensity() = 1.3e3;  // 1,300 kg/m³
        ctx.instance->getHostClothYoungModulus() = 2.5e6;  // 2.5 MPa
        ctx.instance->getHostClothPoissonRate() = 0.35;
        ctx.instance->getHostClothThickness() = 7e-4;  // 0.7 mm
    #endif

    Scalar stretch_modulus = ctx.instance->getHostClothYoungModulus() / (1.0 - std::pow(ctx.instance->getHostClothPoissonRate(), 2.0));
    Scalar shear_modulus = ctx.instance->getHostClothYoungModulus() / (2.0 * (1.0 + ctx.instance->getHostClothPoissonRate()));
    Scalar bend_modulus = ctx.instance->getHostClothYoungModulus() * std::pow(ctx.instance->getHostClothThickness(), 3.0) / (12.0 * (1.0 - std::pow(ctx.instance->getHostClothPoissonRate(), 2.0)));
    ctx.instance->getHostStretchStiff() = 1.0 * stretch_modulus;
    ctx.instance->getHostShearStiff() = 1.0 * shear_modulus;
    ctx.instance->getHostBendStiff() = 1.0 * bend_modulus;
}


void Simulator::init_BodyClothMesh() {

    omp_set_num_threads(8);

    cloth_mesh_loaded = false;
    body_mesh_loaded = false;
    static_mesh_loaded = false;

#if defined(GPUIPC_ANIMATION)

    int frame_start = 0;
    int frame_end = 50;
    ctx.instance->getHostSimulationFrameRange() = frame_end - frame_start + 1;

    // Try to load cloth mesh
    if (LOADMESH::CGAL_readObj(ctx.assets_dir_clothmesh + ctx.clothmeshname + ".obj", ctx.simMesh_ptr->CGAL_clothmesh_orig)) {
        cloth_mesh_loaded = true;
    }

    // Try to load body mesh
    if (LOADMESH::CGAL_readObj(ctx.assets_dir_bodymesh + ctx.bodytposename + ".obj", ctx.simMesh_ptr->CGAL_bodymesh_tpose)) {
        if (LOADMESH::CGAL_readObjBody(ctx.assets_dir_bodymesh + ctx.bodymeshname + "%d.obj", frame_start, frame_end, ctx.simMesh_ptr->CGAL_bodymesh_total)) {
            body_mesh_loaded = true;
        }
    }

    // Try to load static mesh
    if (LOADMESH::CGAL_readObj(ctx.assets_dir_staticmesh + ctx.staticmeshname + ".obj", ctx.simMesh_ptr->CGAL_staticmesh)) {
        static_mesh_loaded = true;
    }

#endif

#if defined(GPUIPC_DEFORMER)
    if (LOADMESH::CGAL_readObj(ctx.assets_dir_clothmesh + ctx.clothmeshname + ".obj", ctx.simMesh_ptr->CGAL_clothmesh_orig)) {
        cloth_mesh_loaded = true;
    }

#endif

#if defined(GPUIPC_HTTP)

    LOADMESH::CGAL_readSimJson_toMesh(
        ctx.simMesh_ptr->httpjson,
        ctx.simMesh_ptr->CGAL_clothmesh_orig,
        ctx.simMesh_ptr->CGAL_bodymesh_tpose,
        ctx.simMesh_ptr->CGAL_bodymesh_total
    );

    ctx.instance->getHostSimulationFrameRange() = ctx.simMesh_ptr->CGAL_bodymesh_total.size();

    cloth_mesh_loaded = true;
    body_mesh_loaded = true;
    
#endif

    // If neither cloth nor body mesh is loaded, exit
    if (!cloth_mesh_loaded) {
        std::cerr << "Error: cloth mesh is not loaded." << std::endl;
        exit(EXIT_FAILURE);
    }

    MergeDuplicateMesh();

    GetReducedMappedMesh();

    AvoidBodyMeshIntersection();

    LinearInterpolationMesh();

    TransformCGALClothtoVector();

    TransformCGALBodytoVector();

    MergeCGALClothBodytoSurface();

    GetSurfaceInfo();

}


void Simulator::init_Scene() {

    SetSurfaceMassVelTypeCons();

    init_ClothFEM();

    AddSoftTargets();

    AddStitchPairs();

    SimulatorCUDAMalloc();

    SimulatorCUDAMemHtoD();

    SortTotalMesh();

    SortClothMesh();

    SortMASPreconditioner();

    SortSoftTargets();

    SortStitchPairs();

    BuildLBVH();

    // Firstly compute xtilta
    __INTEGRATOR__::computeXTilta(ctx.instance);

    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getHostSurfVertPos().data(), ctx.instance->getCudaSurfVertPos(),
                            ctx.simMesh_ptr->numSurfVerts * sizeof(Scalar3), cudaMemcpyDeviceToHost));

}





/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////


void Simulator::MergeDuplicateMesh() {
    // If cloth mesh is loaded, process it
    if (cloth_mesh_loaded) {
        LOADMESH::convertCGALVertsToVector(ctx.simMesh_ptr->CGAL_clothmesh_orig, ctx.simMesh_ptr->clothverts_orig);
        LOADMESH::convertCGALFacesToVector(ctx.simMesh_ptr->CGAL_clothmesh_orig, ctx.simMesh_ptr->clothfaces_orig);

        // Merge duplicate vertices
        ctx.simMesh_ptr->index_mapping_orig2fuse = LOADMESH::CGAL_MergeDuplicateVertices(
            ctx.simMesh_ptr->CGAL_clothmesh_orig,
            ctx.simMesh_ptr->CGAL_clothmesh_fuse, 1e-6);
    }
}

void Simulator::GetReducedMappedMesh() {

    // If body mesh is loaded, take mapped reduced body mesh
    if (body_mesh_loaded) {

        // If cloth mesh is loaded, get mapped body mesh
        if (cloth_mesh_loaded) {
            // Map body mesh to cloth mesh
            LOADMESH::CGAL_getReducedMappedMesh(
                ctx.simMesh_ptr->CGAL_clothmesh_fuse,
                ctx.simMesh_ptr->CGAL_bodymesh_total[0],
                0.01, 0.01, 0.01,
                ctx.simMesh_ptr->CGAL_bodymesh_mapped,
                ctx.simMesh_ptr->bodymeshmapped_faceidx
            );
        } else {
            // If cloth mesh is not loaded, set bodymesh_mapped to bodymesh_total[0]
            ctx.simMesh_ptr->CGAL_bodymesh_mapped = ctx.simMesh_ptr->CGAL_bodymesh_total[0];
        }
    }
}

void Simulator::AvoidBodyMeshIntersection() {
    if (cloth_mesh_loaded) {

        // Avoid cloth self-intersections
        LOADMESH::CGAL_avoidClothSelfIntersection(ctx.simMesh_ptr->CGAL_clothmesh_fuse);

        // If body mesh is loaded, avoid cloth-body intersections
        if (body_mesh_loaded) {
            LOADMESH::CGAL_avoidBodyClothIntersections(ctx.simMesh_ptr->CGAL_clothmesh_fuse, ctx.simMesh_ptr->CGAL_bodymesh_tpose);
        }

        // Avoid body self-intersections
        if (body_mesh_loaded) {
            LOADMESH::CGAL_avoidBodySelfIntersection(ctx.simMesh_ptr->CGAL_bodymesh_mapped);
        }
    }
}

void Simulator::LinearInterpolationMesh() {
    // If body mesh is loaded, perform interpolation
    if (body_mesh_loaded) {
        // Interpolate body mesh
        CGAL_Mesh cgal_bodymesh_fstart = ctx.simMesh_ptr->CGAL_bodymesh_total[0];
        LOADMESH::CGAL_mesh_linear_interpolation(
            ctx.simMesh_ptr->CGAL_bodymesh_tpose,
            cgal_bodymesh_fstart,
            ctx.interpolation_frames,
            ctx.simMesh_ptr->CGAL_bodymesh_total
        );
    }
}


void Simulator::TransformCGALClothtoVector() {

    // If cloth mesh is loaded, process cloth mesh
    if (cloth_mesh_loaded) {
        LOADMESH::convertCGALVertsToVector(ctx.simMesh_ptr->CGAL_clothmesh_fuse, ctx.simMesh_ptr->clothverts_fuse);
        LOADMESH::convertCGALFacesToVector(ctx.simMesh_ptr->CGAL_clothmesh_fuse, ctx.simMesh_ptr->clothfaces_fuse);
        LOADMESH::extractTriBendEdgesFaces_CGAL(ctx.simMesh_ptr->CGAL_clothmesh_fuse, ctx.simMesh_ptr->triBendEdges, ctx.simMesh_ptr->triBendVerts);

        ctx.instance->getHostNumClothVerts() = ctx.simMesh_ptr->clothverts_fuse.size();
        ctx.instance->getHostNumClothFaces() = ctx.simMesh_ptr->clothfaces_fuse.size();
        ctx.simMesh_ptr->numTriElements = ctx.simMesh_ptr->clothfaces_fuse.size();
        ctx.simMesh_ptr->triangles = ctx.simMesh_ptr->clothfaces_fuse;
        ctx.instance->getHostNumTriElements() = ctx.simMesh_ptr->numTriElements;
        ctx.instance->getHostNumTriBendEdges() = ctx.simMesh_ptr->triBendEdges.size();
        ctx.instance->getHostNumTriEdges() = num_edges(ctx.simMesh_ptr->CGAL_clothmesh_fuse);
    } else {
        ctx.instance->getHostNumClothVerts() = 0;
        ctx.instance->getHostNumClothFaces() = 0;
        ctx.simMesh_ptr->numTriElements = 0;
        ctx.instance->getHostNumTriElements() = 0;
        ctx.instance->getHostNumTriBendEdges() = 0;
    }
}


void Simulator::TransformCGALBodytoVector() {

    if (body_mesh_loaded) {
        // Convert body meshes to vector format
        ctx.simMesh_ptr->bodyverts_total.resize(ctx.simMesh_ptr->CGAL_bodymesh_total.size());
        ctx.simMesh_ptr->bodyfaces_total.resize(ctx.simMesh_ptr->CGAL_bodymesh_total.size());
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < ctx.simMesh_ptr->CGAL_bodymesh_total.size(); ++i) {
            std::vector<Scalar3> vec_bodyverts;
            std::vector<uint3> vec_bodyfaces;
            LOADMESH::convertCGALVertsToVector(ctx.simMesh_ptr->CGAL_bodymesh_total[i], vec_bodyverts);
            LOADMESH::convertCGALFacesToVector(ctx.simMesh_ptr->CGAL_bodymesh_total[i], vec_bodyfaces);
            ctx.simMesh_ptr->bodyverts_total[i] = vec_bodyverts;
            ctx.simMesh_ptr->bodyfaces_total[i] = vec_bodyfaces;
        }
        ctx.instance->getHostNumBodyVerts() = ctx.simMesh_ptr->bodyverts_total[0].size();
        ctx.instance->getHostNumBodyFaces() = ctx.simMesh_ptr->bodyfaces_total[0].size();
        ctx.simMesh_ptr->numBoundTargets = ctx.simMesh_ptr->bodyverts_total[0].size();
        ctx.instance->getHostNumBoundTargets() = ctx.simMesh_ptr->numBoundTargets;
    } else {
        ctx.instance->getHostNumBodyVerts() = 0;
        ctx.instance->getHostNumBodyFaces() = 0;
        ctx.simMesh_ptr->numBoundTargets = 0;
        ctx.instance->getHostNumBoundTargets() = 0;
    }

}

void Simulator::MergeCGALClothBodytoSurface() {

    // Merge cloth and body meshes into surfmesh
    if (cloth_mesh_loaded && body_mesh_loaded && static_mesh_loaded) {
        // surfmesh = clothmesh + bodymesh + staticmesh
        CGAL_Mesh temp_surfmesh;
        LOADMESH::CGAL_MergeMesh(temp_surfmesh, ctx.simMesh_ptr->CGAL_clothmesh_fuse, ctx.simMesh_ptr->CGAL_bodymesh_total[0]);
        LOADMESH::CGAL_MergeMesh(ctx.simMesh_ptr->CGAL_surfmesh, temp_surfmesh, ctx.simMesh_ptr->CGAL_staticmesh);

    } else if (cloth_mesh_loaded && body_mesh_loaded) {
        // surfmesh = clothmesh + bodymesh
        LOADMESH::CGAL_MergeMesh(ctx.simMesh_ptr->CGAL_surfmesh, ctx.simMesh_ptr->CGAL_clothmesh_fuse, ctx.simMesh_ptr->CGAL_bodymesh_total[0]);
    
    } else if (cloth_mesh_loaded && static_mesh_loaded) {
        // surfmesh = clothmesh + staticmesh
        LOADMESH::CGAL_MergeMesh(ctx.simMesh_ptr->CGAL_surfmesh, ctx.simMesh_ptr->CGAL_clothmesh_fuse, ctx.simMesh_ptr->CGAL_staticmesh);
    
    } else if (cloth_mesh_loaded) {
        // surfmesh = clothmesh
        ctx.simMesh_ptr->CGAL_surfmesh = ctx.simMesh_ptr->CGAL_clothmesh_fuse;
    
    } else {
        std::cout << "should include one single clothmesh!!!!!!!" << std::endl;
        exit(EXIT_FAILURE);
    }

    LOADMESH::convertCGALVertsToVector(ctx.simMesh_ptr->CGAL_surfmesh, ctx.simMesh_ptr->surffverts);
    LOADMESH::convertCGALFacesToVector(ctx.simMesh_ptr->CGAL_surfmesh, ctx.simMesh_ptr->surffaces);

    // Set surfmesh counts
    ctx.simMesh_ptr->numSurfVerts = ctx.simMesh_ptr->surffverts.size();
    ctx.simMesh_ptr->surfVertPos = ctx.simMesh_ptr->surffverts;
    ctx.instance->getHostSurfVertPos() = ctx.simMesh_ptr->surffverts;
    ctx.simMesh_ptr->surfFaceIds = ctx.simMesh_ptr->surffaces;
}

void Simulator::GetSurfaceInfo() {
    // Get surface edges and vertices
    LOADMESH::getSurface(
        ctx.simMesh_ptr->numSurfVerts,
        ctx.simMesh_ptr->surfFaceIds,
        ctx.simMesh_ptr->surfVertIds,
        ctx.simMesh_ptr->surfEdgeIds);

    
    ctx.instance->getHostNumVertices() = ctx.simMesh_ptr->numSurfVerts;
    ctx.instance->getHostNumSurfVerts() = ctx.simMesh_ptr->surfVertIds.size();
    ctx.instance->getHostNumSurfFaces() = ctx.simMesh_ptr->surfFaceIds.size();
    ctx.instance->getHostNumSurfEdges() = ctx.simMesh_ptr->surfEdgeIds.size();

}





void Simulator::init_ClothFEM() {
    if (!cloth_mesh_loaded) { return; }

    Scalar massSum = 0;
    for (int i = 0; i < ctx.simMesh_ptr->numTriElements; i++) {
        __MATHUTILS__::Matrix2x2S DM;
        __FEMENERGY__::__calculateDm2D_Scalar(ctx.simMesh_ptr->surfVertPos.data(), ctx.simMesh_ptr->triangles[i], DM);
        __MATHUTILS__::Matrix2x2S DMInverse;
        __MATHUTILS__::__Inverse2x2(DM, DMInverse);

        Scalar area = __MATHUTILS__::calculateArea(ctx.simMesh_ptr->surfVertPos.data(), ctx.simMesh_ptr->triangles[i]);
        area *= ctx.instance->getHostClothThickness();
        ctx.simMesh_ptr->area.push_back(area);
        ctx.simMesh_ptr->masses[ctx.simMesh_ptr->triangles[i].x] += ctx.instance->getHostClothDensity() * area / 3;
        ctx.simMesh_ptr->masses[ctx.simMesh_ptr->triangles[i].y] += ctx.instance->getHostClothDensity() * area / 3;
        ctx.simMesh_ptr->masses[ctx.simMesh_ptr->triangles[i].z] += ctx.instance->getHostClothDensity() * area / 3;

        massSum += area * ctx.instance->getHostClothDensity();
        ctx.simMesh_ptr->triDMInverse.push_back(DMInverse);
    }

    ctx.simMesh_ptr->meanMass = massSum / ctx.simMesh_ptr->numSurfVerts;
}








void Simulator::SetSurfaceMassVelTypeCons() {
    ctx.simMesh_ptr->masses.resize(ctx.simMesh_ptr->numSurfVerts, 0);
    ctx.simMesh_ptr->velocities.resize(ctx.simMesh_ptr->numSurfVerts, make_Scalar3(0,0,0));

    for (int i = 0; i < ctx.simMesh_ptr->numSurfVerts; i++) {

        __MATHUTILS__::Matrix3x3S constraint;

        // set clothmesh constraint Identity, btype as 0
        if (cloth_mesh_loaded && i < ctx.instance->getHostNumClothVerts()) {
            __MATHUTILS__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
            ctx.simMesh_ptr->constraints.push_back(constraint);
            ctx.simMesh_ptr->boundaryTypies.push_back(0); // Set cloth boundary as 0

        // set bodymesh constraint Identity, btype as 2
        } else if (body_mesh_loaded 
            && (i >= ctx.instance->getHostNumClothVerts())) {
            __MATHUTILS__::__set_Mat_val(constraint, 1, 0, 0, 0, 1, 0, 0, 0, 1);
            ctx.simMesh_ptr->constraints.push_back(constraint);
            ctx.simMesh_ptr->boundaryTypies.push_back(2); // Set body boundary as 2
            ctx.simMesh_ptr->boundaryTargetIndex.push_back(i);

        // set staticmesh constraint 0, btype as 3        
        } else if (static_mesh_loaded 
            && (i >= (ctx.instance->getHostNumClothVerts() + ctx.instance->getHostNumBodyVerts()))) {
            ctx.simMesh_ptr->masses[i] = 1;
            __MATHUTILS__::__set_Mat_val(constraint, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            ctx.simMesh_ptr->constraints.push_back(constraint);
            ctx.simMesh_ptr->boundaryTypies.push_back(3); // Unknown boundary type
        }
    }

    // If body mesh is loaded, initialize boundary targets
    if (body_mesh_loaded) {
        ctx.simMesh_ptr->boundaryTargetVertPos.clear();
        ctx.simMesh_ptr->boundaryTargetVertPos = ctx.simMesh_ptr->bodyverts_total[0];
        ctx.instance->getHostBoundaryTargetPos() = ctx.simMesh_ptr->bodyverts_total[0];
    }
}


void Simulator::AddSoftTargets() {
    // Only process soft targets and stitch pairs if cloth mesh is loaded
    if (cloth_mesh_loaded && ctx.do_addSoftTargets) {

        std::vector<uint32_t> softtarget_vertex_indices;
        LOADMESH::CGAL_getSoftTargetConstraintsPoints(ctx.simMesh_ptr->CGAL_clothmesh_fuse, softtarget_vertex_indices);
        ctx.simMesh_ptr->softTargetIdsBeforeSort = softtarget_vertex_indices;
        ctx.instance->getHostNumSoftTargets() = ctx.simMesh_ptr->softTargetIdsBeforeSort.size();

        for (uint32_t i = 0; i < softtarget_vertex_indices.size(); ++i) {
            std::cout << ctx.simMesh_ptr->softTargetIdsBeforeSort[i] << ", ";
        }
        std::cout << "++++++++++++++++++++++++++" << std::endl;

        // 计算softtarget在第一帧bodymesh上的uv投影 并且记录到projectedFaceIds和projectedUVs中
        LOADMESH::CGAL_computeProjectionUVs(
            ctx.simMesh_ptr->clothverts_fuse,
            ctx.simMesh_ptr->clothfaces_fuse,
            ctx.simMesh_ptr->bodyverts_total[0],
            ctx.simMesh_ptr->bodyfaces_total[0],
            ctx.simMesh_ptr->softTargetIdsBeforeSort,
            ctx.simMesh_ptr->projectedFaceIds,
            ctx.simMesh_ptr->projectedUVs);

        // 按照softtarget在bodymesh上的uv投影 偏移0.005距离 计算所有的帧的bodymesh投影 然后记录到projectedPositions中
        ctx.simMesh_ptr->projectedPositions.resize(ctx.simMesh_ptr->bodyverts_total.size());
        for (int i = 0; i < ctx.simMesh_ptr->bodyverts_total.size(); ++i) {
            LOADMESH::CGAL_computeProjectedPositions(
                ctx.simMesh_ptr->bodyverts_total[i],
                ctx.simMesh_ptr->bodyfaces_total[i],
                ctx.simMesh_ptr->projectedFaceIds,
                ctx.simMesh_ptr->projectedUVs,
                0.005,
                ctx.simMesh_ptr->projectedPositions[i]);
        }
        // initscene中我们只要第一帧的bodymesh投影信息
        ctx.instance->getHostSoftTargetPos() = ctx.simMesh_ptr->projectedPositions[0];
        
    }
}

void Simulator::AddStitchPairs() {
    if (cloth_mesh_loaded && ctx.do_addStitchPairs) {

        if (!((ctx.clothmeshname == "drag_majia") || (ctx.clothmeshname == "lianti_x_majia_cloth_new"))) {
            std::cout << "stitch is specific option right now, only support one cloth!!!!!!!!!" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (ctx.do_addStitchPairs && ctx.do_addSoftTargets) {
            std::cout << "we do not support stitch and softtargets simutaneously!!!!!!!!!!" << std::endl;
            exit(EXIT_FAILURE);
        }

        ctx.simMesh_ptr->stitchPairsBeforeSort.push_back(make_uint3(
            ctx.simMesh_ptr->index_mapping_orig2fuse[308], ctx.simMesh_ptr->index_mapping_orig2fuse[1305], 1));
        ctx.simMesh_ptr->stitchPairsBeforeSort.push_back(make_uint3(
            ctx.simMesh_ptr->index_mapping_orig2fuse[297], ctx.simMesh_ptr->index_mapping_orig2fuse[1294], 1));
        ctx.simMesh_ptr->stitchPairsBeforeSort.push_back(make_uint3(
            ctx.simMesh_ptr->index_mapping_orig2fuse[286], ctx.simMesh_ptr->index_mapping_orig2fuse[1283], 1));

        ctx.instance->getHostNumStitchPairs() = ctx.simMesh_ptr->stitchPairsBeforeSort.size();

    }
}

void Simulator::SortSoftTargets() {
    // 把softtargets的目标点的id也要更新重新sortmesh一下 获取sort之后的新的id
    if (ctx.do_addSoftTargets) {   
        std::vector<uint32_t> tempSortMapIndex(ctx.simMesh_ptr->clothverts_fuse.size());
        std::vector<uint32_t> tempTargetIdsAfterSort;
        tempSortMapIndex.clear();
        CUDA_SAFE_CALL(cudaMemcpy(tempSortMapIndex.data(), ctx.instance->getCudaSortMapIdOtoS(), ctx.simMesh_ptr->clothverts_fuse.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        tempTargetIdsAfterSort.clear();
        for (auto& id : ctx.simMesh_ptr->softTargetIdsBeforeSort) {
            std::cout << id << ", ";
            tempTargetIdsAfterSort.push_back(tempSortMapIndex[id]);
        }
        std::cout << "\n" << "###############################################" << std::endl;
        ctx.instance->getHostSoftTargetIdsAfterSort().clear();
        for (auto& id : tempTargetIdsAfterSort) {
            std::cout << id << ", ";
            ctx.instance->getHostSoftTargetIdsAfterSort().push_back(id);
        }
        std::cout << "\n" << "###############################################" << std::endl;

        CUDAMallocSafe(ctx.instance->getCudaSoftTargetIndex(), ctx.instance->getHostNumSoftTargets());
        CUDAMallocSafe(ctx.instance->getCudaSoftTargetVertPos(), ctx.instance->getHostNumSoftTargets());

        CUDAMemcpyHToDSafe(ctx.instance->getCudaSoftTargetIndex(), ctx.instance->getHostSoftTargetIdsAfterSort());
        CUDAMemcpyHToDSafe(ctx.instance->getCudaSoftTargetVertPos(), ctx.instance->getHostSoftTargetPos());

    }
}

void Simulator::SortStitchPairs() {
    if (ctx.do_addStitchPairs) {

        std::vector<uint32_t> tempSortMapIndex(ctx.simMesh_ptr->clothverts_fuse.size());
        std::vector<uint3> tempStitchIdsAfterSort;
        tempSortMapIndex.clear();
        CUDA_SAFE_CALL(cudaMemcpy(tempSortMapIndex.data(), ctx.instance->getCudaSortMapIdOtoS(), ctx.simMesh_ptr->clothverts_fuse.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        tempStitchIdsAfterSort.clear();
        for (auto& id : ctx.simMesh_ptr->stitchPairsBeforeSort) {
            std::cout << id.x << " " << id.y << " " << id.z << ", ";
            tempStitchIdsAfterSort.push_back(
                make_uint3(tempSortMapIndex[id.x], tempSortMapIndex[id.y], id.z));
        }
        std::cout << "\n" << "=============================================" << std::endl;
        ctx.instance->getHostStitchPairsAfterSort().clear();
        for (auto& id : tempStitchIdsAfterSort) {
            std::cout << id.x << " " << id.y << " " << id.z << ", ";
            ctx.instance->getHostStitchPairsAfterSort().push_back(id);
        }
        std::cout << "\n" << "=============================================" << std::endl;

        CUDAMallocSafe(ctx.instance->getCudaStitchPairsIndex(), ctx.instance->getHostNumStitchPairs());

        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaStitchPairsIndex(), ctx.instance->getHostStitchPairsAfterSort().data(), ctx.instance->getHostNumStitchPairs() * sizeof(uint3), cudaMemcpyHostToDevice));


    }
}

void Simulator::SortTotalMesh() {

    // 按照surfmesh的AABB大小 对clothmesh进行sortMesh将相互作用的顶点放到临近位置
    AABB* bvs_AABB = ctx.LBVH_CD_ptr->lbvh_f.getSceneSize();
    std::vector<Scalar3> vec_upper_bvs(1);
    std::vector<Scalar3> vec_lower_bvs(1);
    CUDAMemcpyDToHSafe(vec_upper_bvs, &bvs_AABB->upper);
    CUDAMemcpyDToHSafe(vec_lower_bvs, &bvs_AABB->lower);
    Scalar3 _upper_bvs = vec_upper_bvs[0];
    Scalar3 _lower_bvs = vec_lower_bvs[0];

    __SORTMESH__::sortMesh(
        ctx.instance->getCudaSurfVertPos(), ctx.instance->getCudaMortonCodeHash(),
        ctx.instance->getCudaSortMapIdStoO(), ctx.instance->getCudaSortMapIdOtoS(),
        ctx.instance->getCudaOriginVertPos(), ctx.instance->getCudaTempScalar(),
        ctx.instance->getCudaVertMass(), ctx.instance->getCudaTempMat3x3(),
        ctx.instance->getCudaConstraintsMat(), ctx.instance->getCudaBoundaryType(),
        ctx.instance->getCudaTempBoundaryType(), ctx.instance->getCudaTetElement(),
        ctx.instance->getCudaTriElement(), ctx.instance->getCudaSurfFaceIds(),
        ctx.instance->getCudaSurfEdgeIds(), ctx.instance->getCudaTriBendEdges(),
        ctx.instance->getCudaTriBendVerts(), ctx.instance->getCudaSurfVertIds(),
        ctx.instance->getHostNumTetElements(), ctx.instance->getHostNumTriElements(),
        ctx.instance->getHostNumSurfVerts(), ctx.instance->getHostNumSurfFaces(),
        ctx.instance->getHostNumSurfEdges(), ctx.instance->getHostNumTriBendEdges(),
        _upper_bvs, _lower_bvs, 
        ctx.simMesh_ptr->clothverts_fuse.size());

    // 初始化sort后的rest vertpos
    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaRestVertPos(), ctx.instance->getCudaOriginVertPos(),
                            ctx.simMesh_ptr->numSurfVerts * sizeof(Scalar3), cudaMemcpyDeviceToDevice));

}

void Simulator::SortClothMesh() {
    // 拿到sortmesh的vert的id信息 用于后面output恢复到原始的位置
    ctx.simMesh_ptr->SortMapIdStoO.resize(ctx.simMesh_ptr->clothverts_fuse.size());
	ctx.simMesh_ptr->SortMapIdOtoS.resize(ctx.simMesh_ptr->clothverts_fuse.size());
	CUDA_SAFE_CALL(cudaMemcpy(ctx.simMesh_ptr->SortMapIdStoO.data(), ctx.instance->getCudaSortMapIdStoO(), ctx.simMesh_ptr->clothverts_fuse.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(ctx.simMesh_ptr->SortMapIdOtoS.data(), ctx.instance->getCudaSortMapIdOtoS(), ctx.simMesh_ptr->clothverts_fuse.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_SAFE_CALL(cudaMemcpy(ctx.simMesh_ptr->surfFaceIds.data(), ctx.instance->getCudaSurfFaceIds(),
                            ctx.instance->getHostNumSurfFaces() * sizeof(uint3), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(ctx.simMesh_ptr->surfEdgeIds.data(), ctx.instance->getCudaSurfEdgeIds(),
                            ctx.instance->getHostNumSurfEdges() * sizeof(uint2), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(ctx.simMesh_ptr->surfVertIds.data(), ctx.instance->getCudaSurfVertIds(),
                            ctx.instance->getHostNumSurfVerts() * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // 记录sort后的faceid
    for (const auto& face : ctx.simMesh_ptr->surfFaceIds) {
        if (face.x < ctx.simMesh_ptr->SortMapIdStoO.size() && face.y < ctx.simMesh_ptr->SortMapIdStoO.size() && face.z < ctx.simMesh_ptr->SortMapIdStoO.size()) {
            ctx.instance->getHostClothFacesAfterSort().push_back(face);
        } else {
            uint3 bodyFace;
            bodyFace.x = face.x - ctx.simMesh_ptr->SortMapIdStoO.size();
            bodyFace.y = face.y - ctx.simMesh_ptr->SortMapIdStoO.size();
            bodyFace.z = face.z - ctx.simMesh_ptr->SortMapIdStoO.size();
            ctx.instance->getHostBodyFacesAfterSort().push_back(bodyFace);
        }
    }
}

void Simulator::SortMASPreconditioner() {

    // 如果启用MAS Preconditioner 也要对preconditioner进行sort
    if (ctx.instance->getHostPrecondType() == 1) {
        int neighborListSize = LOADMESH::getVertNeighbors(
            ctx.simMesh_ptr->numSurfVerts,
            ctx.simMesh_ptr->numTetElements,
            ctx.simMesh_ptr->vertNeighbors,
            ctx.simMesh_ptr->neighborList,
            ctx.simMesh_ptr->neighborStart,
            ctx.simMesh_ptr->neighborNum,
            ctx.simMesh_ptr->triangles,
            ctx.simMesh_ptr->tetrahedras);

        ctx.PCG_ptr->MP.CUDA_MALLOC_MAS_PRECONDITIONER(ctx.simMesh_ptr->numSurfVerts, neighborListSize,
                                                 ctx.instance->getCudaCollisionPairs());

        ctx.PCG_ptr->MP.hostMASNeighborListSize = neighborListSize;

        CUDAMemcpyHToDSafe(ctx.PCG_ptr->MP.cudaNeighborListInit, ctx.simMesh_ptr->neighborList);
        CUDAMemcpyHToDSafe(ctx.PCG_ptr->MP.cudaNeighborStart, ctx.simMesh_ptr->neighborStart);
        CUDAMemcpyHToDSafe(ctx.PCG_ptr->MP.cudaNeighborNumInit, ctx.simMesh_ptr->neighborNum);

        __SORTMESH__::sortPreconditioner(
            ctx.PCG_ptr->MP.cudaNeighborList, ctx.PCG_ptr->MP.cudaNeighborListInit,
            ctx.PCG_ptr->MP.cudaNeighborNum, ctx.PCG_ptr->MP.cudaNeighborNumInit,
            ctx.PCG_ptr->MP.cudaNeighborStart, ctx.PCG_ptr->MP.cudaNeighborStartTemp,
            ctx.instance->getCudaSortMapIdStoO(), ctx.instance->getCudaSortMapIdOtoS(),
            ctx.PCG_ptr->MP.hostMASNeighborListSize, ctx.simMesh_ptr->clothverts_fuse.size());
    }
}


void Simulator::SimulatorCUDAMalloc() {

    ctx.instance->getHostMaxCCDCollisionPairsNum() =
        ctx.collision_detection_buff_scale *
        (((Scalar)(ctx.instance->getHostNumSurfFaces() * 15 + ctx.instance->getHostNumSurfEdges() * 10)) *
         std::max((ctx.instance->getHostIPCDt() / 0.01), 2.0));
    ctx.instance->getHostMaxCollisionPairsNum() =
        ctx.collision_detection_buff_scale *
        (ctx.instance->getHostNumSurfVerts() * 3 + ctx.instance->getHostNumSurfEdges() * 2) * 3;
    ctx.instance->getHostMaxTetTriMortonCodeNum() = ctx.instance->getHostNumVertices();

    ctx.instance->getHostCpNumLast(0) = 0;
    ctx.instance->getHostCpNumLast(1) = 0;
    ctx.instance->getHostCpNumLast(2) = 0;
    ctx.instance->getHostCpNumLast(3) = 0;
    ctx.instance->getHostCpNumLast(4) = 0;

    ctx.instance->getHostMinKappaCoef() = 1e11;
    ctx.instance->getHostMeanMass() = ctx.simMesh_ptr->meanMass;
    ctx.instance->getHostMeanVolume() = ctx.simMesh_ptr->meanVolume;

    ctx.PCG_ptr->PrecondType = ctx.instance->getHostPrecondType();

    // Allocate device memory based on whether cloth or body mesh is loaded
    CUDAMallocSafe(ctx.instance->getCudaSurfVertPos(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaOriginVertPos(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaVertVel(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaTempScalar3Mem(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaXTilta(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaFb(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaVertMass(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaMortonCodeHash(), ctx.instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(ctx.instance->getCudaSortMapIdStoO(), ctx.instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(ctx.instance->getCudaBoundaryType(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaTempBoundaryType(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaSortMapIdOtoS(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaConstraintsMat(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaTempScalar(), ctx.instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(ctx.instance->getCudaTempMat3x3(), ctx.instance->getHostMaxCollisionPairsNum());

    if (body_mesh_loaded) {
        CUDAMallocSafe(ctx.instance->getCudaBoundTargetIndex(), ctx.simMesh_ptr->numBoundTargets);
        CUDAMallocSafe(ctx.instance->getCudaBoundTargetVertPos(), ctx.simMesh_ptr->numBoundTargets);
        CUDAMallocSafe(ctx.instance->getCudaPrevBoundTargetVertPos(), ctx.simMesh_ptr->numBoundTargets);
        CUDAMallocSafe(ctx.instance->getCudaBoundaryStopDist(), ctx.simMesh_ptr->numBoundTargets);
    }

    if (cloth_mesh_loaded) {
        CUDAMallocSafe(ctx.instance->getCudaRestVertPos(), ctx.simMesh_ptr->numSurfVerts);
        CUDAMallocSafe(ctx.instance->getCudaTriBendEdges(), ctx.simMesh_ptr->triBendEdges.size());
        CUDAMallocSafe(ctx.instance->getCudaTriBendVerts(), ctx.simMesh_ptr->triBendEdges.size());
        CUDAMallocSafe(ctx.instance->getCudaTriDmInverses(), ctx.simMesh_ptr->numTriElements);
        CUDAMallocSafe(ctx.instance->getCudaTriArea(), ctx.simMesh_ptr->numTriElements);
        CUDAMallocSafe(ctx.instance->getCudaTriElement(), ctx.simMesh_ptr->numTriElements);
        CUDAMallocSafe(ctx.instance->getCudaTriVerts(), ctx.instance->getHostNumClothVerts());
        CUDAMallocSafe(ctx.instance->getCudaTriEdges(), ctx.instance->getHostNumTriEdges());
    }


    CUDAMallocSafe(ctx.instance->getCudaMoveDir(), ctx.simMesh_ptr->numSurfVerts);
    CUDAMallocSafe(ctx.instance->getCudaMatIndex(), ctx.instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(ctx.instance->getCudaCollisionPairs(), ctx.instance->getHostMaxCollisionPairsNum());
    CUDAMallocSafe(ctx.instance->getCudaCCDCollisionPairs(), ctx.instance->getHostMaxCCDCollisionPairsNum());
    CUDAMallocSafe(ctx.instance->getCudaEnvCollisionPairs(), ctx.instance->getHostNumSurfVerts());
    CUDAMallocSafe(ctx.instance->getCudaCPNum(), 5);
    CUDAMallocSafe(ctx.instance->getCudaGPNum(), 1);
    CUDAMallocSafe(ctx.instance->getCudaGroundNormal(), 5);
    CUDAMallocSafe(ctx.instance->getCudaGroundOffset(), 5);

    CUDAMallocSafe(ctx.instance->getCudaSurfFaceIds(), ctx.instance->getHostNumSurfFaces());
    CUDAMallocSafe(ctx.instance->getCudaSurfEdgeIds(), ctx.instance->getHostNumSurfEdges());
    CUDAMallocSafe(ctx.instance->getCudaSurfVertIds(), ctx.instance->getHostNumSurfVerts());

    CUDAMallocSafe(ctx.instance->getCudaCloseCPNum(), 1);
    CUDAMallocSafe(ctx.instance->getCudaCloseGPNum(), 1);

    ctx.PCG_ptr->CUDA_MALLOC_PCGSOLVER(ctx.simMesh_ptr->numSurfVerts);
    ctx.BH_ptr->CUDA_MALLOC_BLOCKHESSIAN(
        ctx.instance->getHostNumTetElements(), ctx.instance->getHostNumSurfVerts(),
        ctx.instance->getHostNumSurfFaces(), ctx.instance->getHostNumSurfEdges(),
        ctx.instance->getHostNumTriElements(), ctx.instance->getHostNumTriBendEdges());

    ctx.LBVH_CD_ptr->initBVH(ctx.instance, ctx.instance->getCudaBoundaryType());

}


void Simulator::SimulatorCUDAMemHtoD() {


    // Copy data to device
    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaSurfVertPos(), ctx.simMesh_ptr->surfVertPos.data(), sizeof(Scalar3) * ctx.simMesh_ptr->surfVertPos.size(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaVertVel(), ctx.simMesh_ptr->velocities.data(), sizeof(Scalar3) * ctx.simMesh_ptr->velocities.size(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaVertMass(), ctx.simMesh_ptr->masses.data(), sizeof(Scalar) * ctx.simMesh_ptr->masses.size(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaOriginVertPos(), ctx.simMesh_ptr->surfVertPos.data(), sizeof(Scalar3) * ctx.simMesh_ptr->surfVertPos.size(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaConstraintsMat(), ctx.simMesh_ptr->constraints.data(), sizeof(__MATHUTILS__::Matrix3x3S) * ctx.simMesh_ptr->constraints.size(), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaBoundaryType(), ctx.simMesh_ptr->boundaryTypies.data(), sizeof(int) * ctx.simMesh_ptr->boundaryTypies.size(), cudaMemcpyHostToDevice));

    if (cloth_mesh_loaded) {
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaTriElement(), ctx.simMesh_ptr->triangles.data(), sizeof(uint3) * ctx.simMesh_ptr->triangles.size(), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaTriArea(), ctx.simMesh_ptr->area.data(), sizeof(Scalar) * ctx.simMesh_ptr->area.size(), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaTriDmInverses(), ctx.simMesh_ptr->triDMInverse.data(), sizeof(__MATHUTILS__::Matrix2x2S) * ctx.simMesh_ptr->triDMInverse.size(), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaTriBendEdges(), ctx.simMesh_ptr->triBendEdges.data(), sizeof(uint2) * ctx.simMesh_ptr->triBendEdges.size(), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaTriBendVerts(), ctx.simMesh_ptr->triBendVerts.data(), sizeof(uint2) * ctx.simMesh_ptr->triBendVerts.size(), cudaMemcpyHostToDevice));
    }

    if (body_mesh_loaded) {
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaBoundTargetIndex(), ctx.simMesh_ptr->boundaryTargetIndex.data(), sizeof(uint32_t) * ctx.simMesh_ptr->boundaryTargetIndex.size(), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaBoundTargetVertPos(), ctx.simMesh_ptr->boundaryTargetVertPos.data(), sizeof(Scalar3) * ctx.simMesh_ptr->boundaryTargetVertPos.size(), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaPrevBoundTargetVertPos(), ctx.simMesh_ptr->boundaryTargetVertPos.data(), sizeof(Scalar3) * ctx.simMesh_ptr->boundaryTargetVertPos.size(), cudaMemcpyHostToDevice));
    }


    std::vector<Scalar> h_offset = {0, -1, 1, -1, 1}; // TODO: 似乎只有一个y轴起到了作用
    std::vector<Scalar3> H_normal = {make_Scalar3(0, 1, 0), make_Scalar3(1, 0, 0),
                                     make_Scalar3(-1, 0, 0), make_Scalar3(0, 0, 1),
                                     make_Scalar3(0, 0, -1)};
    CUDAMemcpyHToDSafe(ctx.instance->getCudaGroundOffset(), h_offset);
    CUDAMemcpyHToDSafe(ctx.instance->getCudaGroundNormal(), H_normal);

    CUDAMemcpyHToDSafe(ctx.instance->getCudaSurfFaceIds(), ctx.simMesh_ptr->surfFaceIds);
    CUDAMemcpyHToDSafe(ctx.instance->getCudaSurfEdgeIds(), ctx.simMesh_ptr->surfEdgeIds);
    CUDAMemcpyHToDSafe(ctx.instance->getCudaSurfVertIds(), ctx.simMesh_ptr->surfVertIds);

}


void Simulator::BuildLBVH() {

    // 构建BVH
    ctx.LBVH_CD_ptr->buildBVH(ctx.instance);

    // lower/upper of bbx, bbox diag square
    ctx.instance->getHostBboxDiagSize2() = __MATHUTILS__::__vec3_squaredNorm(
        __MATHUTILS__::__vec3_minus(ctx.LBVH_CD_ptr->lbvh_f.scene.upper, ctx.LBVH_CD_ptr->lbvh_f.scene.lower));
    ctx.instance->getHostDTol() = very_small_number() * ctx.instance->getHostBboxDiagSize2();
    ctx.instance->getHostDHat() = std::pow(ctx.instance->getHostRelativeDHat(), 2) * ctx.instance->getHostBboxDiagSize2(); // d^_sqr * bbx
    ctx.instance->getHostFDHat() = 1e-6 * ctx.instance->getHostBboxDiagSize2();

    printf("bboxDiagSize2: %f\n", ctx.instance->getHostBboxDiagSize2());
    printf("bbox upper: %f, %f, %f\n", ctx.LBVH_CD_ptr->lbvh_f.scene.upper.x, ctx.LBVH_CD_ptr->lbvh_f.scene.upper.y, ctx.LBVH_CD_ptr->lbvh_f.scene.upper.z);
    printf("bbox lower: %f, %f, %f\n", ctx.LBVH_CD_ptr->lbvh_f.scene.lower.x, ctx.LBVH_CD_ptr->lbvh_f.scene.lower.y, ctx.LBVH_CD_ptr->lbvh_f.scene.lower.z);

    // 根据LBVH找到首先先做一次碰撞检测 构建collision pair几何 构建IPC能量
    ctx.LBVH_CD_ptr->buildCP(ctx.instance);
    ctx.LBVH_CD_ptr->buildGP(ctx.instance);

    // 如果我们在最开始就检测self intersection 检测到了就直接退出程序
    // clothmesh的selfinter clothmesh对于bodymesh的interse 不会检测body自己的selfinter 
    if (__GPUIPC__::isIntersected(ctx.instance, ctx.LBVH_CD_ptr)) {
        printf("init cloth self intersection please fix the problem again \n");
        exit(EXIT_FAILURE);
    }

}



/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////





void Simulator::display_without_opengl_animation() {

    // 如果加载了布料网格，保存模拟的布料顶点到 output_clothverts_total 中
    if (cloth_mesh_loaded) {
        std::vector<Scalar3> output_clothverts_currframe(
            ctx.instance->getHostSurfVertPos().begin(),
            ctx.instance->getHostSurfVertPos().begin() + ctx.simMesh_ptr->SortMapIdStoO.size()
        );
        ctx.instance->getHostOutputClothVertsTotal().push_back(output_clothverts_currframe);
    }

    ++ctx.instance->getHostSimulationFrameId();

    // 保存模拟结果
    if (ctx.instance->getHostSimulationFrameId() >= ctx.instance->getHostSimulationFrameRange() + ctx.interpolation_frames) {

        std::cout << "Total time for Simulation: " << ctx.simMesh_ptr->simulation_totaltime << " seconds" << std::endl;

        // 如果加载了布料网格，恢复布料顶点到排序前的位置
        if (cloth_mesh_loaded) {
            // 还原 clothverts 至 sortMesh 之前
            LOADMESH::saveSurfaceMesh_restore_from_sort(
                ctx.simMesh_ptr->SortMapIdStoO, 
                ctx.instance->getHostOutputClothVertsTotal());
            LOADMESH::saveSurfaceMesh_restore_from_fuse(
                ctx.simMesh_ptr->clothfaces_orig,
                ctx.simMesh_ptr->index_mapping_orig2fuse,
                ctx.instance->getHostOutputClothVertsTotal());

            std::vector<std::vector<Scalar3>> output_clothverts_total_subset(
                ctx.instance->getHostOutputClothVertsTotal().begin() + ctx.interpolation_frames,  // 从 ctx.interpolation_frames 开始
                ctx.instance->getHostOutputClothVertsTotal().end()
            );

    #if defined(GPUIPC_ANIMATION)

            std::string outputdir = "output_" + ctx.clothmeshname + "_" + ctx.getCurrentTime();
            std::filesystem::create_directories(outputdir);
            for (int i = 0; i < output_clothverts_total_subset.size(); i++) {
                LOADMESH::saveMeshToOBJ(output_clothverts_total_subset[i], ctx.simMesh_ptr->clothfaces_orig, outputdir + "/" + ctx.clothmeshname + "_" + std::to_string(i) + ".obj");
            }

    #endif

    #if defined(GPUIPC_HTTP)

            LOADMESH::CGAL_saveClothMesh_toJson(
                output_clothverts_total_subset,
                ctx.simMesh_ptr->clothfaces_orig,
                ctx.simMesh_ptr->output_httpjson
            );

    #endif
        }

        ctx.simMesh_ptr->simulation_finished = true;
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // 在 IPC_Solver 计算之前，确保目标位置数据已经更新并传输到设备端
    if (ctx.do_addSoftTargets && cloth_mesh_loaded) {
        ctx.instance->getHostSoftTargetPos() = ctx.simMesh_ptr->projectedPositions[ctx.instance->getHostSimulationFrameId()];
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaSoftTargetVertPos(), ctx.instance->getHostSoftTargetPos().data(),
                                  ctx.instance->getHostNumSoftTargets() * sizeof(Scalar3), cudaMemcpyHostToDevice));
    }

    if (body_mesh_loaded) {
        ctx.instance->getHostBoundaryTargetPos() = ctx.simMesh_ptr->bodyverts_total[ctx.instance->getHostSimulationFrameId()];
        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaBoundTargetVertPos(), ctx.instance->getHostBoundaryTargetPos().data(),
                                  ctx.instance->getHostNumBoundTargets() * sizeof(Scalar3), cudaMemcpyHostToDevice));

        std::vector<Scalar3> output_bodyverts_currframe;
        if (cloth_mesh_loaded) {
            // 如果同时加载了布料网格和身体网格
            output_bodyverts_currframe.assign(
                ctx.instance->getHostSurfVertPos().begin() + ctx.simMesh_ptr->SortMapIdStoO.size(),
                ctx.instance->getHostSurfVertPos().end()
            );
        } else {
            // 如果只加载了身体网格
            output_bodyverts_currframe.assign(
                ctx.instance->getHostSurfVertPos().begin(),
                ctx.instance->getHostSurfVertPos().end()
            );
        }

        CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getCudaPrevBoundTargetVertPos(), output_bodyverts_currframe.data(),
                                  ctx.instance->getHostNumBoundTargets() * sizeof(Scalar3), cudaMemcpyHostToDevice));
    }

    // 每一帧的具体 IPC 计算
    for (int i = 0; i < ctx.instance->getHostNumSubsteps(); i++) {
        __INTEGRATOR__::IPC_Solver(ctx.instance, ctx.BH_ptr, ctx.PCG_ptr, ctx.LBVH_CD_ptr);
    }

    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getHostSurfVertPos().data(), ctx.instance->getCudaSurfVertPos(),
                            ctx.instance->getHostNumVertices() * sizeof(Scalar3), cudaMemcpyDeviceToHost));

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<Scalar> elapsed_time = end_time - start_time;
    ctx.simMesh_ptr->simulation_totaltime += elapsed_time.count();

    // exit(EXIT_SUCCESS);
}





void Simulator::display_without_opengl_drag() {

    ++ctx.instance->getHostSimulationFrameId();

    if (ctx.glRender_ptr->isDragging && ctx.glRender_ptr->dragTargetId != -1) {

        ctx.instance->getHostNumSoftTargets() += 1;
        ctx.instance->getHostSoftTargetIdsAfterSort().push_back(ctx.glRender_ptr->dragTargetId);
        ctx.instance->getHostSoftTargetPos().push_back(make_Scalar3(
            ctx.glRender_ptr->dragTargetPosition.x, 
            ctx.glRender_ptr->dragTargetPosition.y, 
            ctx.glRender_ptr->dragTargetPosition.z));
        
        CUDAMallocSafe(ctx.instance->getCudaSoftTargetIndex(), ctx.instance->getHostNumSoftTargets());
        CUDAMallocSafe(ctx.instance->getCudaSoftTargetVertPos(), ctx.instance->getHostNumSoftTargets());
        CUDAMemcpyHToDSafe(ctx.instance->getCudaSoftTargetIndex(), ctx.instance->getHostSoftTargetIdsAfterSort());
        CUDAMemcpyHToDSafe(ctx.instance->getCudaSoftTargetVertPos(), ctx.instance->getHostSoftTargetPos());

    }

    __INTEGRATOR__::IPC_Solver(ctx.instance, ctx.BH_ptr, ctx.PCG_ptr, ctx.LBVH_CD_ptr);

    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getHostSurfVertPos().data(), ctx.instance->getCudaSurfVertPos(),
                ctx.instance->getHostNumVertices() * sizeof(Scalar3), cudaMemcpyDeviceToHost));
    
    CUDA_SAFE_CALL(cudaMemcpy(ctx.instance->getHostStitchPairsAfterSort().data(), ctx.instance->getCudaStitchPairsIndex(),
                ctx.instance->getHostNumStitchPairs() * sizeof(uint3), cudaMemcpyDeviceToHost));

    if (ctx.glRender_ptr->isDragging && ctx.glRender_ptr->dragTargetId != -1) {

        ctx.instance->getHostNumSoftTargets() -= 1;
        ctx.instance->getHostSoftTargetIdsAfterSort().pop_back();
        ctx.instance->getHostSoftTargetPos().pop_back();
        CUDAFreeSafe(ctx.instance->getCudaSoftTargetVertPos());
        CUDAFreeSafe(ctx.instance->getCudaSoftTargetIndex());

        CUDAMallocSafe(ctx.instance->getCudaSoftTargetIndex(), ctx.instance->getHostNumSoftTargets());
        CUDAMallocSafe(ctx.instance->getCudaSoftTargetVertPos(), ctx.instance->getHostNumSoftTargets());
        CUDAMemcpyHToDSafe(ctx.instance->getCudaSoftTargetIndex(), ctx.instance->getHostSoftTargetIdsAfterSort());
        CUDAMemcpyHToDSafe(ctx.instance->getCudaSoftTargetVertPos(), ctx.instance->getHostSoftTargetPos());
    }

    // exit(EXIT_SUCCESS);
}


void Simulator::display(void) {
    
    // auto& ctx.instance = GeometryManager::ctx.instance;
    CHECK_ERROR(ctx.instance, "geoctx.instance not initialized");

    ctx.glRender_ptr->draw_Scene3D();

    if (ctx.glRender_ptr->stopRender) return;

#if defined(GPUIPC_ANIMATION)
    display_without_opengl_animation();
#endif
#if defined(GPUIPC_DRAG)
    display_without_opengl_drag();
#endif

}

