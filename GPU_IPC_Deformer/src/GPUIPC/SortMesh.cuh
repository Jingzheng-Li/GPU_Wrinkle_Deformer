
#pragma once

#include "PCGSolver.cuh"
#include "UTILS/GeometryManager.hpp"
#include "UTILS/MathUtils.cuh"

namespace __SORTMESH__
{


void sortMesh(
    Scalar3* cudaVertPos,
    uint64_t* cudaMortonCodeHash,
    uint32_t* cudaSortMapIdStoO,
    uint32_t* cudaSortMapIdOtoS,
    Scalar3* cudaOriginVertPos,
    Scalar* cudaTempScalar,
    Scalar* cudaVertMass,
    __MATHUTILS__::Matrix3x3S* cudaTempMat3x3,
    __MATHUTILS__::Matrix3x3S* cudaConstraintsMat,
    int* cudaBoundaryType,
    int* cudaTempBoundaryType,
    uint4* cudaTetElement,
    uint3* cudaTriElement,
    uint3* cudaSurfFace,
    uint2* cudaSurfEdge,
    uint2* cudaTriBendEdges,
    uint2* cudaTriBendVerts,
    uint32_t* cudaSurfVert,
    int hostNumTetElements,
    int hostNumTriElements,
    int hostNumSurfVerts,
    int hostNumSurfFaces,
    int hostNumSurfEdges,
    int hostNumTriBendEdges,
    const Scalar3 _upper_bvs,
    const Scalar3 _lower_bvs,
    int updateVertNum
);
    

void sortPreconditioner(
    unsigned int* cudaNeighborList,
    unsigned int* cudaNeighborListInit,
    unsigned int* cudaNeighborNum,
    unsigned int* cudaNeighborNumInit,
    unsigned int* cudaNeighborStart,
    unsigned int* cudaNeighborStartTemp,
    uint32_t* cudaSortMapIdStoO,
    uint32_t* cudaSortMapIdOtoS,
    int MASNeighborListSize,
    int updateVertNum
);


} // namespace __SORTMESH__


