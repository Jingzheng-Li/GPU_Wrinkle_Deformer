
#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "UTILS/GeometryManager.hpp"
#include "BlockHessian.cuh"

#include "GIPCFricUtils.cuh"


namespace __GIPCFRICTION__ {

__global__ void _getFrictionEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                const Scalar3* o_vertexes,
                                                const int4* _collisionPair, int cpNum, Scalar dt,
                                                const Scalar2* distCoord,
                                                const __MATHUTILS__::Matrix3x2S* tanBasis,
                                                const Scalar* lastH, Scalar fricDHat, Scalar eps);

__global__ void _getFrictionEnergy_gd_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                   const Scalar3* o_vertexes,
                                                   const Scalar3* _normal,
                                                   const uint32_t* _collisionPair_gd, int gpNum,
                                                   Scalar dt, const Scalar* lastH, Scalar eps);

void calFrictionGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient);

void calFrictionHessian(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr);

void buildFrictionSets(std::unique_ptr<GeometryManager>& instance);

}; // __GIPCFRICTION__

