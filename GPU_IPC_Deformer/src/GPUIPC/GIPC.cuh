

#pragma once

#include "BlockHessian.cuh"
#include "LBVH.cuh"
#include "PCGSolver.cuh"

#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "MathUtils.cuh"

namespace __GPUIPC__ {

__device__ Scalar __calBarrierSelfConsDis(const Scalar3* _vertexes, const int4 MMCVIDI);

__global__ void _getBarrierEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                               const Scalar3* rest_vertexes, int4* _collisionPair,
                                               Scalar _Kappa, Scalar _dHat, int cpNum);

__global__ void _computeGroundEnergy_Reduction(Scalar* squeue, const Scalar3* vertexes,
                                               const Scalar* g_offset, const Scalar3* g_normal,
                                               const uint32_t* _environment_collisionPair,
                                               Scalar dHat, Scalar Kappa, int number);

void calKineticGradient(Scalar3* _vertexes, Scalar3* _xTilta, Scalar3* _gradient, Scalar* _masses,
                        int numbers);

void calBarrierGradientAndHessian(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr,
                                  Scalar3* _gradient, Scalar mKappa);

Scalar ground_largestFeasibleStepSize(std::unique_ptr<GeometryManager>& instance, Scalar slackness,
                                      Scalar* mqueue);

Scalar self_largestFeasibleStepSize(std::unique_ptr<GeometryManager>& instance, Scalar slackness,
                                    Scalar* mqueue, int numbers);

Scalar cfl_largestSpeed(std::unique_ptr<GeometryManager>& instance, Scalar* mqueue);

bool isIntersected(std::unique_ptr<GeometryManager>& instance,
                   std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr);

void computeCloseGroundVal(std::unique_ptr<GeometryManager>& instance);

void computeSelfCloseVal(std::unique_ptr<GeometryManager>& instance);

bool checkCloseGroundVal(std::unique_ptr<GeometryManager>& instance);

bool checkSelfCloseVal(std::unique_ptr<GeometryManager>& instance);

void initKappa(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr,
               std::unique_ptr<PCGSolver>& PCG_ptr);

void suggestKappa(std::unique_ptr<GeometryManager>& instance, Scalar& kappa);

void upperBoundKappa(std::unique_ptr<GeometryManager>& instance, Scalar& kappa);


};  // namespace __GPUIPC__
