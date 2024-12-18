
#pragma once

#pragma once

#include "BlockHessian.cuh"
#include "LBVH.cuh"
#include "PCGSolver.cuh"

#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"

namespace __INTEGRATOR__ {

void computeXTilta(std::unique_ptr<GeometryManager>& instance);

void changeBoundarytoSIMPoint(std::unique_ptr<GeometryManager>& instance);

void IPC_Solver(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr,
                std::unique_ptr<PCGSolver>& PCG_ptr,
                std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr);

};  // namespace __INTEGRATOR__
