// WrinkleDeformer.cuh

#pragma once

#include "MathUtils.cuh"
#include "CUDAUtils.hpp"

namespace __DEFORMER__ {

/**
 * XPBD 迭代：使用 cooperative groups，
 */
void xpbdIterationAllInOneGPU_Cooperative(
    Scalar3* d_curr_pos,
    Scalar3* d_constraints,
    Scalar   stretch_stiff,
    Scalar   compress_stiff,
    Scalar*  d_lambda,
    Scalar*  d_masses,
    Scalar3* d_dP,
    Scalar*  d_dPw,
    int      nc,
    int      nv,
    Scalar   dt,
    int      xpbd_iters
);

/**
 * deltaMush 平滑 + 权重计算的综合函数
 */
void deltaMushAllInOneGPU(
    Scalar3* d_positions,
    Scalar3* d_newPositions,
    const int* d_adjacencyOwners,
    const int* d_adjacency,
    const int* d_adjStart,
    const int* d_adjCount,
    Scalar*    d_rawWeights,
    Scalar*    d_weights,
    Scalar*    d_sumW,
    int        totalEdges,
    int        nv,
    Scalar     stepSize,
    int        iterations
);

} // namespace __DEFORMER__
