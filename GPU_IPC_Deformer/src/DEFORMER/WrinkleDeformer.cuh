#pragma once

#include "MathUtils.cuh"  // 假设里头定义了 Scalar, Scalar3
#include <cuda_runtime.h>

namespace __DEFORMER__ {

/// GPU 端约束
struct ConstraintGPU {
    int    v1;
    int    v2;
    Scalar restLength;
    Scalar ctype; 
};

// ======== 原有声明：计算 invMass, XPBD, DeltaMushSmoothGPU... ========

void launchComputeInvMassKernel(const Scalar* d_masses, Scalar* d_inv_mass, int nv);

void xpbdIterationLoopCUDA(
    Scalar3* d_curr_pos,
    ConstraintGPU* d_constraints,
    Scalar* d_stretch_stiffness,
    Scalar* d_compress_stiffness,
    Scalar* d_lambda,
    Scalar* d_inv_mass,
    Scalar3* d_dP,
    Scalar*  d_dPw,
    int nc,
    int nv,
    Scalar time_step
);

void deltaMushSmoothGPU(
    Scalar3* d_positions,
    Scalar3* d_newPositions,
    int*    d_adjacency,
    Scalar* d_weights,
    int*    d_adjStart,
    int*    d_adjCount,
    int     nv,
    Scalar  step_size,
    int     iterations
);

// ======== 新增封装函数: computeDeltaMushWeights(...) ========
// 它在内部做两件事：
//   1) 调用 kernel 计算 dist 并原子加到 sumW
//   2) 调用 kernel 做归一化
void computeDeltaMushWeights(
    const Scalar3* d_positions,
    const int*     d_adjacencyOwner,
    const int*     d_adjacency,
    Scalar*        d_rawWeights,
    Scalar*        d_weights,
    Scalar*        d_sumW,
    int            totalEdges,
    int            nv
);

} // namespace __DEFORMER__
