// WrinkleDeformer.cuh

#pragma once

#include "UTILS/GeometryManager.hpp"
#include "UTILS/CUDAUtils.hpp"
#include "MathUtils.cuh"

namespace __DEFORMER__ {

struct ConstraintGPU {
    int    v1;
    int    v2;
    Scalar  restLength;
    Scalar  ctype; // 0/1 之类，这里可能代表弯曲还是拉伸等
};

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
    Scalar time_step);

void deltaMushSmoothGPU(
    Scalar3* d_positions,
    Scalar3* d_newPositions,
    int*    d_adjacency,
    Scalar*  d_weights,
    int*    d_adjStart,
    int*    d_adjCount,
    int     nv,
    Scalar   step_size,
    int     iterations);



};