#pragma once
#include "MathUtils.cuh"
#include <cuda_runtime.h>

namespace __DEFORMER__ {

/**
 * XPBD 迭代：在每个约束上做一次解算，再更新顶点位置。
 * @param d_curr_pos:        顶点当前坐标
 * @param d_constraints:     每个约束 (v1, v2, restLength)
 * @param stretch_stiff:     全局拉伸刚度
 * @param compress_stiff:    全局压缩刚度
 * @param d_lambda:          XPBD 的拉格朗日乘子数组
 * @param d_masses:          每个顶点的质量 (数组)
 * @param d_dP:              写回的累加位移
 * @param d_dPw:             写回的累加权重
 * @param nc:                约束数
 * @param nv:                顶点数
 * @param dt:                时间步
 */
void xpbdIterationLoopCUDA(
    Scalar3* d_curr_pos,
    Scalar3* d_constraints,
    Scalar    stretch_stiff,
    Scalar    compress_stiff,
    Scalar*  d_lambda,
    Scalar*  d_masses,
    Scalar3* d_dP,
    Scalar*  d_dPw,
    int      nc,
    int      nv,
    Scalar   dt
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
    double     stepSize,      // 改为 double
    int        iterations
);

} // namespace __DEFORMER__
