// WrinkleDeformer.cu

#include "WrinkleDeformer.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

namespace __DEFORMER__ {

// -------------------- Cooperative 版本 XPBD --------------------

__global__ void xpbdIterationCooperativeKernel(
    Scalar3* __restrict__ d_curr_pos,
    const Scalar3* __restrict__ d_constraints,
    Scalar  stretch_stiff,
    Scalar  compress_stiff,
    Scalar* __restrict__ d_lambda,
    const Scalar* __restrict__ d_masses,
    Scalar3* __restrict__ d_dP,
    Scalar*  __restrict__ d_dPw,
    int nc,
    int nv,
    Scalar dt,
    int xpbd_iters
) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for(int iter = 0; iter < xpbd_iters; ++iter) {

        // ---- 1) 约束求解阶段 ----
        if(tid < nc) {
            Scalar3 c = d_constraints[tid];
            int v1 = static_cast<int>(c.x);
            int v2 = static_cast<int>(c.y);
            Scalar rest_len = c.z;

            Scalar3 p1 = d_curr_pos[v1];
            Scalar3 p2 = d_curr_pos[v2];

            Scalar3 dir = {
                p2.x - p1.x,
                p2.y - p1.y,
                p2.z - p1.z
            };
            Scalar lengthSq = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
            if (lengthSq > 1e-24) {
                Scalar length = sqrt(lengthSq);
                dir.x /= length;
                dir.y /= length;
                dir.z /= length;

                Scalar stiffness = (length < rest_len) ? compress_stiff : stretch_stiff;
                if (stiffness > 1e-12) {
                    Scalar alpha = (1.0 / stiffness) / (dt * dt);
                    Scalar C = length - rest_len;

                    Scalar mass_v1 = d_masses[v1];
                    Scalar mass_v2 = d_masses[v2];
                    Scalar w1 = (mass_v1 == 0.0) ? 0.0 : (1.0 / mass_v1);
                    Scalar w2 = (mass_v2 == 0.0) ? 0.0 : (1.0 / mass_v2);

                    Scalar denom = w1 + w2 + alpha;
                    if (fabs(denom) < 1e-12) denom = 1e-12;

                    Scalar old_lambda  = d_lambda[tid];
                    Scalar delta_lambda = (-C - alpha * old_lambda) / denom;
                    Scalar new_lambda   = old_lambda + delta_lambda;
                    d_lambda[tid] = new_lambda;

                    Scalar3 dp1 = {
                        -delta_lambda * w1 * dir.x,
                        -delta_lambda * w1 * dir.y,
                        -delta_lambda * w1 * dir.z
                    };
                    Scalar3 dp2 = {
                         delta_lambda * w2 * dir.x,
                         delta_lambda * w2 * dir.y,
                         delta_lambda * w2 * dir.z
                    };

                    atomicAdd(&d_dP[v1].x, dp1.x);
                    atomicAdd(&d_dP[v1].y, dp1.y);
                    atomicAdd(&d_dP[v1].z, dp1.z);

                    atomicAdd(&d_dP[v2].x, dp2.x);
                    atomicAdd(&d_dP[v2].y, dp2.y);
                    atomicAdd(&d_dP[v2].z, dp2.z);

                    atomicAdd(&d_dPw[v1], 1.0);
                    atomicAdd(&d_dPw[v2], 1.0);
                }
            }
        }

        // 全网格同步，以确保所有约束都更新完 d_dP/d_dPw
        grid.sync();

        // ---- 2) 更新顶点位置阶段 ----
        if(tid < nv) {
            Scalar w = d_dPw[tid];
            if (w > 1e-12) {
                Scalar invw = 1.0 / w;
                d_curr_pos[tid].x += d_dP[tid].x * invw;
                d_curr_pos[tid].y += d_dP[tid].y * invw;
                d_curr_pos[tid].z += d_dP[tid].z * invw;
            }
            d_dP[tid]  = {0.0, 0.0, 0.0};
            d_dPw[tid] = 0.0;
        }

        // 再次同步，确保所有顶点都更新完
        grid.sync();
    }
}

// 封装 cooperative 版本的 XPBD 接口
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
) {
    // 根据约束数/顶点数找一个覆盖范围更大的
    int maxN = (nc > nv) ? nc : nv;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int gridSize  = (maxN + threadNum - 1) / threadNum;

    // 传递参数给 cooperative kernel
    void* kernelArgs[] = {
        (void*)&d_curr_pos,
        (void*)&d_constraints,
        (void*)&stretch_stiff,
        (void*)&compress_stiff,
        (void*)&d_lambda,
        (void*)&d_masses,
        (void*)&d_dP,
        (void*)&d_dPw,
        (void*)&nc,
        (void*)&nv,
        (void*)&dt,
        (void*)&xpbd_iters
    };

    // 启动 cooperative kernel
    cudaLaunchCooperativeKernel(
        (void*)xpbdIterationCooperativeKernel,
        dim3(gridSize), dim3(threadNum),
        kernelArgs
    );
    cudaDeviceSynchronize();
}

// -------------------- deltaMush 综合函数 --------------------

// 计算原始权重核
__global__ void computeRawWeightsKernel(
    const Scalar3* d_positions,
    const int*     d_adjOwners,
    const int*     d_adjacency,
    Scalar*        d_rawWeights,
    Scalar*        d_sumW,
    int            totalEdges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalEdges) return;
    int i = d_adjOwners[idx];
    int nb = d_adjacency[idx];

    Scalar3 pi = d_positions[i];
    Scalar3 pn = d_positions[nb];
    Scalar3 diff = {pn.x - pi.x, pn.y - pi.y, pn.z - pi.z};
    double dist = sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
    double w = (dist > 1e-12) ? (1.0 / dist) : 0.0;

    d_rawWeights[idx] = w;
    atomicAdd(&d_sumW[i], w);
}

// 归一化权重核
__global__ void normalizeWeightsKernel(
    const Scalar* d_rawWeights,
    Scalar*       d_weights,
    const Scalar* d_sumW,
    const int*    d_adjOwners,
    int           totalEdges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalEdges) return;
    int i = d_adjOwners[idx];
    double denom = d_sumW[i];
    double w = d_rawWeights[idx];
    d_weights[idx] = (denom > 1e-12) ? (w / denom) : 0.0;
}

// deltaMush 平滑核
__global__ void deltaMushKernel(
    const Scalar3* d_positions,
    Scalar3*       d_newPositions,
    const int*     d_adjacency,
    const Scalar*  d_weights,
    const int*     d_adjStart,
    const int*     d_adjCount,
    double         stepSize,
    int            nv
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv) return;

    int start = d_adjStart[i];
    int count = d_adjCount[i];
    Scalar3 sum_vec = {0.0, 0.0, 0.0};
    for (int k = 0; k < count; k++) {
        int idx = start + k;
        int nb = d_adjacency[idx];
        Scalar w = d_weights[idx];
        sum_vec.x += d_positions[nb].x * w;
        sum_vec.y += d_positions[nb].y * w;
        sum_vec.z += d_positions[nb].z * w;
    }
    Scalar3 pi = d_positions[i];
    Scalar3 diff = {
        sum_vec.x - pi.x,
        sum_vec.y - pi.y,
        sum_vec.z - pi.z
    };
    d_newPositions[i] = {
        pi.x + diff.x * stepSize,
        pi.y + diff.y * stepSize,
        pi.z + diff.z * stepSize
    };
}

// deltaMush 综合函数
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
    double     stepSize,
    int        iterations
) {
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int gridEdges = (totalEdges + threadNum - 1) / threadNum;
    int gridVerts = (nv + threadNum - 1) / threadNum;

    // 计算原始权重
    computeRawWeightsKernel<<<gridEdges, threadNum>>>(
        d_positions, d_adjacencyOwners, d_adjacency,
        d_rawWeights, d_sumW,
        totalEdges
    );
    cudaDeviceSynchronize();

    // 归一化权重
    normalizeWeightsKernel<<<gridEdges, threadNum>>>(
        d_rawWeights, d_weights, d_sumW, d_adjacencyOwners, totalEdges
    );
    cudaDeviceSynchronize();

    // 进行多次平滑迭代
    for (int it = 0; it < iterations; ++it) {
        deltaMushKernel<<<gridVerts, threadNum>>>(
            d_positions, d_newPositions,
            d_adjacency, d_weights,
            d_adjStart, d_adjCount,
            stepSize, nv
        );
        cudaDeviceSynchronize();
        // 交换位置指针
        Scalar3* tmp       = d_positions;
        d_positions        = d_newPositions;
        d_newPositions     = tmp;
    }
}

} // namespace __DEFORMER__
