#include "WrinkleDeformer.cuh"
#include <cuda_runtime.h>
#include <cmath> // for fabsf

namespace __DEFORMER__ {

// ========== 1) computeInvMassKernel ==========

__global__ void computeInvMassKernel(const Scalar* d_masses, Scalar* d_inv_mass, int nv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nv) return;

    Scalar mass = d_masses[idx];
    d_inv_mass[idx] = (mass == 0.0f) ? 0.0f : (1.0f / mass);
}

void launchComputeInvMassKernel(const Scalar* d_masses, Scalar* d_inv_mass, int nv)
{
    int blockSize = 256;
    int grid = (nv + blockSize - 1) / blockSize;
    computeInvMassKernel<<<grid, blockSize>>>(d_masses, d_inv_mass, nv);
    cudaDeviceSynchronize();
}

// ========== 2) XPBD 核心 ==========

__global__ void xpbdConstraintKernel(
    Scalar3*       d_curr_pos, 
    ConstraintGPU* d_constraints,
    Scalar*        d_stretch_stiffness,
    Scalar*        d_compress_stiffness,
    Scalar*        d_lambda,
    Scalar*        d_inv_mass,
    Scalar3*       d_dP, 
    Scalar*        d_dPw,
    int            nc,
    Scalar         time_step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nc) return;

    // 读取约束
    ConstraintGPU c = d_constraints[idx];
    int v1 = c.v1;
    int v2 = c.v2;
    Scalar restLength = c.restLength;

    // 当前位置
    Scalar3 p1 = d_curr_pos[v1];
    Scalar3 p2 = d_curr_pos[v2];

    // dir = p2 - p1
    Scalar3 dir;
    dir.x = p2.x - p1.x;
    dir.y = p2.y - p1.y;
    dir.z = p2.z - p1.z;

    // 长度
    Scalar d = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    if (d < 1e-12f) {
        return;
    }
    // 归一化
    dir.x /= d;
    dir.y /= d;
    dir.z /= d;

    // 选 stiffness
    Scalar stiffness = (d < restLength)
                        ? d_compress_stiffness[idx]
                        : d_stretch_stiffness[idx];
    if (stiffness < 1e-12f) {
        return;
    }

    Scalar alpha = (1.0f / stiffness) / (time_step * time_step);
    Scalar C = (d - restLength);

    Scalar w1 = d_inv_mass[v1];
    Scalar w2 = d_inv_mass[v2];
    Scalar wsum = w1 + w2;
    Scalar denom = wsum + alpha;
    if (fabsf(denom) < 1e-12f) {
        denom = 1e-12f;
    }

    // Δλ
    Scalar oldLambda = d_lambda[idx];
    Scalar delta_lambda = (-C - alpha * oldLambda) / denom;
    Scalar newLambda   = oldLambda + delta_lambda;
    d_lambda[idx] = newLambda;

    // dp1 / dp2
    Scalar3 dp1, dp2;
    dp1.x = -delta_lambda * w1 * dir.x;
    dp1.y = -delta_lambda * w1 * dir.y;
    dp1.z = -delta_lambda * w1 * dir.z;
    dp2.x =  delta_lambda * w2 * dir.x;
    dp2.y =  delta_lambda * w2 * dir.y;
    dp2.z =  delta_lambda * w2 * dir.z;

    // 原子加
    atomicAdd(&d_dP[v1].x, dp1.x);
    atomicAdd(&d_dP[v1].y, dp1.y);
    atomicAdd(&d_dP[v1].z, dp1.z);

    atomicAdd(&d_dP[v2].x, dp2.x);
    atomicAdd(&d_dP[v2].y, dp2.y);
    atomicAdd(&d_dP[v2].z, dp2.z);

    atomicAdd(&d_dPw[v1], 1.0f);
    atomicAdd(&d_dPw[v2], 1.0f);
}

__global__ void xpbdUpdatePositionKernel(
    Scalar3* d_curr_pos,
    Scalar3* d_dP,
    Scalar*  d_dPw,
    int      nv)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nv) return;

    Scalar w = d_dPw[idx];
    if (w > 1e-12f) {
        Scalar invw = 1.0f / w;
        // shift = d_dP[idx] * invw
        d_curr_pos[idx].x += d_dP[idx].x * invw;
        d_curr_pos[idx].y += d_dP[idx].y * invw;
        d_curr_pos[idx].z += d_dP[idx].z * invw;
    }
    // 清零
    d_dP[idx].x = 0.f; 
    d_dP[idx].y = 0.f;
    d_dP[idx].z = 0.f;
    d_dPw[idx]  = 0.f;
}

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
    Scalar time_step)
{
    int blockSize = 256;

    // 约束 kernel
    {
        int gridC = (nc + blockSize - 1) / blockSize;
        xpbdConstraintKernel<<<gridC, blockSize>>>(
            d_curr_pos, d_constraints,
            d_stretch_stiffness, d_compress_stiffness,
            d_lambda, d_inv_mass,
            d_dP, d_dPw,
            nc, time_step
        );
    }

    // 更新位置 kernel
    {
        int gridV = (nv + blockSize - 1) / blockSize;
        xpbdUpdatePositionKernel<<<gridV, blockSize>>>(
            d_curr_pos, d_dP, d_dPw, nv
        );
    }
    cudaDeviceSynchronize();
}

// ========== 3) deltaMushSmoothGPU ==========

static __global__ void computeDistWeightsKernel(
    const Scalar3* d_positions,
    const int*     d_adjacencyOwner,
    const int*     d_adjacency,
    Scalar*        d_rawWeights,
    Scalar*        d_sumW,
    int            totalEdges
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalEdges) return;

    int  i   = d_adjacencyOwner[idx]; 
    int  nb  = d_adjacency[idx];      

    Scalar3 pi = d_positions[i];
    Scalar3 pn = d_positions[nb];

    Scalar3 diff;
    diff.x = pn.x - pi.x;
    diff.y = pn.y - pi.y;
    diff.z = pn.z - pi.z;

    float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
    float w    = (dist > 1e-12f) ? (1.f/dist) : 0.f;

    d_rawWeights[idx] = w;
    atomicAdd(&d_sumW[i], w);
}


// -------------- 归一化 --------------
static __global__ void normalizeWeightsKernel(
    const Scalar* d_rawWeights,
    Scalar*       d_weights,
    const Scalar* d_sumW,
    const int*    d_adjacencyOwner,
    int           totalEdges
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalEdges) return;

    int i = d_adjacencyOwner[idx];
    float denom = d_sumW[i];
    float wraw  = d_rawWeights[idx];
    float wnorm = (denom > 1e-12f) ? (wraw / denom) : 0.f;
    d_weights[idx] = wnorm;
}

void computeDeltaMushWeights(
    const Scalar3* d_positions,
    const int*     d_adjacencyOwner,
    const int*     d_adjacency,
    Scalar*        d_rawWeights,
    Scalar*        d_weights,
    Scalar*        d_sumW,
    int            totalEdges,
    int            nv
)
{
    // 1) 启动 kernel: computeDistWeightsKernel
    {
        int blockSize = 256;
        int grid = (totalEdges + blockSize - 1) / blockSize;
        computeDistWeightsKernel<<<grid, blockSize>>>(
            d_positions,
            d_adjacencyOwner,
            d_adjacency,
            d_rawWeights,
            d_sumW,
            totalEdges
        );
        cudaDeviceSynchronize();
    }

    // 2) 启动 kernel: normalizeWeightsKernel
    {
        int blockSize = 256;
        int grid = (totalEdges + blockSize - 1) / blockSize;
        normalizeWeightsKernel<<<grid, blockSize>>>(
            d_rawWeights,
            d_weights,
            d_sumW,
            d_adjacencyOwner,
            totalEdges
        );
        cudaDeviceSynchronize();
    }
}


__global__ void deltaMushKernel(
    const Scalar3* d_positions,
    Scalar3*       d_newPositions,
    const int*     d_adjacency,
    const Scalar*  d_weights,
    const int*     d_adjStart,
    const int*     d_adjCount,
    Scalar         step_size,
    int            nv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv) return;

    int start = d_adjStart[i];
    int count = d_adjCount[i];

    // 累加邻居
    Scalar3 sum_vec; 
    sum_vec.x = 0.f; sum_vec.y = 0.f; sum_vec.z = 0.f;

    for(int k = 0; k < count; k++){
        int idx = start + k;
        int nbr = d_adjacency[idx];
        Scalar w = d_weights[idx];
        sum_vec.x += d_positions[nbr].x * w;
        sum_vec.y += d_positions[nbr].y * w;
        sum_vec.z += d_positions[nbr].z * w;
    }

    Scalar3 pi = d_positions[i];
    Scalar3 diff;
    diff.x = sum_vec.x - pi.x;
    diff.y = sum_vec.y - pi.y;
    diff.z = sum_vec.z - pi.z;

    // 新位置 = 旧位置 + step_size * diff
    d_newPositions[i].x = pi.x + diff.x * step_size;
    d_newPositions[i].y = pi.y + diff.y * step_size;
    d_newPositions[i].z = pi.z + diff.z * step_size;
}

void deltaMushSmoothGPU(
    Scalar3* d_positions,
    Scalar3* d_newPositions,
    int*    d_adjacency,
    Scalar* d_weights,
    int*    d_adjStart,
    int*    d_adjCount,
    int     nv,
    Scalar  step_size,
    int     iterations)
{
    int blockSize = 256;
    int grid = (nv + blockSize - 1) / blockSize;

    for(int it = 0; it < iterations; ++it) {
        // kernel
        deltaMushKernel<<<grid, blockSize>>>(
            d_positions, d_newPositions,
            d_adjacency, d_weights,
            d_adjStart, d_adjCount,
            step_size, nv
        );
        cudaDeviceSynchronize();

        // 交换指针
        Scalar3* tmp = d_positions;
        d_positions  = d_newPositions;
        d_newPositions = tmp;
    }
}

} // namespace __DEFORMER__
