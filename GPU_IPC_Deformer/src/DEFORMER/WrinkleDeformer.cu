// WrinkleDeformer.cu


#include "WrinkleDeformer.cuh"


namespace __DEFORMER__ {



__global__ void xpbdConstraintKernel(
    Scalar3*      d_curr_pos, 
    ConstraintGPU* d_constraints,
    Scalar*       d_stretch_stiffness,
    Scalar*       d_compress_stiffness,
    Scalar*       d_lambda,     // lagrange_multipliers
    Scalar*       d_inv_mass,
    Scalar3*      d_dP,         // 临时存储的位移累积
    Scalar*       d_dPw,        // 临时存储的权重累积
    int          number,
    Scalar        time_step
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    // 读取约束
    ConstraintGPU c = d_constraints[idx];
    int   v1         = c.v1;
    int   v2         = c.v2;
    Scalar restLength = c.restLength;

    // 读取位置
    Scalar3 p1 = d_curr_pos[v1];
    Scalar3 p2 = d_curr_pos[v2];
    Scalar3 dir = __MATHUTILS__::__vec3_minus(p2, p1);

    Scalar d = __MATHUTILS__::__vec3_norm(dir);
    if (d < 1e-12f) {
        return;
    }
    dir = __MATHUTILS__::__s_vec3_multiply(dir, 1.0 / d);

    // 根据拉伸 or 压缩，选择不同 stiffness
    Scalar stiffness = (d < restLength) ? d_compress_stiffness[idx] : d_stretch_stiffness[idx];
    if (stiffness < 1e-12f) {
        return;
    }

    Scalar alpha = (1.0f / stiffness) / (time_step * time_step);
    Scalar C     = (d - restLength);

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

    // 写回新的 lambda
    d_lambda[idx] = newLambda;

    // 计算顶点的移动
    Scalar3 dp1 = __MATHUTILS__::__s_vec3_multiply(dir, -delta_lambda * w1);
    Scalar3 dp2 = __MATHUTILS__::__s_vec3_multiply(dir, delta_lambda * w2);

    // 原子加到 d_dP, d_dPw
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
    int     number)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    Scalar w = d_dPw[idx];
    if (w < 1e-12f) {
        return;
    }

    Scalar invw = 1.0f / w;
    Scalar3 shift = d_dP[idx];
    shift = __MATHUTILS__::__s_vec3_multiply(shift, invw);
    d_curr_pos[idx] = __MATHUTILS__::__vec3_add(d_curr_pos[idx], shift);

    // 把 d_dP 和 d_dPw 清零，方便下次迭代
    d_dP[idx]  = make_Scalar3(0.0f, 0.0f, 0.0f);
    d_dPw[idx] = 0.0f;
}

__global__ void deltaMushKernel(
    const Scalar3* d_positions,
    Scalar3*       d_newPositions,
    const int*    d_adjacency,
    const Scalar*  d_weights,
    const int*    d_adjStart,   // 每个顶点的邻接起点
    const int*    d_adjCount,   // 每个顶点邻接数量
    Scalar         step_size,
    int           nv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nv) return;

    int start = d_adjStart[i];
    int count = d_adjCount[i];
    Scalar3 sum_vec = make_Scalar3(0.f,0.f,0.f);

    for(int k = 0; k < count; k++){
        int nbr = d_adjacency[start + k];
        Scalar w = d_weights[start + k];
        // 加权累加
        Scalar3 pnbr = d_positions[nbr];
        sum_vec = __MATHUTILS__::__vec3_add(sum_vec, __MATHUTILS__::__s_vec3_multiply(pnbr, w));
    }

    Scalar3 pi = d_positions[i];
    Scalar3 diff = __MATHUTILS__::__vec3_minus(sum_vec, pi); 
    // 移动 step_size
    Scalar3 newp = __MATHUTILS__::__vec3_add(pi, __MATHUTILS__::__s_vec3_multiply(diff, step_size));

    d_newPositions[i] = newp;
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
    // 1) 计算约束并行
    int blockSize = 256;
    int gridC = (nc + blockSize - 1) / blockSize;
    xpbdConstraintKernel<<<gridC, blockSize>>>( 
        d_curr_pos, d_constraints,
        d_stretch_stiffness, d_compress_stiffness,
        d_lambda, d_inv_mass,
        d_dP, d_dPw,
        nc, time_step
    );

    // 2) 更新位置
    int gridV = (nv + blockSize - 1) / blockSize;
    xpbdUpdatePositionKernel<<<gridV, blockSize>>>( 
        d_curr_pos, d_dP, d_dPw, nv
    );
}


void deltaMushSmoothGPU(
    Scalar3* d_positions,
    Scalar3* d_newPositions,
    int*    d_adjacency,
    Scalar*  d_weights,
    int*    d_adjStart,
    int*    d_adjCount,
    int     nv,
    Scalar   step_size,
    int     iterations)
{
    int blockSize = 256;
    int grid = (nv + blockSize - 1) / blockSize;

    for(int it = 0; it < iterations; ++it) {
        deltaMushKernel<<<grid, blockSize>>>(
            d_positions, d_newPositions,
            d_adjacency, d_weights,
            d_adjStart, d_adjCount,
            step_size, nv
        );
        cudaDeviceSynchronize();

        // 交换 d_positions, d_newPositions
        Scalar3* tmp = d_positions;
        d_positions = d_newPositions;
        d_newPositions = tmp;
    }
}


};