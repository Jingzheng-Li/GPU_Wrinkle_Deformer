
#include "PCGSolver.cuh"

__global__ void __PCG_AXALL_P(const __MATHUTILS__::Matrix12x12S* Hessians12,
                              const __MATHUTILS__::Matrix9x9S* Hessians9,
                              const __MATHUTILS__::Matrix6x6S* Hessians6,
                              const __MATHUTILS__::Matrix3x3S* Hessians3, 
                              const uint4* D4Index, const uint3* D3Index, 
                              const uint2* D2Index, const uint32_t* D1Index,
                              __MATHUTILS__::Matrix3x3S* P, 
                              int numbers4, int numbers3,
                              int numbers2, int numbers1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers4 + numbers3 + numbers2 + numbers1) return;

    if (idx < numbers4) {
        int Hid = idx / 12;
        int qid = idx % 12;
        int mid = (qid / 3) * 3;
        int tid = qid % 3;
        Scalar Hval;
        Hval = Hessians12[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians12[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D4Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);

    } else if (numbers4 <= idx && idx < numbers3 + numbers4) {
        idx -= numbers4;
        int Hid = idx / 9;
        int qid = idx % 9;
        int mid = (qid / 3) * 3;
        int tid = qid % 3;
        Scalar Hval;
        Hval = Hessians9[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians9[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D3Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);

    } else if (numbers3 + numbers4 <= idx && idx < numbers3 + numbers4 + numbers2) {
        idx -= numbers3 + numbers4;
        int Hid = idx / 6;
        int qid = idx % 6;
        int mid = (qid / 3) * 3;
        int tid = qid % 3;
        Scalar Hval;
        Hval = Hessians6[Hid].m[mid][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[0][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 1][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[1][qid % 3]), Hval);
        Hval = Hessians6[Hid].m[mid + 2][mid + tid];
        atomicAdd(&(P[*(&(D2Index[Hid].x) + qid / 3)].m[2][qid % 3]), Hval);

    } else {
        idx -= numbers2 + numbers3 + numbers4;
        int Hid = idx / 3;
        int qid = idx % 3;
        atomicAdd(&(P[D1Index[Hid]].m[0][qid]), Hessians3[Hid].m[0][qid]);
        atomicAdd(&(P[D1Index[Hid]].m[1][qid]), Hessians3[Hid].m[1][qid]);
        atomicAdd(&(P[D1Index[Hid]].m[2][qid]), Hessians3[Hid].m[2][qid]);
    }
}


__global__ void PCG_add_Reduction_delta0(Scalar* squeue, const __MATHUTILS__::Matrix3x3S* P,
                                         const Scalar3* b,
                                         const __MATHUTILS__::Matrix3x3S* constraint, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    // delta0 = rT @ A @ r
    Scalar3 t_b = b[idx];
    __MATHUTILS__::Matrix3x3S t_constraint = constraint[idx];
    Scalar3 filter_b = __MATHUTILS__::__M3x3_v3_multiply(t_constraint, t_b);
    Scalar temp = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__v3_M3x3_multiply(filter_b, P[idx]), filter_b);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);

}


__global__ void PCG_add_Reduction_deltaN0(Scalar* squeue, const __MATHUTILS__::Matrix3x3S* P,
                                          const Scalar3* b, Scalar3* r, Scalar3* c,
                                          const __MATHUTILS__::Matrix3x3S* constraint,
                                          int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    // deltaN = rT @ c
    Scalar3 t_b = b[idx];
    __MATHUTILS__::Matrix3x3S t_constraint = constraint[idx];
    Scalar3 t_r = __MATHUTILS__::__M3x3_v3_multiply(t_constraint, __MATHUTILS__::__vec3_minus(t_b, r[idx]));
    Scalar3 t_c = __MATHUTILS__::__M3x3_v3_multiply(P[idx], t_r);
    t_c = __MATHUTILS__::__M3x3_v3_multiply(t_constraint, t_c);
    r[idx] = t_r;
    c[idx] = t_c;
    Scalar temp = __MATHUTILS__::__vec3_dot(t_r, t_c);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);
}
























__global__ void PCG_add_Reduction_deltaN(Scalar* squeue, Scalar3* dx, const Scalar3* c, Scalar3* r,
                                         const Scalar3* q, const __MATHUTILS__::Matrix3x3S* P,
                                         Scalar3* s, Scalar alpha, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    // deltaN = rT @ s
    Scalar3 t_c = c[idx];
    Scalar3 t_dx = dx[idx];
    Scalar3 t_r = r[idx];
    Scalar3 t_q = q[idx];
    Scalar3 t_s = s[idx];
    dx[idx] = __MATHUTILS__::__vec3_add(t_dx, __MATHUTILS__::__s_vec3_multiply(t_c, alpha));
    t_r = __MATHUTILS__::__vec3_add(t_r, __MATHUTILS__::__s_vec3_multiply(t_q, -alpha));
    r[idx] = t_r;
    t_s = __MATHUTILS__::__M3x3_v3_multiply(P[idx], t_r);
    s[idx] = t_s;
    Scalar temp = __MATHUTILS__::__vec3_dot(t_r, t_s);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);

}

__global__ void PCG_add_Reduction_tempSum(Scalar* squeue, const Scalar3* c, Scalar3* q,
                                          const __MATHUTILS__::Matrix3x3S* constraint,
                                          int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    // tempSum = qT @ c
    Scalar3 t_c = c[idx];
    Scalar3 t_q = q[idx];
    __MATHUTILS__::Matrix3x3S t_constraint = constraint[idx];
    t_q = __MATHUTILS__::__M3x3_v3_multiply(t_constraint, t_q);
    q[idx] = t_q;
    Scalar temp = __MATHUTILS__::__vec3_dot(t_q, t_c);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);

}

__global__ void PCG_add_Reduction_force(Scalar* squeue, const Scalar3* b, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    // calculate ||b|| norm as init residual
    Scalar3 t_b = b[idx];
    Scalar temp = __MATHUTILS__::__vec3_norm(t_b);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);
}

__global__ void __PCG_FinalStep_UpdateC(const __MATHUTILS__::Matrix3x3S* constraints,
                                        const Scalar3* s, Scalar3* c, Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    // update search diretion c
    Scalar3 tempc = __MATHUTILS__::__vec3_add(s[idx], __MATHUTILS__::__s_vec3_multiply(c[idx], rate));
    c[idx] = __MATHUTILS__::__M3x3_v3_multiply(constraints[idx], tempc);
}
















































__global__ void __PCG_Solve_AXALL_b2(
    const __MATHUTILS__::Matrix12x12S* Hessians12, const __MATHUTILS__::Matrix9x9S* Hessians9,
    const __MATHUTILS__::Matrix6x6S* Hessians6, const __MATHUTILS__::Matrix3x3S* Hessians3,
    const uint4* D4Index, const uint3* D3Index, const uint2* D2Index, const uint32_t* D1Index,
    const Scalar3* c, Scalar3* q, 
    int numbers4, int numbers3, int numbers2, int numbers1,
    int offset4, int offset3, int offset2) {

    // calculate A = M + dt^2 * K, q = A @ c
    if (blockIdx.x < offset4) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numbers4) return;
        __shared__ int offset;
        int Hid = idx / 144;
        int MRid = (idx % 144) / 12;
        int MCid = (idx % 144) % 12;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 12;

        Scalar rdata =
            Hessians12[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (12 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 12) / 12;
        int landidx = (threadIdx.x - offset) % 12;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 12; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } 
    




















    
    
    else if (blockIdx.x >= offset4 && blockIdx.x < offset4 + offset3) {
        int idx = (blockIdx.x - offset4) * blockDim.x + threadIdx.x;
        if (idx >= numbers3) return;
        __shared__ int offset;
        int Hid = idx / 81;
        int MRid = (idx % 81) / 9;
        int MCid = (idx % 81) % 9;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 9;

        Scalar rdata = Hessians9[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (9 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 9) / 9;
        int landidx = (threadIdx.x - offset) % 9;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);  // a bit-mask
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 9; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } 
    
    else if (blockIdx.x >= offset4 + offset3 && blockIdx.x < offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3) * blockDim.x + threadIdx.x;
        if (idx >= numbers2) return;
        __shared__ int offset;
        int Hid = idx / 36;
        int MRid = (idx % 36) / 6;
        int MCid = (idx % 36) % 6;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 6;

        Scalar rdata =
            Hessians6[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (6 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 6) / 6;
        int landidx = (threadIdx.x - offset) % 6;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 6; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } 
    
    else if (blockIdx.x >= offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3 - offset2) * blockDim.x + threadIdx.x;
        if (idx >= numbers1) return;
        __MATHUTILS__::Matrix3x3S H = Hessians3[idx];
        Scalar3 tempC, tempQ;

        tempC.x = c[D1Index[idx]].x;
        tempC.y = c[D1Index[idx]].y;
        tempC.z = c[D1Index[idx]].z;

        tempQ = __MATHUTILS__::__M3x3_v3_multiply(H, tempC);

        atomicAdd(&(q[D1Index[idx]].x), tempQ.x);
        atomicAdd(&(q[D1Index[idx]].y), tempQ.y);
        atomicAdd(&(q[D1Index[idx]].z), tempQ.z);
    }
}






__global__ void __PCG_Solve_AXALL_b2_debugtest(
    const __MATHUTILS__::Matrix12x12S* Hessians12, const __MATHUTILS__::Matrix9x9S* Hessians9,
    const __MATHUTILS__::Matrix6x6S* Hessians6, const __MATHUTILS__::Matrix3x3S* Hessians3,
    const uint4* D4Index, const uint3* D3Index, const uint2* D2Index, const uint32_t* D1Index,
    const Scalar3* c, Scalar3* q, 
    int numbers4, int numbers3, int numbers2, int numbers1,
    int offset4, int offset3, int offset2) {

    // calculate A = M + dt^2 * K, q = A @ c
    if (blockIdx.x < offset4) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= numbers4) return;
        __shared__ int offset;
        int Hid = idx / 144;
        int MRid = (idx % 144) / 12; // rowid
        int MCid = (idx % 144) % 12; // colid

        int vId = MCid / 3; // 顶点编号
        int axisId = MCid % 3; // xyz轴
        int GRtid = idx % 12;

        Scalar rdata =
            Hessians12[Hid].m[MRid][MCid] * (*(&(c[*(&(D4Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (12 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 12) / 12;
        int landidx = (threadIdx.x - offset) % 12;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 12; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D4Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } else if (blockIdx.x >= offset4 && blockIdx.x < offset4 + offset3) {
        int idx = (blockIdx.x - offset4) * blockDim.x + threadIdx.x;
        if (idx >= numbers3) return;
        __shared__ int offset;
        int Hid = idx / 81;
        int MRid = (idx % 81) / 9;
        int MCid = (idx % 81) % 9;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 9;

        Scalar rdata = Hessians9[Hid].m[MRid][MCid] * (*(&(c[*(&(D3Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (9 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 9) / 9;
        int landidx = (threadIdx.x - offset) % 9;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);  // a bit-mask
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 9; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D3Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } else if (blockIdx.x >= offset4 + offset3 && blockIdx.x < offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3) * blockDim.x + threadIdx.x;
        if (idx >= numbers2) return;
        __shared__ int offset;
        int Hid = idx / 36;
        int MRid = (idx % 36) / 6;
        int MCid = (idx % 36) % 6;

        int vId = MCid / 3;
        int axisId = MCid % 3;
        int GRtid = idx % 6;

        Scalar rdata =
            Hessians6[Hid].m[MRid][MCid] * (*(&(c[*(&(D2Index[Hid].x) + vId)].x) + axisId));

        if (threadIdx.x == 0) {
            offset = (6 - GRtid);
        }
        __syncthreads();

        int BRid = (threadIdx.x - offset + 6) / 6;
        int landidx = (threadIdx.x - offset) % 6;
        if (BRid == 0) {
            landidx = threadIdx.x;
        }

        int warpId = threadIdx.x & 0x1f;
        bool bBoundary = (landidx == 0) || (warpId == 0);

        unsigned int mark = __ballot_sync(0xFFFFFFFF, bBoundary);
        mark = __brev(mark);
        unsigned int interval = __MATHUTILS__::__m_min(__clz(mark << (warpId + 1)), 31 - warpId);

        for (int iter = 1; iter < 6; iter <<= 1) {
            Scalar tmp = __shfl_down_sync(0xFFFFFFFF, rdata, iter);
            if (interval >= iter) rdata += tmp;
        }

        if (bBoundary) atomicAdd((&(q[*(&(D2Index[Hid].x) + MRid / 3)].x) + MRid % 3), rdata);

    } else if (blockIdx.x >= offset4 + offset3 + offset2) {
        int idx = (blockIdx.x - offset4 - offset3 - offset2) * blockDim.x + threadIdx.x;
        if (idx >= numbers1) return;
        __MATHUTILS__::Matrix3x3S H = Hessians3[idx];
        Scalar3 tempC, tempQ;

        tempC.x = c[D1Index[idx]].x;
        tempC.y = c[D1Index[idx]].y;
        tempC.z = c[D1Index[idx]].z;

        tempQ = __MATHUTILS__::__M3x3_v3_multiply(H, tempC);

        atomicAdd(&(q[D1Index[idx]].x), tempQ.x);
        atomicAdd(&(q[D1Index[idx]].y), tempQ.y);
        atomicAdd(&(q[D1Index[idx]].z), tempQ.z);
    }
}













































__global__ void __PCG_Solve_AX_mass_b(const Scalar* _masses, const Scalar3* c, Scalar3* q,
                                      int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    // q = M @ c
    Scalar3 tempQ = __MATHUTILS__::__s_vec3_multiply(c[idx], _masses[idx]);
    q[idx] = tempQ;
}


__global__ void __PCG_mass_P(const Scalar* _masses, __MATHUTILS__::Matrix3x3S* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    // P = M
    __MATHUTILS__::__init_Mat3x3(P[idx], 0);
    Scalar mass = _masses[idx];
    P[idx].m[0][0] = mass;
    P[idx].m[1][1] = mass;
    P[idx].m[2][2] = mass;
}


__global__ void __PCG_inverse_P(__MATHUTILS__::Matrix3x3S* P, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    // P = P^-1
    __MATHUTILS__::Matrix3x3S PInverse;
    __MATHUTILS__::__Inverse(P[idx], PInverse);
    P[idx] = PInverse;
}




Scalar My_PCG_add_Reduction_Algorithm(int type, std::unique_ptr<GeometryManager>& instance,
                                      Scalar* _PCGSqueue, Scalar3* _PCGb, Scalar3* _PCGr,
                                      Scalar3* _PCGc, Scalar3* _PCGq, Scalar3* _PCGs,
                                      Scalar3* _PCGz, Scalar3* _PCGdx,
                                      __MATHUTILS__::Matrix3x3S* _PCGP, int vertexNum,
                                      Scalar alpha = 1) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);
    switch (type) {
        case 0:
            PCG_add_Reduction_force<<<blockNum, threadNum, sharedMsize>>>(_PCGSqueue, _PCGb, numbers);
            break;
        case 1:
            // compute delta_0
            PCG_add_Reduction_delta0<<<blockNum, threadNum, sharedMsize>>>(
                _PCGSqueue, _PCGP, _PCGb, instance->getCudaConstraintsMat(), numbers);  
            break;
        case 2:
            // compute delta_N
            PCG_add_Reduction_deltaN0<<<blockNum, threadNum, sharedMsize>>>(
                _PCGSqueue, _PCGP, _PCGb, _PCGr, _PCGc, instance->getCudaConstraintsMat(), numbers); 
            break;
        case 3:
            // compute tempSum
            PCG_add_Reduction_tempSum<<<blockNum, threadNum, sharedMsize>>>(
                _PCGSqueue, _PCGc, _PCGq, instance->getCudaConstraintsMat(), numbers);
            break;
        case 4:
            // update delta_N and delta_x
            PCG_add_Reduction_deltaN<<<blockNum, threadNum, sharedMsize>>>(
                _PCGSqueue, _PCGdx, _PCGc, _PCGr, _PCGq, _PCGP, _PCGs, alpha,
                numbers);
            break;
    }

    // if blockNum greater than 1 then reduct block again 
    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;
    while (numbers > 1) {
        __MATHUTILS__::__reduct_add_Scalar<<<blockNum, threadNum, sharedMsize>>>(_PCGSqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar result;
    cudaMemcpy(&result, _PCGSqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    return result;
}


Scalar __dx_sum(std::vector<Scalar3>& dx) {
    Scalar sum = 0.0;
    for (int i = 0; i < dx.size(); i++) {
        sum += dx[i].x + dx[i].y + dx[i].z;
    }
    return sum;
}


void Solve_PCG_AX_B2(const std::unique_ptr<GeometryManager>& instance, const Scalar3* c, Scalar3* q,
                     const std::unique_ptr<BlockHessian>& BH_ptr, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    // q = M @ c
    __PCG_Solve_AX_mass_b<<<blockNum, threadNum>>>(instance->getCudaVertMass(), c, q, numbers);

    // q += K @ c
    int offset4 = (BH_ptr->hostBHDNum[3] * 144 + threadNum - 1) / threadNum;
    int offset3 = (BH_ptr->hostBHDNum[2] * 81 + threadNum - 1) / threadNum;
    int offset2 = (BH_ptr->hostBHDNum[1] * 36 + threadNum - 1) / threadNum;
    int offset1 = (BH_ptr->hostBHDNum[0] + threadNum - 1) / threadNum;
    blockNum = offset1 + offset2 + offset3 + offset4;

    // printf("BHnum %d %d %d %d \n", BH_ptr->hostBHDNum[0], BH_ptr->hostBHDNum[1], BH_ptr->hostBHDNum[2], BH_ptr->hostBHDNum[3]);
    // printf("offset %d %d %d %d \n", offset1, offset2, offset3, offset4);


    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEvent_t start2, stop2;
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    float milliseconds1 = 0;
    float milliseconds2 = 0;

    cudaEventRecord(start1);
    __PCG_Solve_AXALL_b2<<<blockNum, threadNum>>>(
        BH_ptr->cudaH12x12, BH_ptr->cudaH9x9, BH_ptr->cudaH6x6,
        BH_ptr->cudaH3x3, BH_ptr->cudaD4Index, BH_ptr->cudaD3Index,
        BH_ptr->cudaD2Index, BH_ptr->cudaD1Index, 
        c, q, 
        BH_ptr->hostBHDNum[3] * 144,
        BH_ptr->hostBHDNum[2] * 81, BH_ptr->hostBHDNum[1] * 36,
        BH_ptr->hostBHDNum[0], offset4, offset3, offset2);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1); 
    cudaEventElapsedTime(&milliseconds1, start1, stop1);



    // Scalar3* q_debugtest = q;
    // cudaEventRecord(start2);
    // __PCG_Solve_AXALL_b2_debugtest<<<blockNum, threadNum>>>(
    //     BH_ptr->cudaH12x12, BH_ptr->cudaH9x9, BH_ptr->cudaH6x6,
    //     BH_ptr->cudaH3x3, BH_ptr->cudaD4Index, BH_ptr->cudaD3Index,
    //     BH_ptr->cudaD2Index, BH_ptr->cudaD1Index, 
    //     c, q_debugtest, 
    //     BH_ptr->hostBHDNum[3] * 144,
    //     BH_ptr->hostBHDNum[2] * 81, BH_ptr->hostBHDNum[1] * 36,
    //     BH_ptr->hostBHDNum[0], offset4, offset3, offset2);
    // cudaEventRecord(stop2);
    // cudaEventSynchronize(stop2); 
    // cudaEventElapsedTime(&milliseconds2, start2, stop2);

    // std::vector<Scalar3> dx(vertNum);
    // std::vector<Scalar3> dx_debugtest(vertNum);
    // CUDAMemcpyDToHSafe(dx, q);
    // CUDAMemcpyDToHSafe(dx_debugtest, q_debugtest);
    // Scalar sum = __dx_sum(dx);
    // Scalar sum_debugtest = __dx_sum(dx_debugtest);
    // printf("time compare: %f, %f: \n", milliseconds1, milliseconds2);
    // printf("dx compare: %f %f \n", sum, sum_debugtest);

}

void construct_P2(const std::unique_ptr<GeometryManager>& instance, __MATHUTILS__::Matrix3x3S* P,
                  const std::unique_ptr<BlockHessian>& BH_ptr, int vertNum) {
    int numbers = vertNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    // init diagonal matrix Precond P and init mass P = M
    __PCG_mass_P<<<blockNum, threadNum>>>(instance->getCudaVertMass(), P, numbers);

    numbers = BH_ptr->hostBHDNum[3] * 12 + BH_ptr->hostBHDNum[2] * 9 +
              BH_ptr->hostBHDNum[1] * 6 + BH_ptr->hostBHDNum[0] * 3;
    blockNum = (numbers + threadNum - 1) / threadNum;
    // P = M+h^2*K
    __PCG_AXALL_P<<<blockNum, threadNum>>>(
        BH_ptr->cudaH12x12, BH_ptr->cudaH9x9, BH_ptr->cudaH6x6,
        BH_ptr->cudaH3x3, BH_ptr->cudaD4Index, BH_ptr->cudaD3Index,
        BH_ptr->cudaD2Index, BH_ptr->cudaD1Index, P, BH_ptr->hostBHDNum[3] * 12,
        BH_ptr->hostBHDNum[2] * 9, BH_ptr->hostBHDNum[1] * 6,
        BH_ptr->hostBHDNum[0] * 3);

    // P = P^-1 = (M+h^2*K)^-1
    blockNum = (vertNum + threadNum - 1) / threadNum;
    __PCG_inverse_P<<<blockNum, threadNum>>>(P, vertNum);

}

void PCG_FinalStep_UpdateC(const std::unique_ptr<GeometryManager>& instance, Scalar3* c,
                           const Scalar3* s, const Scalar& rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_FinalStep_UpdateC<<<blockNum, threadNum>>>(instance->getCudaConstraintsMat(), s, c, rate,
                                                     numbers);
}


int PCGSolver::PCG_Process(std::unique_ptr<GeometryManager>& instance,
                           const std::unique_ptr<BlockHessian>& BH_ptr, Scalar3* _mvDir, int vertexNum,
                           int tetrahedraNum, Scalar IPC_dt, Scalar meanVolume, Scalar threshold) {
    
    cudaPCGdx = instance->getCudaMoveDir();
    cudaPCGb = instance->getCudaFb();

    // calculate preconditioner M^-1
    construct_P2(instance, cudaPCGP, BH_ptr, vertexNum);

    // calculate delta0=r^T@z deltaN=r^T@c 
    Scalar delta0 = 0;
    Scalar deltaN = 0;
    Scalar deltaN_prev = 0;
    CUDA_SAFE_CALL(cudaMemset(cudaPCGdx, 0, vertexNum * sizeof(Scalar3)));
    CUDA_SAFE_CALL(cudaMemset(cudaPCGr, 0, vertexNum * sizeof(Scalar3)));

    // delta0 = rT@z0
    delta0 = My_PCG_add_Reduction_Algorithm(1, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
                                            cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
                                            cudaPCGP, vertexNum); 

    // deltaN = r0^T * z0 (初始时 deltaN = delta0)
    deltaN = My_PCG_add_Reduction_Algorithm(2, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
                                            cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
                                            cudaPCGP, vertexNum);

    // threadhold = 1e-3
    Scalar errorRate = threshold; 

    int cgCounts = 0;
    while (cgCounts < 30000 && deltaN > errorRate * delta0) {
        cgCounts++;

        // q_k = A @ p_k = (M+K)@p_k
        Solve_PCG_AX_B2(instance, cudaPCGc, cudaPCGq, BH_ptr, vertexNum);

        // tempSum = p_k^T @ q = p_k^T @ A @ p_k
        Scalar tempSum = My_PCG_add_Reduction_Algorithm(3, instance, cudaPCGSqueue, cudaPCGb,
                                                        cudaPCGr, cudaPCGc, cudaPCGq, cudaPCGs,
                                                        cudaPCGz, cudaPCGdx, cudaPCGP, vertexNum);

        // alpha_k = deltaN / tempSum = (r0^T @ z0) / (p_k^T @ A @ p_k)
        Scalar alpha = deltaN / tempSum; 
        deltaN_prev = deltaN;

        // deltaN = r_{k+1}^T * z_{k+1}
        deltaN = My_PCG_add_Reduction_Algorithm(4, instance, cudaPCGSqueue, cudaPCGb, cudaPCGr,
                                                cudaPCGc, cudaPCGq, cudaPCGs, cudaPCGz, cudaPCGdx,
                                                cudaPCGP, vertexNum, alpha);

        // beta_k = deltaN / deltaO
        Scalar beta = deltaN / deltaN_prev;

        // p_{k+1} = z_{k+1} + beta_k * p_k
        PCG_FinalStep_UpdateC(instance, cudaPCGc, cudaPCGs, beta, vertexNum);

        // printf("%d~~~~~~~~JacobiPCG residual norm: %f \n", cgCounts, deltaN / delta0);

    }

    _mvDir = cudaPCGdx;

    if (cgCounts == 0) {
        printf("indefinite exit\n");
        // exit(EXIT_FAILURE);
    }

    return cgCounts;
}















////////////////////////////////////////////////////////
// MAS Preconditioner PCG
////////////////////////////////////////////////////////

__global__ void PCG_vdv_Reduction(Scalar* squeue, const Scalar3* a, const Scalar3* b, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar temp = __MATHUTILS__::__vec3_dot(a[idx], b[idx]);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);

}

Scalar My_PCG_General_v_v_Reduction_Algorithm(std::unique_ptr<GeometryManager>& instance,
                                              Scalar* _PCGSqueue, Scalar3* A, Scalar3* B,
                                              int vertexNum) {
                                                
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);
    PCG_vdv_Reduction<<<blockNum, threadNum>>>(_PCGSqueue, A, B, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::__reduct_add_Scalar<<<blockNum, threadNum, sharedMsize>>>(_PCGSqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar result;
    cudaMemcpy(&result, _PCGSqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    return result;
}

__global__ void __PCG_Update_Dx_R(const Scalar3* c, Scalar3* dx, const Scalar3* q, Scalar3* r,
                                  Scalar rate, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    dx[idx] = __MATHUTILS__::__vec3_add(dx[idx], __MATHUTILS__::__s_vec3_multiply(c[idx], rate));
    r[idx] = __MATHUTILS__::__vec3_add(r[idx], __MATHUTILS__::__s_vec3_multiply(q[idx], -rate));
}

void PCG_Update_Dx_R(const Scalar3* c, Scalar3* dx, const Scalar3* q, Scalar3* r,
                     const Scalar& rate, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_Update_Dx_R<<<blockNum, threadNum>>>(c, dx, q, r, rate, numbers);
}

__global__ void __PCG_constraintFilter(const __MATHUTILS__::Matrix3x3S* constraints,
                                       const Scalar3* input, Scalar3* output, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    output[idx] = __MATHUTILS__::__M3x3_v3_multiply(constraints[idx], input[idx]);
}

void PCG_constraintFilter(const std::unique_ptr<GeometryManager>& instance, const Scalar3* input,
                          Scalar3* output, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    __PCG_constraintFilter<<<blockNum, threadNum>>>(instance->getCudaConstraintsMat(), input,
                                                    output, numbers);
}

int PCGSolver::MASPCG_Process(std::unique_ptr<GeometryManager>& instance,
                              const std::unique_ptr<BlockHessian>& BH_ptr, Scalar3* _mvDir, int vertexNum,
                              int tetrahedraNum, Scalar IPC_dt, Scalar meanVolume, int cpNum,
                              Scalar threshold) {
    cudaPCGdx = instance->getCudaMoveDir();
    cudaPCGb = instance->getCudaFb();

    MP.setPreconditioner(BH_ptr, instance->getCudaVertMass(), cpNum);
    
    Scalar deltaN = 0;
    Scalar delta0 = 0;
    Scalar deltaO = 0;
    // PCG_initDX(dx, z, 0.5, vertexNum);
    CUDA_SAFE_CALL(cudaMemset(cudaPCGdx, 0x0, vertexNum * sizeof(Scalar3)));
    CUDA_SAFE_CALL(cudaMemset(cudaPCGr, 0x0, vertexNum * sizeof(Scalar3)));

    PCG_constraintFilter(instance, cudaPCGb, cudaPCGFilterTempVec3, vertexNum);

    MP.preconditioning(cudaPCGFilterTempVec3, cudaPCGPrecondTempVec3);
    // Solve_PCG_Preconditioning24(instance, P24, P, restP, filterTempVec3,
    // preconditionTempVec3, vertexNum);
    
    delta0 = My_PCG_General_v_v_Reduction_Algorithm(instance, cudaPCGSqueue, cudaPCGFilterTempVec3,
                                                    cudaPCGPrecondTempVec3, vertexNum);

    CUDA_SAFE_CALL(cudaMemcpy(cudaPCGr, cudaPCGFilterTempVec3, vertexNum * sizeof(Scalar3),
                              cudaMemcpyDeviceToDevice));

    PCG_constraintFilter(instance, cudaPCGPrecondTempVec3, cudaPCGFilterTempVec3, vertexNum);

    CUDA_SAFE_CALL(cudaMemcpy(cudaPCGc, cudaPCGFilterTempVec3, vertexNum * sizeof(Scalar3),
                              cudaMemcpyDeviceToDevice));

    deltaN = My_PCG_General_v_v_Reduction_Algorithm(instance, cudaPCGSqueue, cudaPCGr, cudaPCGc,
                                                    vertexNum);

    Scalar errorRate = threshold; // threadhold = 1e-3
    
    int cgCounts = 0;
    while (cgCounts < 3000 && deltaN > errorRate * delta0) {
        cgCounts++;
        // std::cout << "delta0:   " << delta0 << "      deltaN:   " << deltaN
        // << "      iteration_counts:      " << cgCounts << std::endl;
        // CUDA_SAFE_CALL(cudaMemset(q, 0, vertexNum * sizeof(Scalar3)));
        Solve_PCG_AX_B2(instance, cudaPCGc, cudaPCGq, BH_ptr, vertexNum);
        
        Scalar tempSum = My_PCG_add_Reduction_Algorithm(3, instance, cudaPCGSqueue, cudaPCGb,
                                                        cudaPCGr, cudaPCGc, cudaPCGq, cudaPCGs,
                                                        cudaPCGz, cudaPCGdx, cudaPCGP, vertexNum);
        
        Scalar alpha = deltaN / tempSum;
        deltaO = deltaN;
        // deltaN = 0;
        // CUDA_SAFE_CALL(cudaMemset(s, 0, vertexNum * sizeof(Scalar3)));
        // deltaN = My_PCG_add_Reduction_Algorithm(4, instance, instance,
        // vertexNum, alpha);
        PCG_Update_Dx_R(cudaPCGc, cudaPCGdx, cudaPCGq, cudaPCGr, alpha, vertexNum);
        
        MP.preconditioning(cudaPCGr, cudaPCGs);

        deltaN = My_PCG_General_v_v_Reduction_Algorithm(instance, cudaPCGSqueue, cudaPCGr, cudaPCGs,
                                                        vertexNum);
        
        Scalar rate = deltaN / deltaO;
        PCG_FinalStep_UpdateC(instance, cudaPCGc, cudaPCGs, rate, vertexNum);

        // printf("%d~~~~~~~~MASPCG error rate: %f \n", cgCounts, deltaN / delta0);

    }

    _mvDir = cudaPCGdx;
    // CUDA_SAFE_CALL(cudaMemcpy(z, _mvDir, vertexNum * sizeof(Scalar3),
    // cudaMemcpyDeviceToDevice)); printf("cg counts = %d\n", cgCounts);
    if (cgCounts == 0) {
        printf("indefinite exit\n");
        // exit(EXIT_FAILURE);
    }
    return cgCounts;
}

void PCGSolver::CUDA_MALLOC_PCGSOLVER(const int& vertexNum) {
    CUDAMallocSafe(cudaPCGSqueue, vertexNum);
    CUDAMallocSafe(cudaPCGb, vertexNum);
    CUDAMallocSafe(cudaPCGP, vertexNum);
    CUDAMallocSafe(cudaPCGr, vertexNum);
    CUDAMallocSafe(cudaPCGc, vertexNum);
    CUDAMallocSafe(cudaPCGq, vertexNum);
    CUDAMallocSafe(cudaPCGs, vertexNum);
    CUDAMallocSafe(cudaPCGz, vertexNum);
    CUDAMallocSafe(cudaPCGdx, vertexNum);
    CUDAMallocSafe(cudaPCGTempDx, vertexNum);
    CUDAMallocSafe(cudaPCGPrecondTempVec3, vertexNum);
    CUDAMallocSafe(cudaPCGFilterTempVec3, vertexNum);
}

void PCGSolver::CUDA_FREE_PCGSOLVER() {
    CUDAFreeSafe(cudaPCGSqueue);
    CUDAFreeSafe(cudaPCGb);
    CUDAFreeSafe(cudaPCGP);
    CUDAFreeSafe(cudaPCGr);
    CUDAFreeSafe(cudaPCGc);
    CUDAFreeSafe(cudaPCGq);
    CUDAFreeSafe(cudaPCGs);
    CUDAFreeSafe(cudaPCGz);
    CUDAFreeSafe(cudaPCGdx);
    CUDAFreeSafe(cudaPCGTempDx);
    CUDAFreeSafe(cudaPCGFilterTempVec3);
    CUDAFreeSafe(cudaPCGPrecondTempVec3);

    if (PrecondType == 1) {
        MP.CUDA_FREE_MAS_PRECONDITIONER();
    }
}
