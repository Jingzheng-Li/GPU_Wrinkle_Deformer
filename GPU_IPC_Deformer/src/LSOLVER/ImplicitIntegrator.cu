
#include <fstream>

#include "FEMEnergy.cuh"
#include "GIPC.cuh"
#include "GIPCFriction.cuh"
#include "GeometryManager.hpp"
#include "ImplicitIntegrator.cuh"

namespace __INTEGRATOR__ {




void tempMalloc_closeConstraint(std::unique_ptr<GeometryManager>& instance) {
    CUDAMallocSafe(instance->getCudaCloseConstraintID(), instance->getHostGpNum());
    CUDAMallocSafe(instance->getCudaCloseConstraintVal(), instance->getHostGpNum());
    CUDAMallocSafe(instance->getCudaCloseMConstraintID(), instance->getHostCpNum(0));
    CUDAMallocSafe(instance->getCudaCloseMConstraintVal(), instance->getHostCpNum(0));
}

void tempFree_closeConstraint(std::unique_ptr<GeometryManager>& instance) {
    CUDAFreeSafe(instance->getCudaCloseConstraintID());
    CUDAFreeSafe(instance->getCudaCloseConstraintVal());
    CUDAFreeSafe(instance->getCudaCloseMConstraintID());
    CUDAFreeSafe(instance->getCudaCloseMConstraintVal());
}


__device__ Scalar3 calculateWindForce(Scalar3 position, Scalar3 windDirection, Scalar windStrength,
                                      Scalar noiseFrequency, Scalar noiseAmplitude, Scalar time) {
    // 缩放位置以控制噪声频率，并添加时间参数
    Scalar x = position.x * noiseFrequency + time;
    Scalar y = position.y * noiseFrequency + time;
    Scalar z = position.z * noiseFrequency + time;

    // 为每个分量生成噪声值，使用不同的偏移量以获取不同的噪声样本
    Scalar noiseValueX = __MATHUTILS__::_perlinNoise(x, y, z);
    Scalar noiseValueY = __MATHUTILS__::_perlinNoise(x + 31.4, y + 47.2, z + 12.8); // 偏移以获得不同噪声
    Scalar noiseValueZ = __MATHUTILS__::_perlinNoise(x + 58.6, y + 26.7, z + 73.5);

    // 创建噪声向量并缩放
    Scalar3 noise = make_Scalar3(noiseValueX, noiseValueY, noiseValueZ);
    noise = __MATHUTILS__::__s_vec3_multiply(noise, noiseAmplitude);

    // 基础风力计算
    Scalar3 windForce = __MATHUTILS__::__s_vec3_multiply(windDirection, windStrength);

    // 将噪声添加到风力中
    windForce = __MATHUTILS__::__vec3_add(windForce, noise);

    return windForce;
}

__global__ void _computeXTilta(int* _btype, Scalar3* _velocities, Scalar3* _o_vertexes, Scalar3* _xTilta, 
                               Scalar3 windDirection, Scalar windStrength, Scalar airResistance, 
                               Scalar noiseFrequency, Scalar noiseAmplitude, Scalar noisetime,
                               Scalar ipc_dt, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    Scalar3 gravityForce_DtSq = make_Scalar3(0, 0, 0);
    if (_btype[idx] == 0) {
        Scalar3 gravityForce = make_Scalar3(0, -9.8, 0);
        gravityForce_DtSq = __MATHUTILS__::__s_vec3_multiply(gravityForce, ipc_dt * ipc_dt);
    }

    Scalar3 windForce_DtSq = make_Scalar3(0, 0, 0);
    if (_btype[idx] == 0 && windStrength > 1e-6) {
        Scalar3 windForce = calculateWindForce(_o_vertexes[idx], windDirection, windStrength,
                                               noiseFrequency, noiseAmplitude, noisetime);
        windForce_DtSq = __MATHUTILS__::__s_vec3_multiply(windForce, ipc_dt * ipc_dt);
    }

    Scalar3 airResistanceForce_DtSq = make_Scalar3(0, 0, 0);
    if (_btype[idx] == 0 && airResistance > 1e-6) {
        Scalar3 airResistanceForce = __MATHUTILS__::__s_vec3_multiply(_velocities[idx], -airResistance);
        airResistanceForce_DtSq = __MATHUTILS__::__s_vec3_multiply(airResistanceForce, ipc_dt * ipc_dt);
    }

    Scalar3 constForceTotal_DtSq = __MATHUTILS__::__vec3_add(__MATHUTILS__::__vec3_add(gravityForce_DtSq, windForce_DtSq), airResistanceForce_DtSq);

    // 更新位置xtilta
    _xTilta[idx] = __MATHUTILS__::__vec3_add(
        _o_vertexes[idx],
        __MATHUTILS__::__vec3_add(__MATHUTILS__::__s_vec3_multiply(_velocities[idx], ipc_dt), constForceTotal_DtSq));
}


__global__ void _updateVelocities(Scalar3* _vertexes, Scalar3* _o_vertexes, Scalar3* _velocities, int* btype,
                                  Scalar ipc_dt, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (btype[idx] == 0) {
        _velocities[idx] =
            __MATHUTILS__::__s_vec3_multiply(__MATHUTILS__::__vec3_minus(_vertexes[idx], _o_vertexes[idx]), 1 / ipc_dt);
        _o_vertexes[idx] = _vertexes[idx];
    } else {
        _velocities[idx] = make_Scalar3(0, 0, 0);
        _o_vertexes[idx] = _vertexes[idx];
    }
}

__global__ void _changeBoundarytoSIMPoint(int* _btype, __MATHUTILS__::Matrix3x3S* _constraints, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    if ((_btype[idx]) == 1) {
        _btype[idx] = 0;
        __MATHUTILS__::__set_Mat_val(_constraints[idx], 1, 0, 0, 0, 1, 0, 0, 0, 1);
    }
}

__global__ void _getKineticEnergy_Reduction_3D(Scalar3* _vertexes, Scalar3* _xTilta, Scalar* _energy, Scalar* _masses,
                                               int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep1[];

    if (idx >= number) return;

    Scalar temp =
        __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(_vertexes[idx], _xTilta[idx])) * _masses[idx] * 0.5;

    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = DEFAULT_THREADS_PERWARP;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }
    if (warpTid == 0) {
        tep1[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep1[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }
    if (threadIdx.x == 0) {
        _energy[blockIdx.x] = temp;
    }
}

__global__ void _stepForward(Scalar3* _vertexes, Scalar3* _vertexesTemp, Scalar3* _moveDir, int* bType, Scalar alpha,
                             bool moveBoundary, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (abs(bType[idx]) != 1 || moveBoundary) {
        _vertexes[idx] =
            __MATHUTILS__::__vec3_minus(_vertexesTemp[idx], __MATHUTILS__::__s_vec3_multiply(_moveDir[idx], alpha));
    }
}

__global__ void _getDeltaEnergy_Reduction(Scalar* squeue, const Scalar3* b, const Scalar3* dx, int vertexNum) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep1[];
    int numbers = vertexNum;
    if (idx >= numbers) return;
    // int cfid = tid + CONFLICT_FREE_OFFSET(tid);

    Scalar temp = __MATHUTILS__::__vec3_dot(b[idx], dx[idx]);

    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = DEFAULT_THREADS_PERWARP;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }
    if (warpTid == 0) {
        tep1[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep1[threadIdx.x];
        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void _reduct_MGroundDist(const Scalar3* vertexes, const Scalar* g_offset, const Scalar3* g_normal,
                                    uint32_t* _environment_collisionPair, Scalar2* _queue, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar2 tep2[];

    if (idx >= number) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__vec3_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar tempv = dist * dist;
    Scalar2 temp = make_Scalar2(1.0 / tempv, tempv);

    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = DEFAULT_THREADS_PERWARP;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
        temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        tep2[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep2[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
            temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _queue[blockIdx.x] = temp;
    }
}


__global__ void _reduct_MSelfDist(const Scalar3* _vertexes, int4* _collisionPairs, Scalar2* _queue, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar2 tep2[];

    if (idx >= number) return;
    int4 MMCVIDI = _collisionPairs[idx];
    Scalar tempv = __GPUIPC__::__calBarrierSelfConsDis(_vertexes, MMCVIDI);
    Scalar2 temp = make_Scalar2(1.0 / tempv, tempv);
    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    // int tidNum = DEFAULT_THREADS_PERWARP;
    if (blockIdx.x == gridDim.x - 1) {
        // tidNum = numbers - idof;
        warpNum = ((number - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
        Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
        temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
        temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
    }
    if (warpTid == 0) {
        tep2[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        //	tidNum = warpNum;
        temp = tep2[threadIdx.x];

        //	warpNum = ((tidNum + 31) >> 5);
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
            temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _queue[blockIdx.x] = temp;
    }
}

Scalar2 minMaxGroundDist(std::unique_ptr<GeometryManager>& instance) {
    //_reduct_minGroundDist << <blockNum, threadNum >> > (_vertexes,
    //_groundOffset, _groundNormal, _isChange, _closeConstraintID,
    //_closeConstraintVal, numbers);

    int numbers = instance->getHostGpNum();
    if (numbers < 1) return make_Scalar2(1e32, 0);
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar2) * (threadNum >> 5);

    Scalar2* _queue;
    CUDAMallocSafe(_queue, numbers);
    // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number *
    // sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MGroundDist<<<blockNum, threadNum, sharedMsize>>>(
        instance->getCudaSurfVertPos(), instance->getCudaGroundOffset(), instance->getCudaGroundNormal(),
        instance->getCudaEnvCollisionPairs(), _queue, numbers);
    //_reduct_min_Scalar3_to_Scalar << <blockNum, threadNum, sharedMsize >> >
    //(_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> >
        //(_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB),
    // cudaMemcpyDeviceToDevice);
    Scalar2 minMaxValue;
    cudaMemcpy(&minMaxValue, _queue, sizeof(Scalar2), cudaMemcpyDeviceToHost);
    CUDAFreeSafe(_queue);
    minMaxValue.x = 1.0 / minMaxValue.x;
    return minMaxValue;
}

Scalar calcMinMovement(const Scalar3* _moveDir, Scalar* _queue, const int& number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    /*Scalar* _tempMinMovement;
    CUDAMallocSafe(_tempMinMovement, numbers);*/
    // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number *
    // sizeof(AABB), cudaMemcpyDeviceToDevice));

    __MATHUTILS__::_reduct_max_Scalar3_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(_moveDir, _queue, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> >
        //(_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB),
    // cudaMemcpyDeviceToDevice);
    Scalar minValue;
    cudaMemcpy(&minValue, _queue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // CUDAFreeSafe(_tempMinMovement);
    return minValue;
}

void stepForward(Scalar3* _vertexes, Scalar3* _vertexesTemp, Scalar3* _moveDir, int* bType, Scalar alpha,
                 bool moveBoundary, int numbers) {
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _stepForward<<<blockNum, threadNum>>>(_vertexes, _vertexesTemp, _moveDir, bType, alpha, moveBoundary, numbers);
}

void computeXTilta(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostNumVertices();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _computeXTilta<<<blockNum, threadNum>>>(instance->getCudaBoundaryType(), instance->getCudaVertVel(),
                                            instance->getCudaOriginVertPos(), instance->getCudaXTilta(),
                                            instance->getHostWindDirection(), instance->getHostWindStrength(),
                                            instance->getHostAirResistance(), 
                                            instance->getHostNoiseFrequency(), instance->getHostNoiseAmplitude(), 
                                            instance->getHostSimulationFrameId() * instance->getHostIPCDt(),
                                            instance->getHostIPCDt(), numbers);
}

void updateVelocities(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostNumVertices();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _updateVelocities<<<blockNum, threadNum>>>(instance->getCudaSurfVertPos(), instance->getCudaOriginVertPos(),
                                               instance->getCudaVertVel(), instance->getCudaBoundaryType(),
                                               instance->getHostIPCDt(), numbers);
}

void changeBoundarytoSIMPoint(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostNumVertices();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _changeBoundarytoSIMPoint<<<blockNum, threadNum>>>(instance->getCudaBoundaryType(),
                                                       instance->getCudaConstraintsMat(), numbers);
}

int calculateMovingDirection(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr,
                             std::unique_ptr<PCGSolver>& PCG_ptr, int cpNum, int preconditioner_type) {
    if (preconditioner_type == 0) {
        int cgCount = PCG_ptr->PCG_Process(instance, BH_ptr, instance->getCudaMoveDir(), instance->getHostNumVertices(),
                                           instance->getHostNumTetElements(), instance->getHostIPCDt(),
                                           instance->getHostMeanVolume(), instance->getHostPCGThreshold());
        std::cout << "PCG finish:  " << cgCount << std::endl;
        return cgCount;

    } else if (preconditioner_type == 1) {
        int cgCount = PCG_ptr->MASPCG_Process(instance, BH_ptr, instance->getCudaMoveDir(), instance->getHostNumVertices(),
                                            instance->getHostNumTetElements(), instance->getHostIPCDt(),
                                            instance->getHostMeanVolume(), cpNum, instance->getHostPCGThreshold());
        if (cgCount == 3000) {
            printf("MASPCG failed !!!!!!!!!!!!!!!!!!!!!\n");
            exit(EXIT_FAILURE);
        }
        std::cout << "MASPCG finish:  " << cgCount << std::endl;
        return cgCount;

    } else {
        printf("not support other pcg right now!!!!!!!!!!!!!!!!!!!\n");
        exit(EXIT_FAILURE);
    }
}

Scalar computeGradientAndHessian(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr) {
    __GPUIPC__::calKineticGradient(instance->getCudaSurfVertPos(), instance->getCudaXTilta(), instance->getCudaFb(),
                                   instance->getCudaVertMass(), instance->getHostNumVertices());

    CUDA_SAFE_CALL(cudaMemset(instance->getCudaCPNum(), 0, 5 * sizeof(uint32_t)));

    __GPUIPC__::calBarrierGradientAndHessian(instance, BH_ptr, instance->getCudaFb(), instance->getHostKappa());

#ifdef USE_GIPCFRICTION
    __GIPCFRICTION__::calFrictionGradient(instance, instance->getCudaFb());
    __GIPCFRICTION__::calFrictionHessian(instance, BH_ptr);
#endif

    __FEMENERGY__::calculate_bending_gradient_hessian(instance, BH_ptr);

    __FEMENERGY__::calculate_triangle_cons_gradient_hessian(instance, BH_ptr);

    __FEMENERGY__::computeGroundGradientAndHessian(instance, BH_ptr, instance->getCudaFb());

    __FEMENERGY__::computeBoundConstraintGradientAndHessian(instance, BH_ptr, instance->getCudaFb());

    __FEMENERGY__::computeSoftConstraintGradientAndHessian(instance, BH_ptr, instance->getCudaFb());

    __FEMENERGY__::computeStitchConstraintGradientAndHessian(instance, BH_ptr, instance->getCudaFb());

    return 0.0;
}

Scalar Energy_Add_Reduction_Algorithm(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<PCGSolver>& PCG_ptr,
                                      int type) {
    int numbers = 0;

    if (type == 0 || type == 3) {
        numbers = instance->getHostNumVertices();
    } else if (type == 2) {
        numbers = instance->getHostCpNum(0);
    } else if (type == 4) {
        numbers = instance->getHostGpNum();
    } else if (type == 5) {
        numbers = instance->getHostCpNumLast(0);
    } else if (type == 6) {
        numbers = instance->getHostGpNumLast();
    } else if (type == 7 || type == 1) {
        numbers = instance->getHostNumTetElements();
    } else if (type == 8) {
        numbers = instance->getHostNumTriElements();
    } else if (type == 9) {
        numbers = instance->getHostNumTriBendEdges();
    } else if (type == 10) {
        numbers = instance->getHostNumBoundTargets();
    } else if (type == 11) {
        numbers = instance->getHostNumSoftTargets();
    } else if (type == 12) {
        numbers = instance->getHostNumStitchPairs();
    }
    if (numbers == 0) return 0;

    Scalar* queue = PCG_ptr->cudaPCGSqueue;
    // CUDAMallocSafe(queue, numbers);*/

    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);
    switch (type) {
        case 0:
            _getKineticEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                instance->getCudaSurfVertPos(), instance->getCudaXTilta(), queue, instance->getCudaVertMass(), numbers);
            break;
        case 1:
            __FEMENERGY__::_getFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaTetElement(), instance->getCudaTetDmInverses(),
                instance->getCudaTetVolume(), numbers, instance->getHostLengthRate(), instance->getHostVolumeRate());
            break;
        case 2:
            __GPUIPC__::_getBarrierEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaRestVertPos(), instance->getCudaCollisionPairs(),
                instance->getHostKappa(), instance->getHostDHat(), numbers);
            break;
        case 3:
            _getDeltaEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(queue, instance->getCudaFb(),
                                                                            instance->getCudaMoveDir(), numbers);
            break;
        case 4:
            __GPUIPC__::_computeGroundEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaGroundOffset(), instance->getCudaGroundNormal(),
                instance->getCudaEnvCollisionPairs(), instance->getHostDHat(), instance->getHostKappa(), numbers);
            break;
        case 5:
            __GIPCFRICTION__::_getFrictionEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaOriginVertPos(),
                instance->getCudaCollisionPairsLastH(), numbers, instance->getHostIPCDt(), instance->getCudaDistCoord(),
                instance->getCudaTanBasis(), instance->getCudaLambdaLastHScalar(),
                instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
                sqrt(instance->getHostFDHat()) * instance->getHostIPCDt());
            break;
        case 6:
            __GIPCFRICTION__::_getFrictionEnergy_gd_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaOriginVertPos(), instance->getCudaGroundNormal(),
                instance->getCudaCollisionPairsLastHGd(), numbers, instance->getHostIPCDt(),
                instance->getCudaLambdaLastHScalarGd(), 
                sqrt(instance->getHostFDHat()) * instance->getHostIPCDt());
            break;
        case 7:
            __FEMENERGY__::_getRestStableNHKEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaTetVolume(), numbers, instance->getHostLengthRate(),
                instance->getHostVolumeRate());
            break;
        case 8:
            __FEMENERGY__::_get_triangleFEMEnergy_Reduction_3D<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaTriElement(), instance->getCudaTriDmInverses(),
                instance->getCudaTriArea(), numbers, instance->getHostStretchStiff(), instance->getHostShearStiff());
            break;
        case 9:
            __FEMENERGY__::_getBendingEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaRestVertPos(), instance->getCudaTriBendEdges(),
                instance->getCudaTriBendVerts(), numbers, instance->getHostBendStiff());
            break;
        case 10:
            __FEMENERGY__::_computeBoundConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaBoundTargetVertPos(),
                instance->getCudaBoundTargetIndex(), instance->getHostBoundMotionRate(),
                instance->getHostAnimationFullRate(), numbers);
            break;
        case 11:
            __FEMENERGY__::_computeSoftConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), instance->getCudaSoftTargetVertPos(),
                instance->getCudaSoftTargetIndex(), instance->getHostSoftStiffness(),
                instance->getHostAnimationFullRate(), numbers);
            break;
        case 12:
            __FEMENERGY__::_computeStitchConstraintEnergy_Reduction<<<blockNum, threadNum, sharedMsize>>>(
                queue, instance->getCudaSurfVertPos(), 
                instance->getCudaStitchPairsIndex(), instance->getHostStitchStiffness(),
                instance->getHostAnimationFullRate(), numbers);
            break;
    }


    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::__reduct_add_Scalar<<<blockNum, threadNum, sharedMsize>>>(queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar result;
    cudaMemcpy(&result, queue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // CUDAFreeSafe(queue);
    return result;
}

Scalar computeEnergy(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<PCGSolver>& PCG_ptr) {

    Scalar Energy = Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 0);

    Energy += instance->getHostIPCDt() * instance->getHostIPCDt() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 1);

    Energy += instance->getHostIPCDt() * instance->getHostIPCDt() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 8);

    Energy += instance->getHostIPCDt() * instance->getHostIPCDt() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 9);

    Energy += Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 10);

    Energy += Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 11);

    Energy += Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 12);

    Energy += Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 2);

    Energy += instance->getHostKappa() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 4);

#ifdef USE_GIPCFRICTION
    Energy += instance->getHostFrictionRate() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 5);

    Energy += instance->getHostFrictionRate() * Energy_Add_Reduction_Algorithm(instance, PCG_ptr, 6);

#endif

    return Energy;
}

bool lineSearch(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<PCGSolver>& PCG_ptr,
                std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr, Scalar& alpha, const Scalar& cfl_alpha) {

    Scalar lastEnergyVal = computeEnergy(instance, PCG_ptr);

    CUDA_SAFE_CALL(cudaMemcpy(instance->getCudaTempScalar3Mem(), instance->getCudaSurfVertPos(),
                              instance->getHostNumVertices() * sizeof(Scalar3), cudaMemcpyDeviceToDevice));

    stepForward(instance->getCudaSurfVertPos(), instance->getCudaTempScalar3Mem(), instance->getCudaMoveDir(),
                instance->getCudaBoundaryType(), alpha, false, instance->getHostNumVertices());

    // 先检查前进的一步是否有穿透 有就减半
    LBVH_CD_ptr->buildBVH(instance);
    int numOfIntersect = 0;
    int insectNum = 0;
    bool checkInterset = true;
    while (checkInterset && __GPUIPC__::isIntersected(instance, LBVH_CD_ptr)) {
        printf("type 0 intersection happened!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:  %d\n", insectNum);
        insectNum++;
        alpha /= 2.0;
        numOfIntersect++;
        alpha = __MATHUTILS__::__m_min(cfl_alpha, alpha);
        stepForward(instance->getCudaSurfVertPos(), instance->getCudaTempScalar3Mem(), instance->getCudaMoveDir(),
                    instance->getCudaBoundaryType(), alpha, false, instance->getHostNumVertices());
        LBVH_CD_ptr->buildBVH(instance);
    }

    // 然后检查能量是否有正确下降
    LBVH_CD_ptr->buildCP(instance);
    LBVH_CD_ptr->buildGP(instance);
    int numOfLineSearch = 0;
    Scalar LFStepSize = alpha;
    Scalar currEnergyVal = computeEnergy(instance, PCG_ptr);
    while (currEnergyVal > lastEnergyVal && alpha > 1e-3 * LFStepSize) {
        alpha /= 2.0;
        ++numOfLineSearch;
        stepForward(instance->getCudaSurfVertPos(), instance->getCudaTempScalar3Mem(), instance->getCudaMoveDir(),
                    instance->getCudaBoundaryType(), alpha, false, instance->getHostNumVertices());
        LBVH_CD_ptr->buildBVH(instance);
        LBVH_CD_ptr->buildCP(instance);
        LBVH_CD_ptr->buildGP(instance);
        currEnergyVal = computeEnergy(instance, PCG_ptr);
    }
    if (numOfLineSearch > 8)
        printf("type 2 energy not drop down correctly!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");

    // 如果能量没有正确下降 且步长被进一步缩小 就重新检测一次碰撞集 保证无穿透
    if (alpha < LFStepSize) {
        bool needRecomputeCS = false;
        while (checkInterset && __GPUIPC__::isIntersected(instance, LBVH_CD_ptr)) {
            printf("type 3 intersection happened!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!:  %d\n", insectNum);
            insectNum++;
            alpha /= 2.0;
            numOfIntersect++;
            alpha = __MATHUTILS__::__m_min(cfl_alpha, alpha);
            stepForward(instance->getCudaSurfVertPos(), instance->getCudaTempScalar3Mem(), instance->getCudaMoveDir(),
                        instance->getCudaBoundaryType(), alpha, false, instance->getHostNumVertices());
            LBVH_CD_ptr->buildBVH(instance);
            needRecomputeCS = true;
        }
        if (needRecomputeCS) {
            LBVH_CD_ptr->buildCP(instance);
            LBVH_CD_ptr->buildGP(instance);
        }
    }

    return true;
}

void postLineSearch(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr,
                    std::unique_ptr<PCGSolver>& PCG_ptr) {
    if (instance->getHostKappa() == 0.0) {
        __GPUIPC__::initKappa(instance, BH_ptr, PCG_ptr);
    } else {
        bool updateKappa_closeground = __GPUIPC__::checkCloseGroundVal(instance);
        bool updateKappa_selfclose = __GPUIPC__::checkSelfCloseVal(instance);
        bool updateKappa = updateKappa_closeground || updateKappa_selfclose;
        if (updateKappa) {
            instance->getHostKappa() *= 2.0;
            __GPUIPC__::upperBoundKappa(instance, instance->getHostKappa());
        }
        tempFree_closeConstraint(instance);
        tempMalloc_closeConstraint(instance);
        CUDA_SAFE_CALL(cudaMemset(instance->getCudaCloseCPNum(), 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(instance->getCudaCloseGPNum(), 0, sizeof(uint32_t)));
        __GPUIPC__::computeCloseGroundVal(instance);
        __GPUIPC__::computeSelfCloseVal(instance);
    }
}




__global__ void _calculate_BoundaryStopCriterion(
    const Scalar3* vertexes, const uint32_t* targetInd,
    const Scalar3* targetVert_prev, const Scalar3* targetVert, 
    Scalar2 *stopdist, int numbers) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    uint32_t vInd = targetInd[idx];
    Scalar3 vertpos = vertexes[vInd];
    Scalar3 targetpos_prev = targetVert_prev[idx];
    Scalar3 targetpos = targetVert[idx];

    // printf("vertpos: %f, %f, %f \n", vertpos.x, vertpos.y, vertpos.z);
    // printf("targetpos_prev: %f, %f, %f \n", targetpos_prev.x, targetpos_prev.y, targetpos_prev.z);
    // printf("targetpos: %f, %f, %f \n", targetpos.x, targetpos.y, targetpos.z);

    Scalar3 diff_current = __MATHUTILS__::__vec3_minus(targetpos, vertpos);
    Scalar3 diff_previous = __MATHUTILS__::__vec3_minus(targetpos, targetpos_prev);
    
    Scalar dist_current = __MATHUTILS__::__vec3_dot(diff_current, diff_current);  // ||x_{k}^{t+1} - x_{k}^i||^2
    Scalar dist_previous = __MATHUTILS__::__vec3_dot(diff_previous, diff_previous);  // ||x_{k}^{t+1} - x_{k}^t||^2

    stopdist[idx].x = dist_current; // Σ ||x_{k}^{t+1}-x_{k}^i||^2
    stopdist[idx].y = dist_previous; // Σ ||x_{k}^{t+1}-x_{k}^t||^2

}


void calculate_BoundaryStopCriterion(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostNumBoundTargets();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    cudaMemset(instance->getCudaBoundaryStopDist(), 0, numbers * sizeof(Scalar2));

    _calculate_BoundaryStopCriterion<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), 
        instance->getCudaBoundTargetIndex(),
        instance->getCudaPrevBoundTargetVertPos(), 
        instance->getCudaBoundTargetVertPos(),
        instance->getCudaBoundaryStopDist(), 
        numbers
    );

    unsigned int sharedMsize = sizeof(Scalar2) * (threadNum >> 5);
    
    __MATHUTILS__::__reduct_add_Scalar2<<<blockNum, threadNum, sharedMsize>>>(instance->getCudaBoundaryStopDist(), numbers);

    Scalar2 h_stopcriterion;
    cudaMemcpy(&h_stopcriterion, instance->getCudaBoundaryStopDist(), sizeof(Scalar2), cudaMemcpyDeviceToHost);

    Scalar etaA = 1.0 - std::sqrt(h_stopcriterion.x / h_stopcriterion.y);
    instance->getHostBoundaryStopCriterion() = etaA;

}









int solve_subIP(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr, std::unique_ptr<PCGSolver>& PCG_ptr,
                std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr) {
    int iterCap = 10000;
    int iter_k = 0;

    int total_CollisionPairs = 0;
    int total_Cg_count = 0;

    CUDA_SAFE_CALL(cudaMemset(instance->getCudaMoveDir(), 0, instance->getHostNumVertices() * sizeof(Scalar3)));

    Scalar totalTimeStep = 0;
    for (; iter_k < iterCap; ++iter_k) {
        total_CollisionPairs += instance->getHostCpNum(0);

        // 更新local Hessian索引
        BH_ptr->updateBHDNum(instance->getHostNumTriElements(),  // tri_Num
                          instance->getHostNumTriBendEdges(), // tri_Bend
                          instance->getHostNumTetElements(),  // tet_number
                          instance->getHostNumStitchPairs(), // stitch number
                          instance->getHostCpNum(2),          // cpNum1 H6x6 P-P
                          instance->getHostCpNum(3),          // cpNum2 H9x9 P-T
                          instance->getHostCpNum(4),          // cpNum3 H12x12 E-E
                          instance->getHostCpNumLast(2),      // last_cpNum1
                          instance->getHostCpNumLast(3),      // last_cpNum2
                          instance->getHostCpNumLast(4)       // last_cpNum3
        );

        // 更新gradient和Hessian
        computeGradientAndHessian(instance, BH_ptr);

        // 检测最小的dx获得收敛条件
        Scalar distToOpt_PN = calcMinMovement(instance->getCudaMoveDir(), PCG_ptr->cudaPCGSqueue, instance->getHostNumVertices());
        Scalar convergeValue = instance->getHostNewtonSolverThreshold() * instance->getHostIPCDt() // vel_error * dt = dis_error
                                * sqrt(instance->getHostBboxDiagSize2()); // map error to mesh size
        std::cout << "distToOpt_PN: " << distToOpt_PN << " convergeValue: " << convergeValue << std::endl;
        bool gradVanish = (distToOpt_PN < convergeValue);
        if (iter_k && gradVanish) break;

        // 通过PCG获得前进方向dx
        total_Cg_count += calculateMovingDirection(instance, BH_ptr, PCG_ptr, instance->getHostCpNum(0), PCG_ptr->PrecondType);
        
        // line search获得前进步长alpha
        Scalar alpha = 1.0, slackness_a = 0.8, slackness_m = 0.8;
        // 遍历所有的地面顶点 计算每个顶点前行的最大可行步长 这一步会非常卡ccd
        alpha = __MATHUTILS__::__m_min(
            alpha, __GPUIPC__::ground_largestFeasibleStepSize(instance, slackness_a, PCG_ptr->cudaPCGSqueue));
        alpha = __MATHUTILS__::__m_min(
            alpha, __GPUIPC__::self_largestFeasibleStepSize(instance, slackness_m, PCG_ptr->cudaPCGSqueue, instance->getHostCpNum(0)));

        Scalar temp_alpha = alpha;
        Scalar alpha_CFL = alpha;
        // std::cout << "alpha after CCD: " << alpha << " " << alpha_CFL << std::endl;

        Scalar ccd_size = 1.0;
        
#ifdef USE_GIPCFRICTION
        ccd_size = 0.8;
#endif

        // 通过最大速度来计算一次filter line search 通常这一步会非常卡ccd
        LBVH_CD_ptr->buildBVH_FULLCCD(instance, temp_alpha);
        LBVH_CD_ptr->buildFullCP(instance, temp_alpha); // 获取所有的ccd pairs，通常getHostCcdCpNum>>getHostCpNum
        if (instance->getHostCcdCpNum() > 0) {
            Scalar maxSpeed = __GPUIPC__::cfl_largestSpeed(instance, PCG_ptr->cudaPCGSqueue);
            alpha_CFL = sqrt(instance->getHostDHat()) / maxSpeed * 0.5;
            alpha = __MATHUTILS__::__m_min(alpha, alpha_CFL);
            if (temp_alpha > 2 * alpha_CFL) {
                alpha = __MATHUTILS__::__m_min(
                    temp_alpha, __GPUIPC__::self_largestFeasibleStepSize(instance, slackness_m, PCG_ptr->cudaPCGSqueue,
                                                                         instance->getHostCcdCpNum()) * ccd_size);
                alpha = __MATHUTILS__::__m_max(alpha, alpha_CFL);
            }
        }
        // std::cout << "alpha after velocity check: " << alpha << " " << alpha_CFL << std::endl;

        // 检测能量下降 通常这一步不应影响ccd
        lineSearch(instance, PCG_ptr, LBVH_CD_ptr, alpha, alpha_CFL);
        std::cout << "alpha after linesearch: " << alpha << std::endl;

        // 重新计算一下kappa
        postLineSearch(instance, BH_ptr, PCG_ptr);

        // CUDA_SAFE_CALL(cudaDeviceSynchronize());

        totalTimeStep += alpha;

    }

    printf("\n\n\n       Kappa: %f         iteration k:  %d\n", instance->getHostKappa(), iter_k);
    printf("       CollPairs: %d         CGCounts:  %d\n\n\n", total_CollisionPairs, total_Cg_count);

    return iter_k;
}

Scalar2 minMaxSelfDist(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostCpNum(0);
    if (numbers < 1) return make_Scalar2(1e32, 0);
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar2) * (threadNum >> 5);

    Scalar2* _queue;
    CUDAMallocSafe(_queue, numbers);
    // CUDA_SAFE_CALL(cudaMemcpy(_tempMinMovement, _moveDir, number *
    // sizeof(AABB), cudaMemcpyDeviceToDevice));
    _reduct_MSelfDist<<<blockNum, threadNum, sharedMsize>>>(instance->getCudaSurfVertPos(),
                                                            instance->getCudaCollisionPairs(), _queue, numbers);
    //_reduct_min_Scalar3_to_Scalar << <blockNum, threadNum, sharedMsize >> >
    //(_moveDir, _tempMinMovement, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> >
        //(_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar2<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB),
    // cudaMemcpyDeviceToDevice);
    Scalar2 minValue;
    cudaMemcpy(&minValue, _queue, sizeof(Scalar2), cudaMemcpyDeviceToHost);
    CUDAFreeSafe(_queue);
    minValue.x = 1.0 / minValue.x;
    return minValue;
}




void IPC_Solver(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr, std::unique_ptr<PCGSolver>& PCG_ptr,
                std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr) {


    int totalNT = 0;
    Scalar totalTime = 0;

    cudaEvent_t start, end0;
    cudaEventCreate(&start);
    cudaEventCreate(&end0);
    cudaEventRecord(start);

    __GPUIPC__::upperBoundKappa(instance, instance->getHostKappa());
    if (instance->getHostKappa() < very_small_number()) {
        __GPUIPC__::suggestKappa(instance, instance->getHostKappa());
    }
    __GPUIPC__::initKappa(instance, BH_ptr, PCG_ptr);

#ifdef USE_GIPCFRICTION
    CUDAMallocSafe(instance->getCudaLambdaLastHScalar(), instance->getHostCpNum(0));
    CUDAMallocSafe(instance->getCudaDistCoord(), instance->getHostCpNum(0));
    CUDAMallocSafe(instance->getCudaTanBasis(), instance->getHostCpNum(0));
    CUDAMallocSafe(instance->getCudaCollisionPairsLastH(), instance->getHostCpNum(0));
    CUDAMallocSafe(instance->getCudaMatIndexLast(), instance->getHostCpNum(0));
    CUDAMallocSafe(instance->getCudaLambdaLastHScalarGd(), instance->getHostGpNum());
    CUDAMallocSafe(instance->getCudaCollisionPairsLastHGd(), instance->getHostGpNum());
    __GIPCFRICTION__::buildFrictionSets(instance);
#endif

    instance->getHostAnimationFullRate() = instance->getHostAnimationSubRate();


    while (true) {

        tempMalloc_closeConstraint(instance);
        CUDA_SAFE_CALL(cudaMemset(instance->getCudaCloseCPNum(), 0, sizeof(uint32_t)));
        CUDA_SAFE_CALL(cudaMemset(instance->getCudaCloseGPNum(), 0, sizeof(uint32_t)));

        totalNT += solve_subIP(instance, BH_ptr, PCG_ptr, LBVH_CD_ptr);

        Scalar2 minMaxDist1 = minMaxGroundDist(instance);
        Scalar2 minMaxDist2 = minMaxSelfDist(instance);

        Scalar minDist = __MATHUTILS__::__m_min(minMaxDist1.x, minMaxDist2.x); // 所有的ground+self的最小距离
        Scalar maxDist = __MATHUTILS__::__m_max(minMaxDist1.y, minMaxDist2.y); // 所有的ground+self的最大距离

        bool finishMotion = instance->getHostAnimationFullRate() > 0.99 ? true : false;

        if (finishMotion) {
            if ((instance->getHostCpNum(0) + instance->getHostGpNum()) > 0) {
                if (minDist < instance->getHostDTol()) {
                    std::cout << "the frame break in < dtol: " << minDist 
                            << " fullrate: " << instance->getHostAnimationFullRate() 
                            << std::endl;
                    tempFree_closeConstraint(instance); // 如果ground+self距离太小可能发生穿透就退出
                    break;
                } else if (maxDist < instance->getHostDHat()) {
                    std::cout << "the frame break in < dhat: " << maxDist 
                            << " fullrate: " << instance->getHostAnimationFullRate() 
                            << std::endl;
                    tempFree_closeConstraint(instance); // 如果ground+self最大距离小于dhat也可以退出
                    break;
                } else {
                    tempFree_closeConstraint(instance);
                }
            } else {
                tempFree_closeConstraint(instance);
                break;
            }
        } else {
            tempFree_closeConstraint(instance);
        }

        instance->getHostAnimationFullRate() += instance->getHostAnimationSubRate();


#ifdef USE_GIPCFRICTION
        CUDAFreeSafe(instance->getCudaLambdaLastHScalar());
        CUDAFreeSafe(instance->getCudaDistCoord());
        CUDAFreeSafe(instance->getCudaTanBasis());
        CUDAFreeSafe(instance->getCudaCollisionPairsLastH());
        CUDAFreeSafe(instance->getCudaMatIndexLast());
        CUDAFreeSafe(instance->getCudaLambdaLastHScalarGd());
        CUDAFreeSafe(instance->getCudaCollisionPairsLastHGd());

        CUDAMallocSafe(instance->getCudaLambdaLastHScalar(), instance->getHostCpNum(0));
        CUDAMallocSafe(instance->getCudaDistCoord(), instance->getHostCpNum(0));
        CUDAMallocSafe(instance->getCudaTanBasis(), instance->getHostCpNum(0));
        CUDAMallocSafe(instance->getCudaCollisionPairsLastH(), instance->getHostCpNum(0));
        CUDAMallocSafe(instance->getCudaMatIndexLast(), instance->getHostCpNum(0));
        CUDAMallocSafe(instance->getCudaLambdaLastHScalarGd(), instance->getHostGpNum());
        CUDAMallocSafe(instance->getCudaCollisionPairsLastHGd(), instance->getHostGpNum());

        __GIPCFRICTION__::buildFrictionSets(instance);
#endif
    }

#ifdef USE_GIPCFRICTION
    CUDAFreeSafe(instance->getCudaLambdaLastHScalar());
    CUDAFreeSafe(instance->getCudaDistCoord());
    CUDAFreeSafe(instance->getCudaTanBasis());
    CUDAFreeSafe(instance->getCudaCollisionPairsLastH());
    CUDAFreeSafe(instance->getCudaMatIndexLast());
    CUDAFreeSafe(instance->getCudaLambdaLastHScalarGd());
    CUDAFreeSafe(instance->getCudaCollisionPairsLastHGd());
#endif

    updateVelocities(instance);
    computeXTilta(instance);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());


    cudaEventRecord(end0);
    float tttime;
    cudaEventElapsedTime(&tttime, start, end0);
    totalTime += tttime;
    printf("\n\n\n\n\n\n   one Frame finished  average time cost: %f,  frame id: %d \n\n\n\n\n\n", totalTime / totalNT, instance->getHostSimulationFrameId());

}

};  // namespace __INTEGRATOR__
