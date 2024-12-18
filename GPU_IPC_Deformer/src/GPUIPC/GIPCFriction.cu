
#include "GIPCFriction.cuh"

namespace __GIPCFRICTION__ {


__device__ Scalar __cal_Friction_energy(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                        int4 MMCVIDI, Scalar dt, Scalar2 distCoord,
                                        __MATHUTILS__::Matrix3x2S tanBasis, Scalar lastH,
                                        Scalar fricDHat, Scalar eps) {
    Scalar3 relDX3D;

    if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        __GIPCFRICUTILS__::computeRelDX_EE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), 
            distCoord.x, distCoord.y, relDX3D);
    } 
    
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z == -1 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PP(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), 
            relDX3D);

    }
    
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            distCoord.x, relDX3D);

    } 
    
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PT(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), 
            distCoord.x, distCoord.y, relDX3D);
    }

    else if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.w = -MMCVIDI.w - 1;
        __GIPCFRICUTILS__::computeRelDX_EE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), 
            distCoord.x, distCoord.y, relDX3D);

    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z < 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.z = -MMCVIDI.z - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        __GIPCFRICUTILS__::computeRelDX_PP(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), 
            relDX3D);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        __GIPCFRICUTILS__::computeRelDX_PE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            distCoord.x, relDX3D);
    }

    else {
        printf("################################ Error: invalid collision conditions");
        // exit(EXIT_FAILURE);
    }



    __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis);
    Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(__MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D));
    if (relDXSqNorm > fricDHat) { // ||uk||^2 > fricDHat
        return lastH * sqrt(relDXSqNorm); // lambda_k * ||uk||
    } else {
        Scalar f0;
        __GIPCFRICUTILS__::f0_SF(relDXSqNorm, eps, f0); // eps = eps_v * h
        return lastH * f0;
    }
}

__device__ Scalar __cal_Friction_gd_energy(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                           const Scalar3* _normal, uint32_t gidx, Scalar dt,
                                           Scalar lastH, Scalar eps) {
    Scalar3 normal = *_normal;
    Scalar3 Vdiff = __MATHUTILS__::__vec3_minus(_vertexes[gidx], _o_vertexes[gidx]);
    Scalar3 VProj = __MATHUTILS__::__vec3_minus(
        Vdiff, __MATHUTILS__::__s_vec3_multiply(normal, __MATHUTILS__::__vec3_dot(Vdiff, normal)));
    Scalar VProjMag2 = __MATHUTILS__::__vec3_squaredNorm(VProj);
    if (VProjMag2 > eps * eps) {
        return lastH * (sqrt(VProjMag2) - eps * 0.5);

    } else {
        return lastH * VProjMag2 / eps * 0.5;
    }
}



__global__ void _calFrictionGradient(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                     const int4* _last_collisionPair, Scalar3* _gradient,
                                     int numbers, Scalar dt, Scalar2* distCoord,
                                     __MATHUTILS__::Matrix3x2S* tanBasis, Scalar eps2,
                                     Scalar* lastH, Scalar coef) {

    Scalar eps = std::sqrt(eps2);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return; // numbers = cpNumLast
    int4 MMCVIDI = _last_collisionPair[idx];

    Scalar3 relDX3D;

    if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        __GIPCFRICUTILS__::computeRelDX_EE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
            distCoord[idx].y, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, f1_div_relDXNorm);
        }
        __MATHUTILS__::Vector12S TTTDX;
        __GIPCFRICUTILS__::liftRelDXTanToMesh_EE(relDX, tanBasis[idx], distCoord[idx].x,
                                               distCoord[idx].y, TTTDX);
        TTTDX = __MATHUTILS__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z == -1 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PP(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, f1_div_relDXNorm);
        }

        __MATHUTILS__::Vector6S TTTDX;
        __GIPCFRICUTILS__::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
        TTTDX = __MATHUTILS__::__s_vec6_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
        }
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            distCoord[idx].x, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, f1_div_relDXNorm);
        }
        __MATHUTILS__::Vector9S TTTDX;
        __GIPCFRICUTILS__::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
        TTTDX = __MATHUTILS__::__s_vec9_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
        }
    
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PT(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x, distCoord[idx].y, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);

        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, f1_div_relDXNorm);
        }
        __MATHUTILS__::Vector12S TTTDX;
        __GIPCFRICUTILS__::liftRelDXTanToMesh_PT(relDX, tanBasis[idx], distCoord[idx].x,
                                                distCoord[idx].y, TTTDX);
        TTTDX = __MATHUTILS__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);

        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }

    }

    else if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.w = -MMCVIDI.w - 1;
        __GIPCFRICUTILS__::computeRelDX_EE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
            distCoord[idx].y, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, f1_div_relDXNorm);
        }
        __MATHUTILS__::Vector12S TTTDX;
        __GIPCFRICUTILS__::liftRelDXTanToMesh_EE(relDX, tanBasis[idx], distCoord[idx].x,
                                               distCoord[idx].y, TTTDX);
        TTTDX = __MATHUTILS__::__s_vec12_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
            atomicAdd(&(_gradient[MMCVIDI.w].x), TTTDX.v[9]);
            atomicAdd(&(_gradient[MMCVIDI.w].y), TTTDX.v[10]);
            atomicAdd(&(_gradient[MMCVIDI.w].z), TTTDX.v[11]);
        }
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z < 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.z = -MMCVIDI.z - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        __GIPCFRICUTILS__::computeRelDX_PP(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, f1_div_relDXNorm);
        }

        __MATHUTILS__::Vector6S TTTDX;
        __GIPCFRICUTILS__::liftRelDXTanToMesh_PP(relDX, tanBasis[idx], TTTDX);
        TTTDX = __MATHUTILS__::__s_vec6_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
        }
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;

        __GIPCFRICUTILS__::computeRelDX_PE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            distCoord[idx].x, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        if (relDXSqNorm > eps2) {
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, 1.0 / sqrt(relDXSqNorm));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            relDX = __MATHUTILS__::__s_vec2_multiply(relDX, f1_div_relDXNorm);
        }
        __MATHUTILS__::Vector9S TTTDX;
        __GIPCFRICUTILS__::liftRelDXTanToMesh_PE(relDX, tanBasis[idx], distCoord[idx].x, TTTDX);
        TTTDX = __MATHUTILS__::__s_vec9_multiply(TTTDX, lastH[idx] * coef);
        {
            atomicAdd(&(_gradient[MMCVIDI.x].x), TTTDX.v[0]);
            atomicAdd(&(_gradient[MMCVIDI.x].y), TTTDX.v[1]);
            atomicAdd(&(_gradient[MMCVIDI.x].z), TTTDX.v[2]);
            atomicAdd(&(_gradient[MMCVIDI.y].x), TTTDX.v[3]);
            atomicAdd(&(_gradient[MMCVIDI.y].y), TTTDX.v[4]);
            atomicAdd(&(_gradient[MMCVIDI.y].z), TTTDX.v[5]);
            atomicAdd(&(_gradient[MMCVIDI.z].x), TTTDX.v[6]);
            atomicAdd(&(_gradient[MMCVIDI.z].y), TTTDX.v[7]);
            atomicAdd(&(_gradient[MMCVIDI.z].z), TTTDX.v[8]);
        }
    
    }

    else {
        printf("################################ Error: invalid collision conditions");
        // exit(EXIT_FAILURE);
    }

}


__global__ void _calFrictionGradient_gd(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                        const Scalar3* _normal,
                                        const uint32_t* _last_collisionPair_gd,
                                        Scalar3* _gradient, int numbers, Scalar dt, Scalar eps2,
                                        Scalar* lastH, Scalar coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar eps = sqrt(eps2);
    Scalar3 normal = *_normal;
    uint32_t gidx = _last_collisionPair_gd[idx];
    Scalar3 Vdiff = __MATHUTILS__::__vec3_minus(_vertexes[gidx], _o_vertexes[gidx]);
    Scalar3 VProj = __MATHUTILS__::__vec3_minus(
        Vdiff, __MATHUTILS__::__s_vec3_multiply(normal, __MATHUTILS__::__vec3_dot(Vdiff, normal)));
    Scalar VProjMag2 = __MATHUTILS__::__vec3_squaredNorm(VProj);
    if (VProjMag2 > eps2) {
        Scalar3 gdf = __MATHUTILS__::__s_vec3_multiply(VProj, coef * lastH[idx] / sqrt(VProjMag2));
        /*atomicAdd(&(_gradient[gidx].x), gdf.x);
        atomicAdd(&(_gradient[gidx].y), gdf.y);
        atomicAdd(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = __MATHUTILS__::__vec3_add(_gradient[gidx], gdf);
    } else {
        Scalar3 gdf = __MATHUTILS__::__s_vec3_multiply(VProj, coef * lastH[idx] / eps);
        /*atomicAdd(&(_gradient[gidx].x), gdf.x);
        atomicAdd(&(_gradient[gidx].y), gdf.y);
        atomicAdd(&(_gradient[gidx].z), gdf.z);*/
        _gradient[gidx] = __MATHUTILS__::__vec3_add(_gradient[gidx], gdf);
    }
}




__global__ void _calFrictionHessian(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                    const int4* _last_collisionPair,
                                    __MATHUTILS__::Matrix12x12S* H12x12,
                                    __MATHUTILS__::Matrix9x9S* H9x9,
                                    __MATHUTILS__::Matrix6x6S* H6x6, uint4* D4Index, uint3* D3Index,
                                    uint2* D2Index, uint32_t* _cpNum, int numbers, Scalar dt,
                                    Scalar2* distCoord, __MATHUTILS__::Matrix3x2S* tanBasis,
                                    Scalar eps2, Scalar* lastH, Scalar coef, int offset4,
                                    int offset3, int offset2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _last_collisionPair[idx];
    Scalar eps = sqrt(eps2);
    Scalar3 relDX3D;

    // EE friction
    if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        __GIPCFRICUTILS__::computeRelDX_EE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
            distCoord[idx].y, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        Scalar relDXNorm = sqrt(relDXSqNorm);
        __MATHUTILS__::Matrix12x2S T;
        __GIPCFRICUTILS__::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __MATHUTILS__::Matrix2x2S M2;
        if (relDXSqNorm > eps2) {
            __MATHUTILS__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __MATHUTILS__::__Mat2x2_minus(
                M2,
                __MATHUTILS__::__s_Mat2x2_multiply(__MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                                                   1 / (relDXSqNorm * relDXNorm)));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            Scalar f2;
            __GIPCFRICUTILS__::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
            } else {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                   eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::Matrix12x2S TM2 = __MATHUTILS__::__M12x2_M2x2_Multiply(T, projH);

        __MATHUTILS__::Matrix12x12S HessianBlock =
            __MATHUTILS__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T), coef * lastH[idx]);
        int Hidx = atomicAdd(_cpNum + 4, 1);
        Hidx += offset4;
        H12x12[Hidx] = HessianBlock;
        D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
    }

    // PP Friction
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z == -1 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PP(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        Scalar relDXNorm = sqrt(relDXSqNorm);
        __MATHUTILS__::Matrix6x2S T;
        __GIPCFRICUTILS__::computeT_PP(tanBasis[idx], T);
        __MATHUTILS__::Matrix2x2S M2;
        if (relDXSqNorm > eps2) {
            __MATHUTILS__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __MATHUTILS__::__Mat2x2_minus(
                M2, __MATHUTILS__::__s_Mat2x2_multiply(
                        __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                        1 / (relDXSqNorm * relDXNorm)));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            Scalar f2;
            __GIPCFRICUTILS__::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
            } else {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }
        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                    eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::Matrix6x2S TM2 = __MATHUTILS__::__M6x2_M2x2_Multiply(T, projH);

        __MATHUTILS__::Matrix6x6S HessianBlock =
            __MATHUTILS__::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T), coef * lastH[idx]);

        int Hidx = atomicAdd(_cpNum + 2, 1);
        Hidx += offset2;
        H6x6[Hidx] = HessianBlock;
        D2Index[Hidx] = make_uint2(MMCVIDI.x, MMCVIDI.y);
    }

    // PE Friction
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            distCoord[idx].x, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        Scalar relDXNorm = sqrt(relDXSqNorm);
        __MATHUTILS__::Matrix9x2S T;
        __GIPCFRICUTILS__::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
        __MATHUTILS__::Matrix2x2S M2;
        if (relDXSqNorm > eps2) {
            __MATHUTILS__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __MATHUTILS__::__Mat2x2_minus(
                M2, __MATHUTILS__::__s_Mat2x2_multiply(
                        __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                        1 / (relDXSqNorm * relDXNorm)));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            Scalar f2;
            __GIPCFRICUTILS__::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
            } else {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }
        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                    eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::Matrix9x2S TM2 = __MATHUTILS__::__M9x2_M2x2_Multiply(T, projH);

        __MATHUTILS__::Matrix9x9S HessianBlock =
            __MATHUTILS__::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T), coef * lastH[idx]);
        int Hidx = atomicAdd(_cpNum + 3, 1);
        Hidx += offset3;
        H9x9[Hidx] = HessianBlock;
        D3Index[Hidx] = make_uint3(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z);
    }

    // PT Friction
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        __GIPCFRICUTILS__::computeRelDX_PT(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]),
            distCoord[idx].x, distCoord[idx].y, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        Scalar relDXNorm = sqrt(relDXSqNorm);
        __MATHUTILS__::Matrix12x2S T;
        __GIPCFRICUTILS__::computeT_PT(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __MATHUTILS__::Matrix2x2S M2;
        if (relDXSqNorm > eps2) {
            __MATHUTILS__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __MATHUTILS__::__Mat2x2_minus(
                M2, __MATHUTILS__::__s_Mat2x2_multiply(
                        __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                        1 / (relDXSqNorm * relDXNorm)));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            Scalar f2;
            __GIPCFRICUTILS__::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
            } else {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }
        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                    eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::Matrix12x2S TM2 = __MATHUTILS__::__M12x2_M2x2_Multiply(T, projH);

        __MATHUTILS__::Matrix12x12S HessianBlock = __MATHUTILS__::__s_M12x12_Multiply(
            __M12x2_M12x2T_Multiply(TM2, T), coef * lastH[idx]);
        int Hidx = atomicAdd(_cpNum + 4, 1);
        Hidx += offset4;
        H12x12[Hidx] = HessianBlock;
        D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);

    }

    // Parallel EE Friction
    else if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.w = -MMCVIDI.w - 1;
        __GIPCFRICUTILS__::computeRelDX_EE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _o_vertexes[MMCVIDI.w]), distCoord[idx].x,
            distCoord[idx].y, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        Scalar relDXNorm = sqrt(relDXSqNorm);
        __MATHUTILS__::Matrix12x2S T;
        __GIPCFRICUTILS__::computeT_EE(tanBasis[idx], distCoord[idx].x, distCoord[idx].y, T);
        __MATHUTILS__::Matrix2x2S M2;
        if (relDXSqNorm > eps2) {
            __MATHUTILS__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __MATHUTILS__::__Mat2x2_minus(
                M2,
                __MATHUTILS__::__s_Mat2x2_multiply(__MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                                                   1 / (relDXSqNorm * relDXNorm)));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            Scalar f2;
            __GIPCFRICUTILS__::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
            } else {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }

        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                   eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::Matrix12x2S TM2 = __MATHUTILS__::__M12x2_M2x2_Multiply(T, projH);

        __MATHUTILS__::Matrix12x12S HessianBlock =
            __MATHUTILS__::__s_M12x12_Multiply(__M12x2_M12x2T_Multiply(TM2, T), coef * lastH[idx]);
        int Hidx = atomicAdd(_cpNum + 4, 1);
        Hidx += offset4;
        H12x12[Hidx] = HessianBlock;
        D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
    }

    // Parallel PP Friction
    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z < 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.z = -MMCVIDI.z - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        __GIPCFRICUTILS__::computeRelDX_PP(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]), relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        Scalar relDXNorm = sqrt(relDXSqNorm);
        __MATHUTILS__::Matrix6x2S T;
        __GIPCFRICUTILS__::computeT_PP(tanBasis[idx], T);
        __MATHUTILS__::Matrix2x2S M2;
        if (relDXSqNorm > eps2) {
            __MATHUTILS__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __MATHUTILS__::__Mat2x2_minus(
                M2, __MATHUTILS__::__s_Mat2x2_multiply(
                        __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                        1 / (relDXSqNorm * relDXNorm)));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            Scalar f2;
            __GIPCFRICUTILS__::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
            } else {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }
        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                    eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::Matrix6x2S TM2 = __MATHUTILS__::__M6x2_M2x2_Multiply(T, projH);

        __MATHUTILS__::Matrix6x6S HessianBlock =
            __MATHUTILS__::__s_M6x6_Multiply(__M6x2_M6x2T_Multiply(TM2, T), coef * lastH[idx]);

        int Hidx = atomicAdd(_cpNum + 2, 1);
        Hidx += offset2;
        H6x6[Hidx] = HessianBlock;
        D2Index[Hidx] = make_uint2(MMCVIDI.x, MMCVIDI.y);
    }

    // parallel PE Friction
    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        __GIPCFRICUTILS__::computeRelDX_PE(
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.x], _o_vertexes[MMCVIDI.x]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _o_vertexes[MMCVIDI.y]),
            __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _o_vertexes[MMCVIDI.z]),
            distCoord[idx].x, relDX3D);

        __MATHUTILS__::Matrix2x3S tB_T = __MATHUTILS__::__Transpose3x2(tanBasis[idx]);
        Scalar2 relDX = __MATHUTILS__::__M2x3_v3_multiply(tB_T, relDX3D);
        Scalar relDXSqNorm = __MATHUTILS__::__vec2_squaredNorm(relDX);
        Scalar relDXNorm = sqrt(relDXSqNorm);
        __MATHUTILS__::Matrix9x2S T;
        __GIPCFRICUTILS__::computeT_PE(tanBasis[idx], distCoord[idx].x, T);
        __MATHUTILS__::Matrix2x2S M2;
        if (relDXSqNorm > eps2) {
            __MATHUTILS__::__set_Mat_identity(M2);
            M2.m[0][0] /= relDXNorm;
            M2.m[1][1] /= relDXNorm;
            M2 = __MATHUTILS__::__Mat2x2_minus(
                M2, __MATHUTILS__::__s_Mat2x2_multiply(
                        __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                        1 / (relDXSqNorm * relDXNorm)));
        } else {
            Scalar f1_div_relDXNorm;
            __GIPCFRICUTILS__::f1_SF_div_relDXNorm(relDXSqNorm, eps, f1_div_relDXNorm);
            Scalar f2;
            __GIPCFRICUTILS__::f2_SF(relDXSqNorm, eps, f2);
            if (f2 != f1_div_relDXNorm && relDXSqNorm) {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
                M2 = __MATHUTILS__::__Mat2x2_minus(
                    M2, __MATHUTILS__::__s_Mat2x2_multiply(
                            __MATHUTILS__::__v2_vec2_toMat2x2(relDX, relDX),
                            (f1_div_relDXNorm - f2) / relDXSqNorm));
            } else {
                __MATHUTILS__::__set_Mat_identity(M2);
                M2.m[0][0] *= f1_div_relDXNorm;
                M2.m[1][1] *= f1_div_relDXNorm;
            }
        }
        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(M2.m[0][0], M2.m[0][1], M2.m[1][0], M2.m[1][1], eigenValues,
                                    eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::Matrix9x2S TM2 = __MATHUTILS__::__M9x2_M2x2_Multiply(T, projH);

        __MATHUTILS__::Matrix9x9S HessianBlock =
            __MATHUTILS__::__s_M9x9_Multiply(__M9x2_M9x2T_Multiply(TM2, T), coef * lastH[idx]);
        int Hidx = atomicAdd(_cpNum + 3, 1);
        Hidx += offset3;
        H9x9[Hidx] = HessianBlock;
        D3Index[Hidx] = make_uint3(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z);
    }

    else {
        printf("################################ Error: invalid collision conditions");
        // exit(EXIT_FAILURE);
    }
}



__global__ void _calFrictionHessian_gd(const Scalar3* _vertexes, const Scalar3* _o_vertexes,
                                       const Scalar3* _normal,
                                       const uint32_t* _last_collisionPair_gd,
                                       __MATHUTILS__::Matrix3x3S* H3x3, uint32_t* D1Index,
                                       int numbers, Scalar dt, Scalar eps2, Scalar* lastH,
                                       Scalar coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar eps = sqrt(eps2);
    int gidx = _last_collisionPair_gd[idx];
    Scalar multiplier_vI = coef * lastH[idx];
    __MATHUTILS__::Matrix3x3S H_vI;

    Scalar3 Vdiff = __MATHUTILS__::__vec3_minus(_vertexes[gidx], _o_vertexes[gidx]);
    Scalar3 normal = *_normal;
    Scalar3 VProj = __MATHUTILS__::__vec3_minus(
        Vdiff, __MATHUTILS__::__s_vec3_multiply(normal, __MATHUTILS__::__vec3_dot(Vdiff, normal)));
    Scalar VProjMag2 = __MATHUTILS__::__vec3_squaredNorm(VProj);

    if (VProjMag2 > eps2) {
        Scalar VProjMag = sqrt(VProjMag2);

        __MATHUTILS__::Matrix2x2S projH;
        __MATHUTILS__::__set_Mat2x2_val_column(projH, make_Scalar2(0, 0), make_Scalar2(0, 0));

        Scalar eigenValues[2];
        int eigenNum = 0;
        Scalar2 eigenVecs[2];
        __MATHUTILS__::__makePD2x2(
            VProj.x * VProj.x * -multiplier_vI / VProjMag2 / VProjMag + (multiplier_vI / VProjMag),
            VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
            VProj.x * VProj.z * -multiplier_vI / VProjMag2 / VProjMag,
            VProj.z * VProj.z * -multiplier_vI / VProjMag2 / VProjMag + (multiplier_vI / VProjMag),
            eigenValues, eigenNum, eigenVecs);
        for (int i = 0; i < eigenNum; i++) {
            if (eigenValues[i] > 0) {
                __MATHUTILS__::Matrix2x2S eigenMatrix =
                    __MATHUTILS__::__v2_vec2_toMat2x2(eigenVecs[i], eigenVecs[i]);
                eigenMatrix = __MATHUTILS__::__s_Mat2x2_multiply(eigenMatrix, eigenValues[i]);
                projH = __MATHUTILS__::__Mat2x2_add(projH, eigenMatrix);
            }
        }

        __MATHUTILS__::__set_Mat_val(H_vI, projH.m[0][0], 0, projH.m[0][1], 0, 0, 0, projH.m[1][0],
                                     0, projH.m[1][1]);
    } else {
        __MATHUTILS__::__set_Mat_val(H_vI, (multiplier_vI / eps), 0, 0, 0, 0, 0, 0, 0,
                                     (multiplier_vI / eps));
    }

    H3x3[idx] = H_vI;
    D1Index[idx] = gidx;
}


__global__ void _getFrictionEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                const Scalar3* o_vertexes,
                                                const int4* _collisionPair, int cpNumLast, Scalar dt,
                                                const Scalar2* distCoord,
                                                const __MATHUTILS__::Matrix3x2S* tanBasis,
                                                const Scalar* lastH, Scalar fricDHat, Scalar eps

) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ Scalar tep1[];
    int numbers = cpNumLast;
    if (idx >= numbers) return;

    Scalar temp = __cal_Friction_energy(vertexes, o_vertexes, _collisionPair[idx], dt,
                                        distCoord[idx], tanBasis[idx], lastH[idx], fricDHat, eps);

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

__global__ void _getFrictionEnergy_gd_Reduction_3D(Scalar* squeue, const Scalar3* vertexes,
                                                   const Scalar3* o_vertexes,
                                                   const Scalar3* _normal,
                                                   const uint32_t* _collisionPair_gd, int gpNum,
                                                   Scalar dt, const Scalar* lastH, Scalar eps

) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;

    extern __shared__ Scalar tep1[];
    int numbers = gpNum;
    if (idx >= numbers) return;

    Scalar temp = __cal_Friction_gd_energy(vertexes, o_vertexes, _normal, _collisionPair_gd[idx],
                                           dt, lastH[idx], eps);

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


__global__ void _calFrictionLastH_gd(const Scalar3* _vertexes, const Scalar* g_offset,
                                     const Scalar3* g_normal,
                                     const uint32_t* _collisionPair_environment,
                                     Scalar* lambda_lastH_gd, uint32_t* _collisionPair_last_gd,
                                     Scalar dHat, Scalar Kappa, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    Scalar3 normal = *g_normal;
    int gidx = _collisionPair_environment[idx];
    Scalar dist = __MATHUTILS__::__vec3_dot(normal, _vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;

    Scalar t = dist2 - dHat;
    Scalar g_b = t * log(dist2 / dHat) * -2.0 - (t * t) / dist2;

    lambda_lastH_gd[idx] = -Kappa * 2.0 * sqrt(dist2) * g_b;
    _collisionPair_last_gd[idx] = gidx;
}



__global__ void _calFrictionLastH_DistAndTan(const Scalar3* _vertexes,
                                             const int4* _collisionPair, Scalar* lambda_lastH,
                                             Scalar2* distCoord,
                                             __MATHUTILS__::Matrix3x2S* tanBasis,
                                             int4* _collisionPair_last, Scalar dHat, Scalar Kappa,
                                             uint32_t* _cpNum_last, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _collisionPair[idx];
    Scalar dis;
    int last_index = -1;
    
    if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        last_index = atomicAdd(_cpNum_last, 1);
        atomicAdd(_cpNum_last + 4, 1);
        __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                _vertexes[MMCVIDI.w], dis);
        __GIPCFRICUTILS__::computeClosestPoint_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                distCoord[last_index]);
        __GIPCFRICUTILS__::computeTangentBasis_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                tanBasis[last_index]);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z == -1 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        last_index = atomicAdd(_cpNum_last, 1);
        atomicAdd(_cpNum_last + 2, 1);
        __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
        distCoord[last_index].x = 0;
        distCoord[last_index].y = 0;
        __GIPCFRICUTILS__::computeTangentBasis_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                tanBasis[last_index]);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        last_index = atomicAdd(_cpNum_last, 1);
        atomicAdd(_cpNum_last + 3, 1);
        __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                dis);
        __GIPCFRICUTILS__::computeClosestPoint_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z],
                                                distCoord[last_index].x);
        distCoord[last_index].y = 0;
        __GIPCFRICUTILS__::computeTangentBasis_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z], tanBasis[last_index]);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        last_index = atomicAdd(_cpNum_last, 1);
        atomicAdd(_cpNum_last + 4, 1);
        __MATHUTILS__::_d_PT(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                _vertexes[MMCVIDI.w], dis);
        __GIPCFRICUTILS__::computeClosestPoint_PT(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                distCoord[last_index]);
        __GIPCFRICUTILS__::computeTangentBasis_PT(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                tanBasis[last_index]);
    }

    else if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.w = -MMCVIDI.w - 1;
        atomicAdd(_cpNum_last + 4, 1);
        __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                _vertexes[MMCVIDI.w], dis);
        __GIPCFRICUTILS__::computeClosestPoint_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                distCoord[last_index]);
        __GIPCFRICUTILS__::computeTangentBasis_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w],
                                                tanBasis[last_index]);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z < 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.z = -MMCVIDI.z - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        last_index = atomicAdd(_cpNum_last, 1);
        atomicAdd(_cpNum_last + 2, 1);
        __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
        distCoord[last_index].x = 0;
        distCoord[last_index].y = 0;
        __GIPCFRICUTILS__::computeTangentBasis_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                tanBasis[last_index]);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        last_index = atomicAdd(_cpNum_last, 1);
        atomicAdd(_cpNum_last + 3, 1);
        __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z],
                                dis);
        __GIPCFRICUTILS__::computeClosestPoint_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z],
                                                distCoord[last_index].x);
        distCoord[last_index].y = 0;
        __GIPCFRICUTILS__::computeTangentBasis_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y],
                                                _vertexes[MMCVIDI.z], tanBasis[last_index]);
    }

    else {
        printf("################################ Error: invalid collision conditions");
        // exit(EXIT_FAILURE);
    }

    if (last_index >= 0) {
        lambda_lastH[last_index] = -Kappa * 2.0 * sqrt(dis) *
                                    (log(dis / dHat) * log(dis / dHat) * (2 * dis - 2 * dHat) +
                                    (2 * log(dis / dHat) * (dis - dHat) * (dis - dHat)) / dis);
        _collisionPair_last[last_index] = _collisionPair[idx];
    }
}


void buildFrictionSets(std::unique_ptr<GeometryManager>& instance) {
    CUDA_SAFE_CALL(cudaMemset(instance->getCudaCPNum(), 0, 5 * sizeof(uint32_t)));
    int numbers = instance->getHostCpNum(0);
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calFrictionLastH_DistAndTan<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaCollisionPairs(),
        instance->getCudaLambdaLastHScalar(), instance->getCudaDistCoord(),
        instance->getCudaTanBasis(), instance->getCudaCollisionPairsLastH(),
        instance->getHostDHat(), instance->getHostKappa(), instance->getCudaCPNum(),
        instance->getHostCpNum(0));
    CUDA_SAFE_CALL(cudaMemcpy(&instance->getHostCpNumLast(0), instance->getCudaCPNum(),
                              5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    numbers = instance->getHostGpNum();
    blockNum = (numbers + threadNum - 1) / threadNum;
    _calFrictionLastH_gd<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaEnvCollisionPairs(),
        instance->getCudaLambdaLastHScalarGd(), instance->getCudaCollisionPairsLastHGd(),
        instance->getHostDHat(), instance->getHostKappa(), instance->getHostGpNum());
    instance->getHostGpNumLast() = instance->getHostGpNum();
}


void calFrictionGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient) {
    int numbers = instance->getHostCpNumLast(0);
    // if (numbers < 1)return;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaOriginVertPos(),
        instance->getCudaCollisionPairsLastH(), _gradient, numbers, instance->getHostIPCDt(),
        instance->getCudaDistCoord(), instance->getCudaTanBasis(),
        instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
        instance->getCudaLambdaLastHScalar(), instance->getHostFrictionRate());

    numbers = instance->getHostGpNumLast();
    // if (numbers < 1)return;
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionGradient_gd<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaOriginVertPos(),
        instance->getCudaGroundNormal(), instance->getCudaCollisionPairsLastHGd(), _gradient,
        numbers, instance->getHostIPCDt(),
        instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
        instance->getCudaLambdaLastHScalarGd(), instance->getHostFrictionRate());
}


void calFrictionHessian(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr) {
    int numbers = instance->getHostCpNumLast(0);
    // if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //

    _calFrictionHessian<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaOriginVertPos(),
        instance->getCudaCollisionPairsLastH(), BH_ptr->cudaH12x12, BH_ptr->cudaH9x9, BH_ptr->cudaH6x6,
        BH_ptr->cudaD4Index, BH_ptr->cudaD3Index, BH_ptr->cudaD2Index, instance->getCudaCPNum(),
        numbers, instance->getHostIPCDt(), instance->getCudaDistCoord(),
        instance->getCudaTanBasis(),
        instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
        instance->getCudaLambdaLastHScalar(), instance->getHostFrictionRate(),
        instance->getHostCpNum(4), instance->getHostCpNum(3), instance->getHostCpNum(2));

    numbers = instance->getHostGpNumLast();
    CUDA_SAFE_CALL(cudaMemcpy(instance->getCudaGPNum(), &instance->getHostGpNumLast(),
                              sizeof(uint32_t), cudaMemcpyHostToDevice));
    blockNum = (numbers + threadNum - 1) / threadNum;

    _calFrictionHessian_gd<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaOriginVertPos(),
        instance->getCudaGroundNormal(), instance->getCudaCollisionPairsLastHGd(), BH_ptr->cudaH3x3,
        BH_ptr->cudaD1Index, numbers, instance->getHostIPCDt(),
        instance->getHostFDHat() * instance->getHostIPCDt() * instance->getHostIPCDt(),
        instance->getCudaLambdaLastHScalarGd(), instance->getHostFrictionRate());
}



}; // __GIPCFRICTION__





namespace __SIMPLEFRICTION__{



} // __SIMPLEFRICTION__

