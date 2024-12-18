
#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"

namespace __GIPCFRICUTILS__ {

__device__ __host__ __MATHUTILS__::Matrix2x2S __M3x2_transpose_self__multiply(
    const __MATHUTILS__::Matrix3x2S& A);

__device__ __host__ Scalar2
__M3x2_transpose_vec3__multiply(const __MATHUTILS__::Matrix3x2S& A, const Scalar3& b);

__device__ __host__ Scalar2 __M2x2_v2__multiply(const __MATHUTILS__::Matrix2x2S& A,
                                                const Scalar2& b);

__device__ __host__ void computeTangentBasis_PT(Scalar3 v0, Scalar3 v1, Scalar3 v2,
                                                Scalar3 v3,
                                                __MATHUTILS__::Matrix3x2S& basis);

__device__ __host__ void computeClosestPoint_PT(Scalar3 v0, Scalar3 v1, Scalar3 v2,
                                                Scalar3 v3, Scalar2& beta);

__device__ __host__ __device__ __host__ void computeRelDX_PT(
    const Scalar3 dx0, const Scalar3 dx1, const Scalar3 dx2, const Scalar3 dx3,
    Scalar beta1, Scalar beta2, Scalar3& relDX);

__device__ __host__ void liftRelDXTanToMesh_PT(const Scalar2& relDXTan,
                                               const __MATHUTILS__::Matrix3x2S& basis,
                                               Scalar beta1, Scalar beta2,
                                               __MATHUTILS__::Vector12S& TTTDX);

__device__ __host__ void computeT_PT(__MATHUTILS__::Matrix3x2S basis, Scalar beta1,
                                     Scalar beta2, __MATHUTILS__::Matrix12x2S& T);

__device__ __host__ void computeTangentBasis_EE(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2, const Scalar3& v3,
                                                __MATHUTILS__::Matrix3x2S& basis);

__device__ __host__ void computeClosestPoint_EE(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2, const Scalar3& v3,
                                                Scalar2& gamma);

__device__ __host__ void computeRelDX_EE(const Scalar3& dx0, const Scalar3& dx1,
                                         const Scalar3& dx2, const Scalar3& dx3,
                                         Scalar gamma1, Scalar gamma2, Scalar3& relDX);

__device__ __host__ void computeT_EE(const __MATHUTILS__::Matrix3x2S& basis,
                                     Scalar gamma1, Scalar gamma2,
                                     __MATHUTILS__::Matrix12x2S& T);

__device__ __host__ void liftRelDXTanToMesh_EE(const Scalar2& relDXTan,
                                               const __MATHUTILS__::Matrix3x2S& basis,
                                               Scalar gamma1, Scalar gamma2,
                                               __MATHUTILS__::Vector12S& TTTDX);

__device__ __host__ void computeTangentBasis_PE(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2,
                                                __MATHUTILS__::Matrix3x2S& basis);

__device__ __host__ void computeClosestPoint_PE(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2, Scalar& yita);

__device__ __host__ void computeRelDX_PE(const Scalar3& dx0, const Scalar3& dx1,
                                         const Scalar3& dx2, Scalar yita, Scalar3& relDX);

__device__ __host__ void liftRelDXTanToMesh_PE(const Scalar2& relDXTan,
                                               const __MATHUTILS__::Matrix3x2S& basis,
                                               Scalar yita,
                                               __MATHUTILS__::Vector9S& TTTDX);

__device__ __host__ void computeT_PE(const __MATHUTILS__::Matrix3x2S& basis, Scalar yita,
                                     __MATHUTILS__::Matrix9x2S& T);

__device__ __host__ void computeTangentBasis_PP(const Scalar3& v0, const Scalar3& v1,
                                                __MATHUTILS__::Matrix3x2S& basis);

__device__ __host__ void computeRelDX_PP(const Scalar3& dx0, const Scalar3& dx1,
                                         Scalar3& relDX);

__device__ __host__ void liftRelDXTanToMesh_PP(const Scalar2& relDXTan,
                                               const __MATHUTILS__::Matrix3x2S& basis,
                                               __MATHUTILS__::Vector6S& TTTDX);

__device__ __host__ void computeT_PP(const __MATHUTILS__::Matrix3x2S& basis,
                                     __MATHUTILS__::Matrix6x2S& T);

// static friction clamping model
// C0 clamping
__device__ __host__ void f0_SF_C0(Scalar x2, Scalar eps_f, Scalar& f0);

__device__ __host__ void f1_SF_div_relDXNorm_C0(Scalar eps_f, Scalar& result);

__device__ __host__ void f2_SF_C0(Scalar eps_f, Scalar& f2);

// C1 clamping
__device__ __host__ void f0_SF_C1(Scalar x2, Scalar eps_f, Scalar& f0);

__device__ __host__ void f1_SF_div_relDXNorm_C1(Scalar x2, Scalar eps_f, Scalar& result);

__device__ __host__ void f2_SF_C1(Scalar x2, Scalar eps_f, Scalar& f2);

// C2 clamping
__device__ __host__ void f0_SF_C2(Scalar x2, Scalar eps_f, Scalar& f0);

__device__ __host__ void f1_SF_div_relDXNorm_C2(Scalar x2, Scalar eps_f, Scalar& result);

__device__ __host__ void f2_SF_C2(Scalar x2, Scalar eps_f, Scalar& f2);

// interfaces
__device__ __host__ void f0_SF(Scalar relDXSqNorm, Scalar eps_f, Scalar& f0);

__device__ __host__ void f1_SF_div_relDXNorm(Scalar relDXSqNorm, Scalar eps_f,
                                             Scalar& result);

__device__ __host__ void f2_SF(Scalar relDXSqNorm, Scalar eps_f, Scalar& f2);

};  // namespace __GIPCFRICUTILS__
