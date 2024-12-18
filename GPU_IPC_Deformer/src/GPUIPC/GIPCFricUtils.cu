
#include "GIPCFricUtils.cuh"

#define SFCLAMPING_ORDER 1

namespace __GIPCFRICUTILS__ {

__device__ __host__ __MATHUTILS__::Matrix2x2S __M3x2_transpose_self__multiply(
    const __MATHUTILS__::Matrix3x2S& A) {
    __MATHUTILS__::Matrix2x2S result;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[k][i] * A.m[k][j];
            }
            result.m[i][j] = temp;
        }
    }
    return result;
}

__device__ __host__ Scalar2
__M3x2_transpose_vec3__multiply(const __MATHUTILS__::Matrix3x2S& A, const Scalar3& b) {
    Scalar x = A.m[0][0] * b.x + A.m[1][0] * b.y + A.m[2][0] * b.z;
    Scalar y = A.m[0][1] * b.x + A.m[1][1] * b.y + A.m[2][1] * b.z;
    return make_Scalar2(x, y);
}

__device__ __host__ Scalar2 __M2x2_v2__multiply(const __MATHUTILS__::Matrix2x2S& A,
                                                const Scalar2& b) {
    Scalar x = A.m[0][0] * b.x + A.m[0][1] * b.y;
    Scalar y = A.m[1][0] * b.x + A.m[1][1] * b.y;
    return make_Scalar2(x, y);
}

__device__ __host__ void computeTangentBasis_PT(Scalar3 v0, Scalar3 v1, Scalar3 v2,
                                                Scalar3 v3,
                                                __MATHUTILS__::Matrix3x2S& basis) {
    Scalar3 v12 = __MATHUTILS__::__vec3_minus(v2, v1);
    Scalar3 v12_normalized = __MATHUTILS__::__vec3_normalized(v12);
    Scalar3 c = __MATHUTILS__::__vec3_normalized(__MATHUTILS__::__vec3_cross(
        __MATHUTILS__::__vec3_cross(v12, __MATHUTILS__::__vec3_minus(v3, v1)), v12));
    basis.m[0][0] = v12_normalized.x;
    basis.m[1][0] = v12_normalized.y;
    basis.m[2][0] = v12_normalized.z;
    basis.m[0][1] = c.x;
    basis.m[1][1] = c.y;
    basis.m[2][1] = c.z;
}

__device__ __host__ void computeClosestPoint_PT(Scalar3 v0, Scalar3 v1, Scalar3 v2,
                                                Scalar3 v3, Scalar2& beta) {
    __MATHUTILS__::Matrix3x2S basis;
    Scalar3 v12 = __MATHUTILS__::__vec3_minus(v2, v1);
    Scalar3 v13 = __MATHUTILS__::__vec3_minus(v3, v1);
    basis.m[0][0] = v12.x;
    basis.m[1][0] = v12.y;
    basis.m[2][0] = v12.z;

    basis.m[0][1] = v13.x;
    basis.m[1][1] = v13.y;
    basis.m[2][1] = v13.z;
    __MATHUTILS__::Matrix2x2S btb = __M3x2_transpose_self__multiply(basis);
    // Eigen::Matrix<Scalar, 2, 3> basis;
    // basis.row(0) = v2 - v1;
    // basis.row(1) = v3 - v1;
    __MATHUTILS__::Matrix2x2S b2b_inv;
    __Inverse2x2(btb, b2b_inv);
    Scalar3 v10 = __MATHUTILS__::__vec3_minus(v0, v1);
    Scalar2 b = __M3x2_transpose_vec3__multiply(basis, v10);
    beta = __M2x2_v2__multiply(b2b_inv, b);
    // beta = (basis * basis.transpose()).ldlt().solve(basis * (v0 - v1).transpose());
}

__device__ __host__ __device__ __host__ void computeRelDX_PT(
    const Scalar3 dx0, const Scalar3 dx1, const Scalar3 dx2, const Scalar3 dx3,
    Scalar beta1, Scalar beta2, Scalar3& relDX) {
    Scalar3 b1_dx12 =
        __MATHUTILS__::__s_vec3_multiply(__MATHUTILS__::__vec3_minus(dx2, dx1), beta1);
    Scalar3 b2_dx13 =
        __MATHUTILS__::__s_vec3_multiply(__MATHUTILS__::__vec3_minus(dx3, dx1), beta2);

    relDX = __MATHUTILS__::__vec3_minus(
        dx0, __MATHUTILS__::__vec3_add(__MATHUTILS__::__vec3_add(b1_dx12, b2_dx13), dx1));
}

__device__ __host__ void liftRelDXTanToMesh_PT(const Scalar2& relDXTan,
                                               const __MATHUTILS__::Matrix3x2S& basis,
                                               Scalar beta1, Scalar beta2,
                                               __MATHUTILS__::Vector12S& TTTDX) {
    Scalar3 relDXTan3D = __M3x2_v2_multiply(basis, relDXTan);
    TTTDX.v[0] = relDXTan3D.x;
    TTTDX.v[1] = relDXTan3D.y;
    TTTDX.v[2] = relDXTan3D.z;

    TTTDX.v[3] = (-1 + beta1 + beta2) * relDXTan3D.x;
    TTTDX.v[4] = (-1 + beta1 + beta2) * relDXTan3D.y;
    TTTDX.v[5] = (-1 + beta1 + beta2) * relDXTan3D.z;

    TTTDX.v[6] = -beta1 * relDXTan3D.x;
    TTTDX.v[7] = -beta1 * relDXTan3D.y;
    TTTDX.v[8] = -beta1 * relDXTan3D.z;

    TTTDX.v[9] = -beta2 * relDXTan3D.x;
    TTTDX.v[10] = -beta2 * relDXTan3D.y;
    TTTDX.v[11] = -beta2 * relDXTan3D.z;
}

__device__ __host__ void computeT_PT(__MATHUTILS__::Matrix3x2S basis, Scalar beta1,
                                     Scalar beta2, __MATHUTILS__::Matrix12x2S& T) {
    T.m[0][0] = basis.m[0][0];
    T.m[1][0] = basis.m[1][0];
    T.m[2][0] = basis.m[2][0];
    T.m[0][1] = basis.m[0][1];
    T.m[1][1] = basis.m[1][1];
    T.m[2][1] = basis.m[2][1];

    T.m[3][0] = (-1 + beta1 + beta2) * basis.m[0][0];
    T.m[4][0] = (-1 + beta1 + beta2) * basis.m[1][0];
    T.m[5][0] = (-1 + beta1 + beta2) * basis.m[2][0];
    T.m[3][1] = (-1 + beta1 + beta2) * basis.m[0][1];
    T.m[4][1] = (-1 + beta1 + beta2) * basis.m[1][1];
    T.m[5][1] = (-1 + beta1 + beta2) * basis.m[2][1];

    T.m[6][0] = -beta1 * basis.m[0][0];
    T.m[7][0] = -beta1 * basis.m[1][0];
    T.m[8][0] = -beta1 * basis.m[2][0];
    T.m[6][1] = -beta1 * basis.m[0][1];
    T.m[7][1] = -beta1 * basis.m[1][1];
    T.m[8][1] = -beta1 * basis.m[2][1];

    T.m[9][0] = -beta2 * basis.m[0][0];
    T.m[10][0] = -beta2 * basis.m[1][0];
    T.m[11][0] = -beta2 * basis.m[2][0];
    T.m[9][1] = -beta2 * basis.m[0][1];
    T.m[10][1] = -beta2 * basis.m[1][1];
    T.m[11][1] = -beta2 * basis.m[2][1];

}

__device__ __host__ void computeTangentBasis_EE(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2, const Scalar3& v3,
                                                __MATHUTILS__::Matrix3x2S& basis) {
    Scalar3 v01 = __MATHUTILS__::__vec3_minus(v1, v0);
    Scalar3 v01_normalized = __MATHUTILS__::__vec3_normalized(v01);
    Scalar3 c = __MATHUTILS__::__vec3_normalized(__MATHUTILS__::__vec3_cross(
        __MATHUTILS__::__vec3_cross(v01, __MATHUTILS__::__vec3_minus(v3, v2)), v01));
    basis.m[0][0] = v01_normalized.x;
    basis.m[1][0] = v01_normalized.y;
    basis.m[2][0] = v01_normalized.z;
    basis.m[0][1] = c.x;
    basis.m[1][1] = c.y;
    basis.m[2][1] = c.z;
}

__device__ __host__ void computeClosestPoint_EE(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2, const Scalar3& v3,
                                                Scalar2& gamma) {
    Scalar3 e20 = __MATHUTILS__::__vec3_minus(v0, v2);
    Scalar3 e01 = __MATHUTILS__::__vec3_minus(v1, v0);
    Scalar3 e23 = __MATHUTILS__::__vec3_minus(v3, v2);
    __MATHUTILS__::Matrix2x2S coefMtr;
    coefMtr.m[0][0] = __MATHUTILS__::__vec3_squaredNorm(e01);
    coefMtr.m[0][1] = -__MATHUTILS__::__vec3_dot(e23, e01);
    coefMtr.m[1][0] = -__MATHUTILS__::__vec3_dot(e23, e01);
    coefMtr.m[1][1] = __MATHUTILS__::__vec3_squaredNorm(e23);

    Scalar2 rhs;
    rhs.x = -__MATHUTILS__::__vec3_dot(e20, e01);
    rhs.y = __MATHUTILS__::__vec3_dot(e20, e23);
    __MATHUTILS__::Matrix2x2S coefMtr_inv;
    __Inverse2x2(coefMtr, coefMtr_inv);
    gamma = __M2x2_v2__multiply(coefMtr_inv, rhs);
}

__device__ __host__ void computeRelDX_EE(const Scalar3& dx0, const Scalar3& dx1,
                                         const Scalar3& dx2, const Scalar3& dx3,
                                         Scalar gamma1, Scalar gamma2, Scalar3& relDX) {
    Scalar3 g1_dx01 =
        __MATHUTILS__::__s_vec3_multiply(__MATHUTILS__::__vec3_minus(dx1, dx0), gamma1);
    Scalar3 g2_dx23 =
        __MATHUTILS__::__s_vec3_multiply(__MATHUTILS__::__vec3_minus(dx3, dx2), gamma2);

    relDX = __MATHUTILS__::__vec3_minus(__MATHUTILS__::__vec3_add(dx0, g1_dx01),
                                   __MATHUTILS__::__vec3_add(dx2, g2_dx23));
}

__device__ __host__ void computeT_EE(const __MATHUTILS__::Matrix3x2S& basis,
                                     Scalar gamma1, Scalar gamma2,
                                     __MATHUTILS__::Matrix12x2S& T) {
    T.m[0][0] = (1.0 - gamma1) * basis.m[0][0];
    T.m[1][0] = (1.0 - gamma1) * basis.m[1][0];
    T.m[2][0] = (1.0 - gamma1) * basis.m[2][0];
    T.m[0][1] = (1.0 - gamma1) * basis.m[0][1];
    T.m[1][1] = (1.0 - gamma1) * basis.m[1][1];
    T.m[2][1] = (1.0 - gamma1) * basis.m[2][1];

    T.m[3][0] = gamma1 * basis.m[0][0];
    T.m[4][0] = gamma1 * basis.m[1][0];
    T.m[5][0] = gamma1 * basis.m[2][0];
    T.m[3][1] = gamma1 * basis.m[0][1];
    T.m[4][1] = gamma1 * basis.m[1][1];
    T.m[5][1] = gamma1 * basis.m[2][1];

    T.m[6][0] = (gamma2 - 1.0) * basis.m[0][0];
    T.m[7][0] = (gamma2 - 1.0) * basis.m[1][0];
    T.m[8][0] = (gamma2 - 1.0) * basis.m[2][0];
    T.m[6][1] = (gamma2 - 1.0) * basis.m[0][1];
    T.m[7][1] = (gamma2 - 1.0) * basis.m[1][1];
    T.m[8][1] = (gamma2 - 1.0) * basis.m[2][1];

    T.m[9][0] = -gamma2 * basis.m[0][0];
    T.m[10][0] = -gamma2 * basis.m[1][0];
    T.m[11][0] = -gamma2 * basis.m[2][0];
    T.m[9][1] = -gamma2 * basis.m[0][1];
    T.m[10][1] = -gamma2 * basis.m[1][1];
    T.m[11][1] = -gamma2 * basis.m[2][1];
}

__device__ __host__ void liftRelDXTanToMesh_EE(const Scalar2& relDXTan,
                                               const __MATHUTILS__::Matrix3x2S& basis,
                                               Scalar gamma1, Scalar gamma2,
                                               __MATHUTILS__::Vector12S& TTTDX) {
    Scalar3 relDXTan3D = __M3x2_v2_multiply(basis, relDXTan);
    TTTDX.v[0] = (1.0 - gamma1) * relDXTan3D.x;
    TTTDX.v[1] = (1.0 - gamma1) * relDXTan3D.y;
    TTTDX.v[2] = (1.0 - gamma1) * relDXTan3D.z;

    TTTDX.v[3] = gamma1 * relDXTan3D.x;
    TTTDX.v[4] = gamma1 * relDXTan3D.y;
    TTTDX.v[5] = gamma1 * relDXTan3D.z;

    TTTDX.v[6] = (gamma2 - 1.0) * relDXTan3D.x;
    TTTDX.v[7] = (gamma2 - 1.0) * relDXTan3D.y;
    TTTDX.v[8] = (gamma2 - 1.0) * relDXTan3D.z;

    TTTDX.v[9] = -gamma2 * relDXTan3D.x;
    TTTDX.v[10] = -gamma2 * relDXTan3D.y;
    TTTDX.v[11] = -gamma2 * relDXTan3D.z;
}

__device__ __host__ void computeTangentBasis_PE(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2,
                                                __MATHUTILS__::Matrix3x2S& basis) {
    Scalar3 v12 = __MATHUTILS__::__vec3_minus(v2, v1);
    Scalar3 v12_normalized = __MATHUTILS__::__vec3_normalized(v12);
    Scalar3 c = __MATHUTILS__::__vec3_normalized(
        __MATHUTILS__::__vec3_cross(v12, __MATHUTILS__::__vec3_minus(v0, v1)));
    basis.m[0][0] = v12_normalized.x;
    basis.m[1][0] = v12_normalized.y;
    basis.m[2][0] = v12_normalized.z;
    basis.m[0][1] = c.x;
    basis.m[1][1] = c.y;
    basis.m[2][1] = c.z;
}
__device__ __host__ void computeClosestPoint_PE(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2, Scalar& yita) {
    Scalar3 e12 = __MATHUTILS__::__vec3_minus(v2, v1);
    yita = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_minus(v0, v1), e12) /
           __MATHUTILS__::__vec3_squaredNorm(e12);
}
__device__ __host__ void computeRelDX_PE(const Scalar3& dx0, const Scalar3& dx1,
                                         const Scalar3& dx2, Scalar yita,
                                         Scalar3& relDX) {
    Scalar3 y_dx12 =
        __MATHUTILS__::__s_vec3_multiply(__MATHUTILS__::__vec3_minus(dx2, dx1), yita);

    relDX = __MATHUTILS__::__vec3_minus(dx0, __MATHUTILS__::__vec3_add(dx1, y_dx12));
}

__device__ __host__ void liftRelDXTanToMesh_PE(const Scalar2& relDXTan,
                                               const __MATHUTILS__::Matrix3x2S& basis,
                                               Scalar yita,
                                               __MATHUTILS__::Vector9S& TTTDX) {
    Scalar3 relDXTan3D = __M3x2_v2_multiply(basis, relDXTan);

    TTTDX.v[0] = relDXTan3D.x;
    TTTDX.v[1] = relDXTan3D.y;
    TTTDX.v[2] = relDXTan3D.z;

    TTTDX.v[3] = (yita - 1.0) * relDXTan3D.x;
    TTTDX.v[4] = (yita - 1.0) * relDXTan3D.y;
    TTTDX.v[5] = (yita - 1.0) * relDXTan3D.z;

    TTTDX.v[6] = -yita * relDXTan3D.x;
    TTTDX.v[7] = -yita * relDXTan3D.y;
    TTTDX.v[8] = -yita * relDXTan3D.z;
}

__device__ __host__ void computeT_PE(const __MATHUTILS__::Matrix3x2S& basis, Scalar yita,
                                     __MATHUTILS__::Matrix9x2S& T) {
    T.m[0][0] = basis.m[0][0];
    T.m[1][0] = basis.m[1][0];
    T.m[2][0] = basis.m[2][0];
    T.m[0][1] = basis.m[0][1];
    T.m[1][1] = basis.m[1][1];
    T.m[2][1] = basis.m[2][1];

    T.m[3][0] = (yita - 1.0) * basis.m[0][0];
    T.m[4][0] = (yita - 1.0) * basis.m[1][0];
    T.m[5][0] = (yita - 1.0) * basis.m[2][0];
    T.m[3][1] = (yita - 1.0) * basis.m[0][1];
    T.m[4][1] = (yita - 1.0) * basis.m[1][1];
    T.m[5][1] = (yita - 1.0) * basis.m[2][1];

    T.m[6][0] = -yita * basis.m[0][0];
    T.m[7][0] = -yita * basis.m[1][0];
    T.m[8][0] = -yita * basis.m[2][0];
    T.m[6][1] = -yita * basis.m[0][1];
    T.m[7][1] = -yita * basis.m[1][1];
    T.m[8][1] = -yita * basis.m[2][1];
}
__device__ __host__ void computeTangentBasis_PP(const Scalar3& v0, const Scalar3& v1,
                                                __MATHUTILS__::Matrix3x2S& basis) {
    Scalar3 v01 = __MATHUTILS__::__vec3_minus(v1, v0);
    Scalar3 xCross;
    xCross.x = 0;
    xCross.y = -v01.z;
    xCross.z = v01.y;
    Scalar3 yCross;
    yCross.x = v01.z;
    yCross.y = 0;
    yCross.z = -v01.x;

    if (__MATHUTILS__::__vec3_squaredNorm(xCross) > __MATHUTILS__::__vec3_squaredNorm(yCross)) {
        Scalar3 xCross_n = __MATHUTILS__::__vec3_normalized(xCross);
        Scalar3 c =
            __MATHUTILS__::__vec3_normalized(__MATHUTILS__::__vec3_cross(v01, xCross));
        basis.m[0][0] = xCross_n.x;
        basis.m[1][0] = xCross_n.y;
        basis.m[2][0] = xCross_n.z;
        basis.m[0][1] = c.x;
        basis.m[1][1] = c.y;
        basis.m[2][1] = c.z;
    } else {
        Scalar3 yCross_n = __MATHUTILS__::__vec3_normalized(yCross);
        Scalar3 c =
            __MATHUTILS__::__vec3_normalized(__MATHUTILS__::__vec3_cross(v01, yCross));
        basis.m[0][0] = yCross_n.x;
        basis.m[1][0] = yCross_n.y;
        basis.m[2][0] = yCross_n.z;
        basis.m[0][1] = c.x;
        basis.m[1][1] = c.y;
        basis.m[2][1] = c.z;
    }
}
__device__ __host__ void computeRelDX_PP(const Scalar3& dx0, const Scalar3& dx1,
                                         Scalar3& relDX) {
    relDX = __MATHUTILS__::__vec3_minus(dx0, dx1);
}

__device__ __host__ void liftRelDXTanToMesh_PP(const Scalar2& relDXTan,
                                               const __MATHUTILS__::Matrix3x2S& basis,
                                               __MATHUTILS__::Vector6S& TTTDX) {
    Scalar3 relDXTan3D = __M3x2_v2_multiply(basis, relDXTan);

    TTTDX.v[0] = relDXTan3D.x;
    TTTDX.v[1] = relDXTan3D.y;
    TTTDX.v[2] = relDXTan3D.z;

    TTTDX.v[3] = -relDXTan3D.x;
    TTTDX.v[4] = -relDXTan3D.y;
    TTTDX.v[5] = -relDXTan3D.z;
}

__device__ __host__ void computeT_PP(const __MATHUTILS__::Matrix3x2S& basis,
                                     __MATHUTILS__::Matrix6x2S& T) {
    T.m[0][0] = basis.m[0][0];
    T.m[1][0] = basis.m[1][0];
    T.m[2][0] = basis.m[2][0];
    T.m[0][1] = basis.m[0][1];
    T.m[1][1] = basis.m[1][1];
    T.m[2][1] = basis.m[2][1];

    T.m[3][0] = -basis.m[0][0];
    T.m[4][0] = -basis.m[1][0];
    T.m[5][0] = -basis.m[2][0];
    T.m[3][1] = -basis.m[0][1];
    T.m[4][1] = -basis.m[1][1];
    T.m[5][1] = -basis.m[2][1];
}
// static friction clamping model
// C0 clamping
__device__ __host__ void f0_SF_C0(Scalar x2, Scalar eps_f, Scalar& f0) {
    f0 = x2 / (2.0 * eps_f) + eps_f / 2.0;
}

__device__ __host__ void f1_SF_div_relDXNorm_C0(Scalar eps_f, Scalar& result) {
    result = 1.0 / eps_f;
}

__device__ __host__ void f2_SF_C0(Scalar eps_f, Scalar& f2) { f2 = 1.0 / eps_f; }

// C1 clamping
__device__ __host__ void f0_SF_C1(Scalar x2, Scalar eps_f, Scalar& f0) {
    f0 = x2 * (-sqrt(x2) / 3.0 + eps_f) / (eps_f * eps_f) + eps_f / 3.0;
}

__device__ __host__ void f1_SF_div_relDXNorm_C1(Scalar x2, Scalar eps_f, Scalar& result) {
    result = (-sqrt(x2) + 2.0 * eps_f) / (eps_f * eps_f);
}

__device__ __host__ void f2_SF_C1(Scalar x2, Scalar eps_f, Scalar& f2) {
    f2 = 2.0 * (eps_f - sqrt(x2)) / (eps_f * eps_f);
}

// C2 clamping
__device__ __host__ void f0_SF_C2(Scalar x2, Scalar eps_f, Scalar& f0) {
    f0 = x2 * (0.25 * x2 - (sqrt(x2) - 1.5 * eps_f) * eps_f) / (eps_f * eps_f * eps_f) +
         eps_f / 4.0;
}

__device__ __host__ void f1_SF_div_relDXNorm_C2(Scalar x2, Scalar eps_f, Scalar& result) {
    result = (x2 - (3.0 * sqrt(x2) - 3.0 * eps_f) * eps_f) / (eps_f * eps_f * eps_f);
}

__device__ __host__ void f2_SF_C2(Scalar x2, Scalar eps_f, Scalar& f2) {
    f2 = 3.0 * (x2 - (2.0 * sqrt(x2) - eps_f) * eps_f) / (eps_f * eps_f * eps_f);
}

// interfaces
__device__ __host__ void f0_SF(Scalar relDXSqNorm, Scalar eps_f, Scalar& f0) {
#if (SFCLAMPING_ORDER == 0)
    f0_SF_C0(relDXSqNorm, eps_f, f0);
#elif (SFCLAMPING_ORDER == 1)
    f0_SF_C1(relDXSqNorm, eps_f, f0);
#elif (SFCLAMPING_ORDER == 2)
    f0_SF_C2(relDXSqNorm, eps_f, f0);
#endif
}

__device__ __host__ void f1_SF_div_relDXNorm(Scalar relDXSqNorm, Scalar eps_f,
                                             Scalar& result) {
#if (SFCLAMPING_ORDER == 0)
    f1_SF_div_relDXNorm_C0(eps_f, result);
#elif (SFCLAMPING_ORDER == 1)
    f1_SF_div_relDXNorm_C1(relDXSqNorm, eps_f, result);
#elif (SFCLAMPING_ORDER == 2)
    f1_SF_div_relDXNorm_C2(relDXSqNorm, eps_f, result);
#endif
}

__device__ __host__ void f2_SF(Scalar relDXSqNorm, Scalar eps_f, Scalar& f2) {
#if (SFCLAMPING_ORDER == 0)
    f2_SF_C0(eps_f, f2);
#elif (SFCLAMPING_ORDER == 1)
    f2_SF_C1(relDXSqNorm, eps_f, f2);
#elif (SFCLAMPING_ORDER == 2)
    f2_SF_C2(relDXSqNorm, eps_f, f2);
#endif
}

};  // namespace __GIPCFRICUTILS__
