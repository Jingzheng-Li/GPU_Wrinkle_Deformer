
#pragma once

#include <cmath>
#include <vector_functions.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#ifdef USE_DOUBLE_PRECISION
    using Scalar = double;
    using Scalar2 = double2;
    using Scalar3 = double3;
    using Scalar4 = double4;
    __host__ __device__ inline Scalar2 make_Scalar2(double x, double y) {
        return make_double2(x, y);
    }
    __host__ __device__ inline Scalar3 make_Scalar3(double x, double y, double z) {
        return make_double3(x, y, z);
    }
    __host__ __device__ inline Scalar4 make_Scalar4(double x, double y, double z, double w) {
        return make_double4(x, y, z, w);
    }
    __host__ __device__ inline Scalar very_small_number() {
        return 1e-15;
    }
#else
    using Scalar = float;
    using Scalar2 = float2;
    using Scalar3 = float3;
    using Scalar4 = float4;
    __host__ __device__ inline Scalar2 make_Scalar2(float x, float y) {
        return make_float2(x, y);
    }
    __host__ __device__ inline Scalar3 make_Scalar3(float x, float y, float z) {
        return make_float3(x, y, z);
    }
    __host__ __device__ inline Scalar4 make_Scalar4(float x, float y, float z, float w) {
        return make_float4(x, y, z, w);
    }
    __host__ __device__ inline Scalar very_small_number() {
        return 1e-6;
    }
#endif




namespace __MATHUTILS__ {

struct Vector4S {
    Scalar v[4];
};

struct Vector6S {
    Scalar v[6];
};

struct Vector9S {
    Scalar v[9];
};

struct Vector12S {
    Scalar v[12];
};

struct Matrix2x2S {
    Scalar m[2][2];
};

struct Matrix3x3S {
    Scalar m[3][3];
};

struct Matrix4x4S {
    Scalar m[4][4];
};

struct Matrix6x6S {
    Scalar m[6][6];
};

struct Matrix9x9S {
    Scalar m[9][9];
};

struct Matrix3x6S {
    Scalar m[3][6];
};

struct Matrix6x3S {
    Scalar m[6][3];
};

struct Matrix3x2S {
    Scalar m[3][2];
};

struct Matrix2x3S {
    Scalar m[2][3];
};

struct Matrix12x12S {
    Scalar m[12][12];
};

struct Matrix24x24S {
    Scalar m[24][24];
};

struct Matrix36x36S {
    Scalar m[36][36];
};

struct Matrix96x96S {
    Scalar m[96][96];
};

struct Matrix9x2S {
    Scalar m[9][2];
};

struct Matrix6x2S {
    Scalar m[6][2];
};

struct Matrix12x2S {
    Scalar m[12][2];
};

struct Matrix9x12S {
    Scalar m[9][12];
};

struct Matrix12x9S {
    Scalar m[12][9];
};

struct Matrix12x6S {
    Scalar m[12][6];
};

struct Matrix6x12S {
    Scalar m[6][12];
};

struct Matrix12x4S {
    Scalar m[12][4];
};

struct Matrix9x4S {
    Scalar m[9][4];
};

struct Matrix4x9S {
    Scalar m[4][9];
};

struct Matrix6x9S {
    Scalar m[6][9];
};

struct Matrix9x6S {
    Scalar m[9][6];
};


struct MasMatrixSym {
    Matrix3x3S M[32 * (32 + 1) / 2];
};


typedef Eigen::Matrix<Scalar, 3, 1> EVector3S;
typedef Eigen::Matrix<Scalar, 4, 1> EVector4S;
typedef Eigen::Matrix<Scalar, 6, 1> EVector6S;
typedef Eigen::Matrix<Scalar, 9, 1> EVector9S;
typedef Eigen::Matrix<Scalar, 12, 1> EVector12S;

typedef Eigen::Matrix<Scalar, 2, 2> EMatrix2x2S;
typedef Eigen::Matrix<Scalar, 3, 3> EMatrix3x3S;
typedef Eigen::Matrix<Scalar, 4, 4> EMatrix4x4S;
typedef Eigen::Matrix<Scalar, 6, 6> EMatrix6x6S;
typedef Eigen::Matrix<Scalar, 9, 9> EMatrix9x9S;

typedef Eigen::Matrix<Scalar, 3, 6> EMatrix3x6S;
typedef Eigen::Matrix<Scalar, 6, 3> EMatrix6x3S;
typedef Eigen::Matrix<Scalar, 3, 2> EMatrix3x2S;

typedef Eigen::Matrix<Scalar, 12, 12> EMatrix12x12S;
typedef Eigen::Matrix<Scalar, 24, 24> EMatrix24x24S;
typedef Eigen::Matrix<Scalar, 36, 36> EMatrix36x36S;
typedef Eigen::Matrix<Scalar, 96, 96> EMatrix96x96S;

typedef Eigen::Matrix<Scalar, 9, 2> EMatrix9x2S;
typedef Eigen::Matrix<Scalar, 6, 2> EMatrix6x2S;
typedef Eigen::Matrix<Scalar, 12, 2> EMatrix12x2S;
typedef Eigen::Matrix<Scalar, 9, 12> EMatrix9x12S;
typedef Eigen::Matrix<Scalar, 12, 9> EMatrix12x9S;
typedef Eigen::Matrix<Scalar, 12, 6> EMatrix12x6S;
typedef Eigen::Matrix<Scalar, 6, 12> EMatrix6x12S;
typedef Eigen::Matrix<Scalar, 12, 4> EMatrix12x4S;
typedef Eigen::Matrix<Scalar, 9, 4> EMatrix9x4S;
typedef Eigen::Matrix<Scalar, 4, 9> EMatrix4x9S;
typedef Eigen::Matrix<Scalar, 6, 9> EMatrix6x9S;
typedef Eigen::Matrix<Scalar, 9, 6> EMatrix9x6S;

__device__ __host__ inline Scalar3 __vec3_from_Evec3(const EVector3S& evec3) {
    Scalar3 vec3;
    vec3.x = evec3(0);
    vec3.y = evec3(1);
    vec3.z = evec3(2);
    return vec3;
}

__device__ __host__ inline EVector3S __Evec3_from_vec3(Scalar3& vec3) {
    EVector3S evec3;
    evec3(0) = vec3.x;
    evec3(1) = vec3.x;
    evec3(2) = vec3.x;
    return evec3;
}


template <typename EVec, typename Vec, int N>
__device__ __host__ inline Vec __vec_from_Evec(const EVec& evec) {
    static_assert(EVec::RowsAtCompileTime == N && EVec::ColsAtCompileTime == 1, "Eigen vector size does not match.");

    Vec vec;
    for (int i = 0; i < N; ++i) {
        vec.v[i] = evec(i);
    }
    return vec;
}

template <typename EVec, typename Vec, int N>
__device__ __host__ inline EVec __Evec_from_vec(const Vec& vec) {
    static_assert(EVec::RowsAtCompileTime == N && EVec::ColsAtCompileTime == 1, "Eigen vector size does not match.");

    EVec evec;
    for (int i = 0; i < N; ++i) {
        evec(i) = vec.v[i];
    }
    return evec;
}

template <typename EMat, typename Mat, int R, int C>
__device__ __host__ inline Mat __Mat_from_EMat(const EMat& emat) {
    static_assert(EMat::RowsAtCompileTime == R && EMat::ColsAtCompileTime == C, "Eigen matrix size does not match.");

    Mat mat;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            mat.m[r][c] = emat(r, c);
        }
    }
    return mat;
}

template <typename EMat, typename Mat, int R, int C>
__device__ __host__ inline EMat __EMat_from_Mat(const Mat& mat) {
    static_assert(EMat::RowsAtCompileTime == R && EMat::ColsAtCompileTime == C, "Eigen matrix size does not match.");

    EMat emat;
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            emat(r, c) = mat.m[r][c];
        }
    }
    return emat;
}


}  // namespace __MATHUTILS__













namespace __MATHUTILS__ {

__device__ __host__ Scalar __PI();

__device__ void __make_Mat12x12_Diagonal_Dominant(__MATHUTILS__::Matrix12x12S& Mat);

__device__ __host__ EMatrix3x3S _Evec_to_crossMatrix(EVector3S v);

__device__ __host__ void __init_Vector4S(Vector4S& v, const Scalar& val);

__device__ __host__ void __init_Vector6S(Vector6S& v, const Scalar& val);

__device__ __host__ void __init_Vector9S(Vector9S& v, const Scalar& val);

__device__ __host__ void __init_Vector12S(Vector12S& v, const Scalar& val);

__device__ __host__ void __init_Mat3x3(Matrix3x3S& M, const Scalar& val);

__device__ __host__ void __init_Mat4x4(Matrix4x4S& M, const Scalar& val);

__device__ __host__ void __init_Mat6x6(Matrix6x6S& M, const Scalar& val);

__device__ __host__ void __init_Mat9x9(Matrix9x9S& M, const Scalar& val);

__device__ __host__ void __init_Mat12x12(Matrix12x12S& M, const Scalar& val);

__device__ __host__ void __identify_Mat3x3(Matrix3x3S& M);

__device__ __host__ void __identify_Mat4x4(Matrix4x4S& M);

__device__ __host__ void __identify_Mat6x6(Matrix6x6S& M);

__device__ __host__ void __identify_Mat9x9(Matrix9x9S& M);

__device__ __host__ void __identify_Mat12x12(Matrix12x12S& M);

__device__ __host__ Scalar __frobenius_norm_Mat3x3(const Matrix3x3S& M);

__device__ __host__ Scalar __frobenius_norm_Mat4x4(const Matrix4x4S& M);

__device__ __host__ Scalar __frobenius_norm_Mat6x6(const Matrix6x6S& M);

__device__ __host__ Scalar __frobenius_norm_Mat9x9(const Matrix9x9S& M);

__device__ __host__ Scalar __frobenius_norm_Mat12x12(const Matrix12x12S& M);

__device__ __host__ Scalar __vec2_norm(const Scalar2& n);

__device__ __host__ Scalar __vec3_norm(const Scalar3& n);

__device__ __host__ Scalar __vec6_norm(const __MATHUTILS__::Vector6S& vec);

__device__ __host__ Scalar __vec9_norm(const __MATHUTILS__::Vector9S& vec);

__device__ __host__ Scalar __vec12_norm(const __MATHUTILS__::Vector12S& vec);






__device__ __host__ Scalar3 __s_vec3_multiply(const Scalar3& a, Scalar b);

__device__ __host__ Scalar2 __s_vec2_multiply(const Scalar2& a, Scalar b);

__device__ __host__ Scalar3 __vec3_normalized(Scalar3 n);

__device__ __host__ Scalar3 __vec3_add(Scalar3 a, Scalar3 b);

__device__ __host__ Vector9S __vec9_add(const Vector9S& a, const Vector9S& b);

__device__ __host__ Vector6S __vec6_add(const Vector6S& a, const Vector6S& b);

__device__ __host__ Scalar3 __vec3_minus(Scalar3 a, Scalar3 b);

__device__ __host__ Scalar2 __vec2_minus(Scalar2 a, Scalar2 b);

__device__ __host__ Scalar3 __vec3_multiply(Scalar3 a, Scalar3 b);

__device__ __host__ Scalar __vec2_multiply(Scalar2 a, Scalar2 b);

__device__ __host__ Scalar __vec3_squaredNorm(Scalar3 a);

__device__ __host__ Scalar __vec2_squaredNorm(Scalar2 a);

__device__ __host__ void __M3x3_M3x3_multiply(const Matrix3x3S& A,
                                          const Matrix3x3S& B,
                                          Matrix3x3S& output);

__device__ __host__ Matrix3x3S __M3x3_M3x3_multiply(const Matrix3x3S& A,
                                                const Matrix3x3S& B);

__device__ __host__ Matrix2x2S __M2x2_M2x2_multiply(const Matrix2x2S& A,
                                                      const Matrix2x2S& B);

__device__ __host__ Scalar __M3x3_Trace(const Matrix3x3S& A);

__device__ __host__ Scalar3 __v3_M3x3_multiply(const Scalar3& n,
                                           const Matrix3x3S& A);

__device__ __host__ Scalar3 __M3x3_v3_multiply(const Matrix3x3S& A,
                                           const Scalar3& n);

__device__ __host__ Scalar3 __M3x2_v2_multiply(const Matrix3x2S& A,
                                               const Scalar2& n);

__device__ __host__ Matrix3x2S __s_Mat3x2_multiply(const Matrix3x2S& A,
                                                   const Scalar& b);

__device__ __host__ Matrix3x2S __Mat3x2_add(const Matrix3x2S& A,
                                            const Matrix3x2S& B);

__device__ __host__ Vector12S __M12x9_v9_multiply(const Matrix12x9S& A,
                                                 const Vector9S& n);

__device__ __host__ Vector12S __M12x6_v6_multiply(const Matrix12x6S& A,
                                                 const Vector6S& n);

__device__ __host__ Vector6S __M6x3_v3_multiply(const Matrix6x3S& A,
                                               const Scalar3& n);

__device__ __host__ Scalar2 __M2x3_v3_multiply(const Matrix2x3S& A,
                                               const Scalar3& n);

__device__ __host__ Vector9S __M9x6_v6_multiply(const Matrix9x6S& A,
                                               const Vector6S& n);

__device__ __host__ Vector12S __M12x12_v12_multiply(const Matrix12x12S& A,
                                                   const Vector12S& n);

__device__ __host__ Vector9S __M9x9_v9_multiply(const Matrix9x9S& A,
                                               const Vector9S& n);

__device__ __host__ Vector6S __M6x6_v6_multiply(const Matrix6x6S& A,
                                               const Vector6S& n);

__device__ __host__ Matrix9x9S __s_Mat9x9_multiply(const Matrix9x9S& A,
                                                   const Scalar& B);

__device__ __host__ Matrix6x6S __s_Mat6x6_multiply(const Matrix6x6S& A,
                                                   const Scalar& B);

__device__ __host__ Scalar __vec3_dot(const Scalar3& a, const Scalar3& b);

__device__ __host__ Scalar3 __vec3_cross(Scalar3 a, Scalar3 b);

__device__ __host__ Matrix3x3S __v_vec_toMat(Scalar3 a, Scalar3 b);

__device__ __host__ Matrix2x2S __v2_vec2_toMat2x2(Scalar2 a, Scalar2 b);

__device__ __host__ Matrix2x2S __s_Mat2x2_multiply(Matrix2x2S A, Scalar b);

__device__ __host__ Matrix3x3S __s_Mat3x3_multiply(Matrix3x3S A, Scalar b);

__device__ __host__ Matrix2x2S __Mat2x2_minus(Matrix2x2S A, Matrix2x2S B);

__device__ __host__ Matrix3x3S __Mat3x3_minus(Matrix3x3S A, Matrix3x3S B);

__device__ __host__ Matrix6x6S __v6_vec6_toMat6x6(Vector6S a, Vector6S b);

__device__ __host__ Matrix9x9S __v9_vec9_toMat9x9(const Vector9S& a,
                                                  const Vector9S& b,
                                                  const Scalar& coe = 1);

__device__ __host__ Matrix12x12S __v12_vec12_toMat12x12(const Vector12S& a, const Vector12S& b);

__device__ __host__ void __add_Mat3x3_to_Mat9x9(
    __MATHUTILS__::Matrix9x9S& Hessian,
    const __MATHUTILS__::Matrix3x3S& block,
    int rowStart,
    int colStart
);

__device__ __host__ __MATHUTILS__::Matrix3x3S __extract_Mat3x3_from_Mat9x9(
    const __MATHUTILS__::Matrix9x9S& Hessian,
    int rowStart,
    int colStart
);

__device__ __host__ void __add_Mat3x3_to_Mat12x12(
    __MATHUTILS__::Matrix12x12S& Hessian,
    const __MATHUTILS__::Matrix3x3S& block,
    int rowStart,
    int colStart
);

__device__ __host__ Scalar __vec12_norm(const __MATHUTILS__::Vector12S& vec);

__device__ __host__ Vector9S __s_vec9_multiply(Vector9S a, Scalar b);

__device__ __host__ Vector12S __s_vec12_multiply(Vector12S a, Scalar b);

__device__ __host__ Vector6S __s_vec6_multiply(Vector6S a, Scalar b);

__device__ __host__ void __Mat_add(const Matrix3x3S& A, const Matrix3x3S& B,
                                   Matrix3x3S& output);

__device__ __host__ void __Mat_add(const Matrix6x6S& A, const Matrix6x6S& B,
                                   Matrix6x6S& output);

__device__ __host__ Matrix3x3S __Mat_add(const Matrix3x3S& A,
                                         const Matrix3x3S& B);

__device__ __host__ Matrix2x2S __Mat2x2_add(const Matrix2x2S& A,
                                            const Matrix2x2S& B);

__device__ __host__ Matrix9x9S __Mat9x9_add(const Matrix9x9S& A,
                                            const Matrix9x9S& B);

__device__ __host__ Matrix12x12S __Mat12x12_add(const Matrix12x12S& A,
                                            const Matrix12x12S& B);

__device__ __host__ Matrix9x12S __Mat9x12_add(const Matrix9x12S& A,
                                              const Matrix9x12S& B);

__device__ __host__ Matrix6x12S __Mat6x12_add(const Matrix6x12S& A,
                                              const Matrix6x12S& B);

__device__ __host__ Matrix6x9S __Mat6x9_add(const Matrix6x9S& A,
                                            const Matrix6x9S& B);

__device__ __host__ Matrix3x6S __Mat3x6_add(const Matrix3x6S& A,
                                            const Matrix3x6S& B);

__device__ __host__ Matrix3x3S __v_to_crossMat(Scalar3 v);

__device__ __host__ void __set_Mat_identity(Matrix2x2S& M);

__device__ __host__ void __set_Mat_val(Matrix3x3S& M, const Scalar& a00,
                                       const Scalar& a01, const Scalar& a02,
                                       const Scalar& a10, const Scalar& a11,
                                       const Scalar& a12, const Scalar& a20,
                                       const Scalar& a21, const Scalar& a22);

__device__ __host__ void __set_Mat_val_row(Matrix3x3S& M, const Scalar3& row0,
                                           const Scalar3& row1,
                                           const Scalar3& row2);

__device__ __host__ void __set_Mat_val_column(Matrix3x3S& M,
                                              const Scalar3& col0,
                                              const Scalar3& col1,
                                              const Scalar3& col2);

__device__ __host__ void __set_Mat3x2_val_column(Matrix3x2S& M,
                                                 const Scalar3& col0,
                                                 const Scalar3& col1);

__device__ __host__ void __set_Mat2x2_val_column(Matrix2x2S& M,
                                                 const Scalar2& col0,
                                                 const Scalar2& col1);

__device__ __host__ void __init_Mat9x12_val(Matrix9x12S& M, const Scalar& val);

__device__ __host__ void __init_Mat6x12_val(Matrix6x12S& M, const Scalar& val);

__device__ __host__ void __init_Mat6x9_val(Matrix6x9S& M, const Scalar& val);

__device__ __host__ void __init_Mat3x6_val(Matrix3x6S& M, const Scalar& val);

__device__ __host__ Matrix3x3S __s_M3x3_multiply(const Matrix3x3S& A,
                                                const Scalar& B);

__device__ __host__ Matrix3x3S __Transpose3x3(Matrix3x3S input);

__device__ __host__ Matrix12x9S __Transpose9x12(const Matrix9x12S& input);

__device__ __host__ Matrix2x3S __Transpose3x2(const Matrix3x2S& input);

__device__ __host__ Matrix9x12S __Transpose12x9(const Matrix12x9S& input);

__device__ __host__ Matrix12x6S __Transpose6x12(const Matrix6x12S& input);

__device__ __host__ Matrix9x6S __Transpose6x9(const Matrix6x9S& input);

__device__ __host__ Matrix6x3S __Transpose3x6(const Matrix3x6S& input);

__device__ __host__ Matrix12x9S __M12x9_M9x9_Multiply(const Matrix12x9S& A,
                                                      const Matrix9x9S& B);

__device__ __host__ Matrix12x6S __M12x6_M6x6_Multiply(const Matrix12x6S& A,
                                                      const Matrix6x6S& B);

__device__ __host__ Matrix9x6S __M9x6_M6x6_Multiply(const Matrix9x6S& A,
                                                    const Matrix6x6S& B);

__device__ __host__ Matrix6x3S __M6x3_M3x3_Multiply(const Matrix6x3S& A,
                                                    const Matrix3x3S& B);

__device__ __host__ Matrix3x2S __M3x2_M2x2_Multiply(const Matrix3x2S& A,
                                                    const Matrix2x2S& B);

__device__ __host__ Matrix12x12S __M12x9_M9x12_Multiply(const Matrix12x9S& A,
                                                        const Matrix9x12S& B);

__device__ __host__ Matrix12x2S __M12x2_M2x2_Multiply(const Matrix12x2S& A,
                                                      const Matrix2x2S& B);

__device__ __host__ Matrix9x2S __M9x2_M2x2_Multiply(const Matrix9x2S& A,
                                                    const Matrix2x2S& B);

__device__ __host__ Matrix6x2S __M6x2_M2x2_Multiply(const Matrix6x2S& A,
                                                    const Matrix2x2S& B);

__device__ __host__ Matrix12x12S __M12x2_M12x2T_Multiply(const Matrix12x2S& A,
                                                         const Matrix12x2S& B);

__device__ __host__ Matrix9x9S __M9x2_M9x2T_Multiply(const Matrix9x2S& A,
                                                     const Matrix9x2S& B);

__device__ __host__ Matrix6x6S __M6x2_M6x2T_Multiply(const Matrix6x2S& A,
                                                     const Matrix6x2S& B);

__device__ __host__ Matrix12x12S __M12x6_M6x12_Multiply(const Matrix12x6S& A,
                                                        const Matrix6x12S& B);

__device__ __host__ Matrix9x9S __M9x6_M6x9_Multiply(const Matrix9x6S& A,
                                                    const Matrix6x9S& B);

__device__ __host__ Matrix6x6S __M6x3_M3x6_Multiply(const Matrix6x3S& A,
                                                    const Matrix3x6S& B);

__device__ __host__ Matrix12x12S __s_M12x12_Multiply(const Matrix12x12S& A,
                                                     const Scalar& B);

__device__ __host__ Matrix9x9S __s_M9x9_Multiply(const Matrix9x9S& A,
                                                 const Scalar& B);

__device__ __host__ Matrix6x6S __s_M6x6_Multiply(const Matrix6x6S& A,
                                                 const Scalar& B);

__device__ __host__ void __Determiant(const Matrix3x3S& input,
                                      Scalar& determinant);

__device__ __host__ Scalar __Determiant(const Matrix3x3S& input);

__device__ __host__ void __Inverse(const Matrix3x3S& input, Matrix3x3S& output);

__device__ __host__ void __Inverse2x2(const Matrix2x2S& input,
                                      Matrix2x2S& output);

__device__ __host__ Scalar __f(const Scalar& x, const Scalar& a,
                               const Scalar& b, const Scalar& c,
                               const Scalar& d);

__device__ __host__ Scalar __df(const Scalar& x, const Scalar& a,
                                const Scalar& b, const Scalar& c);

__device__ __host__ void __NewtonSolverForCubicEquation(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS = 1e-6);

__device__ __host__ void __NewtonSolverForCubicEquation_satbleNeohook(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS = 1e-6);

__device__ __host__ void __SolverForCubicEquation(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS = 1e-6);

__device__ __host__ Vector9S __Mat3x3_to_vec9_Scalar(const Matrix3x3S& F);

__device__ __host__ void __normalized_vec9_Scalar(Vector9S& v9);

__device__ __host__ void __normalized_vec6_Scalar(Vector6S& v6);

__device__ __host__ Vector6S __Mat3x2_to_vec6_Scalar(const Matrix3x2S& F);

__device__ __host__ Matrix3x3S __vec9_to_Mat3x3_Scalar(const Scalar vec9[9]);

__device__ __host__ Matrix2x2S __vec4_to_Mat2x2_Scalar(const Scalar vec4[4]);

__device__ __host__ Scalar __s_clamp(Scalar val, Scalar min_val, Scalar max_val);

__device__ void SVD(const Matrix3x3S& M, Matrix3x3S& Uout, Matrix3x3S& Vout, Matrix3x3S& Sigma);

__device__ __host__ void __makePD2x2(const Scalar& a00, const Scalar& a01,
                                     const Scalar& a10, const Scalar& a11,
                                     Scalar eigenValues[2], int& num,
                                     Scalar2 eigenVectors[2]);

__device__ __host__ void __M12x9_S9x9_MT9x12_Multiply(const Matrix12x9S& A,
                                                      const Matrix9x9S& B,
                                                      Matrix12x12S& output);

__device__ __host__ void __M9x4_S4x4_MT4x9_Multiply(const Matrix9x4S& A,
                                                    const Matrix4x4S& B,
                                                    Matrix9x9S& output);

__device__ __host__ Vector4S __s_vec4_multiply(Vector4S a, Scalar b);

__device__ __host__ Vector9S __M9x4_v4_multiply(const Matrix9x4S& A,
                                               const Vector4S& n);

__device__ __host__ Matrix4x4S __s_Mat4x4_multiply(const Matrix4x4S& A,
                                                   const Scalar& B);

__device__ __host__ Matrix4x4S __v4_vec4_toMat4x4(Vector4S a, Vector4S b);

__device__ __host__ void __s_M_Mat_MT_multiply(const Matrix3x3S& A,
                                               const Matrix3x3S& B,
                                               const Matrix3x3S& C,
                                               const Scalar& coe,
                                               Matrix3x3S& output);

__device__ __host__ inline Scalar __m_min(Scalar a, Scalar b) {
    return a < b ? a : b;
}

__device__ __host__ inline Scalar __m_max(Scalar a, Scalar b) {
    return a < b ? b : a;
}


__device__
void _d_PP(const Scalar3& v0, const Scalar3& v1, Scalar& d);

__device__
void _d_PT(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d);

__device__
void _d_PE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, Scalar& d);

__device__
void _d_EE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d);

__device__
void _d_EEParallel(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d);

__device__
Scalar _compute_epx(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3);

__device__
Scalar _compute_epx_cp(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3);

__device__ __host__
Scalar calculateVolume(const Scalar3* vertexes, const uint4& index);

__device__ __host__
Scalar calculateArea(const Scalar3* vertexes, const uint3& index);

__device__ Scalar _perlinNoise(Scalar x, Scalar y, Scalar z);

}  // namespace __MATHUTILS__



namespace __MATHUTILS__ {

__device__ void __perform_reduct_add_Scalar(
    Scalar* squeue, Scalar temp, int numbers);

__global__ void __reduct_add_Scalar(Scalar* _mem1Dim, int numbers);

__global__ void __reduct_add_Scalar2(Scalar2* _mem2Dim, int numbers);

__global__ void _reduct_min_Scalar(Scalar* _Scalar1Dim, int number);

__global__ void _reduct_max_Scalar(Scalar* _Scalar1Dim, int number);

__global__ void _reduct_max_Scalar2(Scalar2* _Scalar2Dim, int number);

__global__ void _reduct_max_Scalar3_to_Scalar(const Scalar3* _Scalar3Dim, Scalar* _Scalar1Dim, int number);

}  // namespace __MATHUTILS__

