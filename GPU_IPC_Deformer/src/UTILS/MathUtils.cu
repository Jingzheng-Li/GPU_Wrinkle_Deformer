
#include "MathUtils.cuh"


namespace __MATHUTILS__ {

__device__ __host__ Scalar __PI() {
    return 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899;
}

__device__ void __make_Mat12x12_Diagonal_Dominant(__MATHUTILS__::Matrix12x12S& Mat) {
    for (int i = 0; i < 12; ++i) {
        Scalar row_sum = 0.0;
        for (int j = 0; j < 12; ++j) {
            if (j != i) {
                row_sum += std::abs(Mat.m[i][j]);
            }
        }
        if (Mat.m[i][i] < row_sum) {
            Mat.m[i][i] = row_sum + 1e-6;
        }
    }

    for (int i = 0; i < 12; ++i) {
        if (Mat.m[i][i] <= 0) {
            Mat.m[i][i] = 1e-6;
        }
    }
}


__device__ __host__ EMatrix3x3S _Evec_to_crossMatrix(EVector3S v) {
    EMatrix3x3S ret;
    ret << 0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0;
    return ret;
}


__device__ __host__ void __init_Vector4S(Vector4S& v, const Scalar& val) {
    for (int i = 0; i < 4; i++) {
        v.v[i] = val;
    }
}

__device__ __host__ void __init_Vector6S(Vector6S& v, const Scalar& val) {
    for (int i = 0; i < 6; i++) {
        v.v[i] = val;
    }
}

__device__ __host__ void __init_Vector9S(Vector9S& v, const Scalar& val) {
    for (int i = 0; i < 9; i++) {
        v.v[i] = val;
    }
}

__device__ __host__ void __init_Vector12S(Vector12S& v, const Scalar& val) {
    for (int i = 0; i < 12; i++) {
        v.v[i] = val;
    }
}

__device__ __host__ void __init_Mat3x3(Matrix3x3S& M, const Scalar& val) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat4x4(Matrix4x4S& M, const Scalar& val) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat6x6(Matrix6x6S& M, const Scalar& val) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat9x9(Matrix9x9S& M, const Scalar& val) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat12x12(Matrix12x12S& M, const Scalar& val) {
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __identify_Mat3x3(Matrix3x3S& M) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i == j) {
                M.m[i][j] = 1;
            } else {
                M.m[i][j] = 0;
            }
        }
    }
}

__device__ __host__ void __identify_Mat4x4(Matrix4x4S& M) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                M.m[i][j] = 1;
            } else {
                M.m[i][j] = 0;
            }
        }
    }
}

__device__ __host__ void __identify_Mat6x6(Matrix6x6S& M) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            if (i == j) {
                M.m[i][j] = 1;
            } else {
                M.m[i][j] = 0;
            }
        }
    }
}

__device__ __host__ void __identify_Mat9x9(Matrix9x9S& M) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            if (i == j) {
                M.m[i][j] = 1;
            } else {
                M.m[i][j] = 0;
            }
        }
    }
}

__device__ __host__ void __identify_Mat12x12(Matrix12x12S& M) {
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            if (i == j) {
                M.m[i][j] = 1;
            } else {
                M.m[i][j] = 0;
            }
        }
    }
}

__device__ __host__ Scalar __frobenius_norm_Mat3x3(const Matrix3x3S& M) {
    Scalar sum = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            sum += M.m[i][j] * M.m[i][j];
        }
    }
    return sqrt(sum);
}

__device__ __host__ Scalar __frobenius_norm_Mat4x4(const Matrix4x4S& M) {
    Scalar sum = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            sum += M.m[i][j] * M.m[i][j];
        }
    }
    return sqrt(sum);
}

__device__ __host__ Scalar __frobenius_norm_Mat6x6(const Matrix6x6S& M) {
    Scalar sum = 0;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            sum += M.m[i][j] * M.m[i][j];
        }
    }
    return sqrt(sum);
}

__device__ __host__ Scalar __frobenius_norm_Mat9x9(const Matrix9x9S& M) {
    Scalar sum = 0;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            sum += M.m[i][j] * M.m[i][j];
        }
    }
    return sqrt(sum);
}

__device__ __host__ Scalar __frobenius_norm_Mat12x12(const Matrix12x12S& M) {
    Scalar sum = 0;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            sum += M.m[i][j] * M.m[i][j];
        }
    }
    return sqrt(sum);
}

__device__ __host__ Scalar __vec2_norm(const Scalar2& n) {
    return sqrt(n.x * n.x + n.y * n.y);
}

__device__ __host__ Scalar __vec3_norm(const Scalar3& n) {
    return sqrt(n.x * n.x + n.y * n.y + n.z * n.z);
}

__device__ __host__ Scalar __vec6_norm(const __MATHUTILS__::Vector6S& vec) {
    Scalar sum = 0;
    for (int i = 0; i < 6; ++i) {
        sum += vec.v[i] * vec.v[i];
    }
    return sqrt(sum);
}

__device__ __host__ Scalar __vec9_norm(const __MATHUTILS__::Vector9S& vec) {
    Scalar sum = 0;
    for (int i = 0; i < 9; ++i) {
        sum += vec.v[i] * vec.v[i];
    }
    return sqrt(sum);
}

__device__ __host__ Scalar __vec12_norm(const __MATHUTILS__::Vector12S& vec) {
    Scalar sum = 0;
    for (int i = 0; i < 12; ++i) {
        sum += vec.v[i] * vec.v[i];
    }
    return sqrt(sum);
}




__device__ __host__ Scalar2 __s_vec2_multiply(const Scalar2& a, Scalar b) {
    return make_Scalar2(a.x * b, a.y * b);
}

__device__ __host__ Scalar3 __s_vec3_multiply(const Scalar3& a, Scalar b) {
    return make_Scalar3(a.x * b, a.y * b, a.z * b);
}

__device__ __host__ Scalar2 __vec2_normalized(Scalar2 n) {
    Scalar norm = __vec2_norm(n);
    norm = 1 / norm;
    return __s_vec2_multiply(n, norm);
}

__device__ __host__ Scalar3 __vec3_normalized(Scalar3 n) {
    Scalar norm = __vec3_norm(n);
    norm = 1 / norm;
    return __s_vec3_multiply(n, norm);
}

__device__ __host__ Scalar3 __vec3_add(Scalar3 a, Scalar3 b) {
    return make_Scalar3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ Vector6S __vec6_add(const Vector6S& a, const Vector6S& b) {
    Vector6S V;
    for (int i = 0; i < 6; i++) {
        V.v[i] = a.v[i] + b.v[i];
    }
    return V;
}

__device__ __host__ Vector9S __vec9_add(const Vector9S& a, const Vector9S& b) {
    Vector9S V;
    for (int i = 0; i < 9; i++) {
        V.v[i] = a.v[i] + b.v[i];
    }
    return V;
}

__device__ __host__ Scalar2 __vec2_minus(Scalar2 a, Scalar2 b) {
    return make_Scalar2(a.x - b.x, a.y - b.y);
}

__device__ __host__ Scalar3 __vec3_minus(Scalar3 a, Scalar3 b) {
    return make_Scalar3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ Scalar __vec2_multiply(Scalar2 a, Scalar2 b) {
    return a.x * b.x + a.y * b.y;
}

__device__ __host__ Scalar3 __vec3_multiply(Scalar3 a, Scalar3 b) {
    return make_Scalar3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ Scalar __vec2_squaredNorm(Scalar2 a) {
    return a.x * a.x + a.y * a.y;
}

__device__ __host__ Scalar __vec3_squaredNorm(Scalar3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

__device__ __host__ void __M3x3_M3x3_multiply(const Matrix3x3S& A,
                                          const Matrix3x3S& B,
                                          Matrix3x3S& output) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
}

__device__ __host__ Matrix3x3S __M3x3_M3x3_multiply(const Matrix3x3S& A,
                                                const Matrix3x3S& B) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix2x2S __M2x2_M2x2_multiply(const Matrix2x2S& A,
                                                      const Matrix2x2S& B) {
    Matrix2x2S output;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Scalar __M3x3_Trace(const Matrix3x3S& A) {
    return A.m[0][0] + A.m[1][1] + A.m[2][2];
}

__device__ __host__ Scalar3 __v3_M3x3_multiply(const Scalar3& n,
                                           const Matrix3x3S& A) {
    Scalar x = A.m[0][0] * n.x + A.m[1][0] * n.y + A.m[2][0] * n.z;
    Scalar y = A.m[0][1] * n.x + A.m[1][1] * n.y + A.m[2][1] * n.z;
    Scalar z = A.m[0][2] * n.x + A.m[1][2] * n.y + A.m[2][2] * n.z;
    return make_Scalar3(x, y, z);
}

__device__ __host__ Scalar3 __M3x3_v3_multiply(const Matrix3x3S& A,
                                           const Scalar3& n) {
    Scalar x = A.m[0][0] * n.x + A.m[0][1] * n.y + A.m[0][2] * n.z;
    Scalar y = A.m[1][0] * n.x + A.m[1][1] * n.y + A.m[1][2] * n.z;
    Scalar z = A.m[2][0] * n.x + A.m[2][1] * n.y + A.m[2][2] * n.z;
    return make_Scalar3(x, y, z);
}

__device__ __host__ Scalar3 __M3x2_v2_multiply(const Matrix3x2S& A,
                                               const Scalar2& n) {
    Scalar x = A.m[0][0] * n.x + A.m[0][1] * n.y;  // +A.m[0][2] * n.z;
    Scalar y = A.m[1][0] * n.x + A.m[1][1] * n.y;  // +A.m[1][2] * n.z;
    Scalar z = A.m[2][0] * n.x + A.m[2][1] * n.y;  // +A.m[2][2] * n.z;
    return make_Scalar3(x, y, z);
}

__device__ __host__ Matrix3x2S __Mat3x2_add(const Matrix3x2S& A,
                                            const Matrix3x2S& B) {
    Matrix3x2S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[i][j] = A.m[i][j] + B.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix3x2S __s_Mat3x2_multiply(const Matrix3x2S& A,
                                                   const Scalar& b) {
    Matrix3x2S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[i][j] = A.m[i][j] * b;
        }
    }
    return output;
}

__device__ __host__ Vector12S __M12x9_v9_multiply(const Matrix12x9S& A,
                                                 const Vector9S& n) {
    Vector12S v12;
    for (int i = 0; i < 12; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 9; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v12.v[i] = temp;
    }
    return v12;
}

__device__ __host__ Vector12S __M12x6_v6_multiply(const Matrix12x6S& A,
                                                 const Vector6S& n) {
    Vector12S v12;
    for (int i = 0; i < 12; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 6; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v12.v[i] = temp;
    }
    return v12;
}

__device__ __host__ Vector6S __M6x3_v3_multiply(const Matrix6x3S& A,
                                               const Scalar3& n) {
    Vector6S v6;
    for (int i = 0; i < 6; i++) {
        Scalar temp = A.m[i][0] * n.x;
        temp += A.m[i][1] * n.y;
        temp += A.m[i][2] * n.z;

        v6.v[i] = temp;
    }
    return v6;
}

__device__ __host__ Scalar2 __M2x3_v3_multiply(const Matrix2x3S& A,
                                               const Scalar3& n) {
    Scalar2 output;
    output.x = A.m[0][0] * n.x + A.m[0][1] * n.y + A.m[0][2] * n.z;
    output.y = A.m[1][0] * n.x + A.m[1][1] * n.y + A.m[1][2] * n.z;
    return output;
}

__device__ __host__ Vector9S __M9x6_v6_multiply(const Matrix9x6S& A,
                                               const Vector6S& n) {
    Vector9S v9;
    for (int i = 0; i < 9; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 6; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v9.v[i] = temp;
    }
    return v9;
}

__device__ __host__ Vector12S __M12x12_v12_multiply(const Matrix12x12S& A,
                                                   const Vector12S& n) {
    Vector12S v12;
    for (int i = 0; i < 12; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 12; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v12.v[i] = temp;
    }
    return v12;
}

__device__ __host__ Vector9S __M9x9_v9_multiply(const Matrix9x9S& A,
                                               const Vector9S& n) {
    Vector9S v9;
    for (int i = 0; i < 9; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 9; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v9.v[i] = temp;
    }
    return v9;
}


__device__ __host__ void __add_Mat3x3_to_Mat9x9(
    __MATHUTILS__::Matrix9x9S& Hessian,
    const __MATHUTILS__::Matrix3x3S& block,
    int rowStart,
    int colStart
) {
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            Hessian.m[rowStart + i][colStart + j] += block.m[i][j];
        }
    }
}

__device__ __host__ __MATHUTILS__::Matrix3x3S __extract_Mat3x3_from_Mat9x9(
    const __MATHUTILS__::Matrix9x9S& Hessian,
    int rowStart,
    int colStart
) {
    __MATHUTILS__::Matrix3x3S block;
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            block.m[i][j] = Hessian.m[rowStart + i][colStart + j];
        }
    }
    return block;
}


__device__ __host__ void __add_Mat3x3_to_Mat12x12(
    __MATHUTILS__::Matrix12x12S& Hessian,
    const __MATHUTILS__::Matrix3x3S& block,
    int rowStart,
    int colStart
) {
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
            Hessian.m[rowStart + i][colStart + j] += block.m[i][j];
        }
    }
}


__device__ __host__ Vector6S __M6x6_v6_multiply(const Matrix6x6S& A,
                                               const Vector6S& n) {
    Vector6S v6;
    for (int i = 0; i < 6; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 6; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v6.v[i] = temp;
    }
    return v6;
}

__device__ __host__ Matrix9x9S __s_Mat9x9_multiply(const Matrix9x9S& A,
                                                   const Scalar& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            output.m[i][j] = A.m[i][j] * B;
        }
    }
    return output;
}

__device__ __host__ Matrix6x6S __s_Mat6x6_multiply(const Matrix6x6S& A,
                                                   const Scalar& B) {
    Matrix6x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            output.m[i][j] = A.m[i][j] * B;
        }
    }
    return output;
}

__device__ __host__ Scalar __vec3_dot(const Scalar3& a, const Scalar3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ Scalar3 __vec3_cross(Scalar3 a, Scalar3 b) {
    return make_Scalar3(a.y * b.z - a.z * b.y, 
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x);
}

__device__ __host__ Matrix3x3S __v_vec_toMat(Scalar3 a, Scalar3 b) {
    Matrix3x3S M;
    M.m[0][0] = a.x * b.x;
    M.m[0][1] = a.x * b.y;
    M.m[0][2] = a.x * b.z;
    M.m[1][0] = a.y * b.x;
    M.m[1][1] = a.y * b.y;
    M.m[1][2] = a.y * b.z;
    M.m[2][0] = a.z * b.x;
    M.m[2][1] = a.z * b.y;
    M.m[2][2] = a.z * b.z;
    return M;
}

__device__ __host__ Matrix2x2S __v2_vec2_toMat2x2(Scalar2 a, Scalar2 b) {
    Matrix2x2S M;
    M.m[0][0] = a.x * b.x;
    M.m[0][1] = a.x * b.y;
    M.m[1][0] = a.y * b.x;
    M.m[1][1] = a.y * b.y;
    return M;
}

__device__ __host__ Matrix2x2S __s_Mat2x2_multiply(Matrix2x2S A, Scalar b) {
    Matrix2x2S output;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[i][j] = A.m[i][j] * b;
        }
    }
    return output;
}

__device__ __host__ Matrix3x3S __s_Mat3x3_multiply(Matrix3x3S A, Scalar b) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            output.m[i][j] = A.m[i][j] * b;
        }
    }
    return output;
}

__device__ __host__ Matrix2x2S __Mat2x2_minus(Matrix2x2S A, Matrix2x2S B) {
    Matrix2x2S output;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[i][j] = A.m[i][j] - B.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix3x3S __Mat3x3_minus(Matrix3x3S A, Matrix3x3S B) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            output.m[i][j] = A.m[i][j] - B.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix6x6S __v6_vec6_toMat6x6(Vector6S a, Vector6S b) {
    Matrix6x6S M;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            M.m[i][j] = a.v[i] * b.v[j];
        }
    }
    return M;
}

__device__ __host__ Matrix9x9S __v9_vec9_toMat9x9(const Vector9S& a,
                                                  const Vector9S& b,
                                                  const Scalar& coe) {
    Matrix9x9S M;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            M.m[i][j] = a.v[i] * b.v[j] * coe;
        }
    }
    return M;
}

__device__ __host__ Matrix12x12S __v12_vec12_toMat12x12(const Vector12S& a,
                                                  const Vector12S& b) {
    Matrix12x12S M;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            M.m[i][j] = a.v[i] * b.v[j];
        }
    }
    return M;
}

__device__ __host__ Vector9S __s_vec9_multiply(Vector9S a, Scalar b) {
    Vector9S V;
    for (int i = 0; i < 9; i++) V.v[i] = a.v[i] * b;
    return V;
}

__device__ __host__ Vector12S __s_vec12_multiply(Vector12S a, Scalar b) {
    Vector12S V;
    for (int i = 0; i < 12; i++) V.v[i] = a.v[i] * b;
    return V;
}

__device__ __host__ Vector6S __s_vec6_multiply(Vector6S a, Scalar b) {
    Vector6S V;
    for (int i = 0; i < 6; i++) V.v[i] = a.v[i] * b;
    return V;
}

__device__ __host__ void __Mat_add(const Matrix3x3S& A, const Matrix3x3S& B,
                                   Matrix3x3S& output) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
}

__device__ __host__ void __Mat_add(const Matrix6x6S& A, const Matrix6x6S& B,
                                   Matrix6x6S& output) {
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
}

__device__ __host__ Matrix3x3S __Mat_add(const Matrix3x3S& A,
                                         const Matrix3x3S& B) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix2x2S __Mat2x2_add(const Matrix2x2S& A,
                                            const Matrix2x2S& B) {
    Matrix2x2S output;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix9x9S __Mat9x9_add(const Matrix9x9S& A,
                                            const Matrix9x9S& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix12x12S __Mat12x12_add(const Matrix12x12S& A,
                                            const Matrix12x12S& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix9x12S __Mat9x12_add(const Matrix9x12S& A,
                                              const Matrix9x12S& B) {
    Matrix9x12S output;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 12; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix6x12S __Mat6x12_add(const Matrix6x12S& A,
                                              const Matrix6x12S& B) {
    Matrix6x12S output;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 12; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix6x9S __Mat6x9_add(const Matrix6x9S& A,
                                            const Matrix6x9S& B) {
    Matrix6x9S output;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 9; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}

__device__ __host__ Matrix3x6S __Mat3x6_add(const Matrix3x6S& A,
                                            const Matrix3x6S& B) {
    Matrix3x6S output;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 6; j++) output.m[i][j] = A.m[i][j] + B.m[i][j];
    return output;
}


__device__ __host__ Matrix3x3S __v_to_crossMat(Scalar3 v) {
    Matrix3x3S retMat;
    retMat.m[0][0] = 0;      retMat.m[0][1] = -v.z;  retMat.m[0][2] = v.y;
    retMat.m[1][0] = v.z;    retMat.m[1][1] = 0;     retMat.m[1][2] = -v.x;
    retMat.m[2][0] = -v.y;   retMat.m[2][1] = v.x;   retMat.m[2][2] = 0;
    return retMat;
}

__device__ __host__ void __set_Mat_identity(Matrix2x2S& M) {
    M.m[0][0] = 1;
    M.m[1][0] = 0;
    M.m[0][1] = 0;
    M.m[1][1] = 1;
}

__device__ __host__ void __set_Mat_val(Matrix3x3S& M, const Scalar& a00,
                                       const Scalar& a01, const Scalar& a02,
                                       const Scalar& a10, const Scalar& a11,
                                       const Scalar& a12, const Scalar& a20,
                                       const Scalar& a21, const Scalar& a22) {
    M.m[0][0] = a00;
    M.m[0][1] = a01;
    M.m[0][2] = a02;
    M.m[1][0] = a10;
    M.m[1][1] = a11;
    M.m[1][2] = a12;
    M.m[2][0] = a20;
    M.m[2][1] = a21;
    M.m[2][2] = a22;
}

__device__ __host__ void __set_Mat_val_row(Matrix3x3S& M, const Scalar3& row0,
                                           const Scalar3& row1,
                                           const Scalar3& row2) {
    M.m[0][0] = row0.x;
    M.m[0][1] = row0.y;
    M.m[0][2] = row0.z;
    M.m[1][0] = row1.x;
    M.m[1][1] = row1.y;
    M.m[1][2] = row1.z;
    M.m[2][0] = row2.x;
    M.m[2][1] = row2.y;
    M.m[2][2] = row2.z;
}

__device__ __host__ void __set_Mat_val_column(Matrix3x3S& M,
                                              const Scalar3& col0,
                                              const Scalar3& col1,
                                              const Scalar3& col2) {
    M.m[0][0] = col0.x;
    M.m[0][1] = col1.x;
    M.m[0][2] = col2.x;
    M.m[1][0] = col0.y;
    M.m[1][1] = col1.y;
    M.m[1][2] = col2.y;
    M.m[2][0] = col0.z;
    M.m[2][1] = col1.z;
    M.m[2][2] = col2.z;
}

__device__ __host__ void __set_Mat3x2_val_column(Matrix3x2S& M,
                                                 const Scalar3& col0,
                                                 const Scalar3& col1) {
    M.m[0][0] = col0.x;
    M.m[0][1] = col1.x;
    M.m[1][0] = col0.y;
    M.m[1][1] = col1.y;
    M.m[2][0] = col0.z;
    M.m[2][1] = col1.z;
}

__device__ __host__ void __set_Mat2x2_val_column(Matrix2x2S& M,
                                                 const Scalar2& col0,
                                                 const Scalar2& col1) {
    M.m[0][0] = col0.x;
    M.m[0][1] = col1.x;
    M.m[1][0] = col0.y;
    M.m[1][1] = col1.y;
}

__device__ __host__ void __init_Mat9x12_val(Matrix9x12S& M, const Scalar& val) {
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 12; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat6x12_val(Matrix6x12S& M, const Scalar& val) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 12; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat6x9_val(Matrix6x9S& M, const Scalar& val) {
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 9; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ void __init_Mat3x6_val(Matrix3x6S& M, const Scalar& val) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            M.m[i][j] = val;
        }
    }
}

__device__ __host__ Matrix3x3S __s_M3x3_multiply(const Matrix3x3S& A,
                                                const Scalar& B) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) output.m[i][j] = A.m[i][j] * B;
    return output;
}

__device__ __host__ Matrix3x3S __Transpose3x3(Matrix3x3S input) {
    Matrix3x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            output.m[i][j] = input.m[j][i];
        }
    }
    return output;
}

__device__ __host__ Matrix12x9S __Transpose9x12(const Matrix9x12S& input) {
    Matrix12x9S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 12; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix2x3S __Transpose3x2(const Matrix3x2S& input) {
    Matrix2x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix9x12S __Transpose12x9(const Matrix12x9S& input) {
    Matrix9x12S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 9; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix12x6S __Transpose6x12(const Matrix6x12S& input) {
    Matrix12x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 12; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix9x6S __Transpose6x9(const Matrix6x9S& input) {
    Matrix9x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 9; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix6x3S __Transpose3x6(const Matrix3x6S& input) {
    Matrix6x3S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 6; j++) {
            output.m[j][i] = input.m[i][j];
        }
    }
    return output;
}

__device__ __host__ Matrix12x9S __M12x9_M9x9_Multiply(const Matrix12x9S& A,
                                                      const Matrix9x9S& B) {
    Matrix12x9S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 9; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x6S __M12x6_M6x6_Multiply(const Matrix12x6S& A,
                                                      const Matrix6x6S& B) {
    Matrix12x6S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 6; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 6; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix9x6S __M9x6_M6x6_Multiply(const Matrix9x6S& A,
                                                    const Matrix6x6S& B) {
    Matrix9x6S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 6; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 6; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix6x3S __M6x3_M3x3_Multiply(const Matrix6x3S& A,
                                                    const Matrix3x3S& B) {
    Matrix6x3S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix3x2S __M3x2_M2x2_Multiply(const Matrix3x2S& A,
                                                    const Matrix2x2S& B) {
    Matrix3x2S output;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x12S __M12x9_M9x12_Multiply(const Matrix12x9S& A,
                                                        const Matrix9x12S& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 9; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x2S __M12x2_M2x2_Multiply(const Matrix12x2S& A,
                                                      const Matrix2x2S& B) {
    Matrix12x2S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix9x2S __M9x2_M2x2_Multiply(const Matrix9x2S& A,
                                                    const Matrix2x2S& B) {
    Matrix9x2S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix6x2S __M6x2_M2x2_Multiply(const Matrix6x2S& A,
                                                    const Matrix2x2S& B) {
    Matrix6x2S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 2; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x12S __M12x2_M12x2T_Multiply(const Matrix12x2S& A,
                                                         const Matrix12x2S& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix9x9S __M9x2_M9x2T_Multiply(const Matrix9x2S& A,
                                                     const Matrix9x2S& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix6x6S __M6x2_M6x2T_Multiply(const Matrix6x2S& A,
                                                     const Matrix6x2S& B) {
    Matrix6x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 2; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x12S __M12x6_M6x12_Multiply(const Matrix12x6S& A,
                                                        const Matrix6x12S& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 6; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix9x9S __M9x6_M6x9_Multiply(const Matrix9x6S& A,
                                                    const Matrix6x9S& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 6; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix6x6S __M6x3_M3x6_Multiply(const Matrix6x3S& A,
                                                    const Matrix3x6S& B) {
    Matrix6x6S output;
    for (int i = 0; i < 6; i++) {
        for (int j = 0; j < 6; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            output.m[i][j] = temp;
        }
    }
    return output;
}

__device__ __host__ Matrix12x12S __s_M12x12_Multiply(const Matrix12x12S& A,
                                                     const Scalar& B) {
    Matrix12x12S output;
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++) output.m[i][j] = A.m[i][j] * B;
    return output;
}

__device__ __host__ Matrix9x9S __s_M9x9_Multiply(const Matrix9x9S& A,
                                                 const Scalar& B) {
    Matrix9x9S output;
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++) output.m[i][j] = A.m[i][j] * B;
    return output;
}

__device__ __host__ Matrix6x6S __s_M6x6_Multiply(const Matrix6x6S& A,
                                                 const Scalar& B) {
    Matrix6x6S output;
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 6; j++) output.m[i][j] = A.m[i][j] * B;
    return output;
}

__device__ __host__ void __Determiant(const Matrix3x3S& input,
                                      Scalar& determinant) {
    determinant = input.m[0][0] * input.m[1][1] * input.m[2][2] +
                  input.m[1][0] * input.m[2][1] * input.m[0][2] +
                  input.m[2][0] * input.m[0][1] * input.m[1][2] -
                  input.m[2][0] * input.m[1][1] * input.m[0][2] -
                  input.m[0][0] * input.m[1][2] * input.m[2][1] -
                  input.m[0][1] * input.m[1][0] * input.m[2][2];
}

__device__ __host__ Scalar __Determiant(const Matrix3x3S& input) {
    return input.m[0][0] * input.m[1][1] * input.m[2][2] +
           input.m[1][0] * input.m[2][1] * input.m[0][2] +
           input.m[2][0] * input.m[0][1] * input.m[1][2] -
           input.m[2][0] * input.m[1][1] * input.m[0][2] -
           input.m[0][0] * input.m[1][2] * input.m[2][1] -
           input.m[0][1] * input.m[1][0] * input.m[2][2];
}

__device__ __host__ void __Inverse(const Matrix3x3S& input,
                                   Matrix3x3S& result) {
    Scalar eps = very_small_number();
    const int dim = 3;
    Scalar mat[dim][dim * 2];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < 2 * dim; j++) {
            if (j < dim) {
                mat[i][j] = input.m[i][j];  //[i, j];
            } else {
                mat[i][j] = j - dim == i ? 1 : 0;
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        if (abs(mat[i][i]) < eps) {
            int j;
            for (j = i + 1; j < dim; j++) {
                if (abs(mat[j][i]) > eps) break;
            }
            if (j == dim) return;
            for (int r = i; r < 2 * dim; r++) {
                mat[i][r] += mat[j][r];
            }
        }
        Scalar ep = mat[i][i];
        for (int r = i; r < 2 * dim; r++) {
            mat[i][r] /= ep;
        }

        for (int j = i + 1; j < dim; j++) {
            Scalar e = -1 * (mat[j][i] / mat[i][i]);
            for (int r = i; r < 2 * dim; r++) {
                mat[j][r] += e * mat[i][r];
            }
        }
    }

    for (int i = dim - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            Scalar e = -1 * (mat[j][i] / mat[i][i]);
            for (int r = i; r < 2 * dim; r++) {
                mat[j][r] += e * mat[i][r];
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        for (int r = dim; r < 2 * dim; r++) {
            result.m[i][r - dim] = mat[i][r];
        }
    }
}

__device__ __host__ void __Inverse2x2(const Matrix2x2S& input,
                                      Matrix2x2S& result) {
    Scalar eps = very_small_number();
    const int dim = 2;
    Scalar mat[dim][dim * 2];
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < 2 * dim; j++) {
            if (j < dim) {
                mat[i][j] = input.m[i][j];  //[i, j];
            } else {
                mat[i][j] = j - dim == i ? 1 : 0;
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        if (abs(mat[i][i]) < eps) {
            int j;
            for (j = i + 1; j < dim; j++) {
                if (abs(mat[j][i]) > eps) break;
            }
            if (j == dim) return;
            for (int r = i; r < 2 * dim; r++) {
                mat[i][r] += mat[j][r];
            }
        }
        Scalar ep = mat[i][i];
        for (int r = i; r < 2 * dim; r++) {
            mat[i][r] /= ep;
        }

        for (int j = i + 1; j < dim; j++) {
            Scalar e = -1 * (mat[j][i] / mat[i][i]);
            for (int r = i; r < 2 * dim; r++) {
                mat[j][r] += e * mat[i][r];
            }
        }
    }

    for (int i = dim - 1; i >= 0; i--) {
        for (int j = i - 1; j >= 0; j--) {
            Scalar e = -1 * (mat[j][i] / mat[i][i]);
            for (int r = i; r < 2 * dim; r++) {
                mat[j][r] += e * mat[i][r];
            }
        }
    }

    for (int i = 0; i < dim; i++) {
        for (int r = dim; r < 2 * dim; r++) {
            result.m[i][r - dim] = mat[i][r];
        }
    }
}

__device__ __host__ Scalar __f(const Scalar& x, const Scalar& a,
                               const Scalar& b, const Scalar& c,
                               const Scalar& d) {
    Scalar f = a * x * x * x + b * x * x + c * x + d;
    return f;
}

__device__ __host__ Scalar __df(const Scalar& x, const Scalar& a,
                                const Scalar& b, const Scalar& c) {
    Scalar df = 3 * a * x * x + 2 * b * x + c;
    return df;
}

__device__ __host__ void __NewtonSolverForCubicEquation(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS) {
    // Scalar EPS = 1e-6;
    Scalar DX = 0;
    // Scalar results[3];
    num_solutions = 0;
    Scalar specialPoint = -b / a / 3;
    Scalar pos[2];
    int solves = 1;
    Scalar delta = 4 * b * b - 12 * a * c;
    if (delta > 0) {
        pos[0] = (sqrt(delta) - 2 * b) / 6 / a;
        pos[1] = (-sqrt(delta) - 2 * b) / 6 / a;
        Scalar v1 = __f(pos[0], a, b, c, d);
        Scalar v2 = __f(pos[1], a, b, c, d);
        if (std::abs(v1) < EPS * EPS) {
            v1 = 0;
        }
        if (std::abs(v2) < EPS * EPS) {
            v2 = 0;
        }
        Scalar sign = v1 * v2;
        DX = (pos[0] - pos[1]);
        if (sign <= 0) {
            solves = 3;
        } else if (sign > 0) {
            if ((a < 0 && __f(pos[0], a, b, c, d) > 0) ||
                (a > 0 && __f(pos[0], a, b, c, d) < 0)) {
                DX = -DX;
            }
        }
    } else if (delta == 0) {
        if (std::abs(__f(specialPoint, a, b, c, d)) < EPS * EPS) {
            for (int i = 0; i < 3; i++) {
                Scalar tempReuslt = specialPoint;
                results[num_solutions] = tempReuslt;
                num_solutions++;
            }
            return;
        }
        if (a > 0) {
            if (__f(specialPoint, a, b, c, d) > 0) {
                DX = 1;
            } else if (__f(specialPoint, a, b, c, d) < 0) {
                DX = -1;
            }
        } else if (a < 0) {
            if (__f(specialPoint, a, b, c, d) > 0) {
                DX = -1;
            } else if (__f(specialPoint, a, b, c, d) < 0) {
                DX = 1;
            }
        }
    }

    Scalar start = specialPoint - DX;
    Scalar x0 = start;
    // Scalar result[3];

    for (int i = 0; i < solves; i++) {
        Scalar x1 = 0;
        int itCount = 0;
        do {
            if (itCount) x0 = x1;

            x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
            itCount++;

        } while (std::abs(x1 - x0) > EPS && itCount < 100000);
        results[num_solutions] = (x1);
        num_solutions++;
        start = start + DX;
        x0 = start;
    }
}

__device__ __host__ void __NewtonSolverForCubicEquation_satbleNeohook(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS) {
    Scalar DX = 0;
    num_solutions = 0;
    Scalar specialPoint = -b / 3;
    Scalar pos[2];
    int solves = 1;
    Scalar delta = 4 * b * b - 12 * c;
    Scalar sign = -1;
    if (delta > 0) {
        pos[0] = (sqrt(delta) - 2 * b) / 6;
        pos[1] = (-sqrt(delta) - 2 * b) / 6;
        Scalar v1 = __f(pos[0], a, b, c, d);
        Scalar v2 = __f(pos[1], a, b, c, d);
        DX = (pos[0] - pos[1]);
        if ((v1) >= 0) {
            v1 = 0;
            results[1] = pos[0];
            results[2] = pos[0];
        }
        if ((v2) <= 0) {
            v2 = 0;
            results[1] = pos[1];
            results[2] = pos[1];
        }
        sign = v1 * v2;

        if (sign < 0) {
            solves = 3;
        } else {
            if ((v2 <= 0)) {
                DX = -DX;
            }
        }
    } else {
        results[0] = specialPoint;
        results[1] = specialPoint;
        results[2] = specialPoint;
        num_solutions = 3;
        return;
    }

    Scalar start = specialPoint - DX;
    Scalar x0 = start;
    // Scalar result[3];

    for (int i = 0; i < solves; i++) {
        Scalar x1 = 0;
        int itCount = 0;
        do {
            if (itCount) x0 = x1;

            x1 = x0 - ((__f(x0, a, b, c, d)) / (__df(x0, a, b, c)));
            itCount++;

        } while (std::abs(x1 - x0) > EPS && itCount < 100000);
        results[num_solutions] = (x1);
        num_solutions++;
        start = start + DX;
        x0 = start;
    }
    // printf("%f   %f    %f    %f    %f\n", specialPoint, DX, results[0],
    // results[1], results[2]);
    num_solutions = 3;
}

__device__ __host__ void __SolverForCubicEquation(
    const Scalar& a, const Scalar& b, const Scalar& c, const Scalar& d,
    Scalar* results, int& num_solutions, Scalar EPS) {
    Scalar A = b * b - 3 * a * c;
    Scalar B = b * c - 9 * a * d;
    Scalar C = c * c - 3 * b * d;
    Scalar delta = B * B - 4 * A * C;
    num_solutions = 0;
    if (abs(A) < EPS * EPS && abs(B) < EPS * EPS) {
        results[0] = -b / 3.0 / a;
        results[1] = results[0];
        results[2] = results[0];
        num_solutions = 3;
    } else if (abs(delta) <= EPS * EPS) {
        Scalar K = B / A;
        results[0] = -b / a + K;
        results[1] = -K / 2.0;
        results[2] = results[1];
        num_solutions = 3;
    } else if (delta < -EPS * EPS) {
        Scalar T = (2 * A * b - 3 * a * B) / (2 * A * sqrt(A));
        Scalar theta = acos(T);
        results[0] = (-b - 2 * sqrt(A) * cos(theta / 3.0)) / (3 * a);
        results[1] =
            (-b + sqrt(A) * (cos(theta / 3.0) + sqrt(3.0) * sin(theta / 3.0))) /
            (3 * a);
        results[2] =
            (-b + sqrt(A) * (cos(theta / 3.0) - sqrt(3.0) * sin(theta / 3.0))) /
            (3 * a);
        num_solutions = 3;
    } else if (delta > EPS * EPS) {
        Scalar Y1 = A * b + 3 * a * (-B + sqrt(delta)) / 2;
        Scalar Y2 = A * b + 3 * a * (-B - sqrt(delta)) / 2;
        results[0] = -b - cbrt(Y1) - cbrt(Y2);
        num_solutions = 1;
    }
}

__device__ __host__ Vector9S __Mat3x3_to_vec9_Scalar(const Matrix3x3S& F) {
    Vector9S result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.v[i * 3 + j] = F.m[j][i];
        }
    }
    return result;
}

__device__ __host__ void __normalized_vec9_Scalar(Vector9S& v9) {
    Scalar length = 0;
    for (int i = 0; i < 9; i++) {
        length += v9.v[i] * v9.v[i];
    }
    length = 1.0 / sqrt(length);
    for (int i = 0; i < 9; i++) {
        v9.v[i] = v9.v[i] * length;
    }
}

__device__ __host__ void __normalized_vec6_Scalar(Vector6S& v6) {
    Scalar length = 0;
    for (int i = 0; i < 6; i++) {
        length += v6.v[i] * v6.v[i];
    }
    length = 1.0 / sqrt(length);
    for (int i = 0; i < 6; i++) {
        v6.v[i] = v6.v[i] * length;
    }
}

__device__ __host__ Vector6S __Mat3x2_to_vec6_Scalar(const Matrix3x2S& F) {
    Vector6S result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            result.v[i * 3 + j] = F.m[j][i];
        }
    }
    return result;
}

__device__ __host__ Matrix3x3S __vec9_to_Mat3x3_Scalar(const Scalar vec9[9]) {
    Matrix3x3S mat;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mat.m[j][i] = vec9[i * 3 + j];
        }
    }
    return mat;
}

__device__ __host__ Matrix2x2S __vec4_to_Mat2x2_Scalar(const Scalar vec4[4]) {
    Matrix2x2S mat;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            mat.m[j][i] = vec4[i * 2 + j];
        }
    }
    return mat;
}

__device__ __host__ Scalar __s_clamp(Scalar val, Scalar min_val, Scalar max_val) {
    return max(min_val, min(val, max_val));
}

// 没有SVD分解
__device__ void SVD(const Matrix3x3S& F, Matrix3x3S& Uout, Matrix3x3S& Vout, Matrix3x3S& Sigma) {
    // #include "Eigen/Core"
    // #include "Eigen/SVD"
    // const Eigen::JacobiSVD<MATRIX3,Eigen::NoQRPreconditioner>
}




__device__ __host__ void __makePD2x2(const Scalar& a00, const Scalar& a01,
                                     const Scalar& a10, const Scalar& a11,
                                     Scalar eigenValues[2], int& num,
                                     Scalar2 eigenVectors[2]) {
    Scalar b = -(a00 + a11), c = a00 * a11 - a10 * a01;
    Scalar existEv = b * b - 4 * c;
    if ((a01) == 0 || (a10) == 0) {
        if (a00 > 0) {
            eigenValues[num] = a00;
            eigenVectors[num].x = 1;
            eigenVectors[num].y = 0;
            num++;
        }
        if (a11 > 0) {
            eigenValues[num] = a11;
            eigenVectors[num].x = 0;
            eigenVectors[num].y = 1;
            num++;
        }
    } else {
        if (existEv > 0) {
            num = 2;
            eigenValues[0] = (-b - sqrt(existEv)) / 2;
            eigenVectors[0].x = 1;
            eigenVectors[0].y = (eigenValues[0] - a00) / a01;
            Scalar length = sqrt(eigenVectors[0].x * eigenVectors[0].x +
                                 eigenVectors[0].y * eigenVectors[0].y);
            // eigenValues[0] *= length;
            eigenVectors[0].x /= length;
            eigenVectors[0].y /= length;

            eigenValues[1] = (-b + sqrt(existEv)) / 2;
            eigenVectors[1].x = 1;
            eigenVectors[1].y = (eigenValues[1] - a00) / a01;
            length = sqrt(eigenVectors[1].x * eigenVectors[1].x +
                          eigenVectors[1].y * eigenVectors[1].y);
            // eigenValues[1] *= length;
            eigenVectors[1].x /= length;
            eigenVectors[1].y /= length;
        } else if (existEv == 0) {
            num = 1;
            eigenValues[0] = (-b - sqrt(existEv)) / 2;
            eigenVectors[0].x = 1;
            eigenVectors[0].y = (eigenValues[0] - a00) / a01;
            Scalar length = sqrt(eigenVectors[0].x * eigenVectors[0].x +
                                 eigenVectors[0].y * eigenVectors[0].y);
            // eigenValues[0] *= length;
            eigenVectors[0].x /= length;
            eigenVectors[0].y /= length;
        } else {
            num = 0;
        }
    }
}

__device__ __host__ void __M9x4_S4x4_MT4x9_Multiply(const Matrix9x4S& A,
                                                    const Matrix4x4S& B,
                                                    Matrix9x9S& output) {
    // Matrix12x12S output;
    Vector4S tempM;
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 4; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 4; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            tempM.v[j] = temp;
        }

        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 4; k++) {
                temp += A.m[j][k] * tempM.v[k];
            }
            output.m[i][j] = temp;
        }
    }
    // return output;
}

__device__ __host__ void __M12x9_S9x9_MT9x12_Multiply(const Matrix12x9S& A,
                                                      const Matrix9x9S& B,
                                                      Matrix12x12S& output) {
    // Matrix12x12S output;
    Vector9S tempM;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 9; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 9; k++) {
                temp += A.m[i][k] * B.m[j][k];
            }
            tempM.v[j] = temp;
        }

        for (int j = 0; j < 12; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 9; k++) {
                temp += A.m[j][k] * tempM.v[k];
            }
            output.m[i][j] = temp;
        }
    }
    // return output;
}

__device__ __host__ Vector4S __s_vec4_multiply(Vector4S a, Scalar b) {
    Vector4S V;
    for (int i = 0; i < 4; i++) V.v[i] = a.v[i] * b;
    return V;
}

__device__ __host__ Vector9S __M9x4_v4_multiply(const Matrix9x4S& A,
                                               const Vector4S& n) {
    Vector9S v9;
    for (int i = 0; i < 9; i++) {
        Scalar temp = 0;
        for (int j = 0; j < 4; j++) {
            temp += A.m[i][j] * n.v[j];
        }
        v9.v[i] = temp;
    }
    return v9;
}

__device__ __host__ Matrix4x4S __s_Mat4x4_multiply(const Matrix4x4S& A,
                                                   const Scalar& B) {
    Matrix4x4S output;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            output.m[i][j] = A.m[i][j] * B;
        }
    }
    return output;
}

__device__ __host__ Matrix4x4S __v4_vec4_toMat4x4(Vector4S a, Vector4S b) {
    Matrix4x4S M;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            M.m[i][j] = a.v[i] * b.v[j];
        }
    }
    return M;
}

__device__ __host__ void __s_M_Mat_MT_multiply(const Matrix3x3S& A,
                                               const Matrix3x3S& B,
                                               const Matrix3x3S& C,
                                               const Scalar& coe,
                                               Matrix3x3S& output) {
    Scalar tvec3[3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += A.m[i][k] * B.m[k][j];
            }
            // output.m[i][j] = temp;
            tvec3[j] = temp;
        }

        for (int j = 0; j < 3; j++) {
            Scalar temp = 0;
            for (int k = 0; k < 3; k++) {
                temp += C.m[j][k] * tvec3[k];
            }
            output.m[i][j] = temp * coe;
            // tvec3[j] = temp;
        }
    }
}





}  // namespace __MATHUTILS__






// reduction
namespace __MATHUTILS__ {



__device__
void _d_PP(const Scalar3& v0, const Scalar3& v1, Scalar& d)
{
    d = __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(v0, v1));
}

__device__
void _d_PT(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d)
{
    Scalar3 b = __MATHUTILS__::__vec3_cross(__MATHUTILS__::__vec3_minus(v2, v1), __MATHUTILS__::__vec3_minus(v3, v1));
    Scalar3 test = __MATHUTILS__::__vec3_minus(v0, v1);
    Scalar aTb = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_minus(v0, v1), b);//(v0 - v1).dot(b);
    //printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y, test.z, aTb);
    d = aTb * aTb / __MATHUTILS__::__vec3_squaredNorm(b);
}

__device__
void _d_PE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, Scalar& d)
{
    d = __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_cross(__MATHUTILS__::__vec3_minus(v1, v0), __MATHUTILS__::__vec3_minus(v2, v0))) / __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(v2, v1));
}

__device__
void _d_EE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d)
{
    Scalar3 b = __MATHUTILS__::__vec3_cross(__MATHUTILS__::__vec3_minus(v1, v0), __MATHUTILS__::__vec3_minus(v3, v2));//(v1 - v0).cross(v3 - v2);
    Scalar aTb = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __MATHUTILS__::__vec3_squaredNorm(b);
}


__device__
void _d_EEParallel(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3, Scalar& d)
{
    Scalar3 b = __MATHUTILS__::__vec3_cross(__MATHUTILS__::__vec3_cross(__MATHUTILS__::__vec3_minus(v1, v0), __MATHUTILS__::__vec3_minus(v2, v0)), __MATHUTILS__::__vec3_minus(v1, v0));
    Scalar aTb = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_minus(v2, v0), b);//(v2 - v0).dot(b);
    d = aTb * aTb / __MATHUTILS__::__vec3_squaredNorm(b);
}

__device__
Scalar _compute_epx(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3) {
    return 1e-3 * __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(v0, v1)) * __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(v2, v3));
}

__device__
Scalar _compute_epx_cp(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3) {
    return 1e-3 * __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(v0, v1)) * __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(v2, v3));
}

__device__ __host__
Scalar calculateVolume(const Scalar3* vertexes, const uint4& index) {

    Scalar o1x = vertexes[index.y].x - vertexes[index.x].x;
    Scalar o1y = vertexes[index.y].y - vertexes[index.x].y;
    Scalar o1z = vertexes[index.y].z - vertexes[index.x].z;
    Scalar3 OA = make_Scalar3(o1x, o1y, o1z);

    Scalar o2x = vertexes[index.z].x - vertexes[index.x].x;
    Scalar o2y = vertexes[index.z].y - vertexes[index.x].y;
    Scalar o2z = vertexes[index.z].z - vertexes[index.x].z;
    Scalar3 OB = make_Scalar3(o2x, o2y, o2z);

    Scalar o3x = vertexes[index.w].x - vertexes[index.x].x;
    Scalar o3y = vertexes[index.w].y - vertexes[index.x].y;
    Scalar o3z = vertexes[index.w].z - vertexes[index.x].z;
    Scalar3 OC = make_Scalar3(o3x, o3y, o3z);

    Scalar3 heightDir = __MATHUTILS__::__vec3_cross(OA, OB);  // OA.cross(OB);
    Scalar bottomArea = __MATHUTILS__::__vec3_norm(heightDir);      // heightDir.norm();
    heightDir = __MATHUTILS__::__vec3_normalized(heightDir);

    Scalar volum = bottomArea * __MATHUTILS__::__vec3_dot(heightDir, OC) / 6;
    return volum > 0 ? volum : -volum;
}

__device__ __host__
Scalar calculateArea(const Scalar3* vertexes, const uint3& index) {
    Scalar3 v10 = __MATHUTILS__::__vec3_minus(vertexes[index.y], vertexes[index.x]);
    Scalar3 v20 = __MATHUTILS__::__vec3_minus(vertexes[index.z], vertexes[index.x]);
    Scalar area = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v10, v20));
    return 0.5 * area;
}


// 定义 _permutation 数组为设备常量内存
__device__ __constant__ int _permutation[512] = {
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
    49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    // 重复排列数组
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,172,9,
    129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,107,
    49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

__device__ Scalar _noise_fade(Scalar t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ Scalar _noise_lerp(Scalar a, Scalar b, Scalar t) {
    return a + t * (b - a);
}

__device__ Scalar _noise_grad(int hash, Scalar x, Scalar y, Scalar z) {
    int h = hash & 15;
    Scalar u = h < 8 ? x : y;
    Scalar v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}

__device__ Scalar _perlinNoise(Scalar x, Scalar y, Scalar z) {
    int X = (int)floorf(x) & 255;
    int Y = (int)floorf(y) & 255;
    int Z = (int)floorf(z) & 255;

    x -= floorf(x);
    y -= floorf(y);
    z -= floorf(z);

    Scalar u = _noise_fade(x);
    Scalar v = _noise_fade(y);
    Scalar w = _noise_fade(z);

    int A = _permutation[X] + Y;
    int AA = _permutation[A] + Z;
    int AB = _permutation[A + 1] + Z;
    int B = _permutation[X + 1] + Y;
    int BA = _permutation[B] + Z;
    int BB = _permutation[B + 1] + Z;

    Scalar res = _noise_lerp(w,
        _noise_lerp(v,
            _noise_lerp(u, _noise_grad(_permutation[AA], x, y, z), _noise_grad(_permutation[BA], x - 1, y, z)),
            _noise_lerp(u, _noise_grad(_permutation[AB], x, y - 1, z), _noise_grad(_permutation[BB], x - 1, y - 1, z))),
        _noise_lerp(v, 
            _noise_lerp(u, _noise_grad(_permutation[AA + 1], x, y, z - 1), _noise_grad(_permutation[BA + 1], x - 1, y, z - 1)),
            _noise_lerp(u, _noise_grad(_permutation[AB + 1], x, y - 1, z - 1), _noise_grad(_permutation[BB + 1], x - 1, y - 1, z - 1)))
    );

    return res;
}


}  // namespace __MATHUTILS__



// reduction
namespace __MATHUTILS__ {

// 对单block内的所有的thread进行reductadd 首先进行warp内的add将结果储存在shared memory中 然后进行warp之间的add 
__device__ void __perform_reduct_add_Scalar(
    Scalar* squeue, Scalar temp, int numbers) {

    /////////////////////////////////////////////////
    // Block (blockDim.x = 64 threads)
    // +-------------------------------------------------------+
    // | Thread 0 | Thread 1 | ... | Thread 31 | Thread 32 | ... | Thread 63 |
    // |  temp0   |  temp1   | ... |  temp31  |  temp32  | ... |  temp63  |
    // +-------------------------------------------------------+
    /////////////////////////////////////////////////
    // Warp 0 (Threads 0-31):
    // +---------+---------+---------+---------+---------+---------+---------+---------+
    // | temp0   | temp1   | temp2   | ...     | temp30  | temp31  |
    // +---------+---------+---------+---------+---------+---------+---------+---------+
    // After warp-level reduction:
    // Thread 0 holds the sum of temp0 to temp31
    // .......
    // Warp 1 (Threads 32-63):
    // +---------+---------+---------+---------+---------+---------+---------+---------+
    // | temp32  | temp33  | temp34  | ...     | temp62  | temp63  |
    // +---------+---------+---------+---------+---------+---------+---------+---------+
    // After warp-level reduction:
    // Thread 32 holds the sum of temp32 to temp63
    /////////////////////////////////////////////////
    // Shared Memory (`tep1`):
    // +----------+----------+
    // | tep0     | tep1     |
    // +----------+----------+
    // | sum_warp0| sum_warp1|
    // +----------+----------+
    /////////////////////////////////////////////////
    // Block-level reduction:
    // sum_warp0 + sum_warp1 = total_sum_block
    // squeue[blockIdx.x] = total_sum_block
    /////////////////////////////////////////////////

    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);

    extern __shared__ Scalar tep1[];

    int warpNum;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = (blockDim.x >> 5);
    }

    // Warp-level inside 32 threads add serial
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
    }

    // store warp ans to shared memory
    if (warpTid == 0) {
        tep1[warpId] = temp;
    }

    // wait all warps cal finished
    __syncthreads();

    if (threadIdx.x >= warpNum) return;

    // add all warp together 
    if (warpNum > 1) {
        temp = tep1[threadIdx.x];
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }

    // write ans to thread 0
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}


__global__ void __reduct_add_Scalar(Scalar* squeue, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;
    
    Scalar temp = squeue[idx];

    extern __shared__ Scalar tep1[];
    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    if (blockIdx.x == gridDim.x - 1) {
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
        temp = tep1[threadIdx.x];
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp += __shfl_down_sync(0xFFFFFFFF, temp, i);
        }
    }
    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;
    }
}

__global__ void __reduct_add_Scalar2(Scalar2* squeue, int numbers) {
    int idof = blockIdx.x * blockDim.x;  // 计算当前 block 起始位置
    int idx = threadIdx.x + idof;         // 当前线程对应的索引
    if (idx >= numbers) return;           // 防止越界访问
    
    Scalar2 temp = squeue[idx];

    extern __shared__ Scalar2 tep2[];  // 使用共享内存进行 warp 内归约
    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;   // 当前线程在 warp 中的位置
    int warpId = threadIdx.x >> 5;    // 当前线程所属的 warp ID
    int warpNum;

    // 计算每个 block 中的 warp 数量
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((numbers - idof + 31) >> 5);  // 最后一个 block 时的特殊处理
    } else {
        warpNum = blockDim.x >> 5;  // 每个 block 中有多少个 warp
    }

    // 第一阶段：warp 内部归约 (分别对 x 和 y 分量进行归约)
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        temp.x += __shfl_down_sync(0xFFFFFFFF, temp.x, i);  // 对 x 分量进行归约
        temp.y += __shfl_down_sync(0xFFFFFFFF, temp.y, i);  // 对 y 分量进行归约
    }
    
    if (warpTid == 0) {
        tep2[warpId] = temp;  // 每个 warp 内的归约结果存入共享内存
    }

    __syncthreads();  // 等待所有线程完成 warp 内归约

    if (threadIdx.x >= warpNum) return;  // 如果线程超过 warp 数量则退出
    
    // 第二阶段：block 内部归约（多个 warp）
    if (warpNum > 1) {
        temp = tep2[threadIdx.x];  // 加载各个 warp 的中间结果
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp.x += __shfl_down_sync(0xFFFFFFFF, temp.x, i);  // 对 x 分量进行跨 warp 归约
            temp.y += __shfl_down_sync(0xFFFFFFFF, temp.y, i);  // 对 y 分量进行跨 warp 归约
        }
    }

    if (threadIdx.x == 0) {
        squeue[blockIdx.x] = temp;  // 将 block 内归约后的结果存回全局内存
    }
}




__global__ void _reduct_max_Scalar(Scalar* _Scalar1Dim, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar temp = _Scalar1Dim[idx];

    extern __shared__ Scalar tep1[];
    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_max(temp, tempMax);
    }
    if (warpTid == 0) {
        tep1[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        temp = tep1[threadIdx.x];
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_max(temp, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _Scalar1Dim[blockIdx.x] = temp;
    }
}


__global__ void _reduct_min_Scalar(Scalar* _Scalar1Dim, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar temp = _Scalar1Dim[idx];

    extern __shared__ Scalar tep1[];
    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_min(temp, tempMin);
    }
    if (warpTid == 0) {
        tep1[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        temp = tep1[threadIdx.x];
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_min(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        _Scalar1Dim[blockIdx.x] = temp;
    }
}


__global__ void _reduct_max_Scalar2(Scalar2* _Scalar2Dim, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar2 temp = _Scalar2Dim[idx];

    extern __shared__ Scalar2 tep2[];
    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((numbers - idof + 31) >> 5);
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
        temp = tep2[threadIdx.x];
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp.x, i);
            Scalar tempMax = __shfl_down_sync(0xFFFFFFFF, temp.y, i);
            temp.x = __MATHUTILS__::__m_max(temp.x, tempMin);
            temp.y = __MATHUTILS__::__m_max(temp.y, tempMax);
        }
    }
    if (threadIdx.x == 0) {
        _Scalar2Dim[blockIdx.x] = temp;
    }
}


__global__ void _reduct_max_Scalar3_to_Scalar(const Scalar3* _Scalar3Dim, Scalar* _Scalar1Dim,
                                              int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar3 tempMove = _Scalar3Dim[idx];

    Scalar temp = __MATHUTILS__::__m_max(__MATHUTILS__::__m_max(abs(tempMove.x), abs(tempMove.y)),
                                         abs(tempMove.z));

    extern __shared__ Scalar tep1[];
    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((numbers - idof + 31) >> 5);
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < DEFAULT_THREADS_PERWARP; i = (i << 1)) {
        Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
        temp = __MATHUTILS__::__m_max(temp, tempMin);
    }
    if (warpTid == 0) {
        tep1[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        temp = tep1[threadIdx.x];
        for (int i = 1; i < warpNum; i = (i << 1)) {
            Scalar tempMin = __shfl_down_sync(0xFFFFFFFF, temp, i);
            temp = __MATHUTILS__::__m_max(temp, tempMin);
        }
    }
    if (threadIdx.x == 0) {
        _Scalar1Dim[blockIdx.x] = temp;
    }
}

};
