
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <fstream>
#include <iostream>

#include "LBVH.cuh"

// specific idea
// cloth with boundary 0 human body with boundary 2
// CCD and DCD detect at least one vertex with boundary 0
// which means only do cloth self-collision & cloth-body collision

__device__ __host__ inline AABB merge(const AABB& lhs, const AABB& rhs) noexcept {
    AABB merged;
    merged.upper.x = __MATHUTILS__::__m_max(lhs.upper.x, rhs.upper.x);
    merged.upper.y = __MATHUTILS__::__m_max(lhs.upper.y, rhs.upper.y);
    merged.upper.z = __MATHUTILS__::__m_max(lhs.upper.z, rhs.upper.z);
    merged.lower.x = __MATHUTILS__::__m_min(lhs.lower.x, rhs.lower.x);
    merged.lower.y = __MATHUTILS__::__m_min(lhs.lower.y, rhs.lower.y);
    merged.lower.z = __MATHUTILS__::__m_min(lhs.lower.z, rhs.lower.z);
    return merged;
}

__device__ __host__ inline bool overlap(const AABB& lhs, const AABB& rhs, const Scalar& gapL) noexcept {
    if ((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL) return false;
    if ((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL) return false;
    if ((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL) return false;
    return true;
}

__device__ __host__ inline Scalar3 centroid(const AABB& box) noexcept {
    Scalar3 c;
    c.x = (box.upper.x + box.lower.x) * 0.5;
    c.y = (box.upper.y + box.lower.y) * 0.5;
    c.z = (box.upper.z + box.lower.z) * 0.5;
    return c;
}

__device__ __host__ inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ __host__ inline std::uint32_t morton_code(Scalar x, Scalar y, Scalar z,
                                                     Scalar resolution = 1024.0) noexcept {
    x = __MATHUTILS__::__m_min(__MATHUTILS__::__m_max(x * resolution, 0.0), resolution - 1.0);
    y = __MATHUTILS__::__m_min(__MATHUTILS__::__m_max(y * resolution, 0.0), resolution - 1.0);
    z = __MATHUTILS__::__m_min(__MATHUTILS__::__m_max(z * resolution, 0.0), resolution - 1.0);

    const std::uint32_t xx = expand_bits(static_cast<std::uint32_t>(x));
    const std::uint32_t yy = expand_bits(static_cast<std::uint32_t>(y));
    const std::uint32_t zz = expand_bits(static_cast<std::uint32_t>(z));

    std::uint32_t mchash = ((xx << 2) + (yy << 1) + zz);

    return mchash;
}

__device__ __host__ void AABB::combines(const Scalar& x, const Scalar& y, const Scalar& z) {
    lower = make_Scalar3(__MATHUTILS__::__m_min(lower.x, x), __MATHUTILS__::__m_min(lower.y, y),
                         __MATHUTILS__::__m_min(lower.z, z));
    upper = make_Scalar3(__MATHUTILS__::__m_max(upper.x, x), __MATHUTILS__::__m_max(upper.y, y),
                         __MATHUTILS__::__m_max(upper.z, z));
}

__device__ __host__ void AABB::combines(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& xx,
                                        const Scalar& yy, const Scalar& zz) {
    lower = make_Scalar3(__MATHUTILS__::__m_min(lower.x, x), __MATHUTILS__::__m_min(lower.y, y),
                         __MATHUTILS__::__m_min(lower.z, z));
    upper = make_Scalar3(__MATHUTILS__::__m_max(upper.x, xx), __MATHUTILS__::__m_max(upper.y, yy),
                         __MATHUTILS__::__m_max(upper.z, zz));
}

__host__ __device__ void AABB::combines(const AABB& aabb) {
    lower = make_Scalar3(__MATHUTILS__::__m_min(lower.x, aabb.lower.x), __MATHUTILS__::__m_min(lower.y, aabb.lower.y),
                         __MATHUTILS__::__m_min(lower.z, aabb.lower.z));
    upper = make_Scalar3(__MATHUTILS__::__m_max(upper.x, aabb.upper.x), __MATHUTILS__::__m_max(upper.y, aabb.upper.y),
                         __MATHUTILS__::__m_max(upper.z, aabb.upper.z));
}

__host__ __device__ Scalar3 AABB::center() {
    return make_Scalar3((upper.x + lower.x) * 0.5, (upper.y + lower.y) * 0.5, (upper.z + lower.z) * 0.5);
}

__device__ __host__ AABB::AABB() {
    lower = make_Scalar3(1e32, 1e32, 1e32);
    upper = make_Scalar3(-1e32, -1e32, -1e32);
}

__device__ inline int common_upper_bits(const unsigned long long int lhs, const unsigned long long int rhs) noexcept {
    return ::__clzll(lhs ^ rhs);
}

__device__ inline uint2 determine_range(const uint64_t* node_code, const unsigned int num_leaves, unsigned int idx) {
    if (idx == 0) {
        return make_uint2(0, num_leaves - 1);
    }

    // determine direction of the range
    const uint64_t self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx - 1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx + 1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range

    const int delta_min = __MATHUTILS__::__m_min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if (0 <= i_tmp && i_tmp < num_leaves) {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while (delta > delta_min) {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while (t > 0) {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < num_leaves) {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if (delta > delta_min) {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if (d < 0) {
        unsigned int temp_jdx = jdx;
        jdx = idx;
        idx = temp_jdx;
    }
    return make_uint2(idx, jdx);
}

__device__ inline unsigned int find_split(const uint64_t* node_code, const unsigned int num_leaves,
                                          const unsigned int first, const unsigned int last) noexcept {
    const uint64_t first_code = node_code[first];
    const uint64_t last_code = node_code[last];
    if (first_code == last_code) {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split = first;
    int stride = last - first;
    do {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last) {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node) {
                split = middle;
            }
        }
    } while (stride > 1);

    return split;
}

__device__ int _dType_PT(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3) {
    Scalar3 basis0 = __MATHUTILS__::__vec3_minus(v2, v1);
    Scalar3 basis1 = __MATHUTILS__::__vec3_minus(v3, v1);
    Scalar3 basis2 = __MATHUTILS__::__vec3_minus(v0, v1);

    const Scalar3 nVec = __MATHUTILS__::__vec3_cross(basis0, basis1);

    basis1 = __MATHUTILS__::__vec3_cross(basis0, nVec);
    __MATHUTILS__::Matrix3x3S D, D1, D2;

    __MATHUTILS__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
    __MATHUTILS__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z,
                                 nVec.z);
    __MATHUTILS__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z,
                                 nVec.z);

    Scalar2 param[3];
    param[0].x = __MATHUTILS__::__Determiant(D1) / __MATHUTILS__::__Determiant(D);
    param[0].y = __MATHUTILS__::__Determiant(D2) / __MATHUTILS__::__Determiant(D);

    if (param[0].x > 0 && param[0].x < 1 && param[0].y >= 0) {
        return 3;  // PE v1v2
    } else {
        basis0 = __MATHUTILS__::__vec3_minus(v3, v2);
        basis1 = __MATHUTILS__::__vec3_cross(basis0, nVec);
        basis2 = __MATHUTILS__::__vec3_minus(v0, v2);

        __MATHUTILS__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z,
                                     nVec.z);
        __MATHUTILS__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z,
                                     nVec.z);
        __MATHUTILS__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z,
                                     nVec.z);

        param[1].x = __MATHUTILS__::__Determiant(D1) / __MATHUTILS__::__Determiant(D);
        param[1].y = __MATHUTILS__::__Determiant(D2) / __MATHUTILS__::__Determiant(D);

        if (param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0) {
            return 4;  // PE v2v3
        } else {
            basis0 = __MATHUTILS__::__vec3_minus(v1, v3);
            basis1 = __MATHUTILS__::__vec3_cross(basis0, nVec);
            basis2 = __MATHUTILS__::__vec3_minus(v0, v3);

            __MATHUTILS__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y, nVec.y, basis0.z, basis1.z,
                                         nVec.z);
            __MATHUTILS__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y, nVec.y, basis2.z, basis1.z,
                                         nVec.z);
            __MATHUTILS__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y, nVec.y, basis0.z, basis2.z,
                                         nVec.z);

            param[2].x = __MATHUTILS__::__Determiant(D1) / __MATHUTILS__::__Determiant(D);
            param[2].y = __MATHUTILS__::__Determiant(D2) / __MATHUTILS__::__Determiant(D);

            if (param[2].x > 0.0 && param[2].x < 1.0 && param[2].y >= 0.0) {
                return 5;  // PE v3v1
            } else {
                if (param[0].x <= 0.0 && param[2].x >= 1.0) {
                    return 0;  // PP v1
                } else if (param[1].x <= 0.0 && param[0].x >= 1.0) {
                    return 1;  // PP v2
                } else if (param[2].x <= 0.0 && param[1].x >= 1.0) {
                    return 2;  // PP v3
                } else {
                    return 6;  // PT
                }
            }
        }
    }
}

__device__ int _dType_EE(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2, const Scalar3& v3) {
    Scalar3 u = __MATHUTILS__::__vec3_minus(v1, v0);
    Scalar3 v = __MATHUTILS__::__vec3_minus(v3, v2);
    Scalar3 w = __MATHUTILS__::__vec3_minus(v0, v2);

    Scalar a = __MATHUTILS__::__vec3_squaredNorm(u);
    Scalar b = __MATHUTILS__::__vec3_dot(u, v);
    Scalar c = __MATHUTILS__::__vec3_squaredNorm(v);
    Scalar d = __MATHUTILS__::__vec3_dot(u, w);
    Scalar e = __MATHUTILS__::__vec3_dot(v, w);

    Scalar D = a * c - b * b;  // always >= 0
    Scalar tD = D;             // tc = tN / tD, default tD = D >= 0
    Scalar sN, tN;
    int defaultCase = 8;
    sN = (b * e - c * d);
    if (sN <= 0.0) {  // sc < 0 => the s=0 edge is visible
        tN = e;
        tD = c;
        defaultCase = 2;
    } else if (sN >= D) {  // sc > 1  => the s=1 edge is visible
        tN = e + b;
        tD = c;
        defaultCase = 5;
    } else {
        tN = (a * e - b * d);
        if (tN > 0.0 && tN < tD &&
            (__MATHUTILS__::__vec3_dot(w, __MATHUTILS__::__vec3_cross(u, v)) == 0.0 ||
             __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_cross(u, v)) < 1.0e-20 * a * c)) {
            if (sN < D / 2) {
                tN = e;
                tD = c;
                defaultCase = 2;
            } else {
                tN = e + b;
                tD = c;
                defaultCase = 5;
            }
        }
    }

    if (tN <= 0.0) {
        if (-d <= 0.0) {
            return 0;
        } else if (-d >= a) {
            return 3;
        } else {
            return 6;
        }
    } else if (tN >= tD) {
        if ((-d + b) <= 0.0) {
            return 1;
        } else if ((-d + b) >= a) {
            return 4;
        } else {
            return 7;
        }
    }

    return defaultCase;
}

__device__ inline bool _checkPTintersection(const Scalar3* _vertexes, const uint32_t& id0, const uint32_t& id1,
                                            const uint32_t& id2, const uint32_t& id3, const Scalar& dHat,
                                            uint32_t* _cpNum, int* _mInx, int4* _collisionPair,
                                            int4* _ccd_collisionPair) noexcept {
    Scalar3 v0 = _vertexes[id0];
    Scalar3 v1 = _vertexes[id1];
    Scalar3 v2 = _vertexes[id2];
    Scalar3 v3 = _vertexes[id3];

    int dtype = _dType_PT(v0, v1, v2, v3);

    Scalar d = 1e6;  // any large number
    switch (dtype) {
        // P-P collision (-,+,-1,-1) cpNum0++ cpNum2++
        case 0: {
            __MATHUTILS__::_d_PP(v0, v1, d);
            if (d < dHat) {
                // printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x,
                // _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = atomicAdd(_cpNum, 1);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, -1, -1);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        // P-P collision (-,+,-1,-1) cpNum0++ cpNum2++
        case 1: {
            __MATHUTILS__::_d_PP(v0, v2, d);
            if (d < dHat) {
                // printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x,
                // _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = atomicAdd(_cpNum, 1);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, -1);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        // P-P collision (-,+,-1,-1) cpNum0++ cpNum2++
        case 2: {
            __MATHUTILS__::_d_PP(v0, v3, d);
            if (d < dHat) {
                // printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x,
                // _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = atomicAdd(_cpNum, 1);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, -1);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 2, 1);
            }
            break;
        }

        // P-E collision (-,+,+,-1) cpNum0++ cpNum3++
        case 3: {
            __MATHUTILS__::_d_PE(v0, v1, v2, d);
            if (d < dHat) {
                // printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x,
                // _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = atomicAdd(_cpNum, 1);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, -1);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        // P-E collision (-,+,+,-1) cpNum0++ cpNum3++
        case 4: {
            __MATHUTILS__::_d_PE(v0, v2, v3, d);
            if (d < dHat) {
                // printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x,
                // _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = atomicAdd(_cpNum, 1);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, -1);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        // P-E collision (-,+,+,-1) cpNum0++ cpNum3++
        case 5: {
            __MATHUTILS__::_d_PE(v0, v3, v1, d);
            if (d < dHat) {
                // printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x,
                // _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = atomicAdd(_cpNum, 1);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, id1, -1);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 3, 1);
            }
            break;
        }

        // P-T collision (-,+,+,+) cpNum0++ cpNum4++
        case 6: {
            __MATHUTILS__::_d_PT(v0, v1, v2, v3, d);
            if (d < dHat) {
                // printf("%d   %d   %d   %d   %d   %f\n", dtype, idx, _faces[obj_idx].x,
                // _faces[obj_idx].y, _faces[obj_idx].z, d);
                int cdp_idx = atomicAdd(_cpNum, 1);
                _ccd_collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _collisionPair[cdp_idx] = make_int4(-id0 - 1, id1, id2, id3);
                _mInx[cdp_idx] = atomicAdd(_cpNum + 4, 1);
            }
            break;
        }

        default:
            break;
    }
}

__device__ inline bool _checkEEintersection(const Scalar3* _vertexes, const Scalar3* _rest_vertexes,
                                            const uint32_t& id0, const uint32_t& id1, const uint32_t& id2,
                                            const uint32_t& id3, const uint32_t& obj_idx, const Scalar& dHat,
                                            uint32_t* _cpNum, int* MatIndex, int4* _collisionPair,
                                            int4* _ccd_collisionPair, int surfedgeNum) noexcept {
    Scalar3 v0 = _vertexes[id0];
    Scalar3 v1 = _vertexes[id1];
    Scalar3 v2 = _vertexes[id2];
    Scalar3 v3 = _vertexes[id3];

    int dtype = _dType_EE(v0, v1, v2, v3);

    Scalar d = 1e6;  // any large number
    bool smooth = true; // modify 

    // compute e-e cross norm as ||e1xe2||^2
    // compute eps as 10^-3*||e1||^2*||e2||^2
    Scalar eeSqureNCross = __MATHUTILS__::__vec3_squaredNorm(
        __MATHUTILS__::__vec3_cross(__MATHUTILS__::__vec3_minus(v0, v1), __MATHUTILS__::__vec3_minus(v2, v3)));
    Scalar eps_x = __MATHUTILS__::_compute_epx_cp(_rest_vertexes[id0], _rest_vertexes[id1],
                                                    _rest_vertexes[id2], _rest_vertexes[id3]);
    int add_e = (eeSqureNCross < eps_x) ? -static_cast<int>(obj_idx) - 2 : -1;

    switch (dtype) {
        
        // parallel for P-P condition (-,-,-,-) cpNum0++ cpNum4++
        // nonparallel for P-P condition (-,+,#,#) cpNum0++ cpNum2++
        case 0: {
            __MATHUTILS__::_d_PP(v0, v2, d);
            if (d < dHat) {
                if (add_e <= -2) { // parallel condition
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(-id0 - 1, -id2 - 1, -id1 - 1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                } else { // normal condition
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        // parallel for P-P condition (-,-,-,-) cpNum0++ cpNum4++
        // nonparallel for P-P condition (-,+,#,#) cpNum0++ cpNum2++
        case 1: {
            __MATHUTILS__::_d_PP(v0, v3, d);
            if (d < dHat) {

                if (add_e <= -2) {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(-id0 - 1, -id3 - 1, -id1 - 1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                } else {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        // parallel for P-E condition (-,-,+,-) cpNum0++ cpNum4++
        // nonparallel for P-E condition (-,+,+,#) cpNum0++ cpNum3++
        case 2: {
            __MATHUTILS__::_d_PE(v0, v2, v3, d);
            if (d < dHat) {

                if (add_e <= -2) {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(-id0 - 1, -id2 - 1, id3, -id1 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                } else {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id0 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        // parallel for P-P condition (-,-,-,-) cpNum0++ cpNum4++
        // nonparallel for P-P condition (-,+,#,#) cpNum0++ cpNum2++ 
        case 3: {
            __MATHUTILS__::_d_PP(v1, v2, d);
            if (d < dHat) {

                if (add_e <= -2) {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(-id1 - 1, -id2 - 1, -id0 - 1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                } else {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 4: {
            __MATHUTILS__::_d_PP(v1, v3, d);
            if (d < dHat) {

                if (add_e <= -2) {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(-id1 - 1, -id3 - 1, -id0 - 1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                } else {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id3, -1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 2, 1);
                }
            }
            break;
        }

        case 5: {
            __MATHUTILS__::_d_PE(v1, v2, v3, d);
            if (d < dHat) {

                if (add_e <= -2) {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(-id1 - 1, -id2 - 1, id3, -id0 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                } else {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id1 - 1, id2, id3, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 6: {
            __MATHUTILS__::_d_PE(v2, v0, v1, d);
            if (d < dHat) {

                if (add_e <= -2) {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(-id2 - 1, -id0 - 1, id1, -id3 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                } else {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id2 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        case 7: {
            __MATHUTILS__::_d_PE(v3, v0, v1, d);
            if (d < dHat) {

                if (add_e <= -2) {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(-id3 - 1, -id0 - 1, id1, -id2 - 1);
                        MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                } else {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(-id3 - 1, id0, id1, add_e);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 3, 1);
                }
            }
            break;
        }

        // (+,+,+,-) parallel E-E condition cpNum0++ cpNum4++
        // (+,+,+,+) non-parallel E-E condition cpNum0++ cpNum4++
        case 8: {
            __MATHUTILS__::_d_EE(v0, v1, v2, v3, d);
            if (d < dHat) {

                if (add_e <= -2) {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    if (smooth) {
                        _collisionPair[cdp_idx] = make_int4(id0, id1, id2, -id3 - 1);
                        break;
                    }
                    _collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                } else {
                    int cdp_idx = atomicAdd(_cpNum, 1);
                    _ccd_collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    _collisionPair[cdp_idx] = make_int4(id0, id1, id2, id3);
                    MatIndex[cdp_idx] = atomicAdd(_cpNum + 4, 1);
                }
            }
            break;
        }

        default:
            break;
    }
}

__global__ void _reduct_max_box(AABB* _leafBoxes, int number) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    extern __shared__ AABB tep1[];
    if (idx >= number) return;

    AABB temp = _leafBoxes[idx];

    Scalar xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
    Scalar xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;

    int warpTid = threadIdx.x % DEFAULT_THREADS_PERWARP;
    int warpId = (threadIdx.x >> 5);
    int warpNum;
    int tidNum = DEFAULT_THREADS_PERWARP;
    if (blockIdx.x == gridDim.x - 1) {
        warpNum = ((number - idof + 31) >> 5);
        if (warpId == warpNum - 1) {
            tidNum = number - idof - (warpNum - 1) * DEFAULT_THREADS_PERWARP;
        }
    } else {
        warpNum = ((blockDim.x) >> 5);
    }
    for (int i = 1; i < tidNum; i = (i << 1)) {
        temp.combines(__shfl_down_sync(0xFFFFFFFF, xmin, i), __shfl_down_sync(0xFFFFFFFF, ymin, i),
                      __shfl_down_sync(0xFFFFFFFF, zmin, i), __shfl_down_sync(0xFFFFFFFF, xmax, i),
                      __shfl_down_sync(0xFFFFFFFF, ymax, i), __shfl_down_sync(0xFFFFFFFF, zmax, i));
        if (warpTid + i < tidNum) {
            xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
            xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        }
    }
    if (warpTid == 0) {
        tep1[warpId] = temp;
    }
    __syncthreads();
    if (threadIdx.x >= warpNum) return;
    if (warpNum > 1) {
        temp = tep1[threadIdx.x];
        xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
        xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
        for (int i = 1; i < warpNum; i = (i << 1)) {
            temp.combines(__shfl_down_sync(0xFFFFFFFF, xmin, i), __shfl_down_sync(0xFFFFFFFF, ymin, i),
                          __shfl_down_sync(0xFFFFFFFF, zmin, i), __shfl_down_sync(0xFFFFFFFF, xmax, i),
                          __shfl_down_sync(0xFFFFFFFF, ymax, i), __shfl_down_sync(0xFFFFFFFF, zmax, i));
            if (threadIdx.x + i < warpNum) {
                xmin = temp.lower.x, ymin = temp.lower.y, zmin = temp.lower.z;
                xmax = temp.upper.x, ymax = temp.upper.y, zmax = temp.upper.z;
            }
        }
    }
    if (threadIdx.x == 0) {
        _leafBoxes[blockIdx.x] = temp;
    }
}

template <class element_type>
__global__ void _calcLeafBvs(const Scalar3* _vertexes, const element_type* _elements, AABB* _bvs, int surffaceNum,
                             int type = 0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= surffaceNum) return;
    AABB _bv;

    element_type _e = _elements[idx];
    Scalar3 _v = _vertexes[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    if (type == 0) {
        _v = _vertexes[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
    }
    _bvs[idx] = _bv;
}

template <class element_type>
__global__ void _calcLeafBvs_ccd(const Scalar3* _vertexes, const Scalar3* _moveDir, Scalar alpha,
                                 const element_type* _elements, AABB* _bvs, int surffaceNum, int type = 0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= surffaceNum) return;
    AABB _bv;

    element_type _e = _elements[idx];
    Scalar3 _v = _vertexes[_e.x];
    Scalar3 _mvD = _moveDir[_e.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);

    _v = _vertexes[_e.y];
    _mvD = _moveDir[_e.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    if (type == 0) {
        _v = _vertexes[*((uint32_t*)(&_e) + 2)];
        _mvD = _moveDir[*((uint32_t*)(&_e) + 2)];
        _bv.combines(_v.x, _v.y, _v.z);
        _bv.combines(_v.x - _mvD.x * alpha, _v.y - _mvD.y * alpha, _v.z - _mvD.z * alpha);
    }
    _bvs[idx] = _bv;
}

__global__ void _calcMChash(uint64_t* _MChash, AABB* _bvs, int number) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    AABB maxBv = _bvs[0];
    Scalar3 SceneSize =
        make_Scalar3(maxBv.upper.x - maxBv.lower.x, maxBv.upper.y - maxBv.lower.y, maxBv.upper.z - maxBv.lower.z);
    Scalar3 centerP = _bvs[idx + number - 1].center();
    Scalar3 offset = make_Scalar3(centerP.x - maxBv.lower.x, centerP.y - maxBv.lower.y, centerP.z - maxBv.lower.z);

    // printf("%d   %f     %f     %f\n", offset.x, offset.y, offset.z);
    uint64_t mc32 = morton_code(offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z);
    uint64_t mc64 = ((mc32 << DEFAULT_THREADS_PERWARP) | idx);
    _MChash[idx] = mc64;
}

__global__ void _calcLeafNodes(Node* _nodes, const uint32_t* _indices, int number) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    if (idx < number - 1) {
        _nodes[idx].left_idx = 0xFFFFFFFF;
        _nodes[idx].right_idx = 0xFFFFFFFF;
        _nodes[idx].parent_idx = 0xFFFFFFFF;
        _nodes[idx].element_idx = 0xFFFFFFFF;
    }
    int l_idx = idx + number - 1;
    _nodes[l_idx].left_idx = 0xFFFFFFFF;
    _nodes[l_idx].right_idx = 0xFFFFFFFF;
    _nodes[l_idx].parent_idx = 0xFFFFFFFF;
    _nodes[l_idx].element_idx = _indices[idx];
}

__global__ void _calcInternalNodes(Node* _nodes, const uint64_t* _MChash, int number) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number - 1) return;
    const uint2 ij = determine_range(_MChash, number, idx);
    const unsigned int gamma = find_split(_MChash, number, ij.x, ij.y);

    _nodes[idx].left_idx = gamma;
    _nodes[idx].right_idx = gamma + 1;
    if (__MATHUTILS__::__m_min(ij.x, ij.y) == gamma) {
        _nodes[idx].left_idx += number - 1;
    }
    if (__MATHUTILS__::__m_max(ij.x, ij.y) == gamma + 1) {
        _nodes[idx].right_idx += number - 1;
    }
    _nodes[_nodes[idx].left_idx].parent_idx = idx;
    _nodes[_nodes[idx].right_idx].parent_idx = idx;
}

__global__ void _calcInternalAABB(const Node* _nodes, AABB* _bvs, uint32_t* flags, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;
    idx = idx + number - 1;

    uint32_t parent = _nodes[idx].parent_idx;
    while (parent != 0xFFFFFFFF)  // means idx == 0
    {
        const int old = atomicCAS(flags + parent, 0xFFFFFFFF, 0);
        if (old == 0xFFFFFFFF) {
            return;
        }

        const uint32_t lidx = _nodes[parent].left_idx;
        const uint32_t ridx = _nodes[parent].right_idx;

        const AABB lbox = _bvs[lidx];
        const AABB rbox = _bvs[ridx];
        _bvs[parent] = merge(lbox, rbox);

        __threadfence();

        parent = _nodes[parent].parent_idx;
    }
}

__global__ void _sortBvs(const uint32_t* _indices, AABB* _bvs, AABB* _temp_bvs, int number) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    _bvs[idx] = _temp_bvs[_indices[idx]];
}

__global__ void _selfQuery_vf(const int* _btype, const Scalar3* _vertexes, const uint3* _faces,
                              const uint32_t* _surfVerts, const AABB* _bvs, const Node* _nodes, int4* _collisionPair,
                              int4* _ccd_collisionPair, uint32_t* _cpNum, int* MatIndex, Scalar dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;

    AABB _bv;
    idx = _surfVerts[idx];
    _bv.upper = _vertexes[idx];
    _bv.lower = _vertexes[idx];
    // Scalar bboxDiagSize2 = __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(_bvs[0].upper,
    // _bvs[0].lower)); printf("%f\n", bboxDiagSize2);
    Scalar gapl = sqrt(dHat);  // 0.001 * sqrt(bboxDiagSize2);
    // Scalar dHat = gapl * gapl;// *bboxDiagSize2;
    do {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _nodes[node_id].left_idx;
        const uint32_t R_idx = _nodes[node_id].right_idx;

        if (overlap(_bv, _bvs[L_idx], gapl)) {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
                    if (!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2 && _btype[_faces[obj_idx].y] >= 2 &&
                          _btype[_faces[obj_idx].z] >= 2))
                        _checkPTintersection(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z,
                                             dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair);
                }
            } else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (overlap(_bv, _bvs[R_idx], gapl)) {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
                    if (!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2 && _btype[_faces[obj_idx].y] >= 2 &&
                          _btype[_faces[obj_idx].z] >= 2))
                        _checkPTintersection(_vertexes, idx, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z,
                                             dHat, _cpNum, MatIndex, _collisionPair, _ccd_collisionPair);
                }
            } else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}

__global__ void _selfQuery_vf_ccd(const int* _btype, const Scalar3* _vertexes, const Scalar3* moveDir, Scalar alpha,
                                  const uint3* _faces, const uint32_t* _surfVerts, const AABB* _bvs, const Node* _nodes,
                                  int4* _ccd_collisionPair, uint32_t* _cpNum, Scalar dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;

    AABB _bv;
    idx = _surfVerts[idx];
    Scalar3 current_vertex = _vertexes[idx];
    Scalar3 mvD = moveDir[idx];
    _bv.upper = current_vertex;
    _bv.lower = current_vertex;
    _bv.combines(current_vertex.x - mvD.x * alpha, current_vertex.y - mvD.y * alpha, current_vertex.z - mvD.z * alpha);
    // Scalar bboxDiagSize2 = __MATHUTILS__::__squaredNorm(__MATHUTILS__::__minus(_bvs[0].upper,
    // _bvs[0].lower)); printf("%f\n", bboxDiagSize2);
    Scalar gapl = sqrt(dHat);  // 0.001 * sqrt(bboxDiagSize2);
    // Scalar dHat = gapl * gapl;// *bboxDiagSize2;
    do {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _nodes[node_id].left_idx;
        const uint32_t R_idx = _nodes[node_id].right_idx;

        if (overlap(_bv, _bvs[L_idx], gapl)) {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2 && _btype[_faces[obj_idx].y] >= 2 &&
                      _btype[_faces[obj_idx].z] >= 2))
                    if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
                        _ccd_collisionPair[atomicAdd(_cpNum, 1)] =
                            make_int4(-idx - 1, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z);
                    }
            } else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (overlap(_bv, _bvs[R_idx], gapl)) {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (!(_btype[idx] >= 2 && _btype[_faces[obj_idx].x] >= 2 && _btype[_faces[obj_idx].y] >= 2 &&
                      _btype[_faces[obj_idx].z] >= 2))
                    if (idx != _faces[obj_idx].x && idx != _faces[obj_idx].y && idx != _faces[obj_idx].z) {
                        _ccd_collisionPair[atomicAdd(_cpNum, 1)] =
                            make_int4(-idx - 1, _faces[obj_idx].x, _faces[obj_idx].y, _faces[obj_idx].z);
                    }
            } else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}

__global__ void _selfQuery_ee(const int* _btype, const Scalar3* _vertexes, const Scalar3* _rest_vertexes,
                              const uint2* _edges, const AABB* _bvs, const Node* _nodes, int4* _collisionPair,
                              int4* _ccd_collisionPair, uint32_t* _cpNum, int* MatIndex, Scalar dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;

    idx = idx + number - 1;
    AABB _bv = _bvs[idx];
    uint32_t self_eid = _nodes[idx].element_idx;
    Scalar gapl = sqrt(dHat);
    do {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _nodes[node_id].left_idx;
        const uint32_t R_idx = _nodes[node_id].right_idx;

        if (overlap(_bv, _bvs[L_idx], gapl)) {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (self_eid != obj_idx) {
                    if (!(_edges[self_eid].x == _edges[obj_idx].x || _edges[self_eid].x == _edges[obj_idx].y ||
                          _edges[self_eid].y == _edges[obj_idx].x || _edges[self_eid].y == _edges[obj_idx].y ||
                          obj_idx < self_eid)) {
                        // printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y,
                        // _edges[obj_idx].x, _edges[obj_idx].y);
                        if (!(_btype[_edges[self_eid].x] >= 2 && _btype[_edges[self_eid].y] >= 2 &&
                              _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
                            _checkEEintersection(_vertexes, _rest_vertexes, _edges[self_eid].x, _edges[self_eid].y,
                                                 _edges[obj_idx].x, _edges[obj_idx].y, obj_idx, dHat, _cpNum, MatIndex,
                                                 _collisionPair, _ccd_collisionPair, number);
                    }
                }
            } else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (overlap(_bv, _bvs[R_idx], gapl)) {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (self_eid != obj_idx) {
                    if (!(_edges[self_eid].x == _edges[obj_idx].x || _edges[self_eid].x == _edges[obj_idx].y ||
                          _edges[self_eid].y == _edges[obj_idx].x || _edges[self_eid].y == _edges[obj_idx].y ||
                          obj_idx < self_eid)) {
                        // printf("%d   %d   %d   %d\n", _edges[self_eid].x, _edges[self_eid].y,
                        // _edges[obj_idx].x, _edges[obj_idx].y);
                        if (!(_btype[_edges[self_eid].x] >= 2 && _btype[_edges[self_eid].y] >= 2 &&
                              _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
                            _checkEEintersection(_vertexes, _rest_vertexes, _edges[self_eid].x, _edges[self_eid].y,
                                                 _edges[obj_idx].x, _edges[obj_idx].y, obj_idx, dHat, _cpNum, MatIndex,
                                                 _collisionPair, _ccd_collisionPair, number);
                    }
                }
            } else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}

__global__ void _selfQuery_ee_ccd(const int* _btype, const Scalar3* _vertexes, const Scalar3* moveDir, Scalar alpha,
                                  const uint2* _edges, const AABB* _bvs, const Node* _nodes, int4* _ccd_collisionPair,
                                  uint32_t* _cpNum, Scalar dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;
    idx = idx + number - 1;
    AABB _bv = _bvs[idx];
    uint32_t self_eid = _nodes[idx].element_idx;
    uint2 current_edge = _edges[self_eid];

    Scalar gapl = sqrt(dHat);

    do {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _nodes[node_id].left_idx;
        const uint32_t R_idx = _nodes[node_id].right_idx;

        if (overlap(_bv, _bvs[L_idx], gapl)) {
            const auto obj_idx = _nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (self_eid != obj_idx) {
                    if (!(_btype[_edges[self_eid].x] >= 2 && _btype[_edges[self_eid].y] >= 2 &&
                          _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
                        if (!(current_edge.x == _edges[obj_idx].x || current_edge.x == _edges[obj_idx].y ||
                              current_edge.y == _edges[obj_idx].x || current_edge.y == _edges[obj_idx].y ||
                              obj_idx < self_eid)) {
                            _ccd_collisionPair[atomicAdd(_cpNum, 1)] =
                                make_int4(current_edge.x, current_edge.y, _edges[obj_idx].x, _edges[obj_idx].y);
                        }
                }
            } else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (overlap(_bv, _bvs[R_idx], gapl)) {
            const auto obj_idx = _nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (self_eid != obj_idx) {
                    if (!(_btype[_edges[self_eid].x] >= 2 && _btype[_edges[self_eid].y] >= 2 &&
                          _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
                        if (!(current_edge.x == _edges[obj_idx].x || current_edge.x == _edges[obj_idx].y ||
                              current_edge.y == _edges[obj_idx].x || current_edge.y == _edges[obj_idx].y ||
                              obj_idx < self_eid)) {
                            _ccd_collisionPair[atomicAdd(_cpNum, 1)] =
                                make_int4(current_edge.x, current_edge.y, _edges[obj_idx].x, _edges[obj_idx].y);
                        }
                }
            } else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}

///////////////////////////////////////host//////////////////////////////////////////////

AABB calcMaxBV(AABB* _leafBoxes, AABB* _tempLeafBox, const int& number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(AABB) * (threadNum >> 5);

    // AABB* _tempLeafBox;
    // CUDAMallocSafe(_tempLeafBox, number);
    CUDA_SAFE_CALL(cudaMemcpy(_tempLeafBox, _leafBoxes + number - 1, number * sizeof(AABB), cudaMemcpyDeviceToDevice));

    _reduct_max_box<<<blockNum, threadNum, sharedMsize>>>(_tempLeafBox, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        _reduct_max_box<<<blockNum, threadNum, sharedMsize>>>(_tempLeafBox, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    AABB h_bv;
    cudaMemcpy(&h_bv, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToHost);
    // CUDAFreeSafe(_tempLeafBox));
    return h_bv;
}

template <class element_type>
void calcLeafBvs(const Scalar3* _vertexes, const element_type* _faces, AABB* _bvs, const int& surffaceNum,
                 const int& type) {
    int numbers = surffaceNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcLeafBvs<<<blockNum, threadNum>>>(_vertexes, _faces, _bvs + numbers - 1, surffaceNum, type);
}

template <class element_type>
void calcLeafBvs_fullCCD(const Scalar3* _vertexes, const Scalar3* _moveDir, const Scalar& alpha,
                         const element_type* _faces, AABB* _bvs, const int& surffaceNum, const int& type) {
    int numbers = surffaceNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcLeafBvs_ccd<<<blockNum, threadNum>>>(_vertexes, _moveDir, alpha, _faces, _bvs + numbers - 1, surffaceNum,
                                              type);
}

void calcMChash(uint64_t* _MChash, AABB* _bvs, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcMChash<<<blockNum, threadNum>>>(_MChash, _bvs, number);
}

void calcLeafNodes(Node* _nodes, const uint32_t* _indices, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcLeafNodes<<<blockNum, threadNum>>>(_nodes, _indices, number);
}

void calcInternalNodes(Node* _nodes, const uint64_t* _MChash, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcInternalNodes<<<blockNum, threadNum>>>(_nodes, _MChash, number);
}

void calcInternalAABB(const Node* _nodes, AABB* _bvs, uint32_t* flags, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    CUDA_SAFE_CALL(cudaMemset(flags, 0xFFFFFFFF, sizeof(uint32_t) * (numbers - 1)));
    _calcInternalAABB<<<blockNum, threadNum>>>(_nodes, _bvs, flags, numbers);
}

void sortBvs(const uint32_t* _indices, AABB* _bvs, AABB* _temp_bvs, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    cudaMemcpy(_temp_bvs, _bvs + number - 1, sizeof(AABB) * number, cudaMemcpyDeviceToDevice);
    _sortBvs<<<blockNum, threadNum>>>(_indices, _bvs + number - 1, _temp_bvs, number);
}

void selfQuery_ee(const int* _btype, const Scalar3* _vertexes, const Scalar3* _rest_vertexes, const uint2* _edges,
                  const AABB* _bvs, const Node* _nodes, int4* _collisionPairs, int4* _ccd_collisionPairs,
                  uint32_t* _cpNum, int* MatIndex, Scalar dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _selfQuery_ee<<<blockNum, threadNum>>>(_btype, _vertexes, _rest_vertexes, _edges, _bvs, _nodes, _collisionPairs,
                                           _ccd_collisionPairs, _cpNum, MatIndex, dHat, numbers);
}

void fullCCDselfQuery_ee(const int* _btype, const Scalar3* _vertexes, const Scalar3* moveDir, const Scalar& alpha,
                         const uint2* _edges, const AABB* _bvs, const Node* _nodes, int4* _ccd_collisionPairs,
                         uint32_t* _cpNum, Scalar dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _selfQuery_ee_ccd<<<blockNum, threadNum>>>(_btype, _vertexes, moveDir, alpha, _edges, _bvs, _nodes,
                                               _ccd_collisionPairs, _cpNum, dHat, numbers);
}

void selfQuery_vf(const int* _btype, const Scalar3* _vertexes, const uint3* _faces, const uint32_t* _surfVerts,
                  const AABB* _bvs, const Node* _nodes, int4* _collisionPairs, int4* _ccd_collisionPairs,
                  uint32_t* _cpNum, int* MatIndex, Scalar dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _selfQuery_vf<<<blockNum, threadNum>>>(_btype, _vertexes, _faces, _surfVerts, _bvs, _nodes, _collisionPairs,
                                           _ccd_collisionPairs, _cpNum, MatIndex, dHat, numbers);
}

void fullCCDselfQuery_vf(const int* _btype, const Scalar3* _vertexes, const Scalar3* moveDir, const Scalar& alpha,
                         const uint3* _faces, const uint32_t* _surfVerts, const AABB* _bvs, const Node* _nodes,
                         int4* _ccd_collisionPairs, uint32_t* _cpNum, Scalar dHat, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _selfQuery_vf_ccd<<<blockNum, threadNum>>>(_btype, _vertexes, moveDir, alpha, _faces, _surfVerts, _bvs, _nodes,
                                               _ccd_collisionPairs, _cpNum, dHat, numbers);
}

void LBVH::CUDA_FREE_LBVH() {
    CUDAFreeSafe(_indices);
    CUDAFreeSafe(_MChash);
    CUDAFreeSafe(_nodes);
    CUDAFreeSafe(_bvs);
    CUDAFreeSafe(_flags);
    CUDAFreeSafe(_tempLeafBox);
}

void LBVH::CUDA_MALLOC_LBVH(const int& number) {
    CUDAMallocSafe(_indices, (number));
    CUDAMallocSafe(_MChash, (number));
    CUDAMallocSafe(_nodes, (2 * number - 1));
    CUDAMallocSafe(_bvs, (2 * number - 1));
    CUDAMallocSafe(_tempLeafBox, number);
    CUDAMallocSafe(_flags, (number - 1));
    // CUDAMallocSafe(_cpNum, 1);
}


void LBVH_E::init(int* _mbtype, Scalar3* _mVerts, Scalar3* _mRestVerts, uint2* _mSurfEdges, int4* _mCollisionPairs,
                  int4* _ccd_mCollisionPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& surfedgeNum,
                  const int& surfvertNum) {
    this->_btype = _mbtype;
    this->_vertexes = _mVerts;
    this->_rest_vertexes = _mRestVerts;
    this->_surfEdges = _mSurfEdges;
    this->_collisionPair = _mCollisionPairs;
    this->_ccd_collisionPair = _ccd_mCollisionPairs;
    this->_cpNum = _mcpNum;
    this->_MatIndex = _mMatIndex;
    this->surfEdgeNum = surfedgeNum;
    this->surfVertNum = surfvertNum;

    CUDA_MALLOC_LBVH(surfEdgeNum);
}

void LBVH_F::init(int* _mbtype, Scalar3* _mVerts, uint3* _mSurfFaces, uint32_t* _mSurfVerts, int4* _mCollisionPairs,
                  int4* _ccd_mCollisionPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& surffaceNum,
                  const int& surfvertNum) {
    this->_btype = _mbtype;
    this->_vertexes = _mVerts;
    this->_surfFaces = _mSurfFaces;
    this->_surfVerts = _mSurfVerts;
    this->_collisionPair = _mCollisionPairs;
    this->_ccd_collisionPair = _ccd_mCollisionPairs;
    this->_cpNum = _mcpNum;
    this->_MatIndex = _mMatIndex;
    this->surfFaceNum = surffaceNum;
    this->surfVertNum = surfvertNum;
    CUDA_MALLOC_LBVH(surfFaceNum);
}

AABB* LBVH_F::getSceneSize() {
    calcLeafBvs(_vertexes, _surfFaces, _bvs, surfFaceNum, 0);
    calcMaxBV(_bvs, _tempLeafBox, surfFaceNum);
    return _bvs;
}

Scalar LBVH_F::Construct() {
    calcLeafBvs(_vertexes, _surfFaces, _bvs, surfFaceNum, 0);
    scene = calcMaxBV(_bvs, _tempLeafBox, surfFaceNum);
    calcMChash(_MChash, _bvs, surfFaceNum);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices), thrust::device_ptr<uint32_t>(_indices) + surfFaceNum);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash), thrust::device_ptr<uint64_t>(_MChash) + surfFaceNum,
                        thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, surfFaceNum);
    calcLeafNodes(_nodes, _indices, surfFaceNum);
    calcInternalNodes(_nodes, _MChash, surfFaceNum);
    calcInternalAABB(_nodes, _bvs, _flags, surfFaceNum);
    return 0;
}

Scalar LBVH_F::ConstructFullCCD(const Scalar3* moveDir, const Scalar& alpha) {
    calcLeafBvs_fullCCD(_vertexes, moveDir, alpha, _surfFaces, _bvs, surfFaceNum, 0);
    scene = calcMaxBV(_bvs, _tempLeafBox, surfFaceNum);
    calcMChash(_MChash, _bvs, surfFaceNum);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices), thrust::device_ptr<uint32_t>(_indices) + surfFaceNum);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash), thrust::device_ptr<uint64_t>(_MChash) + surfFaceNum,
                        thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, surfFaceNum);
    calcLeafNodes(_nodes, _indices, surfFaceNum);
    calcInternalNodes(_nodes, _MChash, surfFaceNum);
    calcInternalAABB(_nodes, _bvs, _flags, surfFaceNum);
    return 0;
}

Scalar LBVH_E::Construct() {
    calcLeafBvs(_vertexes, _surfEdges, _bvs, surfEdgeNum, 1);
    scene = calcMaxBV(_bvs, _tempLeafBox, surfEdgeNum);
    calcMChash(_MChash, _bvs, surfEdgeNum);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices), thrust::device_ptr<uint32_t>(_indices) + surfEdgeNum);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash), thrust::device_ptr<uint64_t>(_MChash) + surfEdgeNum,
                        thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, surfEdgeNum);
    calcLeafNodes(_nodes, _indices, surfEdgeNum);
    calcInternalNodes(_nodes, _MChash, surfEdgeNum);
    calcInternalAABB(_nodes, _bvs, _flags, surfEdgeNum);
    return 0;
}

Scalar LBVH_E::ConstructFullCCD(const Scalar3* moveDir, const Scalar& alpha) {
    calcLeafBvs_fullCCD(_vertexes, moveDir, alpha, _surfEdges, _bvs, surfEdgeNum, 1);
    scene = calcMaxBV(_bvs, _tempLeafBox, surfEdgeNum);
    calcMChash(_MChash, _bvs, surfEdgeNum);
    thrust::sequence(thrust::device_ptr<uint32_t>(_indices), thrust::device_ptr<uint32_t>(_indices) + surfEdgeNum);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(_MChash), thrust::device_ptr<uint64_t>(_MChash) + surfEdgeNum,
                        thrust::device_ptr<uint32_t>(_indices));
    sortBvs(_indices, _bvs, _tempLeafBox, surfEdgeNum);
    calcLeafNodes(_nodes, _indices, surfEdgeNum);
    calcInternalNodes(_nodes, _MChash, surfEdgeNum);
    calcInternalAABB(_nodes, _bvs, _flags, surfEdgeNum);
    return 0;
}

void LBVH_F::SelfCollisionDetect(Scalar dHat) {
    selfQuery_vf(_btype, _vertexes, _surfFaces, _surfVerts, _bvs, _nodes, _collisionPair, _ccd_collisionPair, _cpNum,
                 _MatIndex, dHat, surfVertNum);
}

void LBVH_E::SelfCollisionDetect(Scalar dHat) {
    selfQuery_ee(_btype, _vertexes, _rest_vertexes, _surfEdges, _bvs, _nodes, _collisionPair, _ccd_collisionPair,
                 _cpNum, _MatIndex, dHat, surfEdgeNum);
}

void LBVH_F::SelfCollisionFullDetect(Scalar dHat, const Scalar3* moveDir, const Scalar& alpha) {
    fullCCDselfQuery_vf(_btype, _vertexes, moveDir, alpha, _surfFaces, _surfVerts, _bvs, _nodes, _ccd_collisionPair,
                        _cpNum, dHat, surfVertNum);
}

void LBVH_E::SelfCollisionFullDetect(Scalar dHat, const Scalar3* moveDir, const Scalar& alpha) {
    fullCCDselfQuery_ee(_btype, _vertexes, moveDir, alpha, _surfEdges, _bvs, _nodes, _ccd_collisionPair, _cpNum, dHat,
                        surfEdgeNum);
}

__device__ bool edgeTriIntersect(const Scalar3& ve0, const Scalar3& ve1, const Scalar3& vt0, const Scalar3& vt1,
                                 const Scalar3& vt2) {
    // printf("check for tri and lines\n");

    __MATHUTILS__::Matrix3x3S coefMtr;
    Scalar3 col0 = __MATHUTILS__::__vec3_minus(vt1, vt0);
    Scalar3 col1 = __MATHUTILS__::__vec3_minus(vt2, vt0);
    Scalar3 col2 = __MATHUTILS__::__vec3_minus(ve0, ve1);

    __MATHUTILS__::__set_Mat_val_column(coefMtr, col0, col1, col2);

    Scalar3 n = __MATHUTILS__::__vec3_cross(col0, col1);
    if (__MATHUTILS__::__vec3_dot(n, __MATHUTILS__::__vec3_minus(ve0, vt0)) *
            __MATHUTILS__::__vec3_dot(n, __MATHUTILS__::__vec3_minus(ve1, vt0)) >
        0) {
        return false;
    }

    Scalar det = __MATHUTILS__::__Determiant(coefMtr);

    if (abs(det) < very_small_number()) {
        return false;
    }

    __MATHUTILS__::Matrix3x3S D1, D2, D3;
    Scalar3 b = __MATHUTILS__::__vec3_minus(ve0, vt0);

    __MATHUTILS__::__set_Mat_val_column(D1, b, col1, col2);
    __MATHUTILS__::__set_Mat_val_column(D2, col0, b, col2);
    __MATHUTILS__::__set_Mat_val_column(D3, col0, col1, b);

    Scalar uvt[3];
    uvt[0] = __MATHUTILS__::__Determiant(D1) / det;
    uvt[1] = __MATHUTILS__::__Determiant(D2) / det;
    uvt[2] = __MATHUTILS__::__Determiant(D3) / det;

    if (uvt[0] >= 0.0 && uvt[1] >= 0.0 && uvt[0] + uvt[1] <= 1.0 && uvt[2] >= 0.0 && uvt[2] <= 1.0) {
        return true;
    } else {
        return false;
    }
}

__device__ __host__ inline bool _overlap(const AABB& lhs, const AABB& rhs, const Scalar& gapL) noexcept {
    if ((rhs.lower.x - lhs.upper.x) >= gapL || (lhs.lower.x - rhs.upper.x) >= gapL) return false;
    if ((rhs.lower.y - lhs.upper.y) >= gapL || (lhs.lower.y - rhs.upper.y) >= gapL) return false;
    if ((rhs.lower.z - lhs.upper.z) >= gapL || (lhs.lower.z - rhs.upper.z) >= gapL) return false;
    return true;
}

__global__ void _CollisionDetectTriEdge(const int* _btype, const Scalar3* _vertexes, const uint2* _edges,
                                        const uint3* _faces, const AABB* _edge_bvs, const Node* _edge_nodes,
                                        int* _isIntesect, Scalar dHat, int number) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= number) return;

    uint32_t stack[64];
    uint32_t* stack_ptr = stack;
    *stack_ptr++ = 0;

    uint3 face = _faces[idx];
    // idx = idx + number - 1;

    AABB _bv;

    Scalar3 _v = _vertexes[face.x];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.y];
    _bv.combines(_v.x, _v.y, _v.z);
    _v = _vertexes[face.z];
    _bv.combines(_v.x, _v.y, _v.z);

    Scalar gapl = 0;  // sqrt(dHat);

    do {
        const uint32_t node_id = *--stack_ptr;
        const uint32_t L_idx = _edge_nodes[node_id].left_idx;
        const uint32_t R_idx = _edge_nodes[node_id].right_idx;

        if (_overlap(_bv, _edge_bvs[L_idx], gapl)) {
            const auto obj_idx = _edge_nodes[L_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y || face.y == _edges[obj_idx].x ||
                      face.y == _edges[obj_idx].y || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y)) {
                    if (!(_btype[face.x] >= 2 && _btype[face.y] >= 2 && _btype[face.z] >= 2 &&
                          _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
                        if (edgeTriIntersect(_vertexes[_edges[obj_idx].x], _vertexes[_edges[obj_idx].y],
                                             _vertexes[face.x], _vertexes[face.y], _vertexes[face.z])) {
                            // atomicAdd(_isIntesect, -1);
                            *_isIntesect = -1;
                            printf("tri-edge intersection error\n tri: %d %d %d,  edge: %d  %d\n", face.x, face.y,
                                   face.z, _edges[obj_idx].x, _edges[obj_idx].y);
                            return;
                        }
                }

            } else  // the node is not a leaf.
            {
                *stack_ptr++ = L_idx;
            }
        }
        if (_overlap(_bv, _edge_bvs[R_idx], gapl)) {
            const auto obj_idx = _edge_nodes[R_idx].element_idx;
            if (obj_idx != 0xFFFFFFFF) {
                if (!(face.x == _edges[obj_idx].x || face.x == _edges[obj_idx].y || face.y == _edges[obj_idx].x ||
                      face.y == _edges[obj_idx].y || face.z == _edges[obj_idx].x || face.z == _edges[obj_idx].y)) {
                    if (!(_btype[face.x] >= 2 && _btype[face.y] >= 2 && _btype[face.z] >= 2 &&
                          _btype[_edges[obj_idx].x] >= 2 && _btype[_edges[obj_idx].y] >= 2))
                        if (edgeTriIntersect(_vertexes[_edges[obj_idx].x], _vertexes[_edges[obj_idx].y],
                                             _vertexes[face.x], _vertexes[face.y], _vertexes[face.z])) {
                            // atomicAdd(_isIntesect, -1);
                            *_isIntesect = -1;
                            printf("tri-edge intersection error\n tri: %d %d %d,  edge: %d  %d\n", face.x, face.y,
                                   face.z, _edges[obj_idx].x, _edges[obj_idx].y);
                            return;
                        }
                }

            } else  // the node is not a leaf.
            {
                *stack_ptr++ = R_idx;
            }
        }
    } while (stack < stack_ptr);
}

void LBVH_EF::init(int* _btype, Scalar3* _mVerts, Scalar3* _rest_vertexes, uint2* _mSurfEdges, uint3* _mSurfFaces,
                   uint32_t* _mSurfVerts, int4* _mCollisionPairs, int4* _ccd_mCollisionPairs, uint32_t* _mcpNum,
                   int* _mMatIndex, AABB* _surfEdge_bvs, Node* _surfEdge_nodes, const int& surfEdgeNum,
                   const int& surfFaceNum, const int& surfVertNum) {
    // 初始化基类 LBVH
    this->_btype = _btype;
    this->_vertexes = _mVerts;
    this->_rest_vertexes = _rest_vertexes;
    this->_surfEdges = _mSurfEdges;
    this->_surfFaces = _mSurfFaces;
    this->_surfVerts = _mSurfVerts;
    this->_collisionPair = _mCollisionPairs;
    this->_ccd_collisionPair = _ccd_mCollisionPairs;
    this->_cpNum = _mcpNum;
    this->_MatIndex = _mMatIndex;
    this->surfEdgeNum = surfEdgeNum;
    this->surfFaceNum = surfFaceNum;
    this->surfVertNum = surfVertNum;
    this->_surfEdge_bvs = _surfEdge_bvs;
    this->_surfEdge_nodes = _surfEdge_nodes;
}

bool LBVH_EF::CollisionDetectTriEdge(Scalar dHat) {
    int numbers = surfFaceNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    int* _isIntersect;
    CUDAMallocSafe(_isIntersect, 1);
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));

    _CollisionDetectTriEdge<<<blockNum, threadNum>>>(_btype, _vertexes, _surfEdges, _surfFaces, _surfEdge_bvs,
                                                     _surfEdge_nodes, _isIntersect, dHat, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDAFreeSafe(_isIntersect);
    if (h_isITST < 0) {
        return true;
    }
    return false;
}

bool LBVH_EF::checkCollisionDetectTriEdge(Scalar dHat) { return CollisionDetectTriEdge(dHat); }

void LBVHCollisionDetector::initBVH(std::unique_ptr<GeometryManager>& instance, int* _btype) {
    // init lbvh e
    lbvh_e.init(_btype, instance->getCudaSurfVertPos(), instance->getCudaRestVertPos(), instance->getCudaSurfEdgeIds(),
                instance->getCudaCollisionPairs(), instance->getCudaCCDCollisionPairs(), instance->getCudaCPNum(),
                instance->getCudaMatIndex(), instance->getHostNumSurfEdges(), instance->getHostNumSurfVerts());

    // init lbvh f
    lbvh_f.init(_btype, instance->getCudaSurfVertPos(), instance->getCudaSurfFaceIds(), instance->getCudaSurfVertIds(),
                instance->getCudaCollisionPairs(), instance->getCudaCCDCollisionPairs(), instance->getCudaCPNum(),
                instance->getCudaMatIndex(), instance->getHostNumSurfFaces(), instance->getHostNumSurfVerts());

    // init lbvh_ef
    lbvh_ef.init(_btype, instance->getCudaSurfVertPos(), instance->getCudaRestVertPos(), instance->getCudaSurfEdgeIds(),
                 instance->getCudaSurfFaceIds(), instance->getCudaSurfVertIds(), instance->getCudaCollisionPairs(),
                 instance->getCudaCCDCollisionPairs(), instance->getCudaCPNum(), instance->getCudaMatIndex(),
                 lbvh_e._bvs, lbvh_e._nodes, instance->getHostNumSurfEdges(), instance->getHostNumSurfFaces(),
                 instance->getHostNumSurfVerts());
}

// consider static object AABB box
void LBVHCollisionDetector::buildBVH(std::unique_ptr<GeometryManager>& instance) {
    lbvh_f.Construct();
    lbvh_e.Construct();
}

// consider moving object AABB with alpha and mvdir
void LBVHCollisionDetector::buildBVH_FULLCCD(std::unique_ptr<GeometryManager>& instance, const Scalar& alpha) {
    lbvh_f.ConstructFullCCD(instance->getCudaMoveDir(), alpha);
    lbvh_e.ConstructFullCCD(instance->getCudaMoveDir(), alpha);
}

// Static collision detection: The buildCP function is mainly responsible for detecting possible collision pairs in the
// current geometric state, regardless of the movement trajectory of the object (this is covered in buildBVH_FULLCCD),
// and update CPNum. Collision type: Mainly detects edge-edge and point-triangle collisions. When the distance between
// these elements is less than the preset threshold dHat, a collision is considered.
void LBVHCollisionDetector::buildCP(std::unique_ptr<GeometryManager>& instance) {
    CUDA_SAFE_CALL(cudaMemset(instance->getCudaCPNum(), 0, 5 * sizeof(uint32_t)));
    lbvh_f.SelfCollisionDetect(instance->getHostDHat());
    lbvh_e.SelfCollisionDetect(instance->getHostDHat());
    CUDA_SAFE_CALL(
        cudaMemcpy(&instance->getHostCpNum(0), instance->getCudaCPNum(), 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

void LBVHCollisionDetector::buildFullCP(std::unique_ptr<GeometryManager>& instance, const Scalar& alpha) {
    CUDA_SAFE_CALL(cudaMemset(instance->getCudaCPNum(), 0, sizeof(uint32_t)));
    lbvh_f.SelfCollisionFullDetect(instance->getHostDHat(), instance->getCudaMoveDir(), alpha);
    lbvh_e.SelfCollisionFullDetect(instance->getHostDHat(), instance->getCudaMoveDir(), alpha);
    CUDA_SAFE_CALL(
        cudaMemcpy(&instance->getHostCcdCpNum(), instance->getCudaCPNum(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
}

















__global__ void _GroundCollisionDetect(const Scalar3* vertexes, const uint32_t* surfVertIds, const Scalar* g_offset,
                                       const Scalar3* g_normal, uint32_t* _environment_collisionPair, uint32_t* _gpNum,
                                       Scalar dHat, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar dist = __MATHUTILS__::__vec3_dot(*g_normal, vertexes[surfVertIds[idx]]) - *g_offset;
    if (dist * dist > dHat) return;

    _environment_collisionPair[atomicAdd(_gpNum, 1)] = surfVertIds[idx];
}


void GroundCollisionDetect(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostNumSurfVerts();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _GroundCollisionDetect<<<blockNum, threadNum>>>(instance->getCudaSurfVertPos(), instance->getCudaSurfVertIds(),
                                                    instance->getCudaGroundOffset(), instance->getCudaGroundNormal(),
                                                    instance->getCudaEnvCollisionPairs(), instance->getCudaGPNum(),
                                                    instance->getHostDHat(), numbers);
}


void LBVHCollisionDetector::buildGP(std::unique_ptr<GeometryManager>& instance) {
    CUDA_SAFE_CALL(cudaMemset(instance->getCudaGPNum(), 0, sizeof(uint32_t)));
    GroundCollisionDetect(instance);
    CUDA_SAFE_CALL(
        cudaMemcpy(&instance->getHostGpNum(), instance->getCudaGPNum(), sizeof(uint32_t), cudaMemcpyDeviceToHost));
}






