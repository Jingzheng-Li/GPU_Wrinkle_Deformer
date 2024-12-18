

#include "ACCD.cuh"

namespace __ACCD__ {

__device__ int _dType_point_triangle(const Scalar3& v0, const Scalar3& v1,
                                     const Scalar3& v2, const Scalar3& v3) {
    Scalar3 basis0 = __MATHUTILS__::__vec3_minus(v2, v1);
    Scalar3 basis1 = __MATHUTILS__::__vec3_minus(v3, v1);
    Scalar3 basis2 = __MATHUTILS__::__vec3_minus(v0, v1);

    const Scalar3 nVec = __MATHUTILS__::__vec3_cross(basis0, basis1);

    basis1 = __MATHUTILS__::__vec3_cross(basis0, nVec);
    __MATHUTILS__::Matrix3x3S D, D1, D2;

    __MATHUTILS__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y,
                                 nVec.y, basis0.z, basis1.z, nVec.z);
    __MATHUTILS__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y,
                                 nVec.y, basis2.z, basis1.z, nVec.z);
    __MATHUTILS__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y,
                                 nVec.y, basis0.z, basis2.z, nVec.z);

    Scalar2 param[3];
    param[0].x = __MATHUTILS__::__Determiant(D1) / __MATHUTILS__::__Determiant(D);
    param[0].y = __MATHUTILS__::__Determiant(D2) / __MATHUTILS__::__Determiant(D);

    if (param[0].x > 0 && param[0].x < 1 && param[0].y >= 0) {
        return 3;  // PE v1v2
    } else {
        basis0 = __MATHUTILS__::__vec3_minus(v3, v2);
        basis1 = __MATHUTILS__::__vec3_cross(basis0, nVec);
        basis2 = __MATHUTILS__::__vec3_minus(v0, v2);

        __MATHUTILS__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y, basis1.y,
                                     nVec.y, basis0.z, basis1.z, nVec.z);
        __MATHUTILS__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y, basis1.y,
                                     nVec.y, basis2.z, basis1.z, nVec.z);
        __MATHUTILS__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y, basis2.y,
                                     nVec.y, basis0.z, basis2.z, nVec.z);

        param[1].x = __MATHUTILS__::__Determiant(D1) / __MATHUTILS__::__Determiant(D);
        param[1].y = __MATHUTILS__::__Determiant(D2) / __MATHUTILS__::__Determiant(D);

        if (param[1].x > 0.0 && param[1].x < 1.0 && param[1].y >= 0.0) {
            return 4;  // PE v2v3
        } else {
            basis0 = __MATHUTILS__::__vec3_minus(v1, v3);
            basis1 = __MATHUTILS__::__vec3_cross(basis0, nVec);
            basis2 = __MATHUTILS__::__vec3_minus(v0, v3);

            __MATHUTILS__::__set_Mat_val(D, basis0.x, basis1.x, nVec.x, basis0.y,
                                         basis1.y, nVec.y, basis0.z, basis1.z, nVec.z);
            __MATHUTILS__::__set_Mat_val(D1, basis2.x, basis1.x, nVec.x, basis2.y,
                                         basis1.y, nVec.y, basis2.z, basis1.z, nVec.z);
            __MATHUTILS__::__set_Mat_val(D2, basis0.x, basis2.x, nVec.x, basis0.y,
                                         basis2.y, nVec.y, basis0.z, basis2.z, nVec.z);

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

__device__ int _dType_edge_edge(const Scalar3& v0, const Scalar3& v1, const Scalar3& v2,
                                const Scalar3& v3) {
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
             __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_cross(u, v)) <
                 1.0e-20 * a * c)) {
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

__device__ __forceinline__ Scalar point_point_distance(const Scalar3& v0,
                                                       const Scalar3& v1) {
    return __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(v0, v1));
}

__device__ __forceinline__ Scalar point_triangle_distance(const Scalar3& v0,
                                                          const Scalar3& v1,
                                                          const Scalar3& v2,
                                                          const Scalar3& v3) {
    Scalar3 b = __MATHUTILS__::__vec3_cross(__MATHUTILS__::__vec3_minus(v2, v1),
                                             __MATHUTILS__::__vec3_minus(v3, v1));
    // Scalar3 test = __MATHUTILS__::__minus(v0, v1);
    Scalar aTb = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_minus(v0, v1),
                                            b);  //(v0 - v1).dot(b);
    // printf("%f   %f   %f          %f   %f   %f   %f\n", b.x, b.y, b.z, test.x, test.y,
    // test.z, aTb);
    return aTb * aTb / __MATHUTILS__::__vec3_squaredNorm(b);
}

__device__ __forceinline__ Scalar point_edge_distance(const Scalar3& v0,
                                                      const Scalar3& v1,
                                                      const Scalar3& v2) {
    return __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_cross(
               __MATHUTILS__::__vec3_minus(v1, v0), __MATHUTILS__::__vec3_minus(v2, v0))) /
           __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(v2, v1));
}

__device__ __forceinline__ Scalar edge_edge_distance(const Scalar3& v0, const Scalar3& v1,
                                                     const Scalar3& v2,
                                                     const Scalar3& v3) {
    Scalar3 b = __MATHUTILS__::__vec3_cross(
        __MATHUTILS__::__vec3_minus(v1, v0),
        __MATHUTILS__::__vec3_minus(v3, v2));  //(v1 - v0).cross(v3 - v2);

    Scalar aTb = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_minus(v2, v0), b);
    return aTb * aTb / __MATHUTILS__::__vec3_squaredNorm(b);
}

__device__ __forceinline__ Scalar _d_EEParallel(const Scalar3& v0, const Scalar3& v1,
                                                const Scalar3& v2, const Scalar3& v3) {
    Scalar3 b = __MATHUTILS__::__vec3_cross(
        __MATHUTILS__::__vec3_cross(__MATHUTILS__::__vec3_minus(v1, v0),
                                     __MATHUTILS__::__vec3_minus(v2, v0)),
        __MATHUTILS__::__vec3_minus(v1, v0));
    Scalar aTb = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_minus(v2, v0), b);
    return aTb * aTb / __MATHUTILS__::__vec3_squaredNorm(b);
}

__device__ Scalar edge_edge_distance_unclassified(const Scalar3& ea0, const Scalar3& ea1,
                                                  const Scalar3& eb0,
                                                  const Scalar3& eb1) {
    switch (_dType_edge_edge(ea0, ea1, eb0, eb1)) {
        case 0:
            return point_point_distance(ea0, eb0);
        case 1:
            return point_point_distance(ea0, eb1);
        case 2:
            return point_edge_distance(ea0, eb0, eb1);
        case 3:
            return point_point_distance(ea1, eb0);
        case 4:
            return point_point_distance(ea1, eb1);
        case 5:
            return point_edge_distance(ea1, eb0, eb1);
        case 6:
            return point_edge_distance(eb0, ea0, ea1);
        case 7:
            return point_edge_distance(eb1, ea0, ea1);
        case 8:
            return edge_edge_distance(ea0, ea1, eb0, eb1);
        default:
            return 1e32;
    }
}

__device__ Scalar point_triangle_distance_unclassified(const Scalar3& p,
                                                       const Scalar3& t0,
                                                       const Scalar3& t1,
                                                       const Scalar3& t2) {
    switch (_dType_point_triangle(p, t0, t1, t2)) {
        case 0:
            return point_point_distance(p, t0);
        case 1:
            return point_point_distance(p, t1);
        case 2:
            return point_point_distance(p, t2);
        case 3:
            return point_edge_distance(p, t0, t1);
        case 4:
            return point_edge_distance(p, t1, t2);
        case 5:
            return point_edge_distance(p, t2, t0);
        case 6:
            return point_triangle_distance(p, t0, t1, t2);
        default:
            return 1e32;
    }
}

__device__ Scalar edge_edge_ccd(const Scalar3& _ea0, const Scalar3& _ea1,
                                const Scalar3& _eb0, const Scalar3& _eb1,
                                const Scalar3& _dea0, const Scalar3& _dea1,
                                const Scalar3& _deb0, const Scalar3& _deb1, Scalar eta,
                                Scalar thickness) {
    Scalar3 ea0 = _ea0, ea1 = _ea1, eb0 = _eb0, eb1 = _eb1, dea0 = _dea0, dea1 = _dea1,
            deb0 = _deb0, deb1 = _deb1;
    Scalar3 temp0 = __MATHUTILS__::__vec3_add(dea0, dea1);
    Scalar3 temp1 = __MATHUTILS__::__vec3_add(deb0, deb1);
    Scalar3 mov =
        __MATHUTILS__::__s_vec3_multiply(__MATHUTILS__::__vec3_add(temp0, temp1), -0.25);

    dea0 = __MATHUTILS__::__vec3_add(dea0, mov);
    dea1 = __MATHUTILS__::__vec3_add(dea1, mov);
    deb0 = __MATHUTILS__::__vec3_add(deb0, mov);
    deb1 = __MATHUTILS__::__vec3_add(deb1, mov);

    Scalar max_disp_mag =
        sqrt(__MATHUTILS__::__m_max(__MATHUTILS__::__vec3_squaredNorm(dea0),
                                    __MATHUTILS__::__vec3_squaredNorm(dea1))) +
        sqrt(__MATHUTILS__::__m_max(__MATHUTILS__::__vec3_squaredNorm(deb0),
                                    __MATHUTILS__::__vec3_squaredNorm(deb1)));
    if (max_disp_mag == 0) return 1.0;

    Scalar dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1);

    Scalar dFunc = dist2_cur - thickness * thickness;
    if (dFunc <= 0) {
        Scalar dists0 = __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(ea0, eb0));
        Scalar dists1 = __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(ea0, eb1));
        Scalar dists2 = __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(ea1, eb0));
        Scalar dists3 = __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(ea1, eb1));

        dist2_cur = __MATHUTILS__::__m_min(__MATHUTILS__::__m_min(dists0, dists1),
                                           __MATHUTILS__::__m_min(dists2, dists3));
        dFunc = dist2_cur - thickness * thickness;
    }
    Scalar dist_cur = sqrt(dist2_cur);
    Scalar gap = eta * dFunc / (dist_cur + thickness);
    Scalar toc = 0.0;
    int count = 0;
    while (true) {
        count++;
        if (count > 50000) return toc;
        // if (count > 5000)
        //     printf("ee  %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n %f  %f
        //     %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n", _dea0.x, _dea0.y, _dea0.z,
        //     _dea1.x, _dea1.y, _dea1.z,
        //         _deb0.x, _deb0.y, _deb0.z, _deb1.x, _deb1.y, _deb1.z, _ea0.x, _ea0.y,
        //         _ea0.z, _ea1.x, _ea1.y, _ea1.z, _eb0.x, _eb0.y, _eb0.z, _eb1.x, _eb1.y,
        //         _eb1.z);
        Scalar toc_lower_bound =
            (1 - eta) * dFunc / ((dist_cur + thickness) * max_disp_mag);
        ea0 = __MATHUTILS__::__vec3_add(
            ea0, __MATHUTILS__::__s_vec3_multiply(dea0, toc_lower_bound));
        ea1 = __MATHUTILS__::__vec3_add(
            ea1, __MATHUTILS__::__s_vec3_multiply(dea1, toc_lower_bound));
        eb0 = __MATHUTILS__::__vec3_add(
            eb0, __MATHUTILS__::__s_vec3_multiply(deb0, toc_lower_bound));
        eb1 = __MATHUTILS__::__vec3_add(
            eb1, __MATHUTILS__::__s_vec3_multiply(deb1, toc_lower_bound));

        dist2_cur = edge_edge_distance_unclassified(ea0, ea1, eb0, eb1);
        dFunc = dist2_cur - thickness * thickness;
        if (dFunc <= 0) {
            Scalar dists0 =
                __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(ea0, eb0));
            Scalar dists1 =
                __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(ea0, eb1));
            Scalar dists2 =
                __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(ea1, eb0));
            Scalar dists3 =
                __MATHUTILS__::__vec3_squaredNorm(__MATHUTILS__::__vec3_minus(ea1, eb1));

            dist2_cur = __MATHUTILS__::__m_min(__MATHUTILS__::__m_min(dists0, dists1),
                                               __MATHUTILS__::__m_min(dists2, dists3));
            dFunc = dist2_cur - thickness * thickness;
        }
        dist_cur = sqrt(dist2_cur);
        if (toc && (dFunc / (dist_cur + thickness) < gap)) {
            break;
        }
        toc += toc_lower_bound;
        if (toc > 1.0) return 1.0;
    }
    return toc;
}

__device__ Scalar point_triangle_ccd(const Scalar3& _p, const Scalar3& _t0,
                                     const Scalar3& _t1, const Scalar3& _t2,
                                     const Scalar3& _dp, const Scalar3& _dt0,
                                     const Scalar3& _dt1, const Scalar3& _dt2, Scalar eta,
                                     Scalar thickness) {
    Scalar3 p = _p, t0 = _t0, t1 = _t1, t2 = _t2, dp = _dp, dt0 = _dt0, dt1 = _dt1,
            dt2 = _dt2;

    Scalar3 temp0 = __MATHUTILS__::__vec3_add(dt0, dt1);
    Scalar3 temp1 = __MATHUTILS__::__vec3_add(dt2, dp);
    Scalar3 mov =
        __MATHUTILS__::__s_vec3_multiply(__MATHUTILS__::__vec3_add(temp0, temp1), -0.25);

    dt0 = __MATHUTILS__::__vec3_add(dt0, mov);
    dt1 = __MATHUTILS__::__vec3_add(dt1, mov);
    dt2 = __MATHUTILS__::__vec3_add(dt2, mov);
    dp = __MATHUTILS__::__vec3_add(dp, mov);

    Scalar disp_mag2_vec0 = __MATHUTILS__::__vec3_squaredNorm(dt0);
    Scalar disp_mag2_vec1 = __MATHUTILS__::__vec3_squaredNorm(dt1);
    Scalar disp_mag2_vec2 = __MATHUTILS__::__vec3_squaredNorm(dt2);

    Scalar max_disp_mag =
        __MATHUTILS__::__vec3_norm(dp) +
        sqrt(__MATHUTILS__::__m_max(
            disp_mag2_vec0, __MATHUTILS__::__m_max(disp_mag2_vec1, disp_mag2_vec2)));
    if (max_disp_mag == 0) return 1.0;

    Scalar dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2);
    Scalar dist_cur = sqrt(dist2_cur);
    Scalar gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness);
    Scalar toc = 0.0;
    int count = 0;
    while (true) {
        count++;
        if (count > 50000) return toc;
        // if (count > 5000)
        //     printf("pt  %f  %f  %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n %f  %f
        //     %f\n%f  %f  %f\n%f  %f  %f\n%f  %f  %f\n\n\n", _dp.x, _dp.y, _dp.z, _dt0.x,
        //     _dt0.y, _dt0.z,
        //         _dt1.x, _dt1.y, _dt1.z, _dt2.x, _dt2.y, _dt2.z, _p.x, _p.y, _p.z,
        //         _t0.x, _t0.y, _t0.z, _t1.x, _t1.y, _t1.z, _t2.x, _t2.y, _t2.z);
        Scalar toc_lower_bound = (1 - eta) * (dist2_cur - thickness * thickness) /
                                 ((dist_cur + thickness) * max_disp_mag);

        p = __MATHUTILS__::__vec3_add(p, __MATHUTILS__::__s_vec3_multiply(dp, toc_lower_bound));
        t0 = __MATHUTILS__::__vec3_add(t0,
                                  __MATHUTILS__::__s_vec3_multiply(dt0, toc_lower_bound));
        t1 = __MATHUTILS__::__vec3_add(t1,
                                  __MATHUTILS__::__s_vec3_multiply(dt1, toc_lower_bound));
        t2 = __MATHUTILS__::__vec3_add(t2,
                                  __MATHUTILS__::__s_vec3_multiply(dt2, toc_lower_bound));

        dist2_cur = point_triangle_distance_unclassified(p, t0, t1, t2);
        dist_cur = sqrt(dist2_cur);
        if (toc && ((dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap)) {
            break;
        }

        toc += toc_lower_bound;
        if (toc > 1.0) {
            return 1.0;
        }
    }
    return toc;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct {
    Scalar3 ad, bd, cd, pd;
    Scalar3 a0, b0, c0, p0;
} NewtonCheckData;

inline __device__ bool _insideTriangle(Scalar3 a, Scalar3 b, Scalar3 c, Scalar3 p) {
    Scalar3 n, da, db, dc;
    Scalar wa, wb, wc;

    Scalar3 ba = __MATHUTILS__::__vec3_minus(b, a);
    Scalar3 ca = __MATHUTILS__::__vec3_minus(c, a);

    n = __MATHUTILS__::__vec3_cross(ba, ca);  // cross(ba, ca);

    da = __MATHUTILS__::__vec3_minus(a, p);
    db = __MATHUTILS__::__vec3_minus(b, p);
    dc = __MATHUTILS__::__vec3_minus(c, p);
    // da = a - p, db = b - p, dc = c - p;
    if ((wa = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_cross(db, dc), n)) < 0.0f)
        return false;
    if ((wb = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_cross(dc, da), n)) < 0.0f)
        return false;
    if ((wc = __MATHUTILS__::__vec3_dot(__MATHUTILS__::__vec3_cross(da, db), n)) < 0.0f)
        return false;

    // Compute barycentric coordinates
    Scalar area2 = __MATHUTILS__::__vec3_dot(n, n);
    wa /= area2, wb /= area2, wc /= area2;

    return true;
}

inline __device__ int solveQuadric(Scalar c[3], Scalar s[2], const Scalar& errorRate) {
    Scalar p, q, D;

    // make sure we have a d2 equation

    if (c[2] < errorRate) {
        if ((c[1]) < errorRate) return 0;
        s[0] = -c[0] / c[1];
        return 1;
    }

    // normal for: x^2 + px + q
    p = c[1] / (2.0f * c[2]);
    q = c[0] / c[2];
    D = p * p - q;

    if ((D) < errorRate) {
        // one Scalar root
        s[0] = s[1] = -p;
        return 1;
    }

    if (D < 0.0f)
        // no real root
        return 0;

    else {
        // two real roots
        Scalar sqrt_D = sqrt(D);
        s[0] = sqrt_D - p;
        s[1] = -sqrt_D - p;
        return 2;
    }
}
inline __device__ int solveCubic(Scalar c[4], Scalar s[3], const Scalar& errorRate) {
    int i, num;
    Scalar sub, A, B, C, sq_A, p, q, cb_p, D;

    if (c[3] < errorRate) {
        return solveQuadric(c, s, errorRate);
    }

    A = c[2] / c[3];
    B = c[1] / c[3];
    C = c[0] / c[3];

    sq_A = A * A;
    Scalar ONE_DIV_3 = 1.0 / 3;
    p = ONE_DIV_3 * (-ONE_DIV_3 * sq_A + B);
    q = 0.5f * (2.0f / 27.0f * A * sq_A - ONE_DIV_3 * A * B + C);

    // use Cardano's formula

    cb_p = p * p * p;
    D = q * q + cb_p;
    if ((D) < errorRate) {
        if ((q) < errorRate) {
            // one triple solution
            s[0] = 0.0f;
            num = 1;
        } else {
            // one single and one Scalar solution
            Scalar u = cbrt(-q);
            s[0] = 2.0f * u;
            s[1] = -u;
            num = 2;
        }
    } else if (D < 0.0f) {
        // casus irreductibilis: three real solutions
        Scalar phi = ONE_DIV_3 * acos(-q / sqrt(-cb_p));
        Scalar t = 2.0f * sqrt(-p);
        s[0] = t * cos(phi);
        s[1] = -t * cos(phi + __MATHUTILS__::__PI() / 3.0f);
        s[2] = -t * cos(phi - __MATHUTILS__::__PI() / 3.0f);
        num = 3;
    } else {
        // one real solution
        Scalar sqrt_D = sqrt(D);
        Scalar u = cbrt(sqrt_D + fabs(q));
        if (q > 0.0f)
            s[0] = -u + p / u;
        else
            s[0] = u - p / u;
        num = 1;
    }

    // resubstitute
    sub = ONE_DIV_3 * A;
    for (i = 0; i < num; i++) s[i] -= sub;
    return num;
}

inline __device__ void _equateCubic_VF(Scalar3 a0, Scalar3 ad, Scalar3 b0, Scalar3 bd,
                                       Scalar3 c0, Scalar3 cd, Scalar3 p0, Scalar3 pd,
                                       Scalar& a, Scalar& b, Scalar& c, Scalar& d,
                                       const Scalar& thickness) {
    Scalar3 dab, dac, dap;
    Scalar3 oab, oac, oap;
    Scalar3 dabXdac, dabXoac, oabXdac, oabXoac;

    dab = __MATHUTILS__::__vec3_minus(bd, ad), dac = __MATHUTILS__::__vec3_minus(cd, ad),
    dap = __MATHUTILS__::__vec3_minus(pd, ad);
    oab = __MATHUTILS__::__vec3_minus(b0, a0), oac = __MATHUTILS__::__vec3_minus(c0, a0),
    oap = __MATHUTILS__::__vec3_minus(p0, a0);

    dabXdac = __MATHUTILS__::__vec3_cross(dab, dac);
    dabXoac = __MATHUTILS__::__vec3_cross(dab, oac);
    oabXdac = __MATHUTILS__::__vec3_cross(oab, dac);
    oabXoac = __MATHUTILS__::__vec3_cross(oab, oac);

    a = __MATHUTILS__::__vec3_dot(dap, dabXdac);
    b = __MATHUTILS__::__vec3_dot(oap, dabXdac) +
        __MATHUTILS__::__vec3_dot(dap, __MATHUTILS__::__vec3_add(dabXoac, oabXdac));
    c = __MATHUTILS__::__vec3_dot(dap, oabXoac) +
        __MATHUTILS__::__vec3_dot(oap, __MATHUTILS__::__vec3_add(dabXoac, oabXdac));
    d = thickness * __MATHUTILS__::__vec3_dot(oap, oabXoac);
}

inline __device__ Scalar IntersectVF(Scalar3 ta0, Scalar3 tb0, Scalar3 tc0, Scalar3 ad,
                                     Scalar3 bd, Scalar3 cd, Scalar3 q0, Scalar3 qd,
                                     const Scalar& errorRate, const Scalar& thickness) {
    Scalar collisionTime = 1.0;

    Scalar a, b, c, d; /* cubic polynomial coefficients */
    _equateCubic_VF(ta0, ad, tb0, bd, tc0, cd, q0, qd, a, b, c, d, thickness);

    // if ((a) < errorRate && (b) < errorRate && (c) < errorRate && (d) < errorRate)
    //     return 1.0;

    Scalar roots[3];
    Scalar coeffs[4];
    coeffs[3] = a, coeffs[2] = b, coeffs[1] = c, coeffs[0] = d;
    int num;  // = solveCubic(coeffs, roots, errorRate);

    __MATHUTILS__::__NewtonSolverForCubicEquation(a, b, c, d, roots, num, errorRate);

    if (num == 0) return 1.0;

    for (int i = 0; i < num; i++) {
        Scalar r = roots[i];
        if (r < 0 || r > 1) continue;

        if (_insideTriangle(
                __MATHUTILS__::__vec3_add(__MATHUTILS__::__s_vec3_multiply(ad, r), ta0),
                __MATHUTILS__::__vec3_add(__MATHUTILS__::__s_vec3_multiply(bd, r), tb0),
                __MATHUTILS__::__vec3_add(__MATHUTILS__::__s_vec3_multiply(cd, r), tc0),
                __MATHUTILS__::__vec3_add(__MATHUTILS__::__s_vec3_multiply(qd, r), q0))) {
            if (collisionTime > r) {
                collisionTime = r;
            }
        }
    }

    return collisionTime;
}

__device__ Scalar doCCDVF(const Scalar3& _p, const Scalar3& _t0, const Scalar3& _t1,
                          const Scalar3& _t2, const Scalar3& _dp, const Scalar3& _dt0,
                          const Scalar3& _dt1, const Scalar3& _dt2, Scalar errorRate,
                          Scalar thickness) {
    Scalar ret =
        IntersectVF(_t0, _t1, _t2, _dt0, _dt1, _dt2, _p, _dp, errorRate, thickness);

    return ret;
}

}; // namespace ACCD