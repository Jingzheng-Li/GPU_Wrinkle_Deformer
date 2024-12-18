
#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"

namespace __ACCD__ {

__device__ Scalar point_triangle_ccd(const Scalar3& _p, const Scalar3& _t0,
                                     const Scalar3& _t1, const Scalar3& _t2,
                                     const Scalar3& _dp, const Scalar3& _dt0,
                                     const Scalar3& _dt1, const Scalar3& _dt2, Scalar eta,
                                     Scalar thickness);

__device__ Scalar edge_edge_ccd(const Scalar3& _ea0, const Scalar3& _ea1,
                                const Scalar3& _eb0, const Scalar3& _eb1,
                                const Scalar3& _dea0, const Scalar3& _dea1,
                                const Scalar3& _deb0, const Scalar3& _deb1, Scalar eta,
                                Scalar thickness);

__device__ Scalar doCCDVF(const Scalar3& _p, const Scalar3& _t0, const Scalar3& _t1,
                          const Scalar3& _t2, const Scalar3& _dp, const Scalar3& _dt0,
                          const Scalar3& _dt1, const Scalar3& _dt2, Scalar errorRate,
                          Scalar thickness);

}; // namespace ACCD