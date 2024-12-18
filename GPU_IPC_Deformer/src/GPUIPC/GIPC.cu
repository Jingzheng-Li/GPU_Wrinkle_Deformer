

#include <fstream>

#include "ACCD.cuh"
#include "FEMEnergy.cuh"
#include "GIPC.cuh"
#include "GIPC_PDerivative.cuh"
#include "GeometryManager.hpp"

namespace __GPUIPC__ {

__device__ Scalar __calBarrierSelfConsDis(const Scalar3* _vertexes, const int4 MMCVIDI) {
    Scalar dis;
    if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) { // (+,+,+,+) edge-edge
        __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
    }
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z == -1 && MMCVIDI.w == -1) { // (-,+,#,#) point-point 
        __MATHUTILS__::_d_PP(_vertexes[-MMCVIDI.x - 1], _vertexes[MMCVIDI.y], dis);
    }
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w == -1) { // (-,+,+,#) point-edge 
        __MATHUTILS__::_d_PE(_vertexes[-MMCVIDI.x - 1], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
    }
    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) { // (-,+,+,+) point-triangle
        __MATHUTILS__::_d_PT(_vertexes[-MMCVIDI.x - 1], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
    }
    else if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) { // (+,+,+,-) parallel edge-edge
        __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[-MMCVIDI.w - 1], dis);
    }
    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z < 0 && MMCVIDI.w < 0) { // (-,-,-,-) parallel point-point
        __MATHUTILS__::_d_PP(_vertexes[-MMCVIDI.x - 1], _vertexes[-MMCVIDI.y - 1], dis);
    }
    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) { // (-,-,+,-) parallel point edge
        __MATHUTILS__::_d_PE(_vertexes[-MMCVIDI.x - 1], _vertexes[-MMCVIDI.y - 1], _vertexes[MMCVIDI.z], dis);
    }
    else {
        printf("################################ Error: invalid collision conditions");
        // exit(EXIT_FAILURE);;
    }
    return dis;
}


__device__ Scalar __calBarrierEnergy(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI,
                                       Scalar _Kappa, Scalar _dHat) {
    Scalar dHat = _dHat;
    Scalar Kappa = _Kappa;

    // dHat = dhat^2, dis = d^2, g = (d/dhat)^2
    // Energy = kappa * (dhat^2 - dhat^2*g)*log^2(g)

    // if nearly parallel, consider modifier additionally
    // Energy = kappa * modifier * (dhat^2 - dhat^2*g)*log^2(g) 

    if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        // (+,+,+,+) edge-edge
        Scalar dis;
        __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
        Scalar I5 = dis / dHat;
        Scalar lenE = dis - dHat;
        return Kappa * lenE * lenE * log(I5) * log(I5);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z == -1 && MMCVIDI.w == -1) {
        // (-,+,#,#) point-point 
        Scalar dis;
        MMCVIDI.x = -MMCVIDI.x - 1;
        __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
        Scalar I5 = dis / dHat;
        Scalar lenE = dis - dHat;
        return Kappa * lenE * lenE * log(I5) * log(I5);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w == -1) {
        // (-,+,+,#) point-edge 
        Scalar dis;
        MMCVIDI.x = -MMCVIDI.x - 1;
        __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
        Scalar I5 = dis / dHat;
        Scalar lenE = dis - dHat;
        return Kappa * lenE * lenE * log(I5) * log(I5);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        // (-,+,+,+) point-triangle
        Scalar dis;
        MMCVIDI.x = -MMCVIDI.x - 1;
        __MATHUTILS__::_d_PT(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
        Scalar I5 = dis / dHat;
        Scalar lenE = dis - dHat;
        return Kappa * lenE * lenE * log(I5) * log(I5);
    }

    else if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        // (+,+,+,-) parallel edge-edge
        MMCVIDI.w = -MMCVIDI.w - 1;
        Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
        Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
        Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1));
        Scalar I1 = c * c;
        if (I1 == 0) return 0;
        Scalar dis;
        __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
        Scalar I2 = dis / dHat;
        Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y],
                                                   _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);
        Scalar modifier_ek = (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1);
        Scalar Energy = Kappa * modifier_ek * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
        if (Energy < 0) printf("parallel edge edge collision energy less than zero!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return Energy;
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z < 0 && MMCVIDI.w < 0) {
        // (-,-,-,-) parallel point-point
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.z = -MMCVIDI.z - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;

        Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
        Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
        Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1));
        Scalar I1 = c * c;
        if (I1 == 0) return 0;
        Scalar dis;
        __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
        Scalar I2 = dis / dHat;
        Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z],
                                                   _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.w]);
        Scalar modifier_ek = (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1);
        Scalar Energy = Kappa * modifier_ek * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
        if (Energy < 0) printf("parallel point point collision energy less than zero!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return Energy;
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        // (-,-,+,-) parallel point edge
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;

        Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
        Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
        Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1));
        Scalar I1 = c * c;
        if (I1 == 0) return 0;
        Scalar dis;
        __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
        Scalar I2 = dis / dHat;
        Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w],
                                                   _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z]);
        Scalar modifier_ek = (-(1 / (eps_x * eps_x)) * I1 * I1 + (2 / eps_x) * I1);
        Scalar Energy = Kappa * modifier_ek * (dHat - dHat * I2) * (dHat - dHat * I2) * log(I2) * log(I2);
        if (Energy < 0) printf("parallel point edge collision energy less than zero!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        return Energy;
    }

    else {
        printf("################################ Error: invalid collision conditions");
        // exit(EXIT_FAILURE);;
    }

    return 0;
}


__device__ void compute_barrier_grad_EE(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI, Scalar3* _gradient,
                              Scalar dHat, Scalar Kappa) {
    Scalar dis;
    __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);
    __MATHUTILS__::Matrix12x9S PFPxT;
    __GIPCDERIVATIVE__::pFpx_ee2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPxT); // vec(PJPx) (3x3x12)
    Scalar I5 = dis / dHat;
    __MATHUTILS__::Vector9S tmp;
    tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
    tmp.v[8] = 2 * dis_sqrt / dhat_sqrt;
    Scalar PbPg = (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, Kappa * PbPg); // PbPg @ vec(PgPJ)

    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply((PFPxT), flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }
}


__device__ void compute_barrier_grad_PP(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI, Scalar3* _gradient,
                              Scalar dHat, Scalar Kappa) {
    Scalar dis;
    __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);
    __MATHUTILS__::Vector6S PFPxT;
    __GIPCDERIVATIVE__::pFpx_pp2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dhat_sqrt, PFPxT);
    Scalar I5 = dis / dHat;
    Scalar I5_sqrt = 2 * dis_sqrt / dhat_sqrt;
    Scalar PbPg = (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
    Scalar flatten_pk1 = I5_sqrt * Kappa * PbPg;

    __MATHUTILS__::Vector6S gradient_vec = __MATHUTILS__::__s_vec6_multiply(PFPxT, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
    }
}


__device__ void compute_barrier_grad_PE(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI, Scalar3* _gradient,
                              Scalar dHat, Scalar Kappa) {
    Scalar dis;
    __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);
    __MATHUTILS__::Matrix9x4S PFPxT;
    __GIPCDERIVATIVE__::pFpx_pe2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dhat_sqrt, PFPxT);
    Scalar I5 = dis / dHat;
    __MATHUTILS__::Vector4S fnn;
    fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;
    fnn.v[3] = 2 * dis_sqrt / dhat_sqrt;
    Scalar PbPg = (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
    __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(fnn, Kappa * PbPg);

    __MATHUTILS__::Vector9S gradient_vec = __MATHUTILS__::__M9x4_v4_multiply(PFPxT, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
    }
}


__device__ void compute_barrier_grad_PT(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI, Scalar3* _gradient,
                              Scalar dHat, Scalar Kappa) {

    Scalar dis;
    __MATHUTILS__::_d_PT(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);
    __MATHUTILS__::Matrix12x9S PFPxT;
    __GIPCDERIVATIVE__::pFpx_pt2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPxT);
    Scalar I5 = dis / dHat;
    __MATHUTILS__::Vector9S tmp;
    tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
    tmp.v[8] = 2 * dis_sqrt / dhat_sqrt;
    Scalar PbPg = (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, Kappa * PbPg);

    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(PFPxT, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }
}


__device__ void compute_barrier_grad_parallelEE(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI, Scalar3* _gradient,
                              Scalar dHat, Scalar Kappa) {

    Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
    Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
    Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1)) /*/ __MATHUTILS__::__v3_norm(v0)*/;
    Scalar I1 = c * c;
    if (I1 == 0) return;
    Scalar dis;
    __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
    Scalar I2 = dis / dHat;
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);

    __MATHUTILS__::Matrix3x3S F;
    __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis_sqrt / dhat_sqrt);
    Scalar3 n1 = make_Scalar3(0, 1, 0);
    Scalar3 n2 = make_Scalar3(0, 0, 1);

    Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y],
                                               _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);

    __MATHUTILS__::Matrix3x3S g1, g2;

    __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g1);
    nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g2);

    __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
    __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

    __MATHUTILS__::Matrix12x9S PFPx;
    __GIPCDERIVATIVE__::pFpx_pee(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPx);

    Scalar p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                (I2 * (eps_x * eps_x));

    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__vec9_add(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                                                __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }
}


__device__ void compute_barrier_grad_parallelPP(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI, Scalar3* _gradient,
                              Scalar dHat, Scalar Kappa) {

    Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
    Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
    Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1)) /*/ __MATHUTILS__::__v3_norm(v0)*/;
    Scalar I1 = c * c;
    if (I1 == 0) return;
    Scalar dis;
    __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
    Scalar I2 = dis / dHat;
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);

    __MATHUTILS__::Matrix3x3S F;
    __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis_sqrt / dhat_sqrt);
    Scalar3 n1 = make_Scalar3(0, 1, 0);
    Scalar3 n2 = make_Scalar3(0, 0, 1);

    Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z],
                                               _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.w]);

    __MATHUTILS__::Matrix3x3S g1, g2;

    __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g1);
    nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g2);

    __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
    __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

    __MATHUTILS__::Matrix12x9S PFPx;
    __GIPCDERIVATIVE__::pFpx_ppp(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPx);

    Scalar p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                (I2 * (eps_x * eps_x));

    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__vec9_add(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                                                __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }
}


__device__ void compute_barrier_grad_parallelPE(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI, Scalar3* _gradient,
                              Scalar dHat, Scalar Kappa) {

    Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
    Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
    Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1)) /*/ __MATHUTILS__::__v3_norm(v0)*/;
    Scalar I1 = c * c;
    if (I1 == 0) return;
    Scalar dis;
    __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
    Scalar I2 = dis / dHat;
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);

    __MATHUTILS__::Matrix3x3S F;
    __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis_sqrt / dhat_sqrt);
    Scalar3 n1 = make_Scalar3(0, 1, 0);
    Scalar3 n2 = make_Scalar3(0, 0, 1);

    Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w],
                                               _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z]);

    __MATHUTILS__::Matrix3x3S g1, g2;

    __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g1);
    nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g2);

    __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
    __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

    __MATHUTILS__::Matrix12x9S PFPx;
    __GIPCDERIVATIVE__::pFpx_ppe(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPx);

    Scalar p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                (I2 * (eps_x * eps_x));

    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__vec9_add(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                                                __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }
}

__global__ void _calBarrierGradient(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, const int4* _collisionPair,
                                    Scalar3* _gradient, Scalar dHat, Scalar Kappa, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _collisionPair[idx];
    Scalar dHat_sqrt = sqrt(dHat);

    if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        compute_barrier_grad_EE(_vertexes, _rest_vertexes, MMCVIDI, _gradient, dHat, Kappa);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z == -1 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        compute_barrier_grad_PP(_vertexes, _rest_vertexes, MMCVIDI, _gradient, dHat, Kappa);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        compute_barrier_grad_PE(_vertexes, _rest_vertexes, MMCVIDI, _gradient, dHat, Kappa);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        compute_barrier_grad_PT(_vertexes, _rest_vertexes, MMCVIDI, _gradient, dHat, Kappa);
    }

    else if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.w = -MMCVIDI.w - 1;
        compute_barrier_grad_parallelEE(_vertexes, _rest_vertexes, MMCVIDI, _gradient, dHat, Kappa);
    }      

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z < 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.z = -MMCVIDI.z - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        compute_barrier_grad_parallelPP(_vertexes, _rest_vertexes, MMCVIDI, _gradient, dHat, Kappa);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        compute_barrier_grad_parallelPE(_vertexes, _rest_vertexes, MMCVIDI, _gradient, dHat, Kappa);
    }

    else {
        printf("################################ Error: invalid collision conditions");
        // exit(EXIT_FAILURE);;
    }

}

__device__ void compute_barrier_Hess_EE(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI,
                                    Scalar3* _gradient, __MATHUTILS__::Matrix12x12S* H12x12, uint4* D4Index,
                                    int* matIndex, Scalar dHat, Scalar Kappa, Scalar gassThreshold, int idx) {
    Scalar dis;
    __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);
    __MATHUTILS__::Matrix12x9S PFPxT;
    __GIPCDERIVATIVE__::pFpx_ee2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPxT);
    Scalar I5 = dis / dHat;
    __MATHUTILS__::Vector9S tmp;
    tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
    tmp.v[8] = 2 * dis_sqrt / dhat_sqrt;

    Scalar PbPg = (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, Kappa * PbPg);
    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply((PFPxT), flatten_pk1);
    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }

    Scalar lambda0 = -Kappa * (4 * dHat * dHat *
                       (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 +
                        I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
    // filtered Hessian, lambda not too small
    if (dis < gassThreshold * dHat) {
        Scalar lambda1 = -Kappa * (4 * dHat * dHat *
                           (4 * gassThreshold + log(gassThreshold) -
                            3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) +
                            6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold +
                            gassThreshold * log(gassThreshold) * log(gassThreshold) -
                            7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) / gassThreshold;
        lambda0 = lambda1;
    }

    __MATHUTILS__::Vector9S q0;
    q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
    q0.v[8] = 1;

    __MATHUTILS__::Matrix9x9S H;
    H = __MATHUTILS__::__s_Mat9x9_multiply(__MATHUTILS__::__v9_vec9_toMat9x9(q0, q0), lambda0); // H = lambda * vec(Q) * vec(Q)
    __MATHUTILS__::Matrix12x12S Hessian;
    __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian); // Hessian = vec(PJPx) @ (lambda * vec(Q) * vec(Q)) @ vec(PJPx)^T

    int Hidx = matIndex[idx];  // atomicAdd(_cpNum + 4, 1);
    H12x12[Hidx] = Hessian;
    D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
}

__device__ void compute_barrier_Hess_PP(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI,
                                    Scalar3* _gradient, __MATHUTILS__::Matrix6x6S* H6x6, uint2* D2Index, int* matIndex,
                                    Scalar dHat, Scalar Kappa, Scalar gassThreshold, int idx) {
    Scalar dis;
    __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
    
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);
    __MATHUTILS__::Vector6S PFPxT;
    __GIPCDERIVATIVE__::pFpx_pp2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dhat_sqrt, PFPxT);
    Scalar I5 = dis / dHat;
    Scalar fnn = dis_sqrt / dhat_sqrt;

    Scalar PbPg = (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
    Scalar flatten_pk1 = 2 * fnn * Kappa * PbPg;

    __MATHUTILS__::Vector6S gradient_vec = __MATHUTILS__::__s_vec6_multiply(PFPxT, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
    }

    Scalar lambda0 = -(4 * Kappa * dHat * dHat *
                       (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 +
                        I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
    if (dis < gassThreshold * dHat) {
        Scalar lambda1 = -(4 * Kappa * dHat * dHat *
                           (4 * gassThreshold + log(gassThreshold) -
                            3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) +
                            6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold +
                            gassThreshold * log(gassThreshold) * log(gassThreshold) -
                            7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) /
                         gassThreshold;
        lambda0 = lambda1;
    }

    Scalar H = lambda0;
    __MATHUTILS__::Matrix6x6S Hessian = __MATHUTILS__::__s_M6x6_Multiply(__MATHUTILS__::__v6_vec6_toMat6x6(PFPxT, PFPxT), H);

    int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 2, 1);
    H6x6[Hidx] = Hessian;
    D2Index[Hidx] = make_uint2(MMCVIDI.x, MMCVIDI.y);
}

__device__ void compute_barrier_Hess_PE(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI,
                                    Scalar3* _gradient, __MATHUTILS__::Matrix9x9S* H9x9, uint3* D3Index, int* matIndex,
                                    Scalar dHat, Scalar Kappa, Scalar gassThreshold, int idx) {
    Scalar dis;
    __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);
    __MATHUTILS__::Matrix9x4S PFPxT;
    __GIPCDERIVATIVE__::pFpx_pe2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dhat_sqrt, PFPxT);
    Scalar I5 = dis / dHat;
    __MATHUTILS__::Vector4S fnn;
    fnn.v[0] = fnn.v[1] = fnn.v[2] = 0;
    fnn.v[3] = 2 * dis_sqrt / dhat_sqrt;

    Scalar PbPg = (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
    __MATHUTILS__::Vector4S flatten_pk1 = __MATHUTILS__::__s_vec4_multiply(fnn, Kappa * PbPg);

    __MATHUTILS__::Vector9S gradient_vec = __MATHUTILS__::__M9x4_v4_multiply(PFPxT, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
    }

    Scalar lambda0 = -(4 * Kappa * dHat * dHat *
                       (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 +
                        I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
    if (dis < gassThreshold * dHat) {
        Scalar lambda1 = -(4 * Kappa * dHat * dHat *
                           (4 * gassThreshold + log(gassThreshold) -
                            3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) +
                            6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold +
                            gassThreshold * log(gassThreshold) * log(gassThreshold) -
                            7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) / gassThreshold;
        lambda0 = lambda1;
    }

    __MATHUTILS__::Vector4S q0;
    q0.v[0] = q0.v[1] = q0.v[2] = 0;
    q0.v[3] = 1;

    __MATHUTILS__::Matrix4x4S H;
    H = __MATHUTILS__::__s_Mat4x4_multiply(__MATHUTILS__::__v4_vec4_toMat4x4(q0, q0), lambda0);

    __MATHUTILS__::Matrix9x9S Hessian;
    __MATHUTILS__::__M9x4_S4x4_MT4x9_Multiply(PFPxT, H, Hessian);

    int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 3, 1);
    H9x9[Hidx] = Hessian;
    D3Index[Hidx] = make_uint3(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z);
}

__device__ void compute_barrier_Hess_PT(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI,
                                    Scalar3* _gradient, __MATHUTILS__::Matrix12x12S* H12x12, uint4* D4Index,
                                    int* matIndex, Scalar dHat, Scalar Kappa, Scalar gassThreshold, int idx) {
    Scalar dis;
    __MATHUTILS__::_d_PT(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);
    __MATHUTILS__::Matrix12x9S PFPxT;
    __GIPCDERIVATIVE__::pFpx_pt2(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPxT);
    Scalar I5 = dis / dHat;
    __MATHUTILS__::Vector9S tmp;
    tmp.v[0] = tmp.v[1] = tmp.v[2] = tmp.v[3] = tmp.v[4] = tmp.v[5] = tmp.v[6] = tmp.v[7] = 0;
    tmp.v[8] = 2 * dis_sqrt / dhat_sqrt;

    Scalar PbPg = (2 * dHat * dHat * log(I5) * (I5 - 1) * (I5 + I5 * log(I5) - 1)) / I5;
    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__s_vec9_multiply(tmp, Kappa * PbPg);

    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(PFPxT, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }


    Scalar lambda0 = -(4 * Kappa * dHat * dHat *
                       (4 * I5 + log(I5) - 3 * I5 * I5 * log(I5) * log(I5) + 6 * I5 * log(I5) - 2 * I5 * I5 +
                        I5 * log(I5) * log(I5) - 7 * I5 * I5 * log(I5) - 2)) / I5;
    if (dis < gassThreshold * dHat) {
        Scalar lambda1 = -(4 * Kappa * dHat * dHat *
                           (4 * gassThreshold + log(gassThreshold) -
                            3 * gassThreshold * gassThreshold * log(gassThreshold) * log(gassThreshold) +
                            6 * gassThreshold * log(gassThreshold) - 2 * gassThreshold * gassThreshold +
                            gassThreshold * log(gassThreshold) * log(gassThreshold) -
                            7 * gassThreshold * gassThreshold * log(gassThreshold) - 2)) / gassThreshold;
        lambda0 = lambda1;
    }

    __MATHUTILS__::Vector9S q0;
    q0.v[0] = q0.v[1] = q0.v[2] = q0.v[3] = q0.v[4] = q0.v[5] = q0.v[6] = q0.v[7] = 0;
    q0.v[8] = 1;

    __MATHUTILS__::Matrix9x9S H;
    H = __MATHUTILS__::__s_Mat9x9_multiply(__MATHUTILS__::__v9_vec9_toMat9x9(q0, q0), lambda0);

    __MATHUTILS__::Matrix12x12S Hessian;
    __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPxT, H, Hessian);

    int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 4, 1);
    H12x12[Hidx] = Hessian;
    D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
}

__device__ void compute_barrier_Hess_parallelEE(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI,
                                    Scalar3* _gradient, __MATHUTILS__::Matrix12x12S* H12x12, uint4* D4Index,
                                    int* matIndex, Scalar dHat, Scalar Kappa, int idx) {

    Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.y], _vertexes[MMCVIDI.x]);
    Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.z]);
    Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1)) /*/ __MATHUTILS__::__v3_norm(v0)*/;
    Scalar I1 = c * c;
    if (I1 == 0) return;
    Scalar dis;
    __MATHUTILS__::_d_EE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dis);
    Scalar I2 = dis / dHat;
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);

    __MATHUTILS__::Matrix3x3S F;
    __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis_sqrt / dhat_sqrt);
    Scalar3 n1 = make_Scalar3(0, 1, 0);
    Scalar3 n2 = make_Scalar3(0, 0, 1);

    Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.y],
                                               _rest_vertexes[MMCVIDI.z], _rest_vertexes[MMCVIDI.w]);

    __MATHUTILS__::Matrix3x3S g1, g2;

    __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g1);
    nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g2);

    __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
    __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

    __MATHUTILS__::Matrix12x9S PFPx;
    __GIPCDERIVATIVE__::pFpx_pee(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPx);

    Scalar p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                (I2 * (eps_x * eps_x));

    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__vec9_add(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                                                __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }

    Scalar lambda10 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
    Scalar lambda11 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar lambda12 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);

    __MATHUTILS__::Matrix3x3S Tx, Ty, Tz;
    __MATHUTILS__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __MATHUTILS__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
    __MATHUTILS__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

    __MATHUTILS__::Vector9S q11 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M3x3_M3x3_multiply(Tx, g1));
    __MATHUTILS__::__normalized_vec9_Scalar(q11);
    __MATHUTILS__::Vector9S q12 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M3x3_M3x3_multiply(Tz, g1));
    __MATHUTILS__::__normalized_vec9_Scalar(q12);

    __MATHUTILS__::Matrix9x9S projectedH;
    __MATHUTILS__::__init_Mat9x9(projectedH, 0);

    __MATHUTILS__::Matrix9x9S M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q11, q11);
    M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, lambda11);
    projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

    M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q12, q12);
    M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, lambda12);
    projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

    Scalar lambda20 = Kappa *
                      (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                       (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) - 2 * I2 * I2 +
                        I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) / (I2 * (eps_x * eps_x));

    Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                       (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                       (I2 * (eps_x * eps_x));

    Scalar eigenValues[2];
    int eigenNum = 0;
    Scalar2 eigenVecs[2];
    __MATHUTILS__::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

    for (int i = 0; i < eigenNum; i++) {
        if (eigenValues[i] > 0) {
            __MATHUTILS__::Matrix3x3S eigenMatrix;
            __MATHUTILS__::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0, 0, eigenVecs[i].y);
            __MATHUTILS__::Vector9S eigenMVec = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(eigenMatrix);
            M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
            M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, eigenValues[i]);
            projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);
        }
    }

    __MATHUTILS__::Matrix12x12S Hessian;
    __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
    int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 4, 1);
    H12x12[Hidx] = Hessian;
    D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
}

__device__ void compute_barrier_Hess_parallelPP(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI,
                                    Scalar3* _gradient, __MATHUTILS__::Matrix12x12S* H12x12, uint4* D4Index,
                                    int* matIndex, Scalar dHat, Scalar Kappa, int idx) {
    
    Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.x]);
    Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.y]);
    Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1)) /*/ __MATHUTILS__::__v3_norm(v0)*/;
    Scalar I1 = c * c;
    if (I1 == 0) return;
    Scalar dis;
    __MATHUTILS__::_d_PP(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], dis);
    Scalar I2 = dis / dHat;
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);

    __MATHUTILS__::Matrix3x3S F;
    __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis_sqrt / dhat_sqrt);
    Scalar3 n1 = make_Scalar3(0, 1, 0);
    Scalar3 n2 = make_Scalar3(0, 0, 1);

    Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.z],
                                               _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.w]);

    __MATHUTILS__::Matrix3x3S g1, g2;

    __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g1);
    nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g2);

    __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
    __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

    __MATHUTILS__::Matrix12x9S PFPx;
    __GIPCDERIVATIVE__::pFpx_ppp(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPx);

    Scalar p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                (I2 * (eps_x * eps_x));

    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__vec9_add(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                                                __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }

    Scalar lambda10 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
    Scalar lambda11 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar lambda12 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);

    __MATHUTILS__::Matrix3x3S Tx, Ty, Tz;
    __MATHUTILS__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __MATHUTILS__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
    __MATHUTILS__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

    __MATHUTILS__::Vector9S q11 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M3x3_M3x3_multiply(Tx, g1));
    __MATHUTILS__::__normalized_vec9_Scalar(q11);
    __MATHUTILS__::Vector9S q12 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M3x3_M3x3_multiply(Tz, g1));
    __MATHUTILS__::__normalized_vec9_Scalar(q12);

    __MATHUTILS__::Matrix9x9S projectedH;
    __MATHUTILS__::__init_Mat9x9(projectedH, 0);

    __MATHUTILS__::Matrix9x9S M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q11, q11);
    M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, lambda11);
    projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

    M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q12, q12);
    M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, lambda12);
    projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

    Scalar lambda20 = Kappa *
                      (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                       (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) - 2 * I2 * I2 +
                        I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) /
                      (I2 * (eps_x * eps_x));

    Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                       (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                       (I2 * (eps_x * eps_x));

    Scalar eigenValues[2];
    int eigenNum = 0;
    Scalar2 eigenVecs[2];
    __MATHUTILS__::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

    for (int i = 0; i < eigenNum; i++) {
        if (eigenValues[i] > 0) {
            __MATHUTILS__::Matrix3x3S eigenMatrix;
            __MATHUTILS__::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0, 0, eigenVecs[i].y);
            __MATHUTILS__::Vector9S eigenMVec = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(eigenMatrix);
            M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
            M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, eigenValues[i]);
            projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);
        }
    }

    __MATHUTILS__::Matrix12x12S Hessian;
    __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
    int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 4, 1);
    H12x12[Hidx] = Hessian;
    D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
}

__device__ void compute_barrier_Hess_parallelPE(const Scalar3* _vertexes, const Scalar3* _rest_vertexes, int4 MMCVIDI,
                                    Scalar3* _gradient, __MATHUTILS__::Matrix12x12S* H12x12, uint4* D4Index,
                                    int* matIndex, Scalar dHat, Scalar Kappa, int idx) {

    Scalar3 v0 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.w], _vertexes[MMCVIDI.x]);
    Scalar3 v1 = __MATHUTILS__::__vec3_minus(_vertexes[MMCVIDI.z], _vertexes[MMCVIDI.y]);
    Scalar c = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_cross(v0, v1)) /*/ __MATHUTILS__::__v3_norm(v0)*/;
    Scalar I1 = c * c;
    if (I1 == 0) return;
    Scalar dis;
    __MATHUTILS__::_d_PE(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], dis);
    Scalar I2 = dis / dHat;
    Scalar dis_sqrt = sqrt(dis);
    Scalar dhat_sqrt = sqrt(dHat);

    __MATHUTILS__::Matrix3x3S F;
    __MATHUTILS__::__set_Mat_val(F, 1, 0, 0, 0, c, 0, 0, 0, dis_sqrt / dhat_sqrt);
    Scalar3 n1 = make_Scalar3(0, 1, 0);
    Scalar3 n2 = make_Scalar3(0, 0, 1);

    Scalar eps_x = __MATHUTILS__::_compute_epx(_rest_vertexes[MMCVIDI.x], _rest_vertexes[MMCVIDI.w],
                                               _rest_vertexes[MMCVIDI.y], _rest_vertexes[MMCVIDI.z]);

    __MATHUTILS__::Matrix3x3S g1, g2;

    __MATHUTILS__::Matrix3x3S nn = __MATHUTILS__::__v_vec_toMat(n1, n1);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g1);
    nn = __MATHUTILS__::__v_vec_toMat(n2, n2);
    __MATHUTILS__::__M3x3_M3x3_multiply(F, nn, g2);

    __MATHUTILS__::Vector9S flatten_g1 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g1);
    __MATHUTILS__::Vector9S flatten_g2 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(g2);

    __MATHUTILS__::Matrix12x9S PFPx;
    __GIPCDERIVATIVE__::pFpx_ppe(_vertexes[MMCVIDI.x], _vertexes[MMCVIDI.y], _vertexes[MMCVIDI.z], _vertexes[MMCVIDI.w], dhat_sqrt, PFPx);

    Scalar p1 = -Kappa * 2 * (2 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar p2 = -Kappa * 2 * (2 * I1 * dHat * dHat * log(I2) * (I1 - 2 * eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                (I2 * (eps_x * eps_x));

    __MATHUTILS__::Vector9S flatten_pk1 = __MATHUTILS__::__vec9_add(__MATHUTILS__::__s_vec9_multiply(flatten_g1, p1),
                                                                __MATHUTILS__::__s_vec9_multiply(flatten_g2, p2));
    __MATHUTILS__::Vector12S gradient_vec = __MATHUTILS__::__M12x9_v9_multiply(PFPx, flatten_pk1);

    {
        atomicAdd(&(_gradient[MMCVIDI.x].x), gradient_vec.v[0]);
        atomicAdd(&(_gradient[MMCVIDI.x].y), gradient_vec.v[1]);
        atomicAdd(&(_gradient[MMCVIDI.x].z), gradient_vec.v[2]);
        atomicAdd(&(_gradient[MMCVIDI.y].x), gradient_vec.v[3]);
        atomicAdd(&(_gradient[MMCVIDI.y].y), gradient_vec.v[4]);
        atomicAdd(&(_gradient[MMCVIDI.y].z), gradient_vec.v[5]);
        atomicAdd(&(_gradient[MMCVIDI.z].x), gradient_vec.v[6]);
        atomicAdd(&(_gradient[MMCVIDI.z].y), gradient_vec.v[7]);
        atomicAdd(&(_gradient[MMCVIDI.z].z), gradient_vec.v[8]);
        atomicAdd(&(_gradient[MMCVIDI.w].x), gradient_vec.v[9]);
        atomicAdd(&(_gradient[MMCVIDI.w].y), gradient_vec.v[10]);
        atomicAdd(&(_gradient[MMCVIDI.w].z), gradient_vec.v[11]);
    }

    Scalar lambda10 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I2 - 1) * (I2 - 1) * (3 * I1 - eps_x)) / (eps_x * eps_x);
    Scalar lambda11 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);
    Scalar lambda12 = -Kappa * (4 * dHat * dHat * log(I2) * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 - 1)) / (eps_x * eps_x);

    __MATHUTILS__::Matrix3x3S Tx, Ty, Tz;
    __MATHUTILS__::__set_Mat_val(Tx, 0, 0, 0, 0, 0, 1, 0, -1, 0);
    __MATHUTILS__::__set_Mat_val(Ty, 0, 0, -1, 0, 0, 0, 1, 0, 0);
    __MATHUTILS__::__set_Mat_val(Tz, 0, 1, 0, -1, 0, 0, 0, 0, 0);

    __MATHUTILS__::Vector9S q11 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M3x3_M3x3_multiply(Tx, g1));
    __MATHUTILS__::__normalized_vec9_Scalar(q11);
    __MATHUTILS__::Vector9S q12 = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(__MATHUTILS__::__M3x3_M3x3_multiply(Tz, g1));
    __MATHUTILS__::__normalized_vec9_Scalar(q12);

    __MATHUTILS__::Matrix9x9S projectedH;
    __MATHUTILS__::__init_Mat9x9(projectedH, 0);

    __MATHUTILS__::Matrix9x9S M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q11, q11);
    M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, lambda11);
    projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

    M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(q12, q12);
    M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, lambda12);
    projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);

    Scalar lambda20 = Kappa *
                      (4 * I1 * dHat * dHat * (I1 - 2 * eps_x) *
                       (4 * I2 + log(I2) - 3 * I2 * I2 * log(I2) * log(I2) + 6 * I2 * log(I2) - 2 * I2 * I2 +
                        I2 * log(I2) * log(I2) - 7 * I2 * I2 * log(I2) - 2)) /
                      (I2 * (eps_x * eps_x));

    Scalar lambdag1g = -Kappa * 4 * c * F.m[2][2] *
                       (4 * dHat * dHat * log(I2) * (I1 - eps_x) * (I2 - 1) * (I2 + I2 * log(I2) - 1)) /
                       (I2 * (eps_x * eps_x));

    Scalar eigenValues[2];
    int eigenNum = 0;
    Scalar2 eigenVecs[2];
    __MATHUTILS__::__makePD2x2(lambda10, lambdag1g, lambdag1g, lambda20, eigenValues, eigenNum, eigenVecs);

    for (int i = 0; i < eigenNum; i++) {
        if (eigenValues[i] > 0) {
            __MATHUTILS__::Matrix3x3S eigenMatrix;
            __MATHUTILS__::__set_Mat_val(eigenMatrix, 0, 0, 0, 0, eigenVecs[i].x, 0, 0, 0, eigenVecs[i].y);
            __MATHUTILS__::Vector9S eigenMVec = __MATHUTILS__::__Mat3x3_to_vec9_Scalar(eigenMatrix);
            M9_temp = __MATHUTILS__::__v9_vec9_toMat9x9(eigenMVec, eigenMVec);
            M9_temp = __MATHUTILS__::__s_Mat9x9_multiply(M9_temp, eigenValues[i]);
            projectedH = __MATHUTILS__::__Mat9x9_add(projectedH, M9_temp);
        }
    }

    __MATHUTILS__::Matrix12x12S Hessian;
    __MATHUTILS__::__M12x9_S9x9_MT9x12_Multiply(PFPx, projectedH, Hessian);
    int Hidx = matIndex[idx];  // int Hidx = atomicAdd(_cpNum + 4, 1);
    H12x12[Hidx] = Hessian;
    D4Index[Hidx] = make_uint4(MMCVIDI.x, MMCVIDI.y, MMCVIDI.z, MMCVIDI.w);
}

__global__ void _calBarrierGradientAndHessian(const Scalar3* _vertexes, const Scalar3* _rest_vertexes,
                                              const int4* _collisionPair, Scalar3* _gradient,
                                              __MATHUTILS__::Matrix12x12S* H12x12, __MATHUTILS__::Matrix9x9S* H9x9,
                                              __MATHUTILS__::Matrix6x6S* H6x6, uint4* D4Index, uint3* D3Index,
                                              uint2* D2Index, uint32_t* _cpNum, int* matIndex, Scalar dHat, Scalar dThreshold,
                                              Scalar Kappa, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _collisionPair[idx];
    Scalar dHat_sqrt = sqrt(dHat);
    Scalar gassThreshold = dThreshold;

    if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        compute_barrier_Hess_EE(_vertexes, _rest_vertexes, MMCVIDI, _gradient, H12x12, D4Index, matIndex, dHat, Kappa, gassThreshold, idx);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z == -1 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        compute_barrier_Hess_PP(_vertexes, _rest_vertexes, MMCVIDI, _gradient, H6x6, D2Index, matIndex, dHat, Kappa, gassThreshold, idx);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w == -1) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        compute_barrier_Hess_PE(_vertexes, _rest_vertexes, MMCVIDI, _gradient, H9x9, D3Index, matIndex, dHat, Kappa, gassThreshold, idx);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w >= 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        compute_barrier_Hess_PT(_vertexes, _rest_vertexes, MMCVIDI, _gradient, H12x12, D4Index, matIndex, dHat, Kappa, gassThreshold, idx);
    }

    else if (MMCVIDI.x >= 0 && MMCVIDI.y >= 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.w = -MMCVIDI.w - 1;
        compute_barrier_Hess_parallelEE(_vertexes, _rest_vertexes, MMCVIDI, _gradient, H12x12, D4Index, matIndex, dHat, Kappa, idx);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z < 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.z = -MMCVIDI.z - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        compute_barrier_Hess_parallelPP(_vertexes, _rest_vertexes, MMCVIDI, _gradient, H12x12, D4Index, matIndex, dHat, Kappa, idx);
    }

    else if (MMCVIDI.x < 0 && MMCVIDI.y < 0 && MMCVIDI.z >= 0 && MMCVIDI.w < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        MMCVIDI.y = -MMCVIDI.y - 1;
        MMCVIDI.w = -MMCVIDI.w - 1;
        compute_barrier_Hess_parallelPE(_vertexes, _rest_vertexes, MMCVIDI, _gradient, H12x12, D4Index, matIndex, dHat, Kappa, idx);
    }

    else {
        printf("################################ Error: invalid collision conditions");
        // exit(EXIT_FAILURE);;
    }

}

__global__ void _checkSelfCloseVal(const Scalar3* _vertexes, int* _isChange, int4* _close_collisionPair,
                                   Scalar* _close_collisionVal, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _close_collisionPair[idx];
    Scalar dist2 = __calBarrierSelfConsDis(_vertexes, MMCVIDI);
    if (dist2 < _close_collisionVal[idx]) {
        *_isChange = 1;
    }
}


__global__ void _calKineticGradient(Scalar3* vertexes, Scalar3* xTilta, Scalar3* gradient, Scalar* masses,
                                    int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;  // numbers of vertices
    Scalar3 deltaX = __MATHUTILS__::__vec3_minus(vertexes[idx], xTilta[idx]);
    gradient[idx] = make_Scalar3(deltaX.x * masses[idx], deltaX.y * masses[idx], deltaX.z * masses[idx]);
    // printf("%f  %f  %f\n", gradient[idx].x, gradient[idx].y, gradient[idx].z);
}

__global__ void _reduct_min_groundTimeStep_to_Scalar(const Scalar3* vertexes, const uint32_t* surfVertIds,
                                                     const Scalar* g_offset, const Scalar3* g_normal,
                                                     const Scalar3* moveDir, Scalar* minStepSizes, Scalar slackness,
                                                     int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    int svI = surfVertIds[idx];
    Scalar temp = 1.0;
    Scalar3 normal = *g_normal;
    // vertex full moving dis projection in a substep, >0 means moving towrads ground
    // coef是一个substep沿着dx前进的投影距离，dist是顶点到ground的距离，如果一个substep中运动coef>dist, temp就会>1代表会产生穿透,
    // 那么alpha=1/temp<1就会启动filter line search，取小于1的alpha来保证一个substep内的运动不会穿透
    Scalar coef = __MATHUTILS__::__vec3_dot(normal, moveDir[svI]);
    if (coef > 0.0) {
        Scalar dist = __MATHUTILS__::__vec3_dot(normal, vertexes[svI]) - *g_offset;  // vertex到ground距离
        temp = coef / (dist * slackness);
    }

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
        minStepSizes[blockIdx.x] = temp;
    }
}


__global__ void _reduct_min_selfTimeStep_to_Scalar(const Scalar3* vertexes, const int4* _ccd_collisionPairs,
                                                   const Scalar3* moveDir, Scalar* minStepSizes, Scalar slackness,
                                                   int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;
    
    Scalar temp = 1.0;
    Scalar CCDDistRatio = 1.0 - slackness;
    int4 MMCVIDI = _ccd_collisionPairs[idx];
    // 计算碰撞的toi,如果temptoc<1,代表一个substep内会产生碰撞，temp就会大于1,启动filter line search,取小于1的alpha来保证一个substep内的运动不会穿透
    if (MMCVIDI.x < 0) {
        MMCVIDI.x = -MMCVIDI.x - 1;
        Scalar temp_toc = __ACCD__::point_triangle_ccd(vertexes[MMCVIDI.x], vertexes[MMCVIDI.y], vertexes[MMCVIDI.z],
                                         vertexes[MMCVIDI.w], __MATHUTILS__::__s_vec3_multiply(moveDir[MMCVIDI.x], -1),
                                         __MATHUTILS__::__s_vec3_multiply(moveDir[MMCVIDI.y], -1),
                                         __MATHUTILS__::__s_vec3_multiply(moveDir[MMCVIDI.z], -1),
                                         __MATHUTILS__::__s_vec3_multiply(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0);
        temp = 1.0 / temp_toc;
    } else {
        Scalar temp_toc = __ACCD__::edge_edge_ccd(vertexes[MMCVIDI.x], vertexes[MMCVIDI.y], vertexes[MMCVIDI.z],
                                          vertexes[MMCVIDI.w], __MATHUTILS__::__s_vec3_multiply(moveDir[MMCVIDI.x], -1),
                                          __MATHUTILS__::__s_vec3_multiply(moveDir[MMCVIDI.y], -1),
                                          __MATHUTILS__::__s_vec3_multiply(moveDir[MMCVIDI.z], -1),
                                          __MATHUTILS__::__s_vec3_multiply(moveDir[MMCVIDI.w], -1), CCDDistRatio, 0);
        temp = 1.0 / temp_toc;
    }


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
        minStepSizes[blockIdx.x] = temp;
    }

}

__global__ void _reduct_max_cfl_to_Scalar(const Scalar3* moveDir, Scalar* max_Scalar_val, uint32_t* mSVI, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar temp = __MATHUTILS__::__vec3_norm(moveDir[mSVI[idx]]);

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
        max_Scalar_val[blockIdx.x] = temp;
    }
}

__global__ void _reduct_Scalar3Sqr_to_Scalar(const Scalar3* A, Scalar* squeue, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar temp = __MATHUTILS__::__vec3_squaredNorm(A[idx]);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);

}

__global__ void _reduct_Scalar3Dot_to_Scalar(const Scalar3* A, const Scalar3* B, Scalar* squeue, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar temp = __MATHUTILS__::__vec3_dot(A[idx], B[idx]);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);

}

__global__ void _getBarrierEnergy_Reduction_3D(Scalar* squeue, const Scalar3* vertexes, const Scalar3* rest_vertexes,
                                               int4* _collisionPair, Scalar _Kappa, Scalar _dHat, int cpNum) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    int numbers = cpNum;
    if (idx >= numbers) return;

    Scalar temp = __calBarrierEnergy(vertexes, rest_vertexes, _collisionPair[idx], _Kappa, _dHat);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);

}

__global__ void _computeGroundEnergy_Reduction(Scalar* squeue, const Scalar3* vertexes, const Scalar* g_offset,
                                               const Scalar3* g_normal, const uint32_t* _environment_collisionPair,
                                               Scalar dHat, Scalar Kappa, int numbers) {
    int idof = blockIdx.x * blockDim.x;
    int idx = threadIdx.x + idof;
    if (idx >= numbers) return;

    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__vec3_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;
    Scalar temp = -(dist2 - dHat) * (dist2 - dHat) * log(dist2 / dHat);

    __MATHUTILS__::__perform_reduct_add_Scalar(squeue, temp, numbers);

}


Scalar self_largestFeasibleStepSize(std::unique_ptr<GeometryManager>& instance, Scalar slackness, Scalar* mqueue,
                                    int numbers) {

    if (numbers < 1) return 1;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    _reduct_min_selfTimeStep_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(
        instance->getCudaSurfVertPos(), instance->getCudaCCDCollisionPairs(), instance->getCudaMoveDir(), mqueue, slackness,
        numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        //_reduct_max_box << <blockNum, threadNum, sharedMsize >> > (_tempLeafBox, numbers);
        __MATHUTILS__::_reduct_max_Scalar<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    // cudaMemcpy(_leafBoxes, _tempLeafBox, sizeof(AABB), cudaMemcpyDeviceToDevice);
    Scalar minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    // printf("                 full ccd time step:  %f\n", 1.0 / minValue);
    // CUDAFreeSafe(_minSteps);
    return 1.0 / minValue;
}

Scalar cfl_largestSpeed(std::unique_ptr<GeometryManager>& instance, Scalar* mqueue) {
    int numbers = instance->getHostNumSurfVerts();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    _reduct_max_cfl_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(instance->getCudaMoveDir(), mqueue,
                                                                    instance->getCudaSurfVertIds(), numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::_reduct_max_Scalar<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    return minValue;
}

Scalar reduction2Kappa(int type, const Scalar3* A, const Scalar3* B, Scalar* _queue, int vertexNum) {
    int numbers = vertexNum;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    if (type == 0) {
        _reduct_Scalar3Dot_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(A, B, _queue, numbers);
    } else if (type == 1) {
        _reduct_Scalar3Sqr_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(A, _queue, numbers);
    }

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::__reduct_add_Scalar<<<blockNum, threadNum, sharedMsize>>>(_queue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar dotValue;
    cudaMemcpy(&dotValue, _queue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    return dotValue;
}

Scalar ground_largestFeasibleStepSize(std::unique_ptr<GeometryManager>& instance, Scalar slackness, Scalar* mqueue) {
    int numbers = instance->getHostNumSurfVerts();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    unsigned int sharedMsize = sizeof(Scalar) * (threadNum >> 5);

    _reduct_min_groundTimeStep_to_Scalar<<<blockNum, threadNum, sharedMsize>>>(
        instance->getCudaSurfVertPos(), instance->getCudaSurfVertIds(), instance->getCudaGroundOffset(),
        instance->getCudaGroundNormal(), instance->getCudaMoveDir(), mqueue, slackness, numbers);

    numbers = blockNum;
    blockNum = (numbers + threadNum - 1) / threadNum;

    while (numbers > 1) {
        __MATHUTILS__::_reduct_max_Scalar<<<blockNum, threadNum, sharedMsize>>>(mqueue, numbers);
        numbers = blockNum;
        blockNum = (numbers + threadNum - 1) / threadNum;
    }
    Scalar minValue;
    cudaMemcpy(&minValue, mqueue, sizeof(Scalar), cudaMemcpyDeviceToHost);
    return 1.0 / minValue;
}


void calBarrierGradientAndHessian(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr, Scalar3* _gradient,
                                  Scalar mKappa) {
    int numbers = instance->getHostCpNum(0);
    if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradientAndHessian<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaRestVertPos(), instance->getCudaCollisionPairs(), _gradient,
        BH_ptr->cudaH12x12, BH_ptr->cudaH9x9, BH_ptr->cudaH6x6, BH_ptr->cudaD4Index, BH_ptr->cudaD3Index, BH_ptr->cudaD2Index,
        instance->getCudaCPNum(), instance->getCudaMatIndex(), instance->getHostDHat(), instance->getHostRelativeDHatThres(), mKappa, numbers);
}


void calBarrierGradient(std::unique_ptr<GeometryManager>& instance, Scalar3* _gradient, Scalar mKappa) {
    int numbers = instance->getHostCpNum(0);
    if (numbers < 1) return;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;

    _calBarrierGradient<<<blockNum, threadNum>>>(instance->getCudaSurfVertPos(), instance->getCudaRestVertPos(),
                                                 instance->getCudaCollisionPairs(), _gradient, instance->getHostDHat(),
                                                 mKappa, numbers);
}

void compute_H_b(Scalar d, Scalar dHat, Scalar& H) {
    Scalar t = d - dHat;
    H = (std::log(d / dHat) * -2.0 - t * 4.0 / d) + 1.0 / (d * d) * (t * t);
}

void suggestKappa(std::unique_ptr<GeometryManager>& instance, Scalar& kappa) {
    Scalar H_b;
    compute_H_b(1.0e-16 * instance->getHostBboxDiagSize2(), instance->getHostDHat(), H_b);
    if (instance->getHostMeanMass() == 0.0) {
        kappa = instance->getHostMinKappaCoef() / (4.0e-16 * instance->getHostBboxDiagSize2() * H_b);
    } else {
        kappa = instance->getHostMinKappaCoef() * instance->getHostMeanMass() /
                (4.0e-16 * instance->getHostBboxDiagSize2() * H_b);
    }
    //    printf("bboxDiagSize2: %f\n", bboxDiagSize2);
    //    printf("H_b: %f\n", H_b);
    //    printf("sug Kappa: %f\n", kappa);
}

void upperBoundKappa(std::unique_ptr<GeometryManager>& instance, Scalar& kappa) {
    Scalar H_b;
    compute_H_b(1.0e-16 * instance->getHostBboxDiagSize2(), instance->getHostDHat(), H_b);
    Scalar kappaMax = 100 * instance->getHostMinKappaCoef() * instance->getHostMeanMass() /
                      (4.0e-16 * instance->getHostBboxDiagSize2() * H_b);
    // printf("max Kappa: %f\n", kappaMax);
    if (instance->getHostMeanMass() == 0.0) {
        kappaMax = 100 * instance->getHostMinKappaCoef() / (4.0e-16 * instance->getHostBboxDiagSize2() * H_b);
    }

    if (kappa > kappaMax) {
        kappa = kappaMax;
    }
}

void calKineticGradient(Scalar3* _vertexes, Scalar3* _xTilta, Scalar3* _gradient, Scalar* _masses, int numbers) {
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calKineticGradient<<<blockNum, threadNum>>>(_vertexes, _xTilta, _gradient, _masses, numbers);
}

void initKappa(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<BlockHessian>& BH_ptr, std::unique_ptr<PCGSolver>& PCG_ptr) {
    if (instance->getHostCpNum(0) > 0) {
        Scalar3* _GE = instance->getCudaFb();
        Scalar3* _gc = instance->getCudaTempScalar3Mem();
        CUDA_SAFE_CALL(cudaMemset(_gc, 0, instance->getHostNumVertices() * sizeof(Scalar3)));
        CUDA_SAFE_CALL(cudaMemset(_GE, 0, instance->getHostNumVertices() * sizeof(Scalar3)));

        calKineticGradient(instance->getCudaSurfVertPos(), instance->getCudaXTilta(), _GE, instance->getCudaVertMass(),
                           instance->getHostNumVertices());

        __FEMENERGY__::calculate_triangle_cons_gradient(
            instance->getCudaTriDmInverses(), instance->getCudaSurfVertPos(), instance->getCudaTriElement(),
            instance->getCudaTriArea(), _GE, instance->getHostNumTriElements(), instance->getHostStretchStiff(),
            instance->getHostShearStiff(), instance->getHostIPCDt());

        __FEMENERGY__::computeBoundConstraintGradient(instance, _GE);

        __FEMENERGY__::computeSoftConstraintGradient(instance, _GE);

        __FEMENERGY__::computeStitchConstraintGradient(instance, _GE);

        __FEMENERGY__::computeGroundGradient(instance, BH_ptr, _gc, 1);

        calBarrierGradient(instance, _gc, 1);
        Scalar gsum = reduction2Kappa(0, _gc, _GE, PCG_ptr->cudaPCGSqueue, instance->getHostNumVertices());
        Scalar gsnorm = reduction2Kappa(1, _gc, _GE, PCG_ptr->cudaPCGSqueue, instance->getHostNumVertices());

        Scalar minKappa = -gsum / gsnorm;
        if (minKappa > 0.0) {
            instance->getHostKappa() = minKappa;
        }
        suggestKappa(instance, minKappa);
        if (instance->getHostKappa() < minKappa) {
            instance->getHostKappa() = minKappa;
        }
        upperBoundKappa(instance, instance->getHostKappa());
    }
}


__global__ void _checkGroundIntersection(const Scalar3* vertexes, const Scalar* g_offset, const Scalar3* g_normal,
                                         const uint32_t* _environment_collisionPair, int* _isIntersect, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__vec3_dot(normal, vertexes[gidx]) - *g_offset;
    // printf("%f  %f\n", *g_offset, dist);
    if (dist < 0) *_isIntersect = -1;
}


bool checkGroundIntersection(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostGpNum();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //

    int* _isIntersect;
    CUDAMallocSafe(_isIntersect, 1);
    CUDA_SAFE_CALL(cudaMemset(_isIntersect, 0, sizeof(int)));
    _checkGroundIntersection<<<blockNum, threadNum>>>(instance->getCudaSurfVertPos(), instance->getCudaGroundOffset(),
                                                      instance->getCudaGroundNormal(),
                                                      instance->getCudaEnvCollisionPairs(), _isIntersect, numbers);

    int h_isITST;
    cudaMemcpy(&h_isITST, _isIntersect, sizeof(int), cudaMemcpyDeviceToHost);
    CUDAFreeSafe(_isIntersect);
    if (h_isITST < 0) {
        return true;
    }
    return false;
}

bool isIntersected(std::unique_ptr<GeometryManager>& instance, std::unique_ptr<LBVHCollisionDetector>& LBVH_CD_ptr) {
    if (checkGroundIntersection(instance)) {
        return true;
    }

    if (LBVH_CD_ptr->lbvh_ef.checkCollisionDetectTriEdge(instance->getHostDHat())) {
        return true;
    }

    return false;
}












__global__ void _computeGroundCloseVal(const Scalar3* vertexes, const Scalar* g_offset, const Scalar3* g_normal,
                                       const uint32_t* _environment_collisionPair, Scalar dTol,
                                       uint32_t* _closeConstraintID, Scalar* _closeConstraintVal,
                                       uint32_t* _close_gpNum, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _environment_collisionPair[idx];
    Scalar dist = __MATHUTILS__::__vec3_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;
    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_gpNum, 1);
        _closeConstraintID[tidx] = gidx;
        _closeConstraintVal[tidx] = dist2;
    }
}

__global__ void _checkGroundCloseVal(const Scalar3* vertexes, const Scalar* g_offset, const Scalar3* g_normal,
                                     int* _isChange, uint32_t* _closeConstraintID, Scalar* _closeConstraintVal,
                                     int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    Scalar3 normal = *g_normal;
    int gidx = _closeConstraintID[idx];
    Scalar dist = __MATHUTILS__::__vec3_dot(normal, vertexes[gidx]) - *g_offset;
    Scalar dist2 = dist * dist;

    if (dist2 < _closeConstraintVal[gidx]) {
        *_isChange = 1;
    }
}


__global__ void _computeSelfCloseVal(const Scalar3* _vertexes, const int4* _collisionPair, int4* _close_collisionPair,
                                 Scalar* _close_collisionVal, uint32_t* _close_cpNum, Scalar dTol, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    int4 MMCVIDI = _collisionPair[idx];
    Scalar dist2 = __calBarrierSelfConsDis(_vertexes, MMCVIDI);
    if (dist2 < dTol) {
        int tidx = atomicAdd(_close_cpNum, 1);
        _close_collisionPair[tidx] = MMCVIDI;
        _close_collisionVal[tidx] = dist2;
    }
}


void computeCloseGroundVal(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostGpNum();
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    _computeGroundCloseVal<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaGroundOffset(), instance->getCudaGroundNormal(),
        instance->getCudaEnvCollisionPairs(), instance->getHostDTol(), instance->getCudaCloseConstraintID(),
        instance->getCudaCloseConstraintVal(), instance->getCudaCloseGPNum(), numbers);
}

bool checkCloseGroundVal(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostCloseGpNum();
    if (numbers < 1) return false;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    int* _isChange;
    CUDAMallocSafe(_isChange, 1);
    _checkGroundCloseVal<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaGroundOffset(), instance->getCudaGroundNormal(), _isChange,
        instance->getCudaCloseConstraintID(), instance->getCudaCloseConstraintVal(), numbers);
    int isChange = 0;
    CUDAMemcpyDToHSafe(isChange, _isChange);
    CUDAFreeSafe(_isChange);

    return (isChange == 1);
}


void computeSelfCloseVal(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostCpNum(0);
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _computeSelfCloseVal<<<blockNum, threadNum>>>(
        instance->getCudaSurfVertPos(), instance->getCudaCollisionPairs(), instance->getCudaCloseMConstraintID(),
        instance->getCudaCloseMConstraintVal(), instance->getCudaCloseCPNum(), instance->getHostDTol(), numbers);
}

bool checkSelfCloseVal(std::unique_ptr<GeometryManager>& instance) {
    int numbers = instance->getHostCloseCpNum();
    if (numbers < 1) return false;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;  //
    int* _isChange;
    CUDAMallocSafe(_isChange, 1);
    CUDA_SAFE_CALL(cudaMemset(_isChange, 0, sizeof(int)));
    _checkSelfCloseVal<<<blockNum, threadNum>>>(instance->getCudaSurfVertPos(), _isChange,
                                                instance->getCudaCloseMConstraintID(),
                                                instance->getCudaCloseMConstraintVal(), numbers);
    int isChange;
    CUDA_SAFE_CALL(cudaMemcpy(&isChange, _isChange, sizeof(int), cudaMemcpyDeviceToHost));
    CUDAFreeSafe(_isChange);

    return (isChange == 1);
}




};  // namespace __GPUIPC__
