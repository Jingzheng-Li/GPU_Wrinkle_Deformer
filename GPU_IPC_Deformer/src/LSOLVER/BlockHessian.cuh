
#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"


class BlockHessian {

public:

    BlockHessian() : cudaD4Index(nullptr), cudaD3Index(nullptr), cudaD2Index(nullptr), cudaD1Index(nullptr),
                    cudaH12x12(nullptr), cudaH9x9(nullptr), cudaH6x6(nullptr), cudaH3x3(nullptr) {}

    ~BlockHessian() {
		std::cout << "deconstruct BlockHessian" << std::endl;
        CUDA_FREE_BLOCKHESSIAN();
    }

	void CUDA_MALLOC_BLOCKHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& edge_number, const int& triangle_num, const int& tri_Edge_number);
	
	void CUDA_FREE_BLOCKHESSIAN();

public:
	void updateBHDNum(
		const int& tri_Num,
		const int& tri_edge_number,
		const int& tet_number,
		const int& stitch_number,
		const uint32_t& cpNum2,
		const uint32_t& cpNum3,
		const uint32_t& cpNum4,
		const uint32_t& last_cpNum2,
		const uint32_t& last_cpNum3,
		const uint32_t& last_cpNum4
	);


public:
	uint4* cudaD4Index; // 4 vert index
	uint3* cudaD3Index; // 3 vert index
	uint2* cudaD2Index; // 2 vert index
	uint32_t* cudaD1Index; // 1 vert index
	__MATHUTILS__::Matrix12x12S* cudaH12x12; // 12x12 block hessian
	__MATHUTILS__::Matrix9x9S* cudaH9x9; // 9x9 block hessian
	__MATHUTILS__::Matrix6x6S* cudaH6x6; // 6x6 block hessian
	__MATHUTILS__::Matrix3x3S* cudaH3x3; // 3x3 block hessian

	uint32_t hostBHDNum[4]; // 0: 3x3 hessian, 1: 6x6 hessian, 2: 3x3 hessian, 3: 9x9 hessian

};


