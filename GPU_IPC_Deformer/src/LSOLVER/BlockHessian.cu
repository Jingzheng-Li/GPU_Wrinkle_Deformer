
#include "BlockHessian.cuh"




void BlockHessian::updateBHDNum(
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
) {
    hostBHDNum[1] = cpNum2 + stitch_number; // H6x6
    hostBHDNum[2] = cpNum3 + tri_Num; // H9x9
    hostBHDNum[3]= cpNum4 + tet_number + tri_edge_number; // H12x12

#ifdef USE_GIPCFRICTION
    hostBHDNum[1] += last_cpNum2;
    hostBHDNum[2] += last_cpNum3;
    hostBHDNum[3] += last_cpNum4;
#endif
}



void BlockHessian::CUDA_MALLOC_BLOCKHESSIAN(const int& tet_number, const int& surfvert_number, const int& surface_number, const int& surfEdge_number, const int& triangle_num, const int& tri_Edge_number) {

    CUDAMallocSafe(cudaH12x12, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number));
    CUDAMallocSafe(cudaH9x9, (2 * (surfvert_number + surfEdge_number) + triangle_num));
    CUDAMallocSafe(cudaH6x6, 2 * (surfvert_number + surfEdge_number));
    CUDAMallocSafe(cudaH3x3, 2 * surfvert_number);

    CUDAMallocSafe(cudaD4Index, (2 * (surfvert_number + surfEdge_number) + tet_number + tri_Edge_number));
    CUDAMallocSafe(cudaD3Index, (2 * (surfEdge_number + surfvert_number)+ triangle_num));
    CUDAMallocSafe(cudaD2Index, 2 * (surfvert_number + surfEdge_number));
    CUDAMallocSafe(cudaD1Index, 2 * surfvert_number);

}

void BlockHessian::CUDA_FREE_BLOCKHESSIAN() {
    CUDAFreeSafe(cudaH12x12);
    CUDAFreeSafe(cudaH9x9);
    CUDAFreeSafe(cudaH6x6);
    CUDAFreeSafe(cudaH3x3);

    CUDAFreeSafe(cudaD4Index);
    CUDAFreeSafe(cudaD3Index);
    CUDAFreeSafe(cudaD2Index);
    CUDAFreeSafe(cudaD1Index);
}


