
#include "SortMesh.cuh"

#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/device_ptr.h>

namespace __SORTMESH__ {


__device__ __host__ 
inline std::uint32_t expand_bits(std::uint32_t v) noexcept {
	v = (v * 0x00010001u) & 0xFF0000FFu;
	v = (v * 0x00000101u) & 0x0F00F00Fu;
	v = (v * 0x00000011u) & 0xC30C30C3u;
	v = (v * 0x00000005u) & 0x49249249u;
	return v;
}


__device__ __host__
inline uint32_t hash_code(int type, Scalar x, Scalar y, Scalar z, Scalar resolution = 1024) noexcept
{
	x = __MATHUTILS__::__m_min(__MATHUTILS__::__m_max(x * resolution, 0.0), resolution - 1.0);
	y = __MATHUTILS__::__m_min(__MATHUTILS__::__m_max(y * resolution, 0.0), resolution - 1.0);
	z = __MATHUTILS__::__m_min(__MATHUTILS__::__m_max(z * resolution, 0.0), resolution - 1.0);

	if (type == -1) {
		const uint32_t xx = expand_bits(static_cast<uint32_t>(x));
		const uint32_t yy = expand_bits(static_cast<uint32_t>(y));
		const uint32_t zz = expand_bits(static_cast<uint32_t>(z));
		std::uint32_t mchash = ((xx << 2) + (yy << 1) + zz);

		return mchash;
	}
	else if (type == 0) {
		return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(y)) * 1024) + static_cast<uint32_t>(x);
	}
	else if (type == 1) {
		return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(z)) * 1024) + static_cast<uint32_t>(x);
	}
	else if (type == 2) {
		return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(z)) * 1024) + static_cast<uint32_t>(y);
	}
	else if (type == 3) {
		return (((static_cast<uint32_t>(z) * 1024) + static_cast<uint32_t>(x)) * 1024) + static_cast<uint32_t>(y);
	}
	else if (type == 4) {
		return (((static_cast<uint32_t>(y) * 1024) + static_cast<uint32_t>(x)) * 1024) + static_cast<uint32_t>(z);
	}
	else {
		return (((static_cast<uint32_t>(x) * 1024) + static_cast<uint32_t>(y)) * 1024) + static_cast<uint32_t>(z);
	}
	//std::uint32_t mchash = (((static_cast<std::uint32_t>(z) * 1024) + static_cast<std::uint32_t>(y)) * 1024) + static_cast<std::uint32_t>(x);//((xx << 2) + (yy << 1) + zz);
	//return mchash;
}


__global__
void _calcVertMChash(uint64_t* _MChash, const Scalar3* _vertexes, const Scalar3 _ubvs, const Scalar3 _lbvs, int number) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;

    Scalar3 SceneSize = make_Scalar3(_ubvs.x - _lbvs.x, _ubvs.y - _lbvs.y, _ubvs.z - _lbvs.z);
    Scalar3 centerP = _vertexes[idx];
    Scalar3 offset = make_Scalar3(centerP.x - _lbvs.x, centerP.y - _lbvs.y, centerP.z - _lbvs.z);
    int type = -1;
    if (type >= 0) {
        if (SceneSize.x > SceneSize.y && SceneSize.y > SceneSize.z) {
            type = 0;
        }
        else if (SceneSize.x > SceneSize.z && SceneSize.z > SceneSize.y) {
            type = 1;
        }
        else if (SceneSize.y > SceneSize.z && SceneSize.z > SceneSize.x) {
            type = 2;
        }
        else if (SceneSize.y > SceneSize.x && SceneSize.x > SceneSize.z) {
            type = 3;
        }
        else if (SceneSize.z > SceneSize.x && SceneSize.x > SceneSize.y) {
            type = 4;
        }
        else {
            type = 5;
        }
    }

    //printf("minSize %f     %f     %f\n", SceneSize.x, SceneSize.y, SceneSize.z);
    uint64_t mc32 = hash_code(type, offset.x / SceneSize.x, offset.y / SceneSize.y, offset.z / SceneSize.z, 1024);
    uint64_t mc64 = ((mc32 << DEFAULT_THREADS_PERWARP) | idx);
    //printf("morton code %lld\n", mc64);
    _MChash[idx] = mc64;
}

__global__
void _updateVertexes(Scalar3* o_vertexes, const Scalar3* _vertexes, Scalar* tempM, const Scalar* mass, __MATHUTILS__::Matrix3x3S* tempCons, int* tempBtype, const __MATHUTILS__::Matrix3x3S* cons, const int* bType, const uint32_t* sortMapIdStoO, uint32_t* sortMapIndex, int number) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= number) return;
    o_vertexes[idx] = _vertexes[sortMapIdStoO[idx]];
    tempM[idx] = mass[sortMapIdStoO[idx]];
    tempCons[idx] = cons[sortMapIdStoO[idx]];
    sortMapIndex[sortMapIdStoO[idx]] = idx;
    tempBtype[idx] = bType[sortMapIdStoO[idx]];
    //printf("original idx: %d        new idx: %d\n", sortMapIdStoO[idx], idx);
}

__global__
void _updateTopology(uint4* tets, uint3* tris, const uint32_t* sortMapIdOtoS, int traNumber, int triNumber) {
    uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < traNumber) {

        tets[idx].x = sortMapIdOtoS[tets[idx].x];
        tets[idx].y = sortMapIdOtoS[tets[idx].y];
        tets[idx].z = sortMapIdOtoS[tets[idx].z];
        tets[idx].w = sortMapIdOtoS[tets[idx].w];
    }
    if (idx < triNumber) {
        tris[idx].x = sortMapIdOtoS[tris[idx].x];
        tris[idx].y = sortMapIdOtoS[tris[idx].y];
        tris[idx].z = sortMapIdOtoS[tris[idx].z];
    }
}

__global__
void _updateSurfaces(uint32_t* sortMapIdStoO, uint3* _faces, int _offset_num, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (_faces[idx].x < _offset_num) {
        _faces[idx].x = sortMapIdStoO[_faces[idx].x];
    }
    else {
        _faces[idx].x = _faces[idx].x;
    }
    if (_faces[idx].y < _offset_num) {
        _faces[idx].y = sortMapIdStoO[_faces[idx].y];
    }else{
        _faces[idx].y = _faces[idx].y;
    }
    if (_faces[idx].z < _offset_num) {
        _faces[idx].z = sortMapIdStoO[_faces[idx].z];
    }
    else {
        _faces[idx].z = _faces[idx].z;
    }
    //printf("sorted face: %d  %d  %d\n", _faces[idx].x, _faces[idx].y, _faces[idx].z);
}

__global__
void _updateSurfEdges(uint32_t* sortMapIdStoO, uint2* _edges, int _offset_num, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (_edges[idx].x < _offset_num) {
        _edges[idx].x = sortMapIdStoO[_edges[idx].x];
    }
    else {
        _edges[idx].x = _edges[idx].x;
    }
    if (_edges[idx].y < _offset_num) {
        _edges[idx].y = sortMapIdStoO[_edges[idx].y];
    }
    else {
        _edges[idx].y = _edges[idx].y;
    }
}

__global__
void _updateTriBendVerts(uint32_t* sortMapIdStoO, uint2* _edges, uint2* _adj_verts, int _offset_num, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (_edges[idx].x < _offset_num) {
        _edges[idx].x = sortMapIdStoO[_edges[idx].x];
    }
    else {
        _edges[idx].x = _edges[idx].x;
    }
    if (_edges[idx].y < _offset_num) {
        _edges[idx].y = sortMapIdStoO[_edges[idx].y];
    }
    else {
        _edges[idx].y = _edges[idx].y;
    }


    if (_adj_verts[idx].x < _offset_num) {
        _adj_verts[idx].x = sortMapIdStoO[_adj_verts[idx].x];
    }
    else {
        _adj_verts[idx].x = _adj_verts[idx].x;
    }
    if (_adj_verts[idx].y < _offset_num) {
        _adj_verts[idx].y = sortMapIdStoO[_adj_verts[idx].y];
    }
    else {
        _adj_verts[idx].y = _adj_verts[idx].y;
    }
}

__global__
void _updateSurfVerts(uint32_t* sortMapIdStoO, uint32_t* _sVerts, int _offset_num, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;
    if (_sVerts[idx] < _offset_num) {
        _sVerts[idx] = sortMapIdStoO[_sVerts[idx]];
    }
    else {
        _sVerts[idx] = _sVerts[idx];
    }
}

void updateSurfEdges(uint32_t* sortMapIdStoO, uint2* _edges, const int& offset_num, const int& numbers) {
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateSurfEdges << <blockNum, threadNum >> > (sortMapIdStoO, _edges, offset_num, numbers);
}

void updateTriBendVerts(uint32_t* sortMapIdStoO, uint2* _tri_edges, uint2* _adj_verts, const int& offset_num, const int& numbers) {
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateTriBendVerts << <blockNum, threadNum >> > (sortMapIdStoO, _tri_edges, _adj_verts, offset_num, numbers);
}


void updateSurfVerts(uint32_t* sortMapIdStoO, uint32_t* _sVerts, const int& offset_num, const int& numbers) {
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateSurfVerts << <blockNum, threadNum >> > (sortMapIdStoO, _sVerts, offset_num, numbers);
}


void updateTopology(uint4* tets, uint3* tris, const uint32_t* sortMapIdOtoS, int traNumber, int triNumber) {
    int numbers = __MATHUTILS__::__m_max(traNumber, triNumber);
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _updateTopology << <blockNum, threadNum >> > (tets, tris, sortMapIdOtoS, traNumber, triNumber);
}

void updateVertexes(Scalar3* o_vertexes, const Scalar3* _vertexes, Scalar* tempM, const Scalar* mass, __MATHUTILS__::Matrix3x3S* tempCons, int* tempBtype, const __MATHUTILS__::Matrix3x3S* cons, const int* bType, const uint32_t* sortMapIdStoO, uint32_t* sortMapIndex, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _updateVertexes << <blockNum, threadNum >> > (o_vertexes, _vertexes, tempM, mass, tempCons, tempBtype, cons, bType, sortMapIdStoO, sortMapIndex, numbers);
}

void updateSurfFaces(uint32_t* sortMapIdStoO, uint3* _faces, const int& offset_num, const int& numbers) {
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateSurfaces << <blockNum, threadNum >> > (sortMapIdStoO, _faces, offset_num, numbers);
}


void calcVertMChash(uint64_t* _MChash, const Scalar3* _vertexes, const Scalar3 _ubvs, const Scalar3 _lbvs, int number) {
    int numbers = number;
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;
    _calcVertMChash << <blockNum, threadNum >> > (_MChash, _vertexes, _ubvs, _lbvs, number);
}



void sortGeometry(
    uint64_t* cudaMortonCodeHash,
    Scalar3* cudaVertPos,
    const Scalar3 _ubvs,
    const Scalar3 _lbvs,
    int vertex_num,
    uint32_t* cudaSortMapIdStoO,
    Scalar3* cudaOriginVertPos,
    Scalar* cudaTempScalar,
    Scalar* cudaVertMass,
    __MATHUTILS__::Matrix3x3S* cudaTempMat3x3,
    __MATHUTILS__::Matrix3x3S* cudaConstraintsMat,
    int* cudaBoundaryType,
    int* cudaTempBoundaryType,
    uint32_t* cudaSortMapIdOtoS,
    uint4* cudaTetElement,
    uint3* cudaTriElement,
    int tetradedra_num,
    int triangle_num) {

    calcVertMChash(cudaMortonCodeHash, cudaVertPos, _ubvs, _lbvs, vertex_num);

    thrust::sequence(thrust::device_ptr<uint32_t>(cudaSortMapIdStoO), 
                     thrust::device_ptr<uint32_t>(cudaSortMapIdStoO) + vertex_num);
    thrust::sort_by_key(thrust::device_ptr<uint64_t>(cudaMortonCodeHash), 
                        thrust::device_ptr<uint64_t>(cudaMortonCodeHash) + vertex_num, 
                        thrust::device_ptr<uint32_t>(cudaSortMapIdStoO));
    updateVertexes(cudaOriginVertPos, cudaVertPos, 
                  cudaTempScalar, cudaVertMass, 
                  cudaTempMat3x3, cudaTempBoundaryType, 
                  cudaConstraintsMat, cudaBoundaryType, 
                  cudaSortMapIdStoO, cudaSortMapIdOtoS, vertex_num);
    
    CUDA_SAFE_CALL(cudaMemcpy(cudaVertPos, cudaOriginVertPos, vertex_num * sizeof(Scalar3), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cudaVertMass, cudaTempScalar, vertex_num * sizeof(Scalar), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cudaConstraintsMat, cudaTempMat3x3, vertex_num * sizeof(__MATHUTILS__::Matrix3x3S), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(cudaBoundaryType, cudaTempBoundaryType, vertex_num * sizeof(int), cudaMemcpyDeviceToDevice));

    updateTopology(cudaTetElement, cudaTriElement, cudaSortMapIdOtoS, tetradedra_num, triangle_num);
}





void sortMesh(
    Scalar3* cudaVertPos,
    uint64_t* cudaMortonCodeHash,
    uint32_t* cudaSortMapIdStoO,
    uint32_t* cudaSortMapIdOtoS,
    Scalar3* cudaOriginVertPos,
    Scalar* cudaTempScalar,
    Scalar* cudaVertMass,
    __MATHUTILS__::Matrix3x3S* cudaTempMat3x3,
    __MATHUTILS__::Matrix3x3S* cudaConstraintsMat,
    int* cudaBoundaryType,
    int* cudaTempBoundaryType,
    uint4* cudaTetElement,
    uint3* cudaTriElement,
    uint3* cudaSurfFace,
    uint2* cudaSurfEdge,
    uint2* cudaTriBendEdges,
    uint2* cudaTriBendVerts,
    uint32_t* cudaSurfVert,
    int hostNumTetElements,
    int hostNumTriElements,
    int hostNumSurfVerts,
    int hostNumSurfFaces,
    int hostNumSurfEdges,
    int hostNumTriBendEdges,
    const Scalar3 _upper_bvs,
    const Scalar3 _lower_bvs,
    int updateVertNum
) {
    sortGeometry(
        cudaMortonCodeHash,
        cudaVertPos,
        _upper_bvs,
        _lower_bvs,
        updateVertNum,
        cudaSortMapIdStoO,
        cudaOriginVertPos,
        cudaTempScalar,
        cudaVertMass,
        cudaTempMat3x3,
        cudaConstraintsMat,
        cudaBoundaryType,
        cudaTempBoundaryType,
        cudaSortMapIdOtoS,
        cudaTetElement,
        cudaTriElement,
        hostNumTetElements,
        hostNumTriElements
    );

    updateSurfFaces(cudaSortMapIdOtoS, cudaSurfFace, updateVertNum, hostNumSurfFaces);
    updateSurfEdges(cudaSortMapIdOtoS, cudaSurfEdge, updateVertNum, hostNumSurfEdges);
    updateTriBendVerts(cudaSortMapIdOtoS, cudaTriBendEdges, cudaTriBendVerts, updateVertNum, hostNumTriBendEdges);
    updateSurfVerts(cudaSortMapIdOtoS, cudaSurfVert, updateVertNum, hostNumSurfVerts);
}




__global__
void _updateNeighborNum(unsigned int* _neighborNumInit, unsigned int* _neighborNum, const uint32_t* sortMapIdOtoS, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    _neighborNum[idx] = _neighborNumInit[sortMapIdOtoS[idx]];
}

__global__
void _updateNeighborList(unsigned int* _neighborListInit, unsigned int* _neighborList, unsigned int* _neighborNum, unsigned int* _neighborStart, unsigned int* _neighborStartTemp, const uint32_t* sortMapIdStoO, const uint32_t* sortMapIdOtoS, int numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numbers) return;

    int startId = _neighborStartTemp[idx];
    int o_startId = _neighborStart[sortMapIdStoO[idx]];
    int neiNum = _neighborNum[idx];
    for (int i = 0; i < neiNum; i++) {
        _neighborList[startId + i] = sortMapIdOtoS[_neighborListInit[o_startId + i]];
    }
}


void updateNeighborInfo(unsigned int* _neighborList, unsigned int* d_neighborListInit, unsigned int* _neighborNum, unsigned int* _neighborNumInit, unsigned int* _neighborStart, unsigned int* _neighborStartTemp, const uint32_t* sortMapIdStoO, const uint32_t* sortMapIdOtoS,  const int& neighborListSize, const int& numbers) {
    const unsigned int threadNum = DEFAULT_THREADS_PERBLOCK;
    int blockNum = (numbers + threadNum - 1) / threadNum;//
    _updateNeighborNum << <blockNum, threadNum >> > (_neighborNumInit, _neighborNum, sortMapIdStoO, numbers);
    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(_neighborNum), thrust::device_ptr<unsigned int>(_neighborNum) + numbers, thrust::device_ptr<unsigned int>(_neighborStartTemp));
    _updateNeighborList << <blockNum, threadNum >> > (d_neighborListInit, _neighborList, _neighborNum, _neighborStart, _neighborStartTemp, sortMapIdStoO, sortMapIdOtoS, numbers);
    
    CUDA_SAFE_CALL(cudaMemcpy(d_neighborListInit, _neighborList, neighborListSize * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_neighborStart, _neighborStartTemp, numbers * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(_neighborNumInit, _neighborNum, numbers * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

}



void sortPreconditioner(
    unsigned int* cudaNeighborList,
    unsigned int* cudaNeighborListInit,
    unsigned int* cudaNeighborNum,
    unsigned int* cudaNeighborNumInit,
    unsigned int* cudaNeighborStart,
    unsigned int* cudaNeighborStartTemp,
    uint32_t* cudaSortMapIdStoO,
    uint32_t* cudaSortMapIdOtoS,
    int MASNeighborListSize,
    int updateVertNum
) {
    updateNeighborInfo(
        cudaNeighborList, 
        cudaNeighborListInit, 
        cudaNeighborNum, 
        cudaNeighborNumInit, 
        cudaNeighborStart, 
        cudaNeighborStartTemp, 
        cudaSortMapIdStoO, 
        cudaSortMapIdOtoS, 
        MASNeighborListSize,
        updateVertNum
    );
}


} // namespace __SORTMESH__


