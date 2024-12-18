
#pragma once

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/GeometryManager.hpp"
#include "UTILS/MathUtils.cuh"

struct AABB {
public:
    Scalar3 upper;
    Scalar3 lower;
    __host__ __device__  AABB();
    __host__ __device__  void combines(const Scalar& x, const Scalar& y, const Scalar& z);
    __host__ __device__  void combines(const Scalar& x, const Scalar& y, const Scalar& z, const Scalar& xx, const Scalar& yy, const Scalar& zz);
    __host__ __device__  void combines(const AABB& aabb);
    __host__ __device__  Scalar3 center();
};

struct Node {
public:
    uint32_t parent_idx;
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t element_idx;
};

class LBVH {
public:
    uint32_t surfVertNum;
    Scalar3* _vertexes;
    AABB* _bvs;
    AABB* _tempLeafBox;
    Node* _nodes;
    uint64_t* _MChash;
    uint32_t* _indices;
    int4* _collisionPair;
    int4* _ccd_collisionPair;
    uint32_t* _cpNum;
    int* _MatIndex;
    uint32_t* _flags;
    AABB scene;
    int* _btype;
public:
    LBVH()
        : surfVertNum(0), _vertexes(nullptr), _bvs(nullptr), _tempLeafBox(nullptr),
          _nodes(nullptr), _MChash(nullptr), _indices(nullptr), _collisionPair(nullptr),
          _ccd_collisionPair(nullptr), _cpNum(nullptr), _MatIndex(nullptr),
          _flags(nullptr), _btype(nullptr) {}
    virtual ~LBVH() {
        std::cout << "deconstruct LBVH" << std::endl;
        CUDA_FREE_LBVH();
    }
    void CUDA_MALLOC_LBVH(const int& number);
    void CUDA_FREE_LBVH();
};


class LBVH_F : public LBVH {
public:
    uint32_t surfFaceNum;
    uint3* _surfFaces;
    uint32_t* _surfVerts;
public:
    void init(int* _btype, Scalar3* _mVerts, uint3* _mSurfFaces, uint32_t* _mSurfVerts, int4* _mCollisionPairs, int4* _ccd_mCollisionPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& surfFaceNum, const int& surfVertNum);
    Scalar Construct();
    AABB* getSceneSize();
    Scalar ConstructFullCCD(const Scalar3* moveDir, const Scalar& alpha);
    void SelfCollisionDetect(Scalar dHat);
    void SelfCollisionFullDetect(Scalar dHat, const Scalar3* moveDir, const Scalar& alpha);
};

class LBVH_E : public LBVH{
public:
    Scalar3* _rest_vertexes;
    uint32_t surfEdgeNum;
    uint2* _surfEdges;
public:
    void init(int* _btype, Scalar3* _mVerts, Scalar3* _rest_vertexes, uint2* _mSurfEdges, int4* _mCollisionPairs, int4* _ccd_mCollisionPairs, uint32_t* _mcpNum, int* _mMatIndex, const int& surfEdgeNum, const int& surfVertNum);
    Scalar Construct();
    Scalar ConstructFullCCD(const Scalar3* moveDir, const Scalar& alpha);
    void SelfCollisionDetect(Scalar dHat);
    void SelfCollisionFullDetect(Scalar dHat, const Scalar3* moveDir, const Scalar& alpha);
};

class LBVH_EF : public LBVH{
public:
    // 来自 LBVH_F 的变量
    uint32_t surfFaceNum;
    uint3* _surfFaces;
    uint32_t* _surfVerts;

    // 来自 LBVH_E 的变量
    Scalar3* _rest_vertexes;
    uint32_t surfEdgeNum;
    uint2* _surfEdges;

    AABB* _surfEdge_bvs;
    Node* _surfEdge_nodes;

public:
    void init(int* _btype, Scalar3* _mVerts, Scalar3* _rest_vertexes, uint2* _mSurfEdges, 
              uint3* _mSurfFaces, uint32_t* _mSurfVerts, int4* _mCollisionPairs, 
              int4* _ccd_mCollisionPairs, uint32_t* _mcpNum, int* _mMatIndex, AABB* _surfEdge_bvs, Node* _surfEdge_nodes,
              const int& surfEdgeNum, const int& surfFaceNum, const int& surfVertNum);

    bool CollisionDetectTriEdge(Scalar dHat);
    bool checkCollisionDetectTriEdge(Scalar dHat);

};


class LBVHCollisionDetector {

public:
    LBVH_E lbvh_e;
    LBVH_F lbvh_f;
    LBVH_EF lbvh_ef;

public:
    void initBVH(std::unique_ptr<GeometryManager>& instance, int* _btype);
	void buildBVH(std::unique_ptr<GeometryManager>& instance);
	void buildBVH_FULLCCD(std::unique_ptr<GeometryManager>& instance, const Scalar& alpha);
	void buildCP(std::unique_ptr<GeometryManager>& instance);
	void buildFullCP(std::unique_ptr<GeometryManager>& instance, const Scalar& alpha);

    void buildGP(std::unique_ptr<GeometryManager>& instance);

};

