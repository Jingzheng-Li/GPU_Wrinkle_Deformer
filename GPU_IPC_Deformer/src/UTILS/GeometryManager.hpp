
#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "CUDAUtils.hpp"
#include "MathUtils.cuh"

struct Node;
class AABB;


class GeometryManager {
public:
    GeometryManager();
    ~GeometryManager() {
        std::cout << "deconstruct GeometryManager" << std::endl;
        CUDA_FREE_GEOMETRYMANAGER();
    }

    void CUDA_FREE_GEOMETRYMANAGER();

private:

    // geometry parameters
    Scalar3* cudaVertPos;
    Scalar3* cudaVertVel;
    Scalar* cudaVertMass;
    uint4* cudaTetElement;
    Scalar* cudaTetVolume;
    Scalar* cudaTriArea;
    __MATHUTILS__::Matrix3x3S* cudaTetDmInverses;
    __MATHUTILS__::Matrix2x2S* cudaTriDmInverses;
    uint32_t* cudaSurfVert;
    uint3* cudaSurfFace;
    uint2* cudaSurfEdge;
    uint3* cudaTriElement;
    uint2* cudaTriBendEdges;
    uint2* cudaTriBendVerts;
    Scalar3* cudaOriginVertPos;
    Scalar3* cudaRestVertPos;
    uint32_t* cudaSortMapIdStoO;
    uint32_t* cudaSortMapIdOtoS;
    uint3* cudaClothFacesAfterSort;
    uint3* cudaBodyFacesAfterSort;


    // soft Constraints
    Scalar3* cudaBoundTargetVertPos;
    uint32_t* cudaBoundTargetIndex;
    uint32_t* cudaSoftTargetIndex;
    Scalar3* cudaSoftTargetVertPos;
    uint3* cudaStitchPairsAfterSort;
    __MATHUTILS__::Matrix3x3S* cudaConstraintsMat;


    // integrator variables
    Scalar3* cudaXTilta;
    Scalar3* cudaFb;
    Scalar3* cudaMoveDir;
    uint64_t* cudaMortonCodeHash;


    // boundary parameters
    int* cudaBoundaryType;
    int* cudaTempBoundaryType;
    uint32_t* cudaCloseConstraintID;
	Scalar* cudaCloseConstraintVal;
	int4* cudaCloseMConstraintID;
	Scalar* cudaCloseMConstraintVal;
    Scalar* cudaTempScalar;
    Scalar3* cudaTempScalar3Mem;
    __MATHUTILS__::Matrix3x3S* cudaTempMat3x3;
    Scalar3* cudaPrevBoundTargetVertPos;
    Scalar2* cudaBoundaryStopDist;


    // collision pairs
    int4* cudaCollisionPairs;
    int4* cudaCCDCollisionPairs;
    uint32_t* cudaEnvCollisionPairs;
    uint32_t* cudaCPNum; // collision pair [5]
    uint32_t* cudaGPNum; // ground pair
    uint32_t* cudaCloseCPNum; // close collision pair
    uint32_t* cudaCloseGPNum; // close ground pair
    Scalar* cudaGroundOffset;
    Scalar3* cudaGroundNormal;
    int* cudaMatIndex;


    // Block Hessian
    uint32_t* cudaD1Index;//pIndex, DpeIndex, DptIndex;
	uint2* cudaD2Index;
	uint3* cudaD3Index;
	uint4* cudaD4Index;
	__MATHUTILS__::Matrix3x3S* cudaH3x3;
	__MATHUTILS__::Matrix6x6S* cudaH6x6;
	__MATHUTILS__::Matrix9x9S* cudaH9x9;
	__MATHUTILS__::Matrix12x12S* cudaH12x12;

    // friction parameters
    Scalar* cudaLambdaLastHScalar;
	Scalar2* cudaDistCoord;
	__MATHUTILS__::Matrix3x2S* cudaTanBasis;
	int4* cudaCollisionPairsLastH;
	int* cudaMatIndexLast;
	Scalar* cudaLambdaLastHScalarGd;
	uint32_t* cudaCollisionPairsLastHGd;


private:

    // Mesh相关信息
    std::vector<Scalar3> hostSurfVertPos;
	std::vector<uint3> hostSurfFaceIds;
	std::vector<uint2> hostSurfEdgeIds;
	std::vector<uint32_t> hostSurfVertIds;
    std::vector<uint3> hostClothFacesBeforeSort;
    std::vector<uint3> hostBodyFacesBeforeSort;
    std::vector<uint3> hostClothFacesAfterSort;
    std::vector<uint3> hostBodyFacesAfterSort;
    std::vector<std::vector<Scalar3>> hostOutputClothVertsTotal;

    // 物理属性参数
    Scalar hostIPCDt;
    Scalar hostBendStiff;
    Scalar hostDensity;
    Scalar hostYoungModulus;
    Scalar hostPoissonRate;
    Scalar hostLengthRateLame;
    Scalar hostVolumeRateLame;
    Scalar hostLengthRate;
    Scalar hostVolumeRate;
    Scalar hostFrictionRate;
    Scalar hostClothThickness;
    Scalar hostClothYoungModulus;
    Scalar hostClothPoissonRate;
    Scalar hostStretchStiff;
    Scalar hostShearStiff;
    Scalar hostClothDensity;
    Scalar hostBoundMotionRate;
    Scalar hostNewtonSolverThreshold;
    Scalar hostPCGThreshold;
    Scalar hostSoftStiffness;
    bool hostStitchBreak;
    Scalar hostStitchBreakThreshold;
    Scalar hostStitchStiffness;
    int hostPrecondType;

    // 动画参数
    Scalar hostAnimationSubRate;
    Scalar hostAnimationFullRate;

    // simulation parameters
    uint32_t hostSimulationFrameId;
    uint32_t hostSimulationFrameRange;

    // 顶点和元素数量
    uint32_t hostNumVertices;
    uint32_t hostNumSurfVerts;
    uint32_t hostNumSurfEdges;
    uint32_t hostNumSurfFaces;
    uint32_t hostNumTriBendEdges;
    uint32_t hostNumTriElements;
    uint32_t hostNumTetElements;
    uint32_t hostNumSoftTargets;
    uint32_t hostNumStitchPairs;
    uint32_t hostNumBoundTargets;
    uint32_t hostNumClothVerts;
    uint32_t hostNumBodyVerts;
    uint32_t hostNumClothFaces;
    uint32_t hostNumBodyFaces;

    // 碰撞相关参数
    int hostMaxTetTriMortonCodeNum;
    int hostMaxCollisionPairsNum;
    int hostMaxCCDCollisionPairsNum;

    // 碰撞对数信息
    uint32_t hostCpNum[5];         // 碰撞对数，包括多种不同类型
    uint32_t hostCcdCpNum;
    uint32_t hostGpNum;
    uint32_t hostCloseCpNum;
    uint32_t hostCloseGpNum;
    uint32_t hostCpNumLast[5];     // 上次迭代的碰撞对数

    // IPC相关参数
    Scalar hostKappa;
    Scalar hostDHat;
    Scalar hostFDHat;
    Scalar hostBboxDiagSize2;
    Scalar hostRelativeDHat;
    Scalar hostRelativeDHatThres;
    Scalar hostDTol;
    Scalar hostMinKappaCoef;
    Scalar hostMeanMass;
    Scalar hostMeanVolume;
    uint32_t hostGpNumLast;
    uint32_t hostNumSubsteps;
    Scalar hostBoundaryStopCriterion;

    // 风力相关参数
    Scalar3 hostWindDirection;
    Scalar hostWindStrength;
    Scalar hostAirResistance;
    Scalar hostNoiseFrequency;
    Scalar hostNoiseAmplitude;


    // 目标和位置数据
    std::vector<uint32_t> hostSoftTargetIdsBeforeSort;
    std::vector<uint32_t> hostSoftTargetIdsAfterSort;
    std::vector<Scalar3> hostSoftTargetPos;
    std::vector<Scalar3> hostBoundaryTargetPos;
    std::vector<uint3> hostStitchPairsBeforeSort;
    std::vector<uint3> hostStitchPairsAfterSort;



public:

    __device__ __host__ inline Scalar3*& getCudaSurfVertPos() { return cudaVertPos; }
    __device__ __host__ inline Scalar3*& getCudaVertVel() { return cudaVertVel; }
    __device__ __host__ inline Scalar*& getCudaVertMass() { return cudaVertMass; }
    __device__ __host__ inline uint4*& getCudaTetElement() { return cudaTetElement; }
    __device__ __host__ inline Scalar*& getCudaTetVolume() { return cudaTetVolume; }
    __device__ __host__ inline Scalar*& getCudaTriArea() { return cudaTriArea; }
    __device__ __host__ inline __MATHUTILS__::Matrix3x3S*& getCudaTetDmInverses() { return cudaTetDmInverses; }
    __device__ __host__ inline __MATHUTILS__::Matrix2x2S*& getCudaTriDmInverses() { return cudaTriDmInverses; }
    __device__ __host__ inline uint32_t*& getCudaSurfVertIds() { return cudaSurfVert; }
    __device__ __host__ inline uint3*& getCudaSurfFaceIds() { return cudaSurfFace; }
    __device__ __host__ inline uint2*& getCudaSurfEdgeIds() { return cudaSurfEdge; }
    __device__ __host__ inline uint3*& getCudaTriElement() { return cudaTriElement; }
    __device__ __host__ inline uint2*& getCudaTriBendEdges() { return cudaTriBendEdges; }
    __device__ __host__ inline uint2*& getCudaTriBendVerts() { return cudaTriBendVerts; }
    __device__ __host__ inline Scalar3*& getCudaBoundTargetVertPos() { return cudaBoundTargetVertPos; }
    __device__ __host__ inline uint32_t*& getCudaBoundTargetIndex() { return cudaBoundTargetIndex; }
    __device__ __host__ inline uint32_t*& getCudaSoftTargetIndex() { return cudaSoftTargetIndex; }
    __device__ __host__ inline Scalar3*& getCudaSoftTargetVertPos() { return cudaSoftTargetVertPos; }
    __device__ __host__ inline __MATHUTILS__::Matrix3x3S*& getCudaConstraintsMat() { return cudaConstraintsMat; }
    __device__ __host__ inline Scalar3*& getCudaOriginVertPos() { return cudaOriginVertPos; }
    __device__ __host__ inline Scalar3*& getCudaRestVertPos() { return cudaRestVertPos; }
    __device__ __host__ inline int4*& getCudaCollisionPairs() { return cudaCollisionPairs; }
    __device__ __host__ inline int4*& getCudaCCDCollisionPairs() { return cudaCCDCollisionPairs; }
    __device__ __host__ inline uint32_t*& getCudaEnvCollisionPairs() { return cudaEnvCollisionPairs; }
    __device__ __host__ inline uint32_t*& getCudaCPNum() { return cudaCPNum; }
    __device__ __host__ inline uint32_t*& getCudaGPNum() { return cudaGPNum; }
    __device__ __host__ inline uint32_t*& getCudaCloseCPNum() { return cudaCloseCPNum; }
    __device__ __host__ inline uint32_t*& getCudaCloseGPNum() { return cudaCloseGPNum; }
    __device__ __host__ inline Scalar*& getCudaGroundOffset() { return cudaGroundOffset; }
    __device__ __host__ inline Scalar3*& getCudaGroundNormal() { return cudaGroundNormal; }
    __device__ __host__ inline int*& getCudaMatIndex() { return cudaMatIndex; }
    __device__ __host__ inline Scalar3*& getCudaXTilta() { return cudaXTilta; }
    __device__ __host__ inline Scalar3*& getCudaFb() { return cudaFb; }
    __device__ __host__ inline Scalar3*& getCudaMoveDir() { return cudaMoveDir; }
    __device__ __host__ inline uint64_t*& getCudaMortonCodeHash() { return cudaMortonCodeHash; }
    __device__ __host__ inline uint32_t*& getCudaSortMapIdStoO() { return cudaSortMapIdStoO; }
    __device__ __host__ inline uint32_t*& getCudaSortMapIdOtoS() { return cudaSortMapIdOtoS; }
    __device__ __host__ inline Scalar*& getCudaTempScalar() { return cudaTempScalar; }
    __device__ __host__ inline Scalar3*& getCudaTempScalar3Mem() { return cudaTempScalar3Mem; }
    __device__ __host__ inline __MATHUTILS__::Matrix3x3S*& getCudaTempMat3x3() { return cudaTempMat3x3; }
    __device__ __host__ inline int*& getCudaBoundaryType() { return cudaBoundaryType; }
    __device__ __host__ inline int*& getCudaTempBoundaryType() { return cudaTempBoundaryType; }
    __device__ __host__ inline uint32_t*& getCudaCloseConstraintID() { return cudaCloseConstraintID; }
    __device__ __host__ inline Scalar*& getCudaCloseConstraintVal() { return cudaCloseConstraintVal; }
    __device__ __host__ inline int4*& getCudaCloseMConstraintID() { return cudaCloseMConstraintID; }
    __device__ __host__ inline Scalar*& getCudaCloseMConstraintVal() { return cudaCloseMConstraintVal; }
    __device__ __host__ inline uint32_t*& getCudaD1Index() { return cudaD1Index; }
    __device__ __host__ inline uint2*& getCudaD2Index() { return cudaD2Index; }
    __device__ __host__ inline uint3*& getCudaD3Index() { return cudaD3Index; }
    __device__ __host__ inline uint4*& getCudaD4Index() { return cudaD4Index; }
    __device__ __host__ inline __MATHUTILS__::Matrix3x3S*& getCudaH3x3() { return cudaH3x3; }
    __device__ __host__ inline __MATHUTILS__::Matrix6x6S*& getCudaH6x6() { return cudaH6x6; }
    __device__ __host__ inline __MATHUTILS__::Matrix9x9S*& getCudaH9x9() { return cudaH9x9; }
    __device__ __host__ inline __MATHUTILS__::Matrix12x12S*& getCudaH12x12() { return cudaH12x12; }
    __device__ __host__ inline uint3*& getCudaClothFacesAfterSort() { return cudaClothFacesAfterSort; }
    __device__ __host__ inline uint3*& getCudaBodyFacesAfterSort() { return cudaBodyFacesAfterSort; }
    __device__ __host__ inline uint3*& getCudaStitchPairsIndex() { return cudaStitchPairsAfterSort; }
    __device__ __host__ inline Scalar3*& getCudaPrevBoundTargetVertPos() { return cudaPrevBoundTargetVertPos; }
    __device__ __host__ inline Scalar2*& getCudaBoundaryStopDist() { return cudaBoundaryStopDist; }


    // 摩擦参数相关变量
    __device__ __host__ inline Scalar*& getCudaLambdaLastHScalar() { return cudaLambdaLastHScalar; }
    __device__ __host__ inline Scalar2*& getCudaDistCoord() { return cudaDistCoord; }
    __device__ __host__ inline __MATHUTILS__::Matrix3x2S*& getCudaTanBasis() { return cudaTanBasis; }
    __device__ __host__ inline int4*& getCudaCollisionPairsLastH() { return cudaCollisionPairsLastH; }
    __device__ __host__ inline int*& getCudaMatIndexLast() { return cudaMatIndexLast; }
    __device__ __host__ inline Scalar*& getCudaLambdaLastHScalarGd() { return cudaLambdaLastHScalarGd; }
    __device__ __host__ inline uint32_t*& getCudaCollisionPairsLastHGd() { return cudaCollisionPairsLastHGd; }




public:



    // 物理属性参数的get函数，返回引用以便修改
    __device__ __host__ inline Scalar& getHostBendStiff() { return hostBendStiff; }
    __device__ __host__ inline Scalar& getHostDensity() { return hostDensity; }
    __device__ __host__ inline Scalar& getHostYoungModulus() { return hostYoungModulus; }
    __device__ __host__ inline Scalar& getHostPoissonRate() { return hostPoissonRate; }
    __device__ __host__ inline Scalar& getHostLengthRateLame() { return hostLengthRateLame; }
    __device__ __host__ inline Scalar& getHostVolumeRateLame() { return hostVolumeRateLame; }
    __device__ __host__ inline Scalar& getHostLengthRate() { return hostLengthRate; }
    __device__ __host__ inline Scalar& getHostVolumeRate() { return hostVolumeRate; }
    __device__ __host__ inline Scalar& getHostFrictionRate() { return hostFrictionRate; }
    __device__ __host__ inline Scalar& getHostClothThickness() { return hostClothThickness; }
    __device__ __host__ inline Scalar& getHostClothYoungModulus() { return hostClothYoungModulus; }
    __device__ __host__ inline Scalar& getHostClothPoissonRate() { return hostClothPoissonRate; }
    __device__ __host__ inline Scalar& getHostStretchStiff() { return hostStretchStiff; }
    __device__ __host__ inline Scalar& getHostShearStiff() { return hostShearStiff; }
    __device__ __host__ inline Scalar& getHostClothDensity() { return hostClothDensity; }
    __device__ __host__ inline Scalar& getHostBoundMotionRate() { return hostBoundMotionRate; }
    __device__ __host__ inline Scalar& getHostNewtonSolverThreshold() { return hostNewtonSolverThreshold; }
    __device__ __host__ inline Scalar& getHostPCGThreshold() { return hostPCGThreshold; }
    __device__ __host__ inline Scalar& getHostSoftStiffness() { return hostSoftStiffness; }
    __device__ __host__ inline Scalar& getHostStitchStiffness() {return hostStitchStiffness; }
    __device__ __host__ inline Scalar& getHostStitchBreakThreshold() {return hostStitchBreakThreshold; }
    __device__ __host__ inline bool& getHostStitchBreak() {return hostStitchBreak; }


    // 动画参数的get函数，返回引用以便修改
    __device__ __host__ inline Scalar& getHostAnimationSubRate() { return hostAnimationSubRate; }
    __device__ __host__ inline Scalar& getHostAnimationFullRate() { return hostAnimationFullRate; }

    // 顶点和元素数量的get函数，返回引用以便修改
    __device__ __host__ inline uint32_t& getHostNumVertices() { return hostNumVertices; }
    __device__ __host__ inline uint32_t& getHostNumSurfVerts() { return hostNumSurfVerts; }
    __device__ __host__ inline uint32_t& getHostNumSurfEdges() { return hostNumSurfEdges; }
    __device__ __host__ inline uint32_t& getHostNumSurfFaces() { return hostNumSurfFaces; }
    __device__ __host__ inline uint32_t& getHostNumTriBendEdges() { return hostNumTriBendEdges; }
    __device__ __host__ inline uint32_t& getHostNumTriElements() { return hostNumTriElements; }
    __device__ __host__ inline uint32_t& getHostNumTetElements() { return hostNumTetElements; }
    __device__ __host__ inline uint32_t& getHostNumSoftTargets() { return hostNumSoftTargets; }
    __device__ __host__ inline uint32_t& getHostNumStitchPairs() { return hostNumStitchPairs; }
    __device__ __host__ inline uint32_t& getHostNumBoundTargets() { return hostNumBoundTargets; }
    __device__ __host__ inline uint32_t& getHostNumClothVerts() { return hostNumClothVerts; }
    __device__ __host__ inline uint32_t& getHostNumBodyVerts() { return hostNumBodyVerts; }    
    __device__ __host__ inline uint32_t& getHostNumClothFaces() { return hostNumClothFaces; }
    __device__ __host__ inline uint32_t& getHostNumBodyFaces() { return hostNumBodyFaces; }

    // 碰撞相关参数的get函数，返回引用以便修改
    __device__ __host__ inline int& getHostMaxTetTriMortonCodeNum() { return hostMaxTetTriMortonCodeNum; }
    __device__ __host__ inline int& getHostMaxCollisionPairsNum() { return hostMaxCollisionPairsNum; }
    __device__ __host__ inline int& getHostMaxCCDCollisionPairsNum() { return hostMaxCCDCollisionPairsNum; }

    // 碰撞对数信息的get函数，返回引用以便修改
    __device__ __host__ inline uint32_t& getHostCpNum(int index) { return hostCpNum[index]; }
    __device__ __host__ inline uint32_t& getHostCcdCpNum() { return hostCcdCpNum; }
    __device__ __host__ inline uint32_t& getHostGpNum() { return hostGpNum; }
    __device__ __host__ inline uint32_t& getHostCloseCpNum() { return hostCloseCpNum; }
    __device__ __host__ inline uint32_t& getHostCloseGpNum() { return hostCloseGpNum; }
    __device__ __host__ inline uint32_t& getHostCpNumLast(int index) { return hostCpNumLast[index]; }

    // IPC 相关参数的get函数，返回引用以便修改
    __device__ __host__ inline Scalar& getHostKappa() { return hostKappa; }
    __device__ __host__ inline Scalar& getHostDHat() { return hostDHat; }
    __device__ __host__ inline Scalar& getHostFDHat() { return hostFDHat; }
    __device__ __host__ inline Scalar& getHostBboxDiagSize2() { return hostBboxDiagSize2; }
    __device__ __host__ inline Scalar& getHostRelativeDHat() { return hostRelativeDHat; }
    __device__ __host__ inline Scalar& getHostRelativeDHatThres() { return hostRelativeDHatThres; }
    __device__ __host__ inline Scalar& getHostDTol() { return hostDTol; }
    __device__ __host__ inline Scalar& getHostMinKappaCoef() { return hostMinKappaCoef; }
    __device__ __host__ inline Scalar& getHostIPCDt() { return hostIPCDt; }
    __device__ __host__ inline Scalar& getHostMeanMass() { return hostMeanMass; }
    __device__ __host__ inline Scalar& getHostMeanVolume() { return hostMeanVolume; }
    __device__ __host__ inline uint32_t& getHostGpNumLast() { return hostGpNumLast; }
    __device__ __host__ inline uint32_t& getHostNumSubsteps() { return hostNumSubsteps; }

    // 目标和位置数据的get函数，返回引用以便修改
    __device__ __host__ inline std::vector<uint32_t>& getHostSoftTargetIdsAfterSort() { return hostSoftTargetIdsAfterSort; }
    __device__ __host__ inline std::vector<Scalar3>& getHostSoftTargetPos() { return hostSoftTargetPos; }
    __device__ __host__ inline std::vector<Scalar3>& getHostBoundaryTargetPos() { return hostBoundaryTargetPos; }
    __device__ __host__ inline std::vector<uint3>& getHostStitchPairsAfterSort() {return hostStitchPairsAfterSort; }

    // Mesh的相关信息
    __device__ __host__ inline std::vector<Scalar3>& getHostSurfVertPos() { return hostSurfVertPos; }
    __device__ __host__ inline std::vector<uint3>& getHostClothFacesAfterSort() { return hostClothFacesAfterSort; }
    __device__ __host__ inline std::vector<uint3>& getHostBodyFacesAfterSort() { return hostBodyFacesAfterSort; }
    __device__ __host__ inline std::vector<std::vector<Scalar3>>& getHostOutputClothVertsTotal() { return hostOutputClothVertsTotal; }
    __device__ __host__ inline Scalar& getHostBoundaryStopCriterion() { return hostBoundaryStopCriterion; }

    // 风力相关参数
    __device__ __host__ inline Scalar3& getHostWindDirection() { return hostWindDirection; }
    __device__ __host__ inline Scalar& getHostWindStrength() { return hostWindStrength; }
    __device__ __host__ inline Scalar& getHostAirResistance() { return hostAirResistance; }
    __device__ __host__ inline Scalar& getHostNoiseFrequency() { return hostNoiseFrequency; }
    __device__ __host__ inline Scalar& getHostNoiseAmplitude() { return hostNoiseAmplitude; }

    // 模拟参数
    __device__ __host__ inline int& getHostPrecondType() { return hostPrecondType; }
    __device__ __host__ inline uint32_t& getHostSimulationFrameId() { return hostSimulationFrameId; }
    __device__ __host__ inline uint32_t& getHostSimulationFrameRange() { return hostSimulationFrameRange; }

};

