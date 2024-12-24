
#include "GeometryManager.hpp"

// std::unique_ptr<GeometryManager> GeometryManager::instance = nullptr;

GeometryManager::GeometryManager() :
    cudaVertPos(nullptr), 
    cudaVertVel(nullptr), 
    cudaVertMass(nullptr), 
    cudaTetElement(nullptr), 
    cudaTetVolume(nullptr), 
    cudaTriArea(nullptr), 
    cudaTetDmInverses(nullptr), 
    cudaTriDmInverses(nullptr), 
    cudaSurfVert(nullptr), 
    cudaSurfFace(nullptr), 
    cudaSurfEdge(nullptr), 
    cudaTriElement(nullptr), 
    cudaTriBendEdges(nullptr), 
    cudaTriBendVerts(nullptr), 
    cudaBoundTargetVertPos(nullptr), 
    cudaBoundTargetIndex(nullptr), 
    cudaSoftTargetIndex(nullptr), 
    cudaSoftTargetVertPos(nullptr), 
    cudaConstraintsMat(nullptr), 
    cudaOriginVertPos(nullptr), 
    cudaRestVertPos(nullptr), 
    cudaCollisionPairs(nullptr), 
    cudaCCDCollisionPairs(nullptr), 
    cudaEnvCollisionPairs(nullptr), 
    cudaCPNum(nullptr), 
    cudaGPNum(nullptr), 
    cudaCloseCPNum(nullptr), 
    cudaCloseGPNum(nullptr), 
    cudaGroundOffset(nullptr), 
    cudaGroundNormal(nullptr), 
    cudaXTilta(nullptr), 
    cudaFb(nullptr), 
    cudaMoveDir(nullptr), 
    cudaMortonCodeHash(nullptr), 
    cudaSortMapIdStoO(nullptr), 
    cudaSortMapIdOtoS(nullptr), 
    cudaTempScalar(nullptr), 
    cudaTempScalar3Mem(nullptr), 
    cudaTempMat3x3(nullptr), 
    cudaBoundaryType(nullptr), 
    cudaTempBoundaryType(nullptr), 
    cudaCloseConstraintID(nullptr), 
    cudaCloseConstraintVal(nullptr), 
    cudaCloseMConstraintID(nullptr), 
    cudaCloseMConstraintVal(nullptr), 
    cudaD1Index(nullptr), 
    cudaD2Index(nullptr), 
    cudaD3Index(nullptr), 
    cudaD4Index(nullptr), 
    cudaH3x3(nullptr), 
    cudaH6x6(nullptr), 
    cudaH9x9(nullptr), 
    cudaH12x12(nullptr), 
    cudaLambdaLastHScalar(nullptr), 
    cudaDistCoord(nullptr), 
    cudaTanBasis(nullptr), 
    cudaCollisionPairsLastH(nullptr), 
    cudaMatIndexLast(nullptr), 
    cudaLambdaLastHScalarGd(nullptr), 
    cudaCollisionPairsLastHGd(nullptr),
    cudaTriVerts(nullptr),
    cudaTriEdges(nullptr),
    cudaClothFacesAfterSort(nullptr),
    cudaBodyFacesAfterSort(nullptr),
    cudaStitchPairsAfterSort(nullptr),
    cudaPrevBoundTargetVertPos(nullptr),
    cudaBoundaryStopDist(nullptr),


    // host variables 初始化
    hostIPCDt(0.0),
    hostBendStiff(0.0),
    hostDensity(0.0),
    hostYoungModulus(0.0),
    hostPoissonRate(0.0),
    hostLengthRateLame(0.0),
    hostVolumeRateLame(0.0),
    hostLengthRate(0.0),
    hostVolumeRate(0.0),
    hostFrictionRate(0.0),
    hostClothThickness(0.0),
    hostClothYoungModulus(0.0),
    hostStretchStiff(0.0),
    hostShearStiff(0.0),
    hostClothDensity(0.0),
    hostBoundMotionRate(0.0),
    hostNewtonSolverThreshold(0.0),
    hostPCGThreshold(0.0),
    hostSoftStiffness(0.0),
    hostStitchBreak(false),
    hostStitchBreakThreshold(0.0),
    hostStitchStiffness(0.0),
    hostPrecondType(0),
    hostAnimationSubRate(0.0),
    hostAnimationFullRate(0.0),
    hostSimulationFrameId(0),
    hostSimulationFrameRange(0),
    hostNumVertices(0),
    hostNumSurfVerts(0),
    hostNumSurfEdges(0),
    hostNumSurfFaces(0),
    hostNumTriBendEdges(0),
    hostNumTriElements(0),
    hostNumTetElements(0),
    hostNumSoftTargets(0),
    hostNumStitchPairs(0),
    hostNumBoundTargets(0),
    hostMaxTetTriMortonCodeNum(0),
    hostMaxCollisionPairsNum(0),
    hostMaxCCDCollisionPairsNum(0),
    hostCcdCpNum(0),
    hostGpNum(0),
    hostCloseCpNum(0),
    hostCloseGpNum(0),
    hostKappa(0.0),
    hostDHat(0.0),
    hostFDHat(0.0),
    hostBboxDiagSize2(0.0),
    hostRelativeDHat(0.0),
    hostRelativeDHatThres(0.0),
    hostDTol(0.0),
    hostMinKappaCoef(0.0),
    hostMeanMass(0.0),
    hostMeanVolume(0.0),
    hostGpNumLast(0),
    hostBoundaryStopCriterion(0.0),
    hostCpNum{0, 0, 0, 0, 0},
    hostCpNumLast{0, 0, 0, 0, 0},
    hostWindDirection(make_Scalar3(0, 0, 0)),
    hostWindStrength(0.0),
    hostAirResistance(0.0),
    hostNoiseFrequency(0.0),
    hostNoiseAmplitude(0.0)
    {    }



void GeometryManager::CUDA_FREE_GEOMETRYMANAGER() {
    CUDAFreeSafe(cudaVertPos);
    CUDAFreeSafe(cudaVertVel);
    CUDAFreeSafe(cudaVertMass);
    CUDAFreeSafe(cudaTetElement);
    CUDAFreeSafe(cudaTetVolume);
    CUDAFreeSafe(cudaTriArea);
    CUDAFreeSafe(cudaTetDmInverses);
    CUDAFreeSafe(cudaTriDmInverses);
    CUDAFreeSafe(cudaSurfVert);
    CUDAFreeSafe(cudaSurfFace);
    CUDAFreeSafe(cudaSurfEdge);
    CUDAFreeSafe(cudaTriElement);
    CUDAFreeSafe(cudaTriBendEdges);
    CUDAFreeSafe(cudaTriBendVerts);
    CUDAFreeSafe(cudaBoundTargetVertPos);
    CUDAFreeSafe(cudaBoundTargetIndex);
    CUDAFreeSafe(cudaSoftTargetIndex);
    CUDAFreeSafe(cudaSoftTargetVertPos);
    CUDAFreeSafe(cudaConstraintsMat);
    CUDAFreeSafe(cudaOriginVertPos);
    CUDAFreeSafe(cudaRestVertPos);
    CUDAFreeSafe(cudaCollisionPairs);
    CUDAFreeSafe(cudaCCDCollisionPairs);
    CUDAFreeSafe(cudaEnvCollisionPairs);
    CUDAFreeSafe(cudaCPNum);
    CUDAFreeSafe(cudaGPNum);
    CUDAFreeSafe(cudaCloseCPNum);
    CUDAFreeSafe(cudaCloseGPNum);
    CUDAFreeSafe(cudaGroundOffset);
    CUDAFreeSafe(cudaGroundNormal);
    CUDAFreeSafe(cudaXTilta);
    // CUDAFreeSafe(cudaFb); // 在cudaPCGb中提前被释放
    // CUDAFreeSafe(cudaMoveDir); // 在cudaPCGdx中被提前释放
    CUDAFreeSafe(cudaMortonCodeHash);
    CUDAFreeSafe(cudaSortMapIdStoO);
    CUDAFreeSafe(cudaSortMapIdOtoS);
    CUDAFreeSafe(cudaTempScalar);
    CUDAFreeSafe(cudaTempScalar3Mem);
    CUDAFreeSafe(cudaTempMat3x3);
    CUDAFreeSafe(cudaBoundaryType);
    CUDAFreeSafe(cudaTempBoundaryType);
    CUDAFreeSafe(cudaCloseConstraintID);
    CUDAFreeSafe(cudaCloseConstraintVal);
    CUDAFreeSafe(cudaCloseMConstraintID);
    CUDAFreeSafe(cudaCloseMConstraintVal);
    CUDAFreeSafe(cudaD1Index);
    CUDAFreeSafe(cudaD2Index);
    CUDAFreeSafe(cudaD3Index);
    CUDAFreeSafe(cudaD4Index);
    CUDAFreeSafe(cudaH3x3);
    CUDAFreeSafe(cudaH6x6);
    CUDAFreeSafe(cudaH9x9);
    CUDAFreeSafe(cudaH12x12);
    CUDAFreeSafe(cudaLambdaLastHScalar);
    CUDAFreeSafe(cudaDistCoord);
    CUDAFreeSafe(cudaTanBasis);
    CUDAFreeSafe(cudaCollisionPairsLastH);
    CUDAFreeSafe(cudaMatIndexLast);
    CUDAFreeSafe(cudaLambdaLastHScalarGd);
    CUDAFreeSafe(cudaCollisionPairsLastHGd);
    CUDAFreeSafe(cudaTriVerts);
    CUDAFreeSafe(cudaTriEdges);
    CUDAFreeSafe(cudaClothFacesAfterSort);
    CUDAFreeSafe(cudaBodyFacesAfterSort);
    CUDAFreeSafe(cudaStitchPairsAfterSort);
    CUDAFreeSafe(cudaPrevBoundTargetVertPos);
    CUDAFreeSafe(cudaBoundaryStopDist);
}


