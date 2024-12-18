
#include <memory>

#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "BlockHessian.cuh"


class MASPreconditioner {

public:

	MASPreconditioner();
	~MASPreconditioner();

	void CUDA_MALLOC_MAS_PRECONDITIONER(int vertNum, int totalNeighborNum, int4* m_collisionPairs);
	
	void CUDA_FREE_MAS_PRECONDITIONER();

public:
	void computeNumLevels(int vertNum);

	void BuildConnectMaskL0();
	
	void PreparePrefixSumL0();

	void BuildLevel1();
	
	void BuildConnectMaskLx(int level);
	
	void NextLevelCluster(int level);
	
	void PrefixSumLx(int level);
	
	void ComputeNextLevel(int level);
	
	void AggregationKernel();

	int ReorderRealtime(int cpNum);
	
	void PrepareHessian(const std::unique_ptr<BlockHessian>& BH_ptr, const Scalar* masses);

	void setPreconditioner(const std::unique_ptr<BlockHessian>& BH_ptr, const Scalar* masses, int cpNum);

	void BuildCollisionConnection(unsigned int* connectionMsk, int* coarseTableSpace, int level, int cpNum);

	void BuildMultiLevelR(const Scalar3* R);
	
	void SchwarzLocalXSym();

	void CollectFinalZ(Scalar3* Z);

	void preconditioning(const Scalar3* R, Scalar3* Z);



public:
	int hostMASNeighborListSize;
	unsigned int* cudaNeighborList;
	unsigned int* cudaNeighborStart;
	unsigned int* cudaNeighborStartTemp;
	unsigned int* cudaNeighborNum;
	unsigned int* cudaNeighborListInit;
	unsigned int* cudaNeighborNumInit;

private:
	int hostMASTotalNodes;
	int hostMASLevelNum;
	int hostMASTotalNumberClusters;
	int2 hostMAScLevelSize;

	int4* cudaMASCollisionPairs;
	int2* cudaLevelSize;
	int* cudaCoarseSpaceTables;
	int* cudaPrefixOriginal;
	int* cudaPrefixSumOriginal;
	int* cudaGoingNext;
	int* cudaDenseLevel;
	int4* cudaCoarseTable;
	unsigned int* cudaFineConnectMask;
	unsigned int* cudaNextConnectMask;
	unsigned int* cudaNextPrefix;
	unsigned int* cudaNextPrefixSum;

	__MATHUTILS__::Matrix96x96S* cudaMat96;
    __MATHUTILS__::MasMatrixSym* cudaInverseMat96;
	float3* cudaMultiLevelR;
	float3* cudaMultiLevelZ;

};