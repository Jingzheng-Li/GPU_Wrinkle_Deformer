
#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <nlohmann/json.hpp>


#include "UTILS/CUDAUtils.hpp"
#include "UTILS/MathUtils.cuh"
#include "LBVH.cuh"



#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/locate.h>
#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/bbox.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/boost/graph/copy_face_graph.h>
#include <CGAL/Iso_cuboid_3.h>

typedef CGAL::Simple_cartesian<Scalar> CGAL_Kernel;
typedef CGAL_Kernel::Point_3 CGAL_Point_3;
typedef CGAL_Kernel::Vector_3 CGAL_Vector_3;
typedef CGAL_Kernel::Triangle_3 CGAL_Triangle_3;
typedef CGAL::Surface_mesh<CGAL_Point_3> CGAL_Mesh;
typedef CGAL_Mesh::Face_index CGAL_Face_index;
typedef CGAL_Mesh::Vertex_index CGAL_Vertex_index;
typedef CGAL_Mesh::Halfedge_index CGAL_Halfedge_index;
typedef CGAL::AABB_face_graph_triangle_primitive<CGAL_Mesh> CGAL_Primitive;
typedef CGAL::AABB_traits<CGAL_Kernel, CGAL_Primitive> CGAL_AABB_traits;
typedef CGAL::AABB_tree<CGAL_AABB_traits> CGAL_Tree;
typedef CGAL_Tree::Point_and_primitive_id CGAL_Point_and_primitive_id;



struct SIMMesh {

	CGAL_Mesh CGAL_clothmesh_orig;
	CGAL_Mesh CGAL_clothmesh_fuse;
	std::vector<Scalar3> clothverts_orig;
	std::vector<uint3> clothfaces_orig;
	std::vector<Scalar3> clothverts_fuse;
	std::vector<uint3> clothfaces_fuse;


	CGAL_Mesh CGAL_bodymesh_tpose;
	CGAL_Mesh CGAL_bodymesh_mapped;
	std::vector<CGAL_Mesh> CGAL_bodymesh_total;
	std::vector<std::vector<Scalar3>> bodyverts_total;
	std::vector<std::vector<uint3>> bodyfaces_total;

	CGAL_Mesh CGAL_staticmesh;
	CGAL_Mesh CGAL_staticmesh_mapped;

	CGAL_Mesh CGAL_surfmesh;
	std::vector<Scalar3> surffverts;
	std::vector<uint3> surffaces;
	std::vector<Scalar3> surfVertPos;
	std::vector<uint3> surfFaceIds;
	std::vector<uint2> surfEdgeIds;
	std::vector<uint32_t> surfVertIds;


	std::vector<uint32_t> bodymeshmapped_faceidx;
	std::vector<Scalar> animation_motion_rate_total;

	std::vector<uint32_t> projectedFaceIds;
	std::vector<Scalar2> projectedUVs;
	std::vector<std::vector<Scalar3>> projectedPositions;

	std::vector<uint32_t> SortMapIdStoO; // 记录cloth sort后的顶点信息
	std::vector<uint32_t> SortMapIdOtoS;

	std::vector<uint3> stitchPairsBeforeSort;
    std::vector<uint3> stitchPairsAfterSort;
	std::vector<uint32_t> softTargetIdsBeforeSort;
    std::vector<uint32_t> softTargetIdsAfterSort;
	std::vector<Scalar3> softTargetPos;
    std::vector<Scalar3> boundaryTargetPos;

	std::vector<std::vector<Scalar3>> output_clothverts_total; // since we don't change topology face will be same always 


	std::vector<Scalar> volume;
	std::vector<Scalar> area;
	std::vector<Scalar> masses;

	Scalar meanMass;
	Scalar meanVolume;
	
	std::vector<Scalar3> velocities;

	std::vector<uint4> tetrahedras;
	std::vector<uint3> triangles;

	std::vector<int> boundaryTypies;
	std::vector<uint32_t> boundaryTargetIndex;

	std::vector<__MATHUTILS__::Matrix3x3S> DMInverse;
	std::vector<__MATHUTILS__::Matrix2x2S> triDMInverse;
	std::vector<__MATHUTILS__::Matrix3x3S> constraints;

	std::vector<Scalar3> boundaryTargetVertPos;
	std::vector<uint2> triBendEdges; // 一条边的两个顶点索引
	std::vector<uint2> triBendVerts; // 一条边的两个面第三个点的索引

	std::vector<std::vector<unsigned int>> vertNeighbors;
	std::vector<unsigned int> neighborList;
	std::vector<unsigned int> neighborStart;
	std::vector<unsigned int> neighborNum;

	std::vector<int> index_mapping_orig2fuse;

	std::vector<Node> nodes;
	std::vector<AABB> bvs;

	nlohmann::json httpjson;
	nlohmann::json output_httpjson;

	int numSurfVerts;
	int numTetElements;
	int numTriElements;
	int numBoundTargets;

	Scalar simulation_totaltime;
	bool simulation_finished;

    SIMMesh()
        : meanMass(0.0),
          meanVolume(0.0),
          numSurfVerts(0),
          numTetElements(0),
          numTriElements(0),
          numBoundTargets(0),
		  simulation_totaltime(0),
		  simulation_finished(false) {}
};


namespace LOADMESH {

bool CGAL_readObj(const std::string& filename, CGAL_Mesh& mesh);

bool CGAL_writeObj(const std::string& filename, const CGAL_Mesh& mesh);

bool CGAL_readObjBody(
    const std::string& obj_file_pattern,
    const int framerange_start,
    const int framerange_end,
    std::vector<CGAL_Mesh>& meshes
);

void convertCGALVertsToVector(
	const CGAL_Mesh& mesh, 
	std::vector<Scalar3>& verts
);

void convertCGALFacesToVector(
	const CGAL_Mesh& mesh, 
	std::vector<uint3>& faces
);

// used for triangle bending
void extractTriBendEdgesFaces_CGAL(
	const CGAL_Mesh& mesh, 
	std::vector<uint2>& triBendEdges, 
	std::vector<uint2>& triBendVerts
);

bool CGAL_detectBodySelfIntersections(
    CGAL_Mesh& BodyMeshMapped,
    std::set<CGAL_Face_index>& intersectedFaces
);

void CGAL_moveBodySelfIntersectionFaces(
    CGAL_Mesh& BodyMeshMapped,
    const std::set<CGAL_Mesh::Face_index>& intersectedFaces,
    Scalar moveDistance
);

void CGAL_avoidBodySelfIntersection(
    CGAL_Mesh& BodyMeshMapped
);

void CGAL_avoidClothSelfIntersection(
	CGAL_Mesh& clothmesh
);

void CGAL_getReducedMappedMesh(
    const CGAL_Mesh& clothmesh,
    const CGAL_Mesh& bodymesh, 
    const Scalar margin_x, 
    const Scalar margin_y, 
    const Scalar margin_z,
    CGAL_Mesh& BodyMeshMapped,
    std::vector<uint32_t>& bodymeshmapped_faceidx);

void CGAL_getAnimationMotionRate(
    const std::vector<uint32_t>& bodymeshmapped_faceidx,
    const std::vector<std::vector<Scalar3>>& bodyverts_total,
    const std::vector<std::vector<uint3>>& bodyfaces_total,
    const Scalar max_animation_motion_rate,
    std::vector<Scalar>& animation_motion_rate_total);

void CGAL_getSoftTargetConstraintsPoints(
    const CGAL_Mesh& total_mesh,
    std::vector<uint32_t>& boundary_vertex_indices
);

std::vector<int> CGAL_MergeDuplicateVertices(
	const CGAL_Mesh& orig_mesh, 
	CGAL_Mesh& new_mesh,
	const Scalar epsilon);

void CGAL_MergeMesh(
	CGAL_Mesh& totalmesh, 
	const CGAL_Mesh& mesh1, 
	const CGAL_Mesh& mesh2
);

bool saveMeshToOBJ(
	const std::vector<Scalar3> verts, 
	std::vector<uint3> faces, 
	const std::string& filename
);

CGAL_Point_3 CGAL_linear_interpolate(
	const CGAL_Point_3& p_start, 
	const CGAL_Point_3& p_end, 
	Scalar t
);

void CGAL_mesh_linear_interpolation(
    const CGAL_Mesh& bodymesh_tpose,
    const CGAL_Mesh& bodymesh_fstart,
    const int num_interpolations,
    std::vector<CGAL_Mesh>&  bodymesh_total
);


void CGAL_computeProjectionUVs(
    const std::vector<Scalar3>& clothVertices,
    const std::vector<uint3>& clothTriangles,
    const std::vector<Scalar3>& bodyVertices,
    const std::vector<uint3>& bodyTriangles,
    const std::vector<uint32_t>& targetIds,
    std::vector<uint32_t>& projectedFaceIds,
    std::vector<Scalar2>& projectedUVs
);

void CGAL_computeProjectedPositions(
    const std::vector<Scalar3>& bodyVertices,
    const std::vector<uint3>& bodyTriangles,
    const std::vector<uint32_t>& projectedFaceIds,
    const std::vector<Scalar2>& projectedUVs,
    Scalar offsetDistance,
    std::vector<Scalar3>& projectedPositions
);

void CGAL_moveBodyClothIntersectionFaces(
    CGAL_Mesh& bodyMesh,
    const std::set<CGAL_Mesh::Face_index>& intersectedFaces,
    Scalar moveDistance
);

bool CGAL_detectBodyClothIntersections(
    const CGAL_Mesh& clothMesh,
    CGAL_Mesh& bodyMesh,
    std::set<CGAL_Mesh::Face_index>& intersectedFaces
);

void CGAL_avoidBodyClothIntersections(
    const CGAL_Mesh& clothMesh,
    CGAL_Mesh& bodyMesh
);

void CGAL_RestoreOriginalVertices(
	const std::string& simplified_filename, 
	const std::vector<int>& index_mapping_orig2fuse, 
	const std::string& restored_filename
);


void CGAL_convertObjtoJson(
	const std::string& obj_file,
	const std::string& json_file
);


void CGAL_convertBodyObjtoJson(
    const std::string& obj_file_pattern,
    int framerange_start,
    int framerange_end,
    const std::string& json_file
);

void CGAL_readBodyJsontoMesh (
    const std::string& json_file,
    std::vector<CGAL_Mesh>& meshes
);

void CGAL_readSimJson_toMesh(
    const nlohmann::json& httpjson,
    CGAL_Mesh& clothmesh,
    CGAL_Mesh& bodymesh_tpose,
    std::vector<CGAL_Mesh>& bodymesh_total
);

void saveSurfaceMesh_restore_from_sort(
    const std::vector<uint32_t>& sortindex,
    std::vector<std::vector<Scalar3>>& output_clothverts_total
);

void saveSurfaceMesh_restore_from_fuse(
    const std::vector<uint3>& output_clothfaces,
    const std::vector<int>& index_mapping_orig2fuse,
    std::vector<std::vector<Scalar3>>& output_clothverts_total
);

void CGAL_saveClothMesh_toJson(
    const std::vector<std::vector<Scalar3>>& output_clothverts_total,
    const std::vector<uint3>& output_clothfaces,
    nlohmann::json& output_httpjson
);

void CGAL_readOutputClothJson_toMesh(
	const nlohmann::json& outputjson,
    std::vector<std::vector<Scalar3>>& output_clothverts_total,
    std::vector<uint3>& output_clothfaces
);

int getVertNeighbors(
    const int numSurfVerts,
	const int numTetElements,
    std::vector<std::vector<unsigned int>>& vertNeighbors,
    std::vector<unsigned int>& neighborList,
	std::vector<unsigned int>& neighborStart,
	std::vector<unsigned int>& neighborNum,
    std::vector<uint3>& triangles,
    std::vector<uint4>& tetrahedras);

void getSurface(
	const int numSurfVerts,
    std::vector<uint3>& surfFaceIds,
	std::vector<uint32_t>& surfVertIds,
	std::vector<uint2>& surfEdgeIds);

} // namespace LOADMESH
















class LoadObjMesh {

public:

	int vertexOffset;

	bool load_tetrahedraMesh(const std::string& filename, Scalar scale, Scalar3 position_offset);
	bool load_triMesh(const std::string& filename, Scalar scale, Scalar3 transform, int boundaryType);
	
	void getTetSurface();


	std::vector<uint32_t> surfId2TetId;

};
