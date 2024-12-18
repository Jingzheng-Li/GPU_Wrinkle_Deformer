

#include "Deformer.hpp"

#include "CUDAUtils.hpp"
#include "LoadMesh.hpp"



Deformer::Deformer(SimulationContext& context, Simulator& simulator)
    : ctx(context), sim(simulator) {}


void Deformer::DeformerPipeline() {
    getHostMesh(ctx.instance);
}

void Deformer::getHostMesh(std::unique_ptr<GeometryManager>& instance) {

    std::vector<Scalar3> bend_mesh_normals(instance->getHostNumClothVerts());
    CGAL_Mesh bend_mesh;
    LOADMESH::CGAL_readObj("../../Assets/tubemesh_bend.obj", bend_mesh);
    auto vn_map = bend_mesh.add_property_map<CGAL_Vertex_index, CGAL_Vector_3>("v:normal").first;
    CGAL::Polygon_mesh_processing::compute_vertex_normals(bend_mesh, vn_map);
    int temp_bendmesh_size = 0;
    for (auto v : bend_mesh.vertices()) {
        CGAL_Vector_3 normal = vn_map[v];
        bend_mesh_normals[temp_bendmesh_size] = make_Scalar3(normal.x(), normal.y(), normal.z());
        ++temp_bendmesh_size;
    }



    std::vector<Scalar3> clothmesh_verts(instance->getHostNumClothVerts());
    std::vector<Scalar3> clothmesh_restverts(instance->getHostNumClothVerts());
    std::vector<uint3> clothmesh_faces(instance->getHostNumClothFaces());
    std::vector<uint2> clothmesh_bendverts(instance->getHostNumTriBendEdges());
    std::vector<uint2> clothmesh_edges(instance->getHostNumSurfEdges()); // 这个就先这样 后面要把边单独拿出来
    

    CUDAMemcpyDToHSafe(clothmesh_verts, instance->getCudaSurfVertPos());
    CUDAMemcpyDToHSafe(clothmesh_faces, instance->getCudaSurfFaceIds());
    CUDAMemcpyDToHSafe(clothmesh_bendverts, instance->getCudaTriBendVerts());
    CUDAMemcpyDToHSafe(clothmesh_edges, instance->getCudaSurfEdgeIds());

    std::vector<uint2> clothmesh_constraints(clothmesh_bendverts.size() + clothmesh_edges.size()); 
    std::vector<Scalar> clothmesh_constraints_reslen(clothmesh_bendverts.size() + clothmesh_edges.size());
    std::copy(clothmesh_bendverts.begin(), clothmesh_bendverts.end(), clothmesh_constraints.begin());
    std::copy(clothmesh_edges.begin(), clothmesh_edges.end(), clothmesh_constraints.begin() + clothmesh_bendverts.size());

    for (int i = 0; i < clothmesh_constraints.size(); i++) {
        const uint2& cons = clothmesh_constraints[i];
        const Scalar3& vert1 = clothmesh_verts[cons.x];
        const Scalar3& vert2 = clothmesh_verts[cons.y];
        Scalar restlength = __MATHUTILS__::__vec3_norm(__MATHUTILS__::__vec3_minus(vert1, vert2));
        clothmesh_constraints_reslen[i] = restlength;
    }



    std::vector<Scalar3> clothmesh_dP(clothmesh_verts.size());
    std::vector<Scalar> clothmesh_dPw(clothmesh_verts.size());
    std::vector<Scalar3> clothmesh_L(clothmesh_constraints.size());
    std::vector<Scalar3> clothmesh_dL(clothmesh_constraints.size());
    std::cout << "clothmesh_L size: " << clothmesh_L.size() << std::endl;




    Scalar stretchstiffness = 10.0;
    Scalar compressstiffness = 10.0;
    Scalar restlengthscale = 1.0;
    Scalar normalinset = 0.1;



    for (int i = 0; i < 200; i++) {
        std::fill(clothmesh_L.begin(), clothmesh_L.end(), 0);
        
    }










}



