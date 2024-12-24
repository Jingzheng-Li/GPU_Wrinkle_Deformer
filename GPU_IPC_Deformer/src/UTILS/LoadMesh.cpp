

#include <iostream>
#include <unordered_map>
#include <fstream>
#include <set>
#include <queue>
#include <map>
#include<iostream>
#include <cfloat>
#include <cstring>

#include "LoadMesh.hpp"

namespace LOADMESH {

using json = nlohmann::json;

bool CGAL_readObj(const std::string& filename, CGAL_Mesh& mesh) {
	mesh.clear();
    if (!CGAL::IO::read_polygon_mesh(filename, mesh)) {
        std::cerr << "Error: failed to read cgal obj file" << filename << std::endl;
        return false;
    } else {
        std::cout << "successfully read cgal obj file" << filename << std::endl;
        return true;
    }
}


bool CGAL_writeObj(const std::string& filename, const CGAL_Mesh& mesh) {
    if (!CGAL::IO::write_polygon_mesh(filename, mesh)) {
        std::cerr << "Error: failed to write cgal obj file" << filename << std::endl;
        return false;
    } else {
        std::cout << "successfully write cgal obj file" << filename << std::endl;
        return true;
    }
}

void convertCGALVertsToVector(const CGAL_Mesh& mesh, std::vector<Scalar3>& verts) {
    verts.clear();
    for (CGAL_Vertex_index v : mesh.vertices()) {
        CGAL_Point_3 p = mesh.point(v);
		verts.emplace_back(make_Scalar3(p.x(), p.y(), p.z()));
    }
}

void convertCGALFacesToVector(const CGAL_Mesh& mesh, std::vector<uint3>& faces) {
    faces.clear();
    for (CGAL_Face_index f : mesh.faces()) {
        std::vector<unsigned int> faceVertices;
        for (CGAL_Vertex_index v : vertices_around_face(mesh.halfedge(f), mesh)) {
            faceVertices.push_back(v.idx());
        }
        if (faceVertices.size() == 3) {
            faces.emplace_back(make_uint3(faceVertices[0], faceVertices[1], faceVertices[2]));
        }
    }
}


void extractTriBendEdgesFaces_CGAL(const CGAL_Mesh& mesh, std::vector<uint2>& triBendEdges, std::vector<uint2>& triBendVerts) {
    triBendEdges.clear();
    triBendVerts.clear();
    triBendEdges.reserve(mesh.number_of_edges());
    triBendVerts.reserve(mesh.number_of_edges());

    auto find_third_vertex = [&](const CGAL_Face_index& face, const CGAL_Vertex_index& src, const CGAL_Vertex_index& tgt) -> int {
        if (face == mesh.null_face()) {
            std::cout << "wrong bending pairs" << std::endl;
            exit(EXIT_FAILURE);
        }
        auto he = mesh.halfedge(face);
        do {
            auto v = mesh.target(he);
            if (v != src && v != tgt) {
                return static_cast<int>(v);
            }
            he = mesh.next(he);
        } while (he != mesh.halfedge(face));
        std::cout << "wrong bending pairs" << std::endl;
        exit(EXIT_FAILURE);
    };

    for (auto edge : mesh.edges()) {
        auto halfedge1 = mesh.halfedge(edge, 0);
        auto halfedge2 = mesh.halfedge(edge, 1);

        bool is_border1 = mesh.is_border(halfedge1);
        bool is_border2 = mesh.is_border(halfedge2);

        if (!is_border1 && !is_border2) {
            auto source = mesh.source(halfedge1);
            auto target = mesh.target(halfedge1);
            unsigned int source_id = static_cast<unsigned int>(source);
            unsigned int target_id = static_cast<unsigned int>(target);

            if (source_id > target_id) std::swap(source_id, target_id);
            triBendEdges.emplace_back(make_uint2(source_id, target_id));

            auto face1 = mesh.face(halfedge1);
            auto face2 = mesh.face(halfedge2);
            int third_vertex1 = find_third_vertex(face1, source, target);
            int third_vertex2 = find_third_vertex(face2, source, target);

            triBendVerts.emplace_back(make_uint2(third_vertex1, third_vertex2));
        }
    }

    size_t n = triBendEdges.size();
    std::vector<size_t> indices(n);
    for (size_t i = 0; i < n; ++i) indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) -> bool {
        if (triBendEdges[a].x != triBendEdges[b].x)
            return triBendEdges[a].x < triBendEdges[b].x;
        return triBendEdges[a].y < triBendEdges[b].y;
    });

    std::vector<uint2> sortedTriBendEdges;
    std::vector<uint2> sortedTriBendVerts;
    sortedTriBendEdges.reserve(n);
    sortedTriBendVerts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        size_t idx = indices[i];
        sortedTriBendEdges.emplace_back(triBendEdges[idx]);
        sortedTriBendVerts.emplace_back(triBendVerts[idx]);
    }

    triBendEdges = std::move(sortedTriBendEdges);
    triBendVerts = std::move(sortedTriBendVerts);
}




bool CGAL_detectBodySelfIntersections(
    CGAL_Mesh& BodyMeshMapped,
    std::set<CGAL_Face_index>& intersectedFaces
) {
    // 使用 CGAL 的 AABB 树来加速自交检测
    CGAL::AABB_tree<CGAL::AABB_traits<CGAL_Kernel, CGAL::AABB_face_graph_triangle_primitive<CGAL_Mesh>>> tree(faces(BodyMeshMapped).first, faces(BodyMeshMapped).second, BodyMeshMapped);
    tree.accelerate_distance_queries();
    intersectedFaces.clear();

    if (BodyMeshMapped.is_empty()) {
        std::cerr << "Error: BodyMeshMapped is empty." << std::endl;
        return false;
    }

    std::vector<std::pair<CGAL_Face_index, CGAL_Face_index>> intersecting_pairs;
    CGAL::Polygon_mesh_processing::self_intersections(BodyMeshMapped, std::back_inserter(intersecting_pairs));

    for (const auto& pair : intersecting_pairs) {
        intersectedFaces.insert(pair.first);
        intersectedFaces.insert(pair.second);
    }

    return !intersecting_pairs.empty();
}

void CGAL_moveBodySelfIntersectionFaces(
    CGAL_Mesh& BodyMeshMapped,
    const std::set<CGAL_Mesh::Face_index>& intersectedFaces,
    Scalar moveDistance
) {

    for (const auto& face : intersectedFaces) {
        auto v0 = BodyMeshMapped.point(target(halfedge(face, BodyMeshMapped), BodyMeshMapped));
        auto v1 = BodyMeshMapped.point(target(next(halfedge(face, BodyMeshMapped), BodyMeshMapped), BodyMeshMapped));
        auto v2 = BodyMeshMapped.point(target(next(next(halfedge(face, BodyMeshMapped), BodyMeshMapped), BodyMeshMapped), BodyMeshMapped));

        CGAL_Vector_3 normal = CGAL::Polygon_mesh_processing::compute_face_normal(face, BodyMeshMapped);

        BodyMeshMapped.point(target(halfedge(face, BodyMeshMapped), BodyMeshMapped)) = v0 + normal * moveDistance;
        BodyMeshMapped.point(target(next(halfedge(face, BodyMeshMapped), BodyMeshMapped), BodyMeshMapped)) = v1 + normal * moveDistance;
        BodyMeshMapped.point(target(next(next(halfedge(face, BodyMeshMapped), BodyMeshMapped), BodyMeshMapped), BodyMeshMapped)) = v2 + normal * moveDistance;
    }
}


void CGAL_avoidBodySelfIntersection(
    CGAL_Mesh& BodyMeshMapped
) {
    // std::set<CGAL_Face_index> intersectedFaces;
    // Scalar moveDistance = -0.0005; // 每次移动的距离，正值表示沿法线方向移动
    // int maxIterations = 5; // 最大迭代次数
    // int currentIteration = 0;

    // CGAL_Mesh O_BodyMeshMapped = BodyMeshMapped;
    // while (currentIteration < maxIterations) {
    //     bool hasIntersections = CGAL_detectBodySelfIntersections(BodyMeshMapped, intersectedFaces);
    //     if (!hasIntersections) {
    //         break; // 无更多交叉，退出循环
    //     }
    //     std::cout << "Iteration " << currentIteration + 1 << ": " << intersectedFaces.size() << " intersected faces found." << std::endl;

    //     // 移动交叉的面
    //     CGAL_moveBodySelfIntersectionFaces(BodyMeshMapped, intersectedFaces, moveDistance);
    //     currentIteration++;
    // }

    // if (currentIteration == maxIterations) {
    //     std::cout << "Max iterations reached. Some intersections may remain." << std::endl;
    //     BodyMeshMapped = O_BodyMeshMapped;
    // } else {
    //     std::cout << "No more intersections after " << currentIteration << " iterations." << std::endl;
    // }

}

void CGAL_avoidClothSelfIntersection(
    CGAL_Mesh& clothmesh
) {
    
}



void CGAL_moveBodyClothIntersectionFaces(
    CGAL_Mesh& bodyMesh,
    const std::set<CGAL_Mesh::Face_index>& intersectedFaces,
    Scalar moveDistance
) {
    for (const auto& face : intersectedFaces) {
        auto v0 = bodyMesh.point(target(halfedge(face, bodyMesh), bodyMesh));
        auto v1 = bodyMesh.point(target(next(halfedge(face, bodyMesh), bodyMesh), bodyMesh));
        auto v2 = bodyMesh.point(target(next(next(halfedge(face, bodyMesh), bodyMesh), bodyMesh), bodyMesh));

        CGAL_Vector_3 normal = CGAL::Polygon_mesh_processing::compute_face_normal(face, bodyMesh);

        bodyMesh.point(target(halfedge(face, bodyMesh), bodyMesh)) = v0 + normal * moveDistance;
        bodyMesh.point(target(next(halfedge(face, bodyMesh), bodyMesh), bodyMesh)) = v1 + normal * moveDistance;
        bodyMesh.point(target(next(next(halfedge(face, bodyMesh), bodyMesh), bodyMesh), bodyMesh)) = v2 + normal * moveDistance;
    }
}

bool CGAL_detectBodyClothIntersections(
    const CGAL_Mesh& clothMesh,
    CGAL_Mesh& bodyMesh,
    std::set<CGAL_Mesh::Face_index>& intersectedFaces
) {
    // 使用AABB树加速检测人体网格
    CGAL_Tree tree(faces(bodyMesh).first, faces(bodyMesh).second, bodyMesh);
    tree.accelerate_distance_queries();

    intersectedFaces.clear(); // 每次清空交点信息

    for (auto face : faces(clothMesh)) {
        auto v0 = clothMesh.point(target(halfedge(face, clothMesh), clothMesh));
        auto v1 = clothMesh.point(target(next(halfedge(face, clothMesh), clothMesh), clothMesh));
        auto v2 = clothMesh.point(target(next(next(halfedge(face, clothMesh), clothMesh), clothMesh), clothMesh));

        CGAL_Triangle_3 clothTri(v0, v1, v2);

        if (tree.do_intersect(clothTri)) {
            std::vector<CGAL_Mesh::Face_index> faceIntersections;
            tree.all_intersected_primitives(clothTri, std::back_inserter(faceIntersections));

            intersectedFaces.insert(faceIntersections.begin(), faceIntersections.end());
        }
    }

    return !intersectedFaces.empty(); // 返回是否有交点
}

void CGAL_avoidBodyClothIntersections(
    const CGAL_Mesh& clothMesh,
    CGAL_Mesh& bodyMesh
) {
    std::set<CGAL_Mesh::Face_index> intersectedFaces;
    Scalar moveDistance = -0.001; // 每次移动的距离
    int maxIterations = 100; // 最大迭代次数
    int currentIteration = 0;

    // 开始迭代，直到没有交点或者达到最大迭代次数
    while (currentIteration < maxIterations && CGAL_detectBodyClothIntersections(clothMesh, bodyMesh, intersectedFaces)) {
        std::cout << "Iteration " << currentIteration + 1 << ": " << intersectedFaces.size() << " intersected faces found." << std::endl;

        // 移动相交的面
        CGAL_moveBodyClothIntersectionFaces(bodyMesh, intersectedFaces, moveDistance);
        currentIteration++;
    }

    if (currentIteration == maxIterations) {
        std::cout << "Max iterations reached. Some intersections may remain." << std::endl;
    } else {
        std::cout << "No more intersections after " << currentIteration << " iterations." << std::endl;
    }
}

void CGAL_getReducedMappedMesh(
    const CGAL_Mesh& clothmesh,
    const CGAL_Mesh& bodymesh,
    const Scalar margin_x,
    const Scalar margin_y,
    const Scalar margin_z,
    CGAL_Mesh& BodyMeshMapped,
    std::vector<uint32_t>& bodymeshmapped_faceidx) {
    // 1. 计算 clothmesh 的包围盒
    CGAL::Bbox_3 bbox = CGAL::Polygon_mesh_processing::bbox(clothmesh);

    // 2. 扩展包围盒
    bbox = CGAL::Bbox_3(
        bbox.xmin() - margin_x, bbox.ymin() - margin_y, bbox.zmin() - margin_z,
        bbox.xmax() + margin_x, bbox.ymax() + margin_y, bbox.zmax() + margin_z
    );

    // 3. 为 bodymesh 的面片添加索引
    CGAL_Mesh bodymesh_with_indices = bodymesh;
    auto face_id_map = bodymesh_with_indices.add_property_map<CGAL_Face_index, std::size_t>("f:idx").first;
    std::size_t idx = 0;
    for (auto f : faces(bodymesh_with_indices)) {
        face_id_map[f] = idx++;
    }

    // 4. 遍历 bodymesh 的面片，筛选出位于包围盒内的面片
    std::map<CGAL_Point_3, CGAL_Vertex_index> vertex_map;
    for (auto f : faces(bodymesh_with_indices)) {
        bool inside = false;
        for (auto v : vertices_around_face(bodymesh_with_indices.halfedge(f), bodymesh_with_indices)) {
            CGAL_Point_3 p = bodymesh_with_indices.point(v);
            if (p.x() >= bbox.xmin() && p.x() <= bbox.xmax() &&
                p.y() >= bbox.ymin() && p.y() <= bbox.ymax() &&
                p.z() >= bbox.zmin() && p.z() <= bbox.zmax()) {
                inside = true;
                break;
            }
        }
        if (inside) {
            // 添加面片到 BodyMeshMapped
            std::vector<CGAL_Vertex_index> face_vertices;
            for (auto v : vertices_around_face(bodymesh_with_indices.halfedge(f), bodymesh_with_indices)) {
                CGAL_Point_3 p = bodymesh_with_indices.point(v);
                auto res = vertex_map.insert(std::make_pair(p, CGAL_Vertex_index()));
                if (res.second) {
                    // 顶点尚未添加到 BodyMeshMapped
                    CGAL_Vertex_index new_v = BodyMeshMapped.add_vertex(p);
                    res.first->second = new_v;
                }
                face_vertices.push_back(res.first->second);
            }
            BodyMeshMapped.add_face(face_vertices);
            // 保存原始面片的索引
            bodymeshmapped_faceidx.push_back(face_id_map[f]);
        }
    }
}


void CGAL_getAnimationMotionRate(
    const std::vector<uint32_t>& bodymeshmapped_faceidx,
    const std::vector<std::vector<Scalar3>>& bodyverts_total,
    const std::vector<std::vector<uint3>>& bodyfaces_total,
    const Scalar max_animation_motion_rate,
    std::vector<Scalar>& animation_motion_rate_total)
{
    // 获取帧数
    size_t num_frames = bodyverts_total.size();

    // 初始化结果向量
    animation_motion_rate_total.resize(num_frames, 1); // 默认等级为1

    // 假设拓扑结构在所有帧中保持不变，因此只需要获取一次子网格的顶点索引
    // 1. 收集子网格中涉及的顶点索引
    std::set<uint32_t> submesh_vertex_indices;
    const std::vector<uint3>& faces = bodyfaces_total[0]; // 取第一帧的面列表

    for (uint32_t face_idx : bodymeshmapped_faceidx) {
        const uint3& face = faces[face_idx];
        submesh_vertex_indices.insert(face.x);
        submesh_vertex_indices.insert(face.y);
        submesh_vertex_indices.insert(face.z);
    }

    // 将顶点索引集合转换为向量，便于后续操作
    std::vector<uint32_t> submesh_vertex_idx_list(submesh_vertex_indices.begin(), submesh_vertex_indices.end());

    // 2. 计算每一帧之间的变化率
    // 存储每一帧的平均运动量
    std::vector<Scalar> avg_motions(num_frames, 0.0);

    for (size_t i = 1; i < num_frames; ++i) {
        const std::vector<Scalar3>& verts_prev = bodyverts_total[i - 1]; // 前一帧的顶点
        const std::vector<Scalar3>& verts_curr = bodyverts_total[i];     // 当前帧的顶点

        Scalar sum_sq_dist = 0.0; // 子网格顶点的总平方距离

        // 对于子网格中的每个顶点，计算在两帧之间的位移平方和
        for (uint32_t v_idx : submesh_vertex_idx_list) {
            const Scalar3& v_prev = verts_prev[v_idx];
            const Scalar3& v_curr = verts_curr[v_idx];

            Scalar dx = v_curr.x - v_prev.x;
            Scalar dy = v_curr.y - v_prev.y;
            Scalar dz = v_curr.z - v_prev.z;

            Scalar dist_sq = dx * dx + dy * dy + dz * dz;

            sum_sq_dist += dist_sq;
        }

        // 计算平均运动量（每个顶点的平均位移平方和）
        Scalar avg_motion = sum_sq_dist / submesh_vertex_idx_list.size();

        avg_motions[i] = avg_motion;
    }

    // 3. 将平均运动量映射到等级（1 到 max_animation_motion_rate）
    // 首先找到最大运动量，便于归一化
    Scalar max_motion = *std::max_element(avg_motions.begin() + 1, avg_motions.end()); // 跳过第0帧

    if (max_motion == 0.0) {
        // 如果最大运动量为0，说明没有运动，将所有等级设为1
        std::fill(animation_motion_rate_total.begin(), animation_motion_rate_total.end(), 1);
    } else {
        for (size_t i = 1; i < num_frames; ++i) {
            // 归一化平均运动量到[0,1]
            Scalar normalized_motion = avg_motions[i] / max_motion;

            // 映射到等级，使用ceil确保整数部分向上取整
            uint32_t level = static_cast<uint32_t>(std::ceil(normalized_motion * (max_animation_motion_rate - 1))) + 1;

            // 确保等级在1到max_animation_motion_rate之间
            if (level > max_animation_motion_rate) level = max_animation_motion_rate;
            if (level < 1) level = 1;

            animation_motion_rate_total[i] = level;
        }

        // 第0帧的变化率设为1
        animation_motion_rate_total[0] = 1;
    }
}

std::vector<int> CGAL_MergeDuplicateVertices(
	const CGAL_Mesh& orig_mesh, 
	CGAL_Mesh& new_mesh,
	const Scalar epsilon) {

    // 获取所有顶点并存储在向量中
    std::vector<CGAL_Vertex_index> all_vertices;
    for(auto v : orig_mesh.vertices()) {
        all_vertices.push_back(v);
    }

    // 初始化索引映射，默认值为 -1
    std::vector<int> index_mapping_orig2fuse(all_vertices.size(), -1);

    // 创建一个向量存储点及其原始索引
    std::vector<std::pair<CGAL_Point_3, int>> points_with_indices;
    points_with_indices.reserve(all_vertices.size());
    for(int i = 0; i < all_vertices.size(); ++i){
        points_with_indices.emplace_back(orig_mesh.point(all_vertices[i]), i);
    }

    // 按照 x, y, z 坐标对点进行排序
    std::sort(points_with_indices.begin(), points_with_indices.end(),
        [&](const std::pair<CGAL_Point_3, int>& a, const std::pair<CGAL_Point_3, int>& b) -> bool {
            if (a.first.x() != b.first.x()) return a.first.x() < b.first.x();
            if (a.first.y() != b.first.y()) return a.first.y() < b.first.y();
            return a.first.z() < b.first.z();
        });

    // 创建一个向量存储唯一顶点的原始索引
    std::vector<int> unique_vertex_indices;
    unique_vertex_indices.reserve(all_vertices.size());

    // 初始化第一个顶点为唯一顶点
    unique_vertex_indices.push_back(points_with_indices[0].second);
    index_mapping_orig2fuse[points_with_indices[0].second] = 0;

    // 当前唯一顶点计数
    int new_index = 0;

    // 遍历排序后的顶点，合并重合顶点
    for(int i = 1; i < points_with_indices.size(); ++i){
        const CGAL_Point_3& current = points_with_indices[i].first;
        const CGAL_Point_3& previous = points_with_indices[i-1].first;

        // 计算两点之间的欧氏距离平方
        Scalar dx = current.x() - previous.x();
        Scalar dy = current.y() - previous.y();
        Scalar dz = current.z() - previous.z();
        Scalar dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq <= epsilon * epsilon){
            // 如果距离小于等于 epsilon，则认为是重复顶点
            index_mapping_orig2fuse[points_with_indices[i].second] = new_index;
        }
        else{
            // 否则，认为是新的唯一顶点
            new_index++;
            unique_vertex_indices.push_back(points_with_indices[i].second);
            index_mapping_orig2fuse[points_with_indices[i].second] = new_index;
        }
    }

    // 创建一个新的网格用于存储合并后的顶点和面
    std::vector<CGAL_Vertex_index> new_vertex_indices;
    new_vertex_indices.reserve(unique_vertex_indices.size());

    // 将唯一顶点添加到新的网格中，并记录其索引
    for(auto old_idx : unique_vertex_indices){
        new_vertex_indices.push_back(new_mesh.add_vertex(orig_mesh.point(all_vertices[old_idx])));
    }

    // 遍历原始网格中的所有面，并在新网格中重新添加面
    for(auto f : orig_mesh.faces()){
        std::vector<CGAL_Vertex_index> face_vertices;
        // 获取面上的所有顶点
        for(auto v : CGAL::vertices_around_face(orig_mesh.halfedge(f), orig_mesh)){
            // 找到旧顶点的索引
            int old_idx = std::distance(all_vertices.begin(),
                                        std::find(all_vertices.begin(), all_vertices.end(), v));
            // 获取新顶点的索引
            int mapped_idx = index_mapping_orig2fuse[old_idx];
            face_vertices.push_back(new_vertex_indices[mapped_idx]);
        }
        // 添加面到新的网格中
        new_mesh.add_face(face_vertices);
    }

    // 返回旧顶点到新顶点的索引映射
    return index_mapping_orig2fuse;
}


void CGAL_getSoftTargetConstraintsPoints(
    const CGAL_Mesh& total_mesh,
    std::vector<uint32_t>& boundary_vertex_indices
) {
    // Nested function to compute the average y-coordinate of a boundary cycle
    auto compute_average_y = [&total_mesh](const std::vector<uint32_t>& cycle) -> Scalar {
        Scalar sum_y = 0.0;
        for (uint32_t vertex_id : cycle) {
            // Directly use the vertex index without recreating it
            sum_y += total_mesh.point(CGAL_Vertex_index(vertex_id)).y();
        }
        return sum_y / static_cast<Scalar>(cycle.size());
    };

    // Compute connected components and get component IDs for faces
    using FaceIndexMap = boost::property_map<CGAL_Mesh, boost::face_index_t>::const_type;
    FaceIndexMap face_index_map = get(boost::face_index, total_mesh);

    using FaceComponentMap = boost::vector_property_map<int, FaceIndexMap>;
    FaceComponentMap face_component_map(num_faces(total_mesh), face_index_map);

    int num_components = CGAL::Polygon_mesh_processing::connected_components(
        total_mesh, face_component_map
    );

    // Collect boundary cycles
    std::vector<CGAL_Mesh::Halfedge_index> boundary_cycles;
    CGAL::Polygon_mesh_processing::extract_boundary_cycles(
        total_mesh, std::back_inserter(boundary_cycles)
    );

    // Map component ID to its boundary cycles (vertex indices)
    std::vector<std::vector<std::vector<uint32_t>>> component_boundary_cycles(num_components);

    for (CGAL_Mesh::Halfedge_index halfedge : boundary_cycles) {
        // Get component ID by checking the face adjacent to the opposite halfedge
        CGAL_Mesh::Face_index adjacent_face = face(opposite(halfedge, total_mesh), total_mesh);
        int component_id = face_component_map[adjacent_face];

        // Collect vertex indices in the cycle
        std::vector<uint32_t> vertex_indices_in_cycle;
        CGAL_Mesh::Halfedge_index start_halfedge = halfedge;
        do {
            CGAL_Mesh::Vertex_index vertex_index = target(halfedge, total_mesh);
            vertex_indices_in_cycle.push_back(vertex_index);
            halfedge = next(halfedge, total_mesh);
        } while (halfedge != start_halfedge);

        // Move the cycle into the corresponding component's list to avoid copying
        component_boundary_cycles[component_id].push_back(std::move(vertex_indices_in_cycle));
    }

    // For each component, find the boundary cycle with the maximum average y-coordinate
    for (int component_id = 0; component_id < num_components; ++component_id) {
        const auto& boundary_cycles = component_boundary_cycles[component_id];
        if (boundary_cycles.empty()) {
            std::cout << "No boundary cycles found in component " << component_id << std::endl;
            continue;
        }

        Scalar max_average_y = -std::numeric_limits<Scalar>::infinity();
        const std::vector<uint32_t>* max_cycle = nullptr;

        for (const auto& cycle : boundary_cycles) {
            Scalar average_y = compute_average_y(cycle);
            if (average_y > max_average_y) {
                max_average_y = average_y;
                max_cycle = &cycle;
            }
        }

        if (max_cycle) {
            // Add the max_cycle's vertex indices to the boundary_vertex_indices
            boundary_vertex_indices.insert(
                boundary_vertex_indices.end(),
                max_cycle->begin(),
                max_cycle->end()
            );
        } else {
            std::cout << "No cycles found in component " << component_id << std::endl;
        }
    }

}




void CGAL_MergeMesh(CGAL_Mesh& totalmesh, const CGAL_Mesh& mesh1, const CGAL_Mesh& mesh2) {
    // 清空 totalmesh 确保它是空的
    totalmesh.clear();
    
    // 1. 复制 mesh1 的顶点和面到 totalmesh
    std::map<CGAL_Vertex_index, CGAL_Vertex_index> vmap1;
    for (auto v : mesh1.vertices()) {
        CGAL_Vertex_index v_new = totalmesh.add_vertex(mesh1.point(v));
        vmap1[v] = v_new; // 记录顶点的映射
    }
    for (auto f : mesh1.faces()) {
        std::vector<CGAL_Vertex_index> vertices;
        for (auto v : CGAL::vertices_around_face(mesh1.halfedge(f), mesh1)) {
            vertices.push_back(vmap1[v]);
        }
        totalmesh.add_face(vertices);
    }

    // 2. 复制 mesh2 的顶点和面到 totalmesh，注意顶点的索引需要偏移
    std::map<CGAL_Vertex_index, CGAL_Vertex_index> vmap2;
    for (auto v : mesh2.vertices()) {
        CGAL_Vertex_index v_new = totalmesh.add_vertex(mesh2.point(v));
        vmap2[v] = v_new; // 记录顶点的映射
    }
    for (auto f : mesh2.faces()) {
        std::vector<CGAL_Vertex_index> vertices;
        for (auto v : CGAL::vertices_around_face(mesh2.halfedge(f), mesh2)) {
            vertices.push_back(vmap2[v]);
        }
        totalmesh.add_face(vertices);
    }
}



bool saveMeshToOBJ(const std::vector<Scalar3> verts, std::vector<uint3> faces, const std::string& filename) {
    std::ofstream objFile(filename);
    if (!objFile.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        return false;
    }

    // 写入顶点
    for (const auto& vertex : verts) {
        objFile << "v " << vertex.x << " " << vertex.y << " " << vertex.z << "\n";
    }

    // 写入面（假设索引从0开始，需要转换为从1开始）
    for (const auto& face : faces) {
        objFile << "f " << (face.x + 1) << " " << (face.y + 1) << " " << (face.z + 1) << "\n";
    }

    objFile.close();
    std::cout << "OBJ saved in " << filename << std::endl;
    return true;
}


CGAL_Point_3 CGAL_linear_interpolate(const CGAL_Point_3& p_start, const CGAL_Point_3& p_end, Scalar t) {
    return CGAL_Point_3(
        (1 - t) * p_start.x() + t * p_end.x(),
        (1 - t) * p_start.y() + t * p_end.y(),
        (1 - t) * p_start.z() + t * p_end.z()
    );
}

void CGAL_mesh_linear_interpolation(
    const CGAL_Mesh& bodymesh_tpose,
    const CGAL_Mesh& bodymesh_fstart,
    const int num_interpolations,
    std::vector<CGAL_Mesh>& bodymesh_total
) {
    size_t num_vertices = CGAL::num_vertices(bodymesh_fstart);
    std::vector<CGAL_Point_3> vertices_start;
    std::vector<CGAL_Point_3> vertices_end;

    // 读取起始和结束网格的点
    for (auto v : bodymesh_fstart.vertices()) {
        vertices_start.push_back(bodymesh_fstart.point(v));
    }
    for (auto v : bodymesh_tpose.vertices()) {
        vertices_end.push_back(bodymesh_tpose.point(v));
    }

    // 插值过程
    for (int frame = 1; frame <= num_interpolations; ++frame) {
        Scalar t = static_cast<Scalar>(frame) / num_interpolations;
        CGAL_Mesh interpolated_mesh = bodymesh_fstart;

        size_t idx = 0;
        for (auto v : interpolated_mesh.vertices()) {
            CGAL_Point_3 p_start = vertices_start[idx];
            CGAL_Point_3 p_end = vertices_end[idx];
            CGAL_Point_3 p_interp = LOADMESH::CGAL_linear_interpolate(p_start, p_end, t);
            interpolated_mesh.point(v) = p_interp;
            ++idx;
        }

        // 将插值后的mesh插入到bodymesh_total中
        bodymesh_total.insert(bodymesh_total.begin(), interpolated_mesh);

    }

}



void CGAL_computeProjectionUVs(
    const std::vector<Scalar3>& clothVertices,
    const std::vector<uint3>& clothTriangles,
    const std::vector<Scalar3>& bodyVertices,
    const std::vector<uint3>& bodyTriangles,
    const std::vector<uint32_t>& targetIds,
    std::vector<uint32_t>& projectedFaceIds,
    std::vector<Scalar2>& projectedUVs)
{
    // 辅助函数，用于计算点到三角形的最近点和对应的重心坐标
    auto ClosestPointOnTriangle = [](const Scalar3& p, const Scalar3& a, const Scalar3& b, const Scalar3& c, Scalar& u, Scalar& v, Scalar& w) -> Scalar3 {
        // 计算向量
        Scalar3 ab = make_Scalar3(b.x - a.x, b.y - a.y, b.z - a.z);
        Scalar3 ac = make_Scalar3(c.x - a.x, c.y - a.y, c.z - a.z);
        Scalar3 ap = make_Scalar3(p.x - a.x, p.y - a.y, p.z - a.z);

        // 计算点积
        Scalar d1 = ab.x * ap.x + ab.y * ap.y + ab.z * ap.z;
        Scalar d2 = ac.x * ap.x + ac.y * ap.y + ac.z * ap.z;
        if (d1 <= 0.0 && d2 <= 0.0) {
            u = 1.0; v = 0.0; w = 0.0;
            return a;
        }

        // 检查是否在顶点B的区域外
        Scalar3 bp = make_Scalar3(p.x - b.x, p.y - b.y, p.z - b.z);
        Scalar d3 = ab.x * bp.x + ab.y * bp.y + ab.z * bp.z;
        Scalar d4 = ac.x * bp.x + ac.y * bp.y + ac.z * bp.z;
        if (d3 >= 0.0 && d4 <= d3) {
            u = 0.0; v = 1.0; w = 0.0;
            return b;
        }

        // 检查是否在边AB上
        Scalar vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            Scalar v1 = d1 / (d1 - d3);
            u = 1.0 - v1; v = v1; w = 0.0;
            return make_Scalar3(a.x + ab.x * v1, a.y + ab.y * v1, a.z + ab.z * v1);
        }

        // 检查是否在顶点C的区域外
        Scalar3 cp = make_Scalar3(p.x - c.x, p.y - c.y, p.z - c.z);
        Scalar d5 = ab.x * cp.x + ab.y * cp.y + ab.z * cp.z;
        Scalar d6 = ac.x * cp.x + ac.y * cp.y + ac.z * cp.z;
        if (d6 >= 0.0 && d5 <= d6) {
            u = 0.0; v = 0.0; w = 1.0;
            return c;
        }

        // 检查是否在边AC上
        Scalar vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            Scalar w1 = d2 / (d2 - d6);
            u = 1.0 - w1; v = 0.0; w = w1;
            return make_Scalar3(a.x + ac.x * w1, a.y + ac.y * w1, a.z + ac.z * w1);
        }

        // 检查是否在边BC上
        Scalar va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            Scalar w1 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            u = 0.0; v = 1.0 - w1; w = w1;
            Scalar3 bc = make_Scalar3(c.x - b.x, c.y - b.y, c.z - b.z);
            return make_Scalar3(b.x + bc.x * w1, b.y + bc.y * w1, b.z + bc.z * w1);
        }

        // 在面内部，计算重心坐标
        Scalar denom = 1.0 / (va + vb + vc);
        v = vb * denom;
        w = vc * denom;
        u = 1.0 - v - w;
        return make_Scalar3(
            a.x * u + b.x * v + c.x * w,
            a.y * u + b.y * v + c.y * w,
            a.z * u + b.z * v + c.z * w
        );
    };

	projectedFaceIds.clear();
	projectedUVs.clear();
    projectedFaceIds.resize(targetIds.size());
    projectedUVs.resize(targetIds.size());

    // 对于每个目标点
    for (size_t idx = 0; idx < targetIds.size(); ++idx) {
        uint32_t targetId = targetIds[idx];
        const Scalar3& targetPoint = clothVertices[targetId];

        Scalar minDistSq = 1e10;
        uint32_t bestFaceId = 0;
        Scalar bestU = 0.0, bestV = 0.0, bestW = 0.0;

        // 遍历所有body的三角形
        for (uint32_t faceId = 0; faceId < bodyTriangles.size(); ++faceId) {
            const uint3& tri = bodyTriangles[faceId];
            const Scalar3& a = bodyVertices[tri.x];
            const Scalar3& b = bodyVertices[tri.y];
            const Scalar3& c = bodyVertices[tri.z];

            Scalar u, v, w;
            Scalar3 closestPoint = ClosestPointOnTriangle(targetPoint, a, b, c, u, v, w);
            Scalar dx = closestPoint.x - targetPoint.x;
            Scalar dy = closestPoint.y - targetPoint.y;
            Scalar dz = closestPoint.z - targetPoint.z;
            Scalar distSq = dx * dx + dy * dy + dz * dz;

            if (distSq < minDistSq) {
                minDistSq = distSq;
                bestFaceId = faceId;
                bestU = u;
                bestV = v;
                bestW = w;
            }
        }

        // 记录最佳三角形和UV坐标
        projectedFaceIds[idx] = bestFaceId;
        projectedUVs[idx] = make_Scalar2(bestV, bestW); // 使用重心坐标(v, w)作为UV
    }
}

void CGAL_computeProjectedPositions(
    const std::vector<Scalar3>& bodyVertices,
    const std::vector<uint3>& bodyTriangles,
    const std::vector<uint32_t>& projectedFaceIds,
    const std::vector<Scalar2>& projectedUVs,
    Scalar offsetDistance,
    std::vector<Scalar3>& projectedPositions
) {
	projectedPositions.clear();
    projectedPositions.resize(projectedFaceIds.size());

    for (size_t idx = 0; idx < projectedFaceIds.size(); ++idx) {
        uint32_t faceId = projectedFaceIds[idx];
        const uint3& tri = bodyTriangles[faceId];

        const Scalar3& a = bodyVertices[tri.x];
        const Scalar3& b = bodyVertices[tri.y];
        const Scalar3& c = bodyVertices[tri.z];

        // 获取重心坐标
        Scalar v = projectedUVs[idx].x;
        Scalar w = projectedUVs[idx].y;
        Scalar u = 1.0 - v - w;

        // 计算投影点的位置
        Scalar3 projectedPoint = make_Scalar3(
            a.x * u + b.x * v + c.x * w,
            a.y * u + b.y * v + c.y * w,
            a.z * u + b.z * v + c.z * w
        );

        // 计算三角形的法线
        Scalar3 ab = make_Scalar3(b.x - a.x, b.y - a.y, b.z - a.z);
        Scalar3 ac = make_Scalar3(c.x - a.x, c.y - a.y, c.z - a.z);
        Scalar3 normal = make_Scalar3(
            ab.y * ac.z - ab.z * ac.y,
            ab.z * ac.x - ab.x * ac.z,
            ab.x * ac.y - ab.y * ac.x
        );

        // 归一化法线
        Scalar norm = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        normal.x /= norm;
        normal.y /= norm;
        normal.z /= norm;

        // 沿法线方向偏移
        projectedPoint.x += normal.x * offsetDistance;
        projectedPoint.y += normal.y * offsetDistance;
        projectedPoint.z += normal.z * offsetDistance;

        // 保存结果
        projectedPositions[idx] = projectedPoint;
    }
}


void tempfunction() {

    // auto assets_dir = std::string{gipc::assets_dir()};

    // // 在这里获取到CGAL_mappedbodymesh
    // SIMMesh temp_simMesh;
    // CGAL_readObj(assets_dir + "tshirt/Tshirt_total.obj", temp_simMesh.CGAL_clothmesh_fuse);
    // CGAL_readObj(assets_dir + "tshirt/body_1.obj", temp_simMesh.CGAL_bodymesh);
    
    // // 修改映射关系为存储顶点序号
    // std::vector<uint32_t> mapped_to_body_map;
    // CGAL_getReducedMappedMesh(
    //     temp_simMesh.CGAL_bodymesh_mapped,
    //     temp_simMesh.CGAL_clothmesh_fuse,
    //     temp_simMesh.CGAL_bodymesh,
    //     0.1, 0.1, 0.1,
    //     mapped_to_body_map
    // );
    // CGAL_writeObj(assets_dir + "mappedbodymesh.obj", temp_simMesh.CGAL_bodymesh_mapped);

    // // 然后把现在读取每帧的geometry 能够正确把bodymesh映射到mappedbodymesh上
    // for (int i = 1; i <= 240; i++) {
    //     std::string bodyFilename = assets_dir + "tshirt/body_" + std::to_string(i) + ".obj";
    //     CGAL_readObj(bodyFilename, temp_simMesh.CGAL_bodymesh);

    //     // 创建一个映射从 bodymesh 的顶点序号到 CGAL_Vertex_index
    //     std::vector<CGAL_Vertex_index> body_vertices_vector;
    //     for(auto v : temp_simMesh.CGAL_bodymesh.vertices()) {
    //         body_vertices_vector.push_back(v);
    //     }

    //     // 更新 mappedbodymesh 的顶点位置
    //     for (uint32_t j = 0; j < mapped_to_body_map.size(); j++) {
    //         uint32_t body_v_index = mapped_to_body_map[j];
    //         if(body_v_index < body_vertices_vector.size()) {
    //             CGAL_Vertex_index body_v = body_vertices_vector[body_v_index];
    //             CGAL_Vertex_index mapped_v = *(temp_simMesh.CGAL_bodymesh_mapped.vertices_begin() + j);
    //             temp_simMesh.CGAL_bodymesh_mapped.point(mapped_v) = temp_simMesh.CGAL_bodymesh.point(body_v);
    //         } else {
    //             std::cerr << "Error: body_v_index out of range for frame " << i << std::endl;
    //         }
    //     }

    //     std::cout << "frame: " << i << std::endl;
    //     CGAL_avoidBodySelfIntersection(temp_simMesh.CGAL_bodymesh_mapped);

    //     CGAL_writeObj(
    //         assets_dir + "mappedbodymesh" + std::to_string(i) + ".obj",
    //         temp_simMesh.CGAL_bodymesh_mapped
    //     );
    // }
}






















void CGAL_RestoreOriginalVertices(const std::string& simplified_filename, const std::vector<int>& index_mapping_orig2fuse, const std::string& restored_filename) {
    // simplified_filename: 合并顶点后的 OBJ 文件
    // index_mapping_orig2fuse: 原始顶点索引到合并后顶点索引的映射
    // restored_filename: 要写入的还原后的 OBJ 文件

    // 首先读取合并后的顶点和面片信息
    std::ifstream infile(simplified_filename);
    if (!infile) {
        std::cerr << "无法打开文件 " << simplified_filename << std::endl;
        return;
    }

    std::vector<Scalar3> simplified_vertices;
    std::vector<std::string> face_lines;
    std::vector<std::string> other_lines;

    std::string line;
    while (std::getline(infile, line)) {
        if (line.substr(0, 2) == "v ") {
            // 处理顶点行
            std::istringstream iss(line.substr(2));
            Scalar3 v;
            iss >> v.x >> v.y >> v.z;
            simplified_vertices.push_back(v);
        } else if (line.substr(0, 2) == "f ") {
            // 处理面片行
            face_lines.push_back(line);
        } else {
            // 其他类型的行，如纹理坐标和法线
            other_lines.push_back(line);
        }
    }

    infile.close();

    // 根据 index_mapping_orig2fuse 还原原始的顶点列表
    std::vector<Scalar3> restored_vertices(index_mapping_orig2fuse.size());
    for (size_t i = 0; i < index_mapping_orig2fuse.size(); ++i) {
        int simplified_index = index_mapping_orig2fuse[i];
        restored_vertices[i] = simplified_vertices[simplified_index];
    }

    // 更新面片的顶点索引，恢复为原始的索引
    std::vector<std::string> new_face_lines;
    for (const auto& face_line : face_lines) {
        std::istringstream iss(face_line.substr(2));
        std::string token;
        std::vector<std::string> tokens;
        while (iss >> token) {
            tokens.push_back(token);
        }

        // 更新顶点索引
        for (size_t i = 0; i < tokens.size(); ++i) {
            std::string& vert = tokens[i];
            // 处理类似 v/vt/vn 的情况
            size_t pos1 = vert.find('/');
            size_t pos2 = std::string::npos;
            if (pos1 != std::string::npos) {
                pos2 = vert.find('/', pos1 + 1);
            }

            std::string v_index_str = vert.substr(0, pos1);

            int v_index = std::stoi(v_index_str);
            // OBJ 文件的索引从 1 开始

            // 找到所有原始顶点索引中，映射到当前合并顶点索引的那些索引
            std::vector<int> original_indices;
            for (size_t j = 0; j < index_mapping_orig2fuse.size(); ++j) {
                if (index_mapping_orig2fuse[j] == v_index - 1) {
                    original_indices.push_back(j + 1); // OBJ 索引从 1 开始
                }
            }

            // 为了简单，我们假设每个合并顶点只对应一个原始顶点索引
            // 但实际上，一个合并顶点可能对应多个原始顶点索引
            // 这里我们随机选择一个原始索引（或根据某种策略）
            int new_v_index = original_indices.front();

            // 构建新的顶点索引字符串
            std::string new_vert;
            new_vert = std::to_string(new_v_index);
            if (pos1 != std::string::npos) {
                new_vert += vert.substr(pos1);
            }

            tokens[i] = new_vert;
        }

        // 重新构建面片行
        std::ostringstream oss;
        oss << "f ";
        for (const auto& t : tokens) {
            oss << t << " ";
        }
        new_face_lines.push_back(oss.str());
    }

    // 将结果写入输出文件
    std::ofstream outfile(restored_filename);
    if (!outfile) {
        std::cerr << "无法打开文件 " << restored_filename << std::endl;
        return;
    }

    // 写入其他行
    for (const auto& other_line : other_lines) {
        outfile << other_line << std::endl;
    }

    // 写入还原的顶点
    for (const auto& v : restored_vertices) {
        outfile << "v " << std::fixed << std::setprecision(6) << v.x << " " << v.y << " " << v.z << std::endl;
    }

    // 写入更新后的面片
    for (const auto& face_line : new_face_lines) {
        outfile << face_line << std::endl;
    }

    outfile.close();

    std::cout << "成功还原了原始的顶点数据。" << std::endl;
}





























void CGAL_convertObjtoJson(
    const std::string& obj_file,
    const std::string& json_file
) {
    CGAL_Mesh mesh;

    // 读取 OBJ 文件
    if (!CGAL::IO::read_polygon_mesh(obj_file, mesh)) {
        throw std::runtime_error("错误: 无法读取 OBJ 文件 " + obj_file);
    }

    // 提取顶点为扁平数组
    std::vector<Scalar> vertices;
    vertices.reserve(mesh.number_of_vertices() * 3);

    for (const auto& vertex : mesh.vertices()) {
        CGAL_Point_3 point = mesh.point(vertex);
        vertices.push_back(point.x());
        vertices.push_back(point.y());
        vertices.push_back(point.z());
    }

    // 提取面
    std::vector<std::vector<int>> faces;
    faces.reserve(mesh.number_of_faces());

    for (const auto& face : mesh.faces()) {
        std::vector<int> face_indices;
        face_indices.reserve(6); // 预估一个面最多6个顶点（可根据需要调整）
        for (const auto& vertex : CGAL::vertices_around_face(mesh.halfedge(face), mesh)) {
            face_indices.push_back(static_cast<int>(vertex));
        }
        faces.emplace_back(std::move(face_indices));
    }

    // 构建 JSON 对象
    json j;
    j["vertices"] = vertices;
    j["faces"] = faces;

    // 写入 JSON 文件，使用无格式化输出
    std::ofstream outfile(json_file, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("错误: 无法打开 JSON 文件以写入: " + json_file);
    }

    outfile << j.dump(); // 无格式化输出
    outfile.close();

    std::cout << "成功将 OBJ 文件转换为 JSON 文件: " << json_file << std::endl;
}


void CGAL_readSimJson_toMesh(
    const nlohmann::json& httpjson,
    CGAL_Mesh& clothmesh,
    CGAL_Mesh& bodymesh_tpose,
    std::vector<CGAL_Mesh>& bodymesh_total
) {

    // Extract and process triangles
    std::vector<int> triangles = httpjson["triangles"].get<std::vector<int>>();
    std::vector<std::array<int, 3>> triangle_indices(triangles.size() / 3);
    std::cout << "Loading body triangles..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < triangles.size(); i += 3) {
        size_t idx = i / 3;
        triangle_indices[idx] = {triangles[i], triangles[i + 1], triangles[i + 2]};
    }
    std::cout << "Finished loading body triangles." << std::endl;

    // Extract and process cloth triangles
    std::vector<int> cloth_triangles = httpjson["cloth_triangles"].get<std::vector<int>>();
    std::vector<std::array<int, 3>> cloth_triangle_indices(cloth_triangles.size() / 3);
    std::cout << "Loading cloth triangles..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < cloth_triangles.size(); i += 3) {
        size_t idx = i / 3;
        cloth_triangle_indices[idx] = {cloth_triangles[i], cloth_triangles[i + 1], cloth_triangles[i + 2]};
    }
    std::cout << "Finished loading cloth triangles." << std::endl;

    // Extract and process t-pose vertices
    std::vector<Scalar> t_pose_vertices_flat = httpjson["t_pose_vertices"].get<std::vector<Scalar>>();
    size_t num_t_pose_vertices = t_pose_vertices_flat.size() / 3;
    std::vector<CGAL_Point_3> t_pose_vertices(num_t_pose_vertices);
    std::cout << "Loading t-pose vertices..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_t_pose_vertices; ++i) {
        t_pose_vertices[i] = CGAL_Point_3(
            t_pose_vertices_flat[3 * i],
            t_pose_vertices_flat[3 * i + 1],
            t_pose_vertices_flat[3 * i + 2]);
    }
    std::cout << "Finished loading t-pose vertices." << std::endl;

    // Extract and process t-pose cloth vertices
    std::vector<Scalar> t_pose_cloth_vertices_flat = httpjson["t_pose_cloth_vertices"].get<std::vector<Scalar>>();
    size_t num_t_pose_cloth_vertices = t_pose_cloth_vertices_flat.size() / 3;
    std::vector<CGAL_Point_3> t_pose_cloth_vertices(num_t_pose_cloth_vertices);
    std::cout << "Loading t-pose cloth vertices..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_t_pose_cloth_vertices; ++i) {
        t_pose_cloth_vertices[i] = CGAL_Point_3(
            t_pose_cloth_vertices_flat[3 * i],
            t_pose_cloth_vertices_flat[3 * i + 1],
            t_pose_cloth_vertices_flat[3 * i + 2]);
    }
    std::cout << "Finished loading t-pose cloth vertices." << std::endl;

    // Build the bodymesh_tpose
    bodymesh_tpose.clear();
    std::vector<CGAL_Vertex_index> vertex_indices(num_t_pose_vertices);
    for (size_t i = 0; i < num_t_pose_vertices; ++i) {
        vertex_indices[i] = bodymesh_tpose.add_vertex(t_pose_vertices[i]);
    }
    for (const auto& tri : triangle_indices) {
        bodymesh_tpose.add_face(
            vertex_indices[tri[0]],
            vertex_indices[tri[1]],
            vertex_indices[tri[2]]);
    }

    // Build the clothmesh
    clothmesh.clear();
    std::vector<CGAL_Vertex_index> cloth_vertex_indices(num_t_pose_cloth_vertices);
    for (size_t i = 0; i < num_t_pose_cloth_vertices; ++i) {
        cloth_vertex_indices[i] = clothmesh.add_vertex(t_pose_cloth_vertices[i]);
    }
    for (const auto& tri : cloth_triangle_indices) {
        clothmesh.add_face(
            cloth_vertex_indices[tri[0]],
            cloth_vertex_indices[tri[1]],
            cloth_vertex_indices[tri[2]]);
    }

    // Build bodymesh_total
    bodymesh_total.clear();
    std::vector<std::vector<Scalar>> anim_vertices = httpjson["anim_vertices"].get<std::vector<std::vector<Scalar>>>();
    size_t num_frames = anim_vertices.size();
    bodymesh_total.resize(num_frames);

    std::cout << "Loading animation frames..." << std::endl;
    #pragma omp parallel for schedule(dynamic)
    for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
        const auto& frame_vertices_flat = anim_vertices[frame_idx];
        size_t num_frame_vertices = frame_vertices_flat.size() / 3;
        CGAL_Mesh frame_mesh;
        std::vector<CGAL_Vertex_index> frame_vertex_indices(num_frame_vertices);

        // Add vertices to the frame mesh
        for (size_t i = 0; i < num_frame_vertices; ++i) {
            CGAL_Point_3 point(
                frame_vertices_flat[3 * i],
                frame_vertices_flat[3 * i + 1],
                frame_vertices_flat[3 * i + 2]);
            frame_vertex_indices[i] = frame_mesh.add_vertex(point);
        }

        // Add faces to the frame mesh
        for (const auto& tri : triangle_indices) {
            frame_mesh.add_face(
                frame_vertex_indices[tri[0]],
                frame_vertex_indices[tri[1]],
                frame_vertex_indices[tri[2]]);
        }

        // Assign the frame mesh to the total body meshes
        bodymesh_total[frame_idx] = std::move(frame_mesh);
        std::cout << "loading frame " << frame_idx << " successfully" << std::endl;
    }
    std::cout << "Finished loading animation frames." << std::endl;
}



void CGAL_convertBodyObjtoJson(
    const std::string& obj_file_pattern,
    int framerange_start,
    int framerange_end,
    const std::string& json_file
) {
    json j;
    j["frames"] = json::array();
    j["faces"] = json::array(); // 添加 faces 到顶层

    std::vector<std::vector<int>> faces; // 存储 faces
    bool faces_extracted = false;
    int face_extraction_frame = -1;

    // 步骤 1：顺序读取帧，提取 faces
    for (int frame = framerange_start; frame <= framerange_end; ++frame) {
        // 生成 OBJ 文件名
        char obj_filename[512];
        std::snprintf(obj_filename, sizeof(obj_filename), obj_file_pattern.c_str(), frame);

        std::cout << "Processing frame " << frame << ": " << obj_filename << std::endl;

        CGAL_Mesh mesh;
        // 读取 OBJ 文件
        if (!CGAL::IO::read_polygon_mesh(obj_filename, mesh)) {
            std::cerr << "错误: 无法读取 OBJ 文件 " << obj_filename << std::endl;
            continue; // 跳过当前帧，继续下一个
        }

        if (!faces_extracted) {
            // 提取 faces
            faces.reserve(mesh.number_of_faces());

            for (const auto& face : mesh.faces()) {
                std::vector<int> face_indices;
                for (const auto& vertex : CGAL::vertices_around_face(mesh.halfedge(face), mesh)) {
                    face_indices.push_back(static_cast<int>(vertex));
                }
                faces.emplace_back(std::move(face_indices));
            }

            // 将 faces 添加到顶层 JSON
            for (const auto& face : faces) {
                j["faces"].emplace_back(face);
            }

            faces_extracted = true;
            face_extraction_frame = frame;
            std::cout << "Extracted faces from frame " << frame << std::endl;

            // 提取并添加当前帧的 vertices
            std::vector<Scalar> vertices;
            vertices.reserve(mesh.number_of_vertices() * 3);

            for (const auto& vertex : mesh.vertices()) {
                CGAL_Point_3 point = mesh.point(vertex);
                vertices.push_back(point.x());
                vertices.push_back(point.y());
                vertices.push_back(point.z());
            }

            // 创建帧的 JSON 对象
            json frame_json;
            frame_json["frame"] = frame;
            frame_json["vertices"] = vertices;

            // 添加到 frames 数组
            j["frames"].emplace_back(std::move(frame_json));

            break; // 只需提取一次 faces
        }
    }

    // 检查是否成功提取 faces
    if (!faces_extracted) {
        throw std::runtime_error("没有成功读取任何 OBJ 文件，因此无法提取 faces 信息。");
    }

    // 步骤 2：并行处理剩余帧，提取 vertices
    std::vector<int> frames_to_process;
    frames_to_process.reserve(framerange_end - framerange_start + 1 - 1); // 排除用于提取 faces 的帧

    for (int frame = framerange_start; frame <= framerange_end; ++frame) {
        if (frame != face_extraction_frame) {
            frames_to_process.push_back(frame);
        }
    }

    size_t total_frames_to_process = frames_to_process.size();
    std::vector<json> temp_frames(total_frames_to_process);

    // 并行处理帧
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < total_frames_to_process; ++i) {
        int frame = frames_to_process[i];
        char obj_filename[512];
        std::snprintf(obj_filename, sizeof(obj_filename), obj_file_pattern.c_str(), frame);

        std::cout << "Processing frame " << frame << " in parallel: " << obj_filename << std::endl;

        CGAL_Mesh mesh;
        // 读取 OBJ 文件
        if (!CGAL::IO::read_polygon_mesh(obj_filename, mesh)) {
            std::cerr << "错误: 无法读取 OBJ 文件 " << obj_filename << std::endl;
            continue; // 跳过此帧
        }

        // 提取 vertices
        std::vector<Scalar> vertices;
        vertices.reserve(mesh.number_of_vertices() * 3);

        for (const auto& vertex : mesh.vertices()) {
            CGAL_Point_3 point = mesh.point(vertex);
            vertices.push_back(point.x());
            vertices.push_back(point.y());
            vertices.push_back(point.z());
        }

        // 创建帧的 JSON 对象
        json frame_json;
        frame_json["frame"] = frame;
        frame_json["vertices"] = vertices;

        // 存储到临时数组
        temp_frames[i] = std::move(frame_json);
    }

    // 合并临时帧到顶层 JSON
    for (const auto& frame_json : temp_frames) {
        if (!frame_json.is_null()) {
            j["frames"].emplace_back(frame_json);
        }
    }

    // 步骤 3：写入 JSON 文件
    std::ofstream outfile(json_file, std::ios::binary);
    if (!outfile.is_open()) {
        throw std::runtime_error("无法打开 JSON 文件以写入: " + json_file);
    }

    outfile << j.dump(); // 无格式化输出
    outfile.close();

    std::cout << "成功将所有 OBJ 文件转换为 JSON 文件: " << json_file << std::endl;
}


void CGAL_readBodyJsontoMesh (
    const std::string& json_file,
    std::vector<CGAL_Mesh>& meshes
) {
    // 步骤 1：打开并读取 JSON 文件
    std::ifstream infile(json_file);
    if (!infile.is_open()) {
        throw std::runtime_error("无法打开 JSON 文件: " + json_file);
    }

    json j;
    infile >> j;
    infile.close();

    // 步骤 2：检查并提取顶层 'faces'
    if (!j.contains("faces") || !j["faces"].is_array()) {
        throw std::runtime_error("JSON 文件缺少顶层 'faces' 字段或 'faces' 不是数组");
    }

    const auto& json_faces = j["faces"];
    std::vector<std::vector<int>> faces;
    faces.reserve(json_faces.size());

    for (const auto& f : json_faces) {
        if (!f.is_array() || f.size() < 3) {
            std::cerr << "警告: 某个面数据不合法，跳过此面" << std::endl;
            continue;
        }

        std::vector<int> face_indices;
        for (const auto& idx : f) {
            if (!idx.is_number_integer()) {
                std::cerr << "警告: 面顶点索引不是整数，跳过此面" << std::endl;
                face_indices.clear();
                break;
            }
            int vertex_idx = idx.get<int>();
            if (vertex_idx < 0) {
                std::cerr << "警告: 顶点索引为负数，跳过此面" << std::endl;
                face_indices.clear();
                break;
            }
            face_indices.push_back(vertex_idx);
        }

        if (!face_indices.empty()) {
            faces.emplace_back(std::move(face_indices));
        }
    }

    if (faces.empty()) {
        throw std::runtime_error("没有有效的 'faces' 数据被提取");
    }

    // 步骤 3：检查并提取 'frames'
    if (!j.contains("frames") || !j["frames"].is_array()) {
        throw std::runtime_error("JSON 文件缺少 'frames' 字段或 'frames' 不是数组");
    }

    const auto& frames = j["frames"];
    size_t num_frames = frames.size();
    meshes.reserve(num_frames);

    // 步骤 4：并行处理每一帧，构建网格
    // 准备一个临时向量来存储网格
    std::vector<CGAL_Mesh> temp_meshes(num_frames);
    std::vector<bool> frame_loaded(num_frames, false); // 标记哪些帧成功加载

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < num_frames; ++i) {
        const auto& frame_json = frames[i];

        // 检查每一帧是否包含 'vertices'
        if (!frame_json.contains("vertices")) {
            std::cerr << "警告: 某一帧缺少 'vertices' 字段，跳过此帧" << std::endl;
            continue;
        }

        const auto& json_vertices = frame_json["vertices"];

        if (!json_vertices.is_array()) {
            std::cerr << "警告: 'vertices' 不是数组，跳过此帧" << std::endl;
            continue;
        }

        // 读取顶点
        CGAL_Mesh mesh;
        std::vector<CGAL_Vertex_index> vertex_indices;
        size_t num_vertices = json_vertices.size() / 3;
        vertex_indices.reserve(num_vertices);

        bool incomplete_vertices = false;
        for (size_t jv = 0; jv < json_vertices.size(); jv += 3) {
            if (jv + 2 >= json_vertices.size()) {
                std::cerr << "警告: 顶点数据不完整，跳过剩余顶点" << std::endl;
                incomplete_vertices = true;
                break;
            }
            Scalar x = json_vertices[jv].get<Scalar>();
            Scalar y = json_vertices[jv + 1].get<Scalar>();
            Scalar z = json_vertices[jv + 2].get<Scalar>();

            CGAL_Vertex_index vi = mesh.add_vertex(CGAL_Point_3(x, y, z));
            vertex_indices.push_back(vi);
        }

        if (incomplete_vertices || vertex_indices.empty()) {
            std::cerr << "警告: 没有顶点被添加，跳过此帧" << std::endl;
            continue;
        }

        // 添加 faces，使用共享的 'faces'
        for (const auto& face : faces) {
            std::vector<CGAL_Vertex_index> face_vertices;
            face_vertices.reserve(face.size());

            bool invalid_index = false;
            for (const auto& idx : face) {
                if (static_cast<size_t>(idx) >= vertex_indices.size()) {
                    std::cerr << "警告: 顶点索引 " << idx << " 超出范围，跳过此面" << std::endl;
                    invalid_index = true;
                    break;
                }
                face_vertices.push_back(vertex_indices[idx]);
            }

            if (invalid_index) {
                continue;
            }

            // 根据顶点数量调用不同的 add_face 重载
            CGAL_Face_index fi;
            if (face_vertices.size() == 3) {
                // 三角形面
                fi = mesh.add_face(face_vertices[0], face_vertices[1], face_vertices[2]);
            }
            else if (face_vertices.size() == 4) {
                // 四边形面
                fi = mesh.add_face(face_vertices[0], face_vertices[1], face_vertices[2], face_vertices[3]);
            }
            else {
                // 多边形面，使用顶点范围
                fi = mesh.add_face(face_vertices);
            }

            if (fi == CGAL::Surface_mesh<CGAL_Point_3>::null_face()) {
                std::cerr << "警告: 无法添加面，可能是因为网格不合法或重复面: ";
                for (const auto& vi : face_vertices) {
                    std::cerr << vi << " ";
                }
                std::cerr << std::endl;
            }
        }

        // 将构建好的网格存储到临时向量中
        temp_meshes[i] = std::move(mesh);
        frame_loaded[i] = true;
    }

    // 步骤 5：收集所有成功加载的网格
    for (size_t i = 0; i < num_frames; ++i) {
        if (frame_loaded[i]) {
            meshes.push_back(std::move(temp_meshes[i]));
            std::cout << "成功加载帧 " << frames[i].value("frame", -1) << " 的网格" << std::endl;
        }
    }

    // 检查是否至少有一个网格被成功加载
    if (meshes.empty()) {
        throw std::runtime_error("没有任何网格被成功加载");
    }

    std::cout << "成功从 JSON 文件加载所有网格: " << json_file << std::endl;
}

bool CGAL_readObjBody(
    const std::string& obj_file_pattern,
    const int framerange_start,
    const int framerange_end,
    std::vector<CGAL_Mesh>& meshes
) {
    meshes.clear();
    meshes.resize(framerange_end - framerange_start + 1);

    bool success = true;

    #pragma omp parallel for schedule(dynamic)
    for (int frame = framerange_start; frame <= framerange_end; ++frame) {
        // 生成 OBJ 文件名
        char obj_filename[512];
        std::snprintf(obj_filename, sizeof(obj_filename), obj_file_pattern.c_str(), frame);

        // 为避免输出混乱，可以使用临界区或其他同步机制
        #pragma omp critical
        {
            std::cout << "Processing frame " << frame << ": " << obj_filename << std::endl;
        }

        CGAL_Mesh mesh;
        // 读取 OBJ 文件
        if (!CGAL::IO::read_polygon_mesh(obj_filename, mesh)) {
            #pragma omp critical
            {
                std::cerr << "错误: 无法读取 OBJ 文件 " << obj_filename << std::endl;
                success = false;
            }
        } else {
            // 正确的索引应该是 frame - framerange_start
            meshes[frame - framerange_start] = mesh;
        }
    }

    return success;
}


void saveSurfaceMesh_restore_from_sort(
    const std::vector<uint32_t>& sortMapIdStoO,
    std::vector<std::vector<Scalar3>>& output_clothverts_total
) {

    // 还原布料顶点的位置
    for (int i = 0; i < output_clothverts_total.size(); i++) {
        int numClothVerts = sortMapIdStoO.size();
        std::vector<Scalar3> restoredClothPositions(numClothVerts);
        for (int sortedIdx = 0; sortedIdx < numClothVerts; ++sortedIdx) {
            int originalIdx = sortMapIdStoO[sortedIdx]; // 将排序后的索引映射回原始索引
            restoredClothPositions[originalIdx] = output_clothverts_total[i][sortedIdx];
        }
        output_clothverts_total[i] = restoredClothPositions;
    }

    std::cout << "Successfully restored sorted mesh data for all frames." << std::endl;
}

void saveSurfaceMesh_restore_from_fuse(
    const std::vector<uint3>& output_clothfaces,
    const std::vector<int>& index_mapping_orig2fuse,
    std::vector<std::vector<Scalar3>>& output_clothverts_total
) {
    // Number of frames to process
    size_t num_frames = output_clothverts_total.size();
    size_t num_original_vertices = index_mapping_orig2fuse.size();

    // Build a simplified index to original index mapping (pick one original index per simplified index)
    std::unordered_map<int, int> simplified_to_original;
    for (size_t original_idx = 0; original_idx < num_original_vertices; ++original_idx) {
        int simplified_index = index_mapping_orig2fuse[original_idx];
        // If not already mapped, assign this original index
        if (simplified_to_original.find(simplified_index) == simplified_to_original.end()) {
            simplified_to_original[simplified_index] = static_cast<int>(original_idx);
        }
    }

    // Process each frame
    for (size_t frame = 0; frame < num_frames; ++frame) {
        // Get the simplified vertices for this frame
        const std::vector<Scalar3>& simplified_vertices = output_clothverts_total[frame];

        // Restore the original vertices using the index mapping
        std::vector<Scalar3> restored_vertices(num_original_vertices);
        for (size_t original_idx = 0; original_idx < num_original_vertices; ++original_idx) {
            int simplified_index = index_mapping_orig2fuse[original_idx];
            restored_vertices[original_idx] = simplified_vertices[simplified_index];
        }
        output_clothverts_total[frame] = restored_vertices;
    }

    std::cout << "Successfully restored fused mesh data for all frames." << std::endl;
}


void CGAL_saveClothMesh_toJson(
    const std::vector<std::vector<Scalar3>>& output_clothverts_total,
    const std::vector<uint3>& output_clothfaces,
    nlohmann::json& output_httpjson
) {
    // 创建一个新的 JSON 对象
    nlohmann::json j;

    // Step 1: 构建 cloth_triangles 列表，直接使用简化后的索引
    j["cloth_triangles"] = nlohmann::json::array();

    std::cout << "正在保存布料三角面到 JSON..." << std::endl;

    for (const auto& face : output_clothfaces) {
        // 将每个面作为一个三元组添加到 JSON 中
        j["cloth_triangles"].push_back({ face.x, face.y, face.z });
    }

    std::cout << "布料三角面保存完成。" << std::endl;

    // Step 2: 构建 cloth_anim_vertices 数据
    j["cloth_anim_vertices"] = nlohmann::json::array();

    size_t total_frames = output_clothverts_total.size();

    std::cout << "开始保存布料动画顶点..." << std::endl;

    for (size_t frame = 0; frame < total_frames; ++frame) {
        const std::vector<Scalar3>& simplified_vertices = output_clothverts_total[frame];

        // 将顶点数据扁平化为一个连续的数组
        std::vector<Scalar> restored_vertices;
        restored_vertices.reserve(simplified_vertices.size() * 3);

        for (const auto& v : simplified_vertices) {
            restored_vertices.push_back(v.x);
            restored_vertices.push_back(v.y);
            restored_vertices.push_back(v.z);
        }

        // 将扁平化的顶点数据存储到 JSON 中
        j["cloth_anim_vertices"].push_back(restored_vertices);
    }

    std::cout << "布料动画顶点保存完成。" << std::endl;

    // 将构建好的 JSON 对象赋值给输出参数
    output_httpjson = j;

    std::cout << "整个 JSON 保存过程完成。" << std::endl;
}


void CGAL_readOutputClothJson_toMesh(
    const nlohmann::json& outputjson,
    std::vector<std::vector<Scalar3>>& output_clothverts_total,
    std::vector<uint3>& output_clothfaces
) {
    // Step 1: 提取并处理 "cloth_triangles"
    if (!outputjson.contains("cloth_triangles")) {
        throw std::runtime_error("JSON 中缺少 'cloth_triangles' 字段。");
    }

    try {
        const auto& cloth_triangles_json = outputjson["cloth_triangles"];
        if (!cloth_triangles_json.is_array()) {
            throw std::runtime_error("'cloth_triangles' 应该是一个数组。");
        }

        size_t num_triangles = cloth_triangles_json.size();
        output_clothfaces.reserve(num_triangles); // 预留空间以提高性能

        std::cout << "正在加载布料三角面..." << std::endl;

        for (size_t i = 0; i < num_triangles; ++i) {
            const auto& tri = cloth_triangles_json[i];
            if (!tri.is_array() || tri.size() != 3) {
                std::cerr << "警告: 'cloth_triangles' 中的元素 " << i << " 不是一个包含 3 个元素的数组，跳过。" << std::endl;
                continue;
            }

            uint3 face;
            try {
                face.x = tri[0].get<unsigned int>();
                face.y = tri[1].get<unsigned int>();
                face.z = tri[2].get<unsigned int>();
            }
            catch (const std::exception& e) {
                std::cerr << "警告: 解析 'cloth_triangles' 中的元素 " << i << " 时出错: " << e.what() << "，跳过。" << std::endl;
                continue;
            }

            output_clothfaces.push_back(face);
        }

        std::cout << "加载了 " << output_clothfaces.size() << " 个布料三角面。" << std::endl;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("无法解析 'cloth_triangles': ") + e.what());
    }

    // Step 2: 提取并处理 "cloth_anim_vertices"
    if (!outputjson.contains("cloth_anim_vertices")) {
        throw std::runtime_error("JSON 中缺少 'cloth_anim_vertices' 字段。");
    }

    try {
        const auto& cloth_anim_vertices_json = outputjson["cloth_anim_vertices"];
        if (!cloth_anim_vertices_json.is_array()) {
            throw std::runtime_error("'cloth_anim_vertices' 应该是一个数组。");
        }

        size_t num_frames = cloth_anim_vertices_json.size();
        output_clothverts_total.resize(num_frames); // 预先调整大小

        std::cout << "正在加载布料动画顶点..." << std::endl;

        for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
            const auto& frame_json = cloth_anim_vertices_json[frame_idx];
            if (!frame_json.is_array()) {
                std::cerr << "警告: 'cloth_anim_vertices' 中的帧 " << frame_idx << " 不是一个数组，跳过。" << std::endl;
                continue;
            }

            size_t num_values = frame_json.size();
            if (num_values % 3 != 0) {
                std::cerr << "警告: 'cloth_anim_vertices' 中的帧 " << frame_idx << " 的顶点数量不是 3 的倍数，跳过。" << std::endl;
                continue;
            }

            size_t num_vertices = num_values / 3;
            std::vector<Scalar3> frame_vertices;
            frame_vertices.reserve(num_vertices);

            for (size_t v_idx = 0; v_idx < num_vertices; ++v_idx) {
                try {
                    Scalar x = frame_json[3 * v_idx].get<Scalar>();
                    Scalar y = frame_json[3 * v_idx + 1].get<Scalar>();
                    Scalar z = frame_json[3 * v_idx + 2].get<Scalar>();
                    frame_vertices.emplace_back(Scalar3{ x, y, z });
                }
                catch (const std::exception& e) {
                    std::cerr << "警告: 解析帧 " << frame_idx << " 的顶点 " << v_idx << " 时出错: " << e.what() << "，使用 (0,0,0)。" << std::endl;
                    frame_vertices.emplace_back(Scalar3{ 0.0, 0.0, 0.0 });
                }
            }

            // 将解析后的顶点数据赋值给输出向量
            output_clothverts_total[frame_idx] = std::move(frame_vertices);
        }

        std::cout << "加载了 " << output_clothverts_total.size() << " 帧布料动画顶点。" << std::endl;
    }
    catch (const std::exception& e) {
        throw std::runtime_error(std::string("无法解析 'cloth_anim_vertices': ") + e.what());
    }
}


int getVertNeighbors(
    const int numSurfVerts,
	const int numTetElements,
    std::vector<std::vector<unsigned int>>& vertNeighbors,
    std::vector<unsigned int>& neighborList,
	std::vector<unsigned int>& neighborStart,
	std::vector<unsigned int>& neighborNum,
    std::vector<uint3>& triangles,
    std::vector<uint4>& tetrahedras) {

    // vertNeightbors：每个表面顶点对应一个vector，记录邻居顶点索引
    // neighborList：储存所有顶点的邻居索引，按顺序排列
	// neighborStart：记录每个顶点在neighborlist中的起始位置
	// neighborNum：记录每个顶点的neigh的数量

	vertNeighbors.resize(numSurfVerts);
	std::set<std::pair<uint32_t, uint32_t>> SFEdges_set;

    // 收集三角形的三条边 储存成一个pair
	for (const auto& cTri : triangles) {
		for (int i = 0; i < 3; i++) {
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y, cTri.x)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x, cTri.y)) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.x, cTri.y));
			}
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z, cTri.y)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y, cTri.z)) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.y, cTri.z));
			}
			if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x, cTri.z)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z, cTri.x)) == SFEdges_set.end()) {
				SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.z, cTri.x));
			}
		}
	}

    // 构建邻接列表
	for (const auto& edgI : SFEdges_set) {
		vertNeighbors[edgI.first].push_back(edgI.second);
		vertNeighbors[edgI.second].push_back(edgI.first);
	}

	neighborNum.resize(numSurfVerts);
	int offset = 0;
	for (int i = 0; i < numSurfVerts; i++) {
		for (int j = 0; j < vertNeighbors[i].size(); j++) {
			neighborList.push_back(vertNeighbors[i][j]);
		}

		neighborStart.push_back(offset);

		offset += vertNeighbors[i].size();
		neighborNum[i] = vertNeighbors[i].size();
	}

    // 返回总的邻居数量
	return neighborStart[numSurfVerts - 1] + neighborNum[numSurfVerts - 1];
}



void getSurface(
    const int numSurfVerts,
    const std::vector<uint3>& surfFaceIds,
	std::vector<uint32_t>& surfVertIds,
	std::vector<uint2>& surfEdgeIds) {

	std::vector<bool> flag(numSurfVerts, false);
	for (const auto& cTri : surfFaceIds) {

		if (!flag[cTri.x]) {
			surfVertIds.push_back(cTri.x);
			flag[cTri.x] = true;
		}
		if (!flag[cTri.y]) {
			surfVertIds.push_back(cTri.y);
			flag[cTri.y] = true;
		}
		if (!flag[cTri.z]) {
			surfVertIds.push_back(cTri.z);
			flag[cTri.z] = true;
		}

	}

	std::set<std::pair<uint64_t, uint64_t>> SFEdges_set;
	for (const auto& cTri : surfFaceIds) {
		for (int i = 0;i < 3;i++) {
			for (int i = 0;i < 3;i++) {
				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y, cTri.x)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x, cTri.y)) == SFEdges_set.end()) {
					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.x, cTri.y));
				}
				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z, cTri.y)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.y, cTri.z)) == SFEdges_set.end()) {
					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.y, cTri.z));
				}
				if (SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.x, cTri.z)) == SFEdges_set.end() && SFEdges_set.find(std::pair<uint32_t, uint32_t>(cTri.z, cTri.x)) == SFEdges_set.end()) {
					SFEdges_set.insert(std::pair<uint32_t, uint32_t>(cTri.z, cTri.x));
				}
			}
		}
	}

	std::vector<std::pair<uint64_t, uint64_t>> tempEdge = std::vector<std::pair<uint64_t, uint64_t>>(SFEdges_set.begin(), SFEdges_set.end());
	for (int i = 0;i < tempEdge.size();i++) {
		surfEdgeIds.push_back(make_uint2(tempEdge[i].first, tempEdge[i].second));
	}
}




}; // LOADMESH namespace














