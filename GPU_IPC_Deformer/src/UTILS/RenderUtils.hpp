#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <limits>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <memory>
#include <string>
#include <iostream>

#include "GeometryManager.hpp"
#include "CUDAUtils.hpp"

class OpenGLRender {

public:

    OpenGLRender(std::unique_ptr<GeometryManager>& instance) : m_instance(instance) {}

    // GLUT Callback Functions
    void idle_func() { glutPostRedisplay(); }

    void reshape_func(GLint width, GLint height) {
        window_width = static_cast<float>(width);
        window_height = static_cast<float>(height);
        glViewport(0, 0, width, height);
    }

    void keyboard_func(unsigned char key, int x, int y) {
        switch (key) {
            case 'w': zTrans += 0.02f; break;
            case 's': zTrans -= 0.02f; break;
            case 'a': xTrans += 0.02f; break;
            case 'd': xTrans -= 0.02f; break;
            case 'q': yTrans -= 0.02f; break;
            case 'e': yTrans += 0.02f; break;
            case 'k': drawsurface = !drawsurface; break;
            case 'f': drawbox = !drawbox; break;
            case 'l': drawline = !drawline; break;
            case ' ': stopRender = !stopRender; break;
            default: break;
        }
        glutPostRedisplay();
    }

    struct Ray {
        glm::vec3 origin;
        glm::vec3 direction;
    };

    glm::vec3 getCameraPosition() {
        glm::mat4 view = getViewMatrix();
        glm::mat4 invView = glm::inverse(view);
        return glm::vec3(invView[3]);
    }

    Ray createPickingRay(int mouseX, int mouseY) {
        glm::mat4 projection = getProjectionMatrix();
        glm::mat4 view = getViewMatrix();

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);

        float x = (2.0f * mouseX) / viewport[2] - 1.0f;
        float y = 1.0f - (2.0f * mouseY) / viewport[3];
        float z = -1.0f; // Near plane

        glm::vec4 ray_nds = glm::vec4(x, y, z, 1.0f);

        glm::vec4 ray_clip = ray_nds;

        glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
        ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);

        glm::vec4 ray_world = glm::inverse(view) * ray_eye;
        glm::vec3 rayDir = glm::normalize(glm::vec3(ray_world));

        glm::vec3 cameraPosition = getCameraPosition();

        Ray ray;
        ray.origin = cameraPosition;
        ray.direction = rayDir;

        return ray;
    }

    bool rayTriangleIntersect(
        const Ray& ray,
        const glm::vec3& v0,
        const glm::vec3& v1,
        const glm::vec3& v2,
        float& outT
    ) {
        const float EPSILON = 1e-6f;
        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;

        glm::vec3 h = glm::cross(ray.direction, edge2);
        float a = glm::dot(edge1, h);
        if (fabs(a) < EPSILON)
            return false; // 平行，射线与三角形平面平行

        float f = 1.0f / a;
        glm::vec3 s = ray.origin - v0;
        float u = f * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f)
            return false;

        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(ray.direction, q);
        if (v < 0.0f || u + v > 1.0f)
            return false;

        // 计算 t，射线参数
        float t = f * glm::dot(edge2, q);
        if (t > EPSILON) { // 射线与三角形相交
            outT = t;
            return true;
        } else {
            // 射线与三角形相交，但在起点之前
            return false;
        }
    }

    bool performRayMeshIntersection(const Ray& ray, glm::vec3& outIntersectionPoint) {
        // 获取衣物网格的顶点和面列表
        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );
        std::vector<uint3> clothFaces = m_instance->getHostClothFacesAfterSort();

        float minDistance = FLT_MAX;
        bool hasIntersect = false;

        // 遍历每个三角形面，进行射线与三角形的相交测试
        for (const auto& face : clothFaces) {
            glm::vec3 v0(
                clothVertices[face.x].x,
                clothVertices[face.x].y,
                clothVertices[face.x].z
            );
            glm::vec3 v1(
                clothVertices[face.y].x,
                clothVertices[face.y].y,
                clothVertices[face.y].z
            );
            glm::vec3 v2(
                clothVertices[face.z].x,
                clothVertices[face.z].y,
                clothVertices[face.z].z
            );

            float t;
            if (rayTriangleIntersect(ray, v0, v1, v2, t)) {
                if (t < minDistance) {
                    minDistance = t;
                    outIntersectionPoint = ray.origin + t * ray.direction;
                    hasIntersect = true;
                }
            }
        }

        return hasIntersect;
    }

    void selectNearestVertexToIntersection() {
        if (!hasIntersection) {
            selectedVertexIndex = -1;
            return;
        }

        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );

        float minDistance = FLT_MAX;
        int nearestVertexIndex = -1;

        for (size_t i = 0; i < clothVertices.size(); ++i) {
            glm::vec3 vertexPos(
                clothVertices[i].x,
                clothVertices[i].y,
                clothVertices[i].z
            );

            float distance = glm::distance(intersectionPoint, vertexPos);

            if (distance < minDistance) {
                minDistance = distance;
                nearestVertexIndex = static_cast<int>(i);
            }
        }

        selectedVertexIndex = nearestVertexIndex;

        // 输出选中顶点的信息（可选）
        std::cout << "Selected Vertex Index: " << selectedVertexIndex << std::endl;
        std::cout << "Vertex Position: ("
                << clothVertices[selectedVertexIndex].x << ", "
                << clothVertices[selectedVertexIndex].y << ", "
                << clothVertices[selectedVertexIndex].z << ")" << std::endl;
    }

    void updateTargetPosition(int mouseX, int mouseY) {
        if (selectedVertexIndex == -1) return;

        glm::mat4 projection = getProjectionMatrix();
        glm::mat4 view = getViewMatrix();

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);

        float winX = static_cast<float>(mouseX);
        float winY = static_cast<float>(viewport[3] - mouseY);

        // 获取选中顶点的世界坐标
        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );
        Scalar3 selectedVertex = clothVertices[selectedVertexIndex];
        glm::vec3 selectedVertexWorldPos(selectedVertex.x, selectedVertex.y, selectedVertex.z);

        // 将选中顶点的世界坐标投影到窗口坐标系，获取 winZ
        glm::vec4 viewportVec = glm::vec4(viewport[0], viewport[1], viewport[2], viewport[3]);
        glm::vec3 selectedVertexScreenPos = glm::project(selectedVertexWorldPos, view, projection, viewportVec);

        float winZ = selectedVertexScreenPos.z;

        // 使用选中顶点的 winZ，结合当前鼠标位置，反投影得到新的目标位置
        glm::vec3 screenPos = glm::vec3(winX, winY, winZ);
        glm::vec3 worldPos = glm::unProject(screenPos, view, projection, viewportVec);

        dragTargetId = selectedVertexIndex;
        dragTargetPosition = worldPos;
    }

    void mouse_func(int button, int state, int x, int y) {
        if (button == GLUT_RIGHT_BUTTON) {
            if (state == GLUT_DOWN) {
                buttonState = 1;
                ox = x;
                oy = y;
            }
        } else if (button == GLUT_LEFT_BUTTON) {
            if (state == GLUT_DOWN) {
                Ray pickingRay = createPickingRay(x, y);
                hasIntersection = performRayMeshIntersection(pickingRay, intersectionPoint);

                if (hasIntersection) {
                    selectNearestVertexToIntersection(); // 选择最近的顶点
                    isDragging = true; // 开始拖拽

                    // 在此处更新目标位置
                    updateTargetPosition(x, y);
                } else {
                    selectedVertexIndex = -1; // 未选中任何顶点
                    isDragging = false;
                }

                buttonState = 2;
                ox = x;
                oy = y;
            } else if (state == GLUT_UP) {
                buttonState = 0;
                isDragging = false; // 结束拖拽
            }
        }
        glutPostRedisplay();
    }

    void motion_func(int x, int y) {
        if (buttonState == 1) { // 右键
            float dx = static_cast<float>(x - ox);
            float dy = static_cast<float>(y - oy);

            yRot += dx / 5.0f;
            xRot += dy / 5.0f;
        } else if (buttonState == 2 && isDragging && selectedVertexIndex != -1) {
            // 更新目标位置
            updateTargetPosition(x, y);
        }

        ox = x;
        oy = y;
        glutPostRedisplay();
    }

    void draw_Scene3D() {
        glEnable(GL_DEPTH_TEST);
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        // 计算矩阵
        glm::mat4 model = glm::mat4(1.0f);
        glm::mat4 view = getViewMatrix();
        glm::mat4 projection = getProjectionMatrix();

        // 计算法线矩阵
        glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(view * model)));

        // 设置 uniform
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(projLoc,  1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix3fv(normalMatrixLoc, 1, GL_FALSE, glm::value_ptr(normalMatrix));

        // 设置光照和视图位置
        glm::vec3 lightPos = glm::vec3(5.0f, 5.0f, 5.0f); // 调整光源位置
        glm::vec3 viewPos = getCameraPosition();

        glUniform3fv(lightPosLoc, 1, glm::value_ptr(lightPos));
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(viewPos));

        if (drawsurface) {
            updateMeshData();

            // 绘制衣物（填充三角形）
            glBindVertexArray(vao_mesh);
            glUniform3f(colorLoc, COLOR_CLOTH_SURFACE.r, COLOR_CLOTH_SURFACE.g, COLOR_CLOTH_SURFACE.b);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(numIndices), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

            // 绘制衣物边框（线框）
            if (drawline) { // 根据 drawline 变量决定是否绘制线框
                glBindVertexArray(vao_mesh_edges);
                glUniform3f(colorLoc, COLOR_CLOTH_EDGES.r, COLOR_CLOTH_EDGES.g, COLOR_CLOTH_EDGES.b);
                glLineWidth(1.0f);
                glDrawElements(GL_LINES, static_cast<GLsizei>(numEdgeIndices), GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            }

            updateBodyMeshData();

            // 绘制身体（填充三角形）
            glBindVertexArray(vao_body);
            glUniform3f(colorLoc, COLOR_BODY_SURFACE.r, COLOR_BODY_SURFACE.g, COLOR_BODY_SURFACE.b);
            glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(numBodyIndices), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);

            // 绘制身体边框（线框）
            if (drawline) { // 根据 drawline 变量决定是否绘制线框
                glBindVertexArray(vao_body_edges);
                glUniform3f(colorLoc, COLOR_BODY_EDGES.r, COLOR_BODY_EDGES.g, COLOR_BODY_EDGES.b);
                glLineWidth(1.0f);
                glDrawElements(GL_LINES, static_cast<GLsizei>(numBodyEdgeIndices), GL_UNSIGNED_INT, 0);
                glBindVertexArray(0);
            }
        }

        // 绘制选中的顶点和目标位置
        if (selectedVertexIndex != -1 && isDragging) {
            updateDragPointData();
            glBindVertexArray(vao_drag_points);

            // 绘制选中顶点
            glPointSize(10.0f);
            glUniform3f(colorLoc, COLOR_SELECTED_VERTEX.r, COLOR_SELECTED_VERTEX.g, COLOR_SELECTED_VERTEX.b); // 黄色
            glDrawArrays(GL_POINTS, 0, 1);

            // 绘制目标位置
            glUniform3f(colorLoc, COLOR_TARGET_POSITION.r, COLOR_TARGET_POSITION.g, COLOR_TARGET_POSITION.b); // 红色
            glDrawArrays(GL_POINTS, 1, 1);
            glBindVertexArray(0);

            // 绘制从选中顶点到目标位置的线
            updateDragLineData();
            glBindVertexArray(vao_drag_lines);
            glUniform3f(colorLoc, COLOR_DRAG_LINE.r, COLOR_DRAG_LINE.g, COLOR_DRAG_LINE.b); // 绿色
            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, 2);
            glBindVertexArray(0);
        }

        // 更新并绘制 Stitch Pairs
        updateStitchPointData();
        updateStitchLineData();

        size_t total_stitch_pairs = m_instance->getHostStitchPairsAfterSort().size();

        // 绘制所有 Stitch Points
        if (total_stitch_pairs > 0) {
            glPointSize(10.0f);
            glBindVertexArray(vao_stitch_points);

            // 绘制第一个点（蓝色）
            glUniform3f(colorLoc, COLOR_STITCH_POINTS.r, COLOR_STITCH_POINTS.g, COLOR_STITCH_POINTS.b);
            glDrawArrays(GL_POINTS, 0, total_stitch_pairs);

            // 绘制第二个点（蓝色）
            glUniform3f(colorLoc, COLOR_STITCH_POINTS.r, COLOR_STITCH_POINTS.g, COLOR_STITCH_POINTS.b);
            glDrawArrays(GL_POINTS, total_stitch_pairs, total_stitch_pairs);

            glBindVertexArray(0);

            // 绘制 Stitch Lines（蓝色），仅针对 z == 1 的 pairs
            if (active_stitch_pairs > 0) {
                glBindVertexArray(vao_stitch_lines);
                glUniform3f(colorLoc, COLOR_STITCH_POINTS.r, COLOR_STITCH_POINTS.g, COLOR_STITCH_POINTS.b); // 蓝色
                glLineWidth(2.0f);
                glDrawArrays(GL_LINES, 0, 2 * active_stitch_pairs);
                glBindVertexArray(0);
            }
        }

        // 绘制盒子
        if (drawbox) {
            glBindVertexArray(vao_box);
            glUniform3f(colorLoc, COLOR_BOX.r, COLOR_BOX.g, COLOR_BOX.b); // 优化后的盒子颜色（柔和的黄色）
            glLineWidth(1.5f);
            glDrawElements(GL_LINES, static_cast<GLsizei>(numBoxIndices), GL_UNSIGNED_INT, 0);
            glBindVertexArray(0);
        }

        glutSwapBuffers();
    }

    void init_OpenGL() {
        GLenum err = glewInit();
        if (GLEW_OK != err) {
            std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
            exit(EXIT_FAILURE);
        }
        std::cerr << "Status: Using GLEW " << glewGetString(GLEW_VERSION) << std::endl;

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glViewport(0, 0, static_cast<int>(window_width), static_cast<int>(window_height));

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_MULTISAMPLE);

        // 编译着色器
        GLuint vertexShader = compileShader(vertexShaderSource, GL_VERTEX_SHADER);
        GLuint fragmentShader = compileShader(fragmentShaderSource, GL_FRAGMENT_SHADER);

        // 链接着色器程序
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);

        // 检查链接错误
        GLint success;
        GLchar infoLog[512];

        glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
        if(!success)
        {
            glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
            std::cerr << "Error: Shader linking failed\n" << infoLog << std::endl;
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // 获取 uniform 位置
        modelLoc = glGetUniformLocation(shaderProgram, "model");
        viewLoc  = glGetUniformLocation(shaderProgram, "view");
        projLoc  = glGetUniformLocation(shaderProgram, "projection");
        normalMatrixLoc = glGetUniformLocation(shaderProgram, "normalMatrix");
        lightPosLoc = glGetUniformLocation(shaderProgram, "lightPos");
        viewPosLoc = glGetUniformLocation(shaderProgram, "viewPos");
        colorLoc = glGetUniformLocation(shaderProgram, "color");

        setupMesh();
        setupBodyMesh();
        setupDragPointVAO();
        setupDragLineVAO();
        setupStitchPointVAO();
        setupStitchLineVAO();
        setupBoxVAO();

        // 设置初始摄像机平移
        yTrans = -1.0f;
        zTrans = -4.0f;
    }

private:

    // Shader source code
    const char* vertexShaderSource = R"glsl(
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 normal;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    uniform mat3 normalMatrix;

    out vec3 FragPos;
    out vec3 Normal;

    void main()
    {
        FragPos = vec3(model * vec4(position, 1.0)); // 世界空间位置
        Normal = normalMatrix * normal; // 转换法线到世界空间
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
    )glsl";

    const char* fragmentShaderSource = R"glsl(
    #version 330 core
    out vec4 FragColor;

    in vec3 FragPos;
    in vec3 Normal;

    uniform vec3 lightPos;
    uniform vec3 viewPos;
    uniform vec3 color;

    void main()
    {
        // Ambient
        float ambientStrength = 0.2;
        vec3 ambient = ambientStrength * color;

        // Diffuse
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(lightPos - FragPos);
        float diff = max(dot(norm, lightDir), 0.0);
        vec3 diffuse = diff * color;

        // Specular
        float specularStrength = 0.2;
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);
        float shininess = 32.0;
        float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
        vec3 specular = specularStrength * spec * vec3(1.0); // 白色高光

        vec3 result = ambient + diffuse + specular;
        FragColor = vec4(result, 1.0);
    }
    )glsl";

    GLuint compileShader(const char* source, GLenum shaderType)
    {
        GLuint shader = glCreateShader(shaderType);

        glShaderSource(shader, 1, &source, NULL);
        glCompileShader(shader);

        // Check for compilation errors
        GLint success;
        GLchar infoLog[512];

        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success)
        {
            glGetShaderInfoLog(shader, 512, NULL, infoLog);
            std::cerr << "Error: Shader compilation failed\n" << infoLog << std::endl;
        }

        return shader;
    }

    void computeNormals(
        const std::vector<glm::vec3>& vertices,
        const std::vector<unsigned int>& indices,
        std::vector<glm::vec3>& normals)
    {
        normals.resize(vertices.size(), glm::vec3(0.0f));

        // 对每个三角形
        for(size_t i = 0; i < indices.size(); i += 3)
        {
            unsigned int index0 = indices[i];
            unsigned int index1 = indices[i + 1];
            unsigned int index2 = indices[i + 2];

            glm::vec3 v0 = vertices[index0];
            glm::vec3 v1 = vertices[index1];
            glm::vec3 v2 = vertices[index2];

            // 计算面法线
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            glm::vec3 faceNormal = glm::normalize(glm::cross(edge1, edge2));

            // 累加到顶点法线
            normals[index0] += faceNormal;
            normals[index1] += faceNormal;
            normals[index2] += faceNormal;
        }

        // 归一化法线
        for(auto& normal : normals)
        {
            normal = glm::normalize(normal);
        }
    }

    void setupMesh()
    {
        // 获取网格数据
        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );
        std::vector<uint3> clothFaces = m_instance->getHostClothFacesAfterSort();

        // 转换顶点数据
        std::vector<glm::vec3> vertices;
        for(const auto& vert : clothVertices)
        {
            vertices.emplace_back(vert.x, vert.y, vert.z);
        }

        // 转换索引数据
        std::vector<unsigned int> indices;
        for(const auto& face : clothFaces)
        {
            indices.push_back(face.x);
            indices.push_back(face.y);
            indices.push_back(face.z);
        }
        numIndices = indices.size();

        // 计算法线
        std::vector<glm::vec3> normals;
        computeNormals(vertices, indices, normals);

        // 交错存储位置和法线数据
        std::vector<float> vertexData;
        for(size_t i = 0; i < vertices.size(); ++i)
        {
            vertexData.push_back(vertices[i].x);
            vertexData.push_back(vertices[i].y);
            vertexData.push_back(vertices[i].z);
            vertexData.push_back(normals[i].x);
            vertexData.push_back(normals[i].y);
            vertexData.push_back(normals[i].z);
        }

        // 创建VAO和VBO
        glGenVertexArrays(1, &vao_mesh);
        glGenBuffers(1, &vbo_vertices);
        glGenBuffers(1, &ebo_indices);

        glBindVertexArray(vao_mesh);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_indices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); // 位置
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); // 法线
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);

        // 设置边框 VAO 和 VBO
        setupMeshEdges();
    }

    void setupMeshEdges() {
        // 获取衣物的边框索引
        std::vector<unsigned int> edgeIndices;
        for (const auto& face : m_instance->getHostClothFacesAfterSort()) {
            edgeIndices.push_back(face.x);
            edgeIndices.push_back(face.y);

            edgeIndices.push_back(face.y);
            edgeIndices.push_back(face.z);

            edgeIndices.push_back(face.z);
            edgeIndices.push_back(face.x);
        }
        numEdgeIndices = edgeIndices.size();

        // 创建 VAO 和 EBO
        glGenVertexArrays(1, &vao_mesh_edges);
        glBindVertexArray(vao_mesh_edges);

        // 复用顶点缓冲区 vbo_vertices
        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);

        // 创建边索引缓冲区
        glGenBuffers(1, &ebo_edge_indices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_edge_indices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeIndices.size() * sizeof(unsigned int), edgeIndices.data(), GL_STATIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); // 位置
        glEnableVertexAttribArray(0);

        // 法线属性（如果需要绘制边框的法线）
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); // 法线
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    void setupBodyMesh()
    {
        // 获取身体网格数据
        std::vector<Scalar3> bodyVertices(
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts(),
            m_instance->getHostSurfVertPos().end()
        );
        std::vector<uint3> bodyFaces = m_instance->getHostBodyFacesAfterSort();

        // 转换顶点数据
        std::vector<glm::vec3> vertices;
        for(const auto& vert : bodyVertices)
        {
            vertices.emplace_back(vert.x, vert.y, vert.z);
        }

        // 转换索引数据
        std::vector<unsigned int> indices;
        for(const auto& face : bodyFaces)
        {
            indices.push_back(face.x);
            indices.push_back(face.y);
            indices.push_back(face.z);
        }
        numBodyIndices = indices.size();

        // 计算法线
        std::vector<glm::vec3> normals;
        computeNormals(vertices, indices, normals);

        // 交错存储位置和法线数据
        std::vector<float> vertexData;
        for(size_t i = 0; i < vertices.size(); ++i)
        {
            vertexData.push_back(vertices[i].x);
            vertexData.push_back(vertices[i].y);
            vertexData.push_back(vertices[i].z);
            vertexData.push_back(normals[i].x);
            vertexData.push_back(normals[i].y);
            vertexData.push_back(normals[i].z);
        }

        // 创建VAO和VBO
        glGenVertexArrays(1, &vao_body);
        glGenBuffers(1, &vbo_body_vertices);
        glGenBuffers(1, &ebo_body_indices);

        glBindVertexArray(vao_body);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_body_vertices);
        glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), vertexData.data(), GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_body_indices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); // 位置
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); // 法线
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);

        // 设置身体边框 VAO 和 VBO
        setupBodyMeshEdges();
    }

    void setupBodyMeshEdges() {
        // 获取身体的边框索引
        std::vector<unsigned int> edgeIndices;
        for (const auto& face : m_instance->getHostBodyFacesAfterSort()) {
            edgeIndices.push_back(face.x);
            edgeIndices.push_back(face.y);

            edgeIndices.push_back(face.y);
            edgeIndices.push_back(face.z);

            edgeIndices.push_back(face.z);
            edgeIndices.push_back(face.x);
        }
        numBodyEdgeIndices = edgeIndices.size();

        // 创建 VAO 和 EBO
        glGenVertexArrays(1, &vao_body_edges);
        glBindVertexArray(vao_body_edges);

        // 复用顶点缓冲区 vbo_body_vertices
        glBindBuffer(GL_ARRAY_BUFFER, vbo_body_vertices);

        // 创建边索引缓冲区
        glGenBuffers(1, &ebo_body_edge_indices);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_body_edge_indices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, edgeIndices.size() * sizeof(unsigned int), edgeIndices.data(), GL_STATIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0); // 位置
        glEnableVertexAttribArray(0);

        // 法线属性（如果需要绘制边框的法线）
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float))); // 法线
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    void setupDragPointVAO()
    {
        glGenVertexArrays(1, &vao_drag_points);
        glGenBuffers(1, &vbo_drag_points);

        glBindVertexArray(vao_drag_points);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_drag_points);
        glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);
    }

    void setupDragLineVAO()
    {
        glGenVertexArrays(1, &vao_drag_lines);
        glGenBuffers(1, &vbo_drag_lines);

        glBindVertexArray(vao_drag_lines);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_drag_lines);
        glBufferData(GL_ARRAY_BUFFER, 2 * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);
    }

    void setupStitchPointVAO()
    {
        glGenVertexArrays(1, &vao_stitch_points);
        glGenBuffers(1, &vbo_stitch_points);

        glBindVertexArray(vao_stitch_points);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_stitch_points);
        
        // 分配足够的缓冲区大小：2 * total_pairs * sizeof(glm::vec3)
        size_t max_pairs = 10000; // 根据实际需求调整
        glBufferData(GL_ARRAY_BUFFER, 2 * max_pairs * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);
    }

    void setupStitchLineVAO()
    {
        glGenVertexArrays(1, &vao_stitch_lines);
        glGenBuffers(1, &vbo_stitch_lines);

        glBindVertexArray(vao_stitch_lines);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_stitch_lines);
        
        // 每对 stitchpair 有一条线，因此需要 2 * max_active_pairs 个点
        size_t max_active_pairs = 10000; // 根据实际需求调整
        glBufferData(GL_ARRAY_BUFFER, 2 * max_active_pairs * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);
    }

    void setupBoxVAO() {
        // 创建 VAO、VBO 和 EBO
        glGenVertexArrays(1, &vao_box);
        glGenBuffers(1, &vbo_box_vertices);
        glGenBuffers(1, &ebo_box_indices);

        glBindVertexArray(vao_box);

        // 定义盒子的8个顶点
        float ox = -1.0f, oy = 0.0f, oz = -1.0f;
        float width = 2.0f, height = 2.0f, length = 2.0f;

        glm::vec3 boxVertices[8] = {
            glm::vec3(ox, oy, oz),
            glm::vec3(ox + width, oy, oz),
            glm::vec3(ox + width, oy + height, oz),
            glm::vec3(ox, oy + height, oz),
            glm::vec3(ox, oy, oz + length),
            glm::vec3(ox + width, oy, oz + length),
            glm::vec3(ox + width, oy + height, oz + length),
            glm::vec3(ox, oy + height, oz + length)
        };

        // 定义盒子的边索引
        unsigned int boxIndices[] = {
            0, 1, 1, 2, 2, 3, 3, 0,       // 前面
            4, 5, 5, 6, 6, 7, 7, 4,       // 后面
            0, 4, 1, 5, 2, 6, 3, 7        // 连接前后面的线
        };
        numBoxIndices = sizeof(boxIndices) / sizeof(boxIndices[0]);

        // 上传顶点数据
        glBindBuffer(GL_ARRAY_BUFFER, vbo_box_vertices);
        glBufferData(GL_ARRAY_BUFFER, sizeof(boxVertices), boxVertices, GL_STATIC_DRAW);

        // 上传索引数据
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_box_indices);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(boxIndices), boxIndices, GL_STATIC_DRAW);

        // 设置顶点属性
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (void*)0);
        glEnableVertexAttribArray(0);

        glBindVertexArray(0);
    }

    void updateMeshData()
    {
        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );
        std::vector<uint3> clothFaces = m_instance->getHostClothFacesAfterSort();

        // 转换顶点数据
        std::vector<glm::vec3> vertices;
        for(const auto& vert : clothVertices)
        {
            vertices.emplace_back(vert.x, vert.y, vert.z);
        }

        // 转换索引数据
        std::vector<unsigned int> indices;
        for(const auto& face : clothFaces)
        {
            indices.push_back(face.x);
            indices.push_back(face.y);
            indices.push_back(face.z);
        }

        // 重新计算法线
        std::vector<glm::vec3> normals;
        computeNormals(vertices, indices, normals);

        // 交错存储位置和法线数据
        std::vector<float> vertexData;
        for(size_t i = 0; i < vertices.size(); ++i)
        {
            vertexData.push_back(vertices[i].x);
            vertexData.push_back(vertices[i].y);
            vertexData.push_back(vertices[i].z);
            vertexData.push_back(normals[i].x);
            vertexData.push_back(normals[i].y);
            vertexData.push_back(normals[i].z);
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertexData.size() * sizeof(float), vertexData.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void updateBodyMeshData()
    {
        std::vector<Scalar3> bodyVertices(
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts(),
            m_instance->getHostSurfVertPos().end()
        );
        std::vector<uint3> bodyFaces = m_instance->getHostBodyFacesAfterSort();

        // 转换顶点数据
        std::vector<glm::vec3> vertices;
        for(const auto& vert : bodyVertices)
        {
            vertices.emplace_back(vert.x, vert.y, vert.z);
        }

        // 转换索引数据
        std::vector<unsigned int> indices;
        for(const auto& face : bodyFaces)
        {
            indices.push_back(face.x);
            indices.push_back(face.y);
            indices.push_back(face.z);
        }

        // 重新计算法线
        std::vector<glm::vec3> normals;
        computeNormals(vertices, indices, normals);

        // 交错存储位置和法线数据
        std::vector<float> vertexData;
        for(size_t i = 0; i < vertices.size(); ++i)
        {
            vertexData.push_back(vertices[i].x);
            vertexData.push_back(vertices[i].y);
            vertexData.push_back(vertices[i].z);
            vertexData.push_back(normals[i].x);
            vertexData.push_back(normals[i].y);
            vertexData.push_back(normals[i].z);
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo_body_vertices);
        glBufferSubData(GL_ARRAY_BUFFER, 0, vertexData.size() * sizeof(float), vertexData.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void updateDragPointData()
    {
        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );
        Scalar3 selectedVertex = clothVertices[selectedVertexIndex];

        glm::vec3 points[2];
        points[0] = glm::vec3(selectedVertex.x, selectedVertex.y, selectedVertex.z);
        points[1] = dragTargetPosition;

        glBindBuffer(GL_ARRAY_BUFFER, vbo_drag_points);
        glBufferSubData(GL_ARRAY_BUFFER, 0, 2 * sizeof(glm::vec3), points);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void updateDragLineData()
    {
        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );
        Scalar3 selectedVertex = clothVertices[selectedVertexIndex];

        glm::vec3 lines[2];
        lines[0] = glm::vec3(selectedVertex.x, selectedVertex.y, selectedVertex.z);
        lines[1] = dragTargetPosition;

        glBindBuffer(GL_ARRAY_BUFFER, vbo_drag_lines);
        glBufferSubData(GL_ARRAY_BUFFER, 0, 2 * sizeof(glm::vec3), lines);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void updateStitchPointData()
    {
        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );
        std::vector<uint3> stitchPairs = m_instance->getHostStitchPairsAfterSort();

        size_t total_pairs = stitchPairs.size();

        // Reserve space for all p1 and p2 points
        std::vector<glm::vec3> p1_points;
        std::vector<glm::vec3> p2_points;
        p1_points.reserve(total_pairs);
        p2_points.reserve(total_pairs);

        // Collect all p1 and p2 points
        for (const auto& pair : stitchPairs) {
            // 确保索引有效
            if (pair.x >= clothVertices.size() || pair.y >= clothVertices.size()) {
                std::cerr << "Error: stitchPair index out of range." << std::endl;
                continue;
            }
            Scalar3 p1 = clothVertices[pair.x];
            Scalar3 p2 = clothVertices[pair.y];
            p1_points.emplace_back(p1.x, p1.y, p1.z);
            p2_points.emplace_back(p2.x, p2.y, p2.z);
        }

        // Combine points: first all p1, then all p2
        std::vector<glm::vec3> points;
        points.reserve(2 * total_pairs);
        points.insert(points.end(), p1_points.begin(), p1_points.end());
        points.insert(points.end(), p2_points.begin(), p2_points.end());

        // Update VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo_stitch_points);
        // Resize buffer if necessary
        GLint bufferSize;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bufferSize);
        if (bufferSize < points.size() * sizeof(glm::vec3)) {
            glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
        }

        glBufferSubData(GL_ARRAY_BUFFER, 0, points.size() * sizeof(glm::vec3), points.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void updateStitchLineData()
    {
        std::vector<Scalar3> clothVertices(
            m_instance->getHostSurfVertPos().begin(),
            m_instance->getHostSurfVertPos().begin() + m_instance->getHostNumClothVerts()
        );
        std::vector<uint3> stitchPairs = m_instance->getHostStitchPairsAfterSort();

        // Reset active_stitch_pairs
        active_stitch_pairs = 0;

        // First, count the number of stitchpairs with flag == 1
        for (const auto& pair : stitchPairs) {
            if (pair.z == 1)
                active_stitch_pairs++;
        }

        // Reserve space
        std::vector<glm::vec3> lines;
        lines.reserve(2 * active_stitch_pairs); // only for active pairs

        // Collect lines where flag == 1
        for (const auto& pair : stitchPairs) {
            if (pair.z == 1) {
                // 确保索引有效
                if (pair.x >= clothVertices.size() || pair.y >= clothVertices.size()) {
                    std::cerr << "Error: stitchPair index out of range." << std::endl;
                    continue;
                }
                Scalar3 p1 = clothVertices[pair.x];
                Scalar3 p2 = clothVertices[pair.y];
                lines.emplace_back(p1.x, p1.y, p1.z); // 起点
                lines.emplace_back(p2.x, p2.y, p2.z); // 终点
            }
        }

        // Update VBO
        glBindBuffer(GL_ARRAY_BUFFER, vbo_stitch_lines);
        // Resize buffer if necessary
        GLint bufferSize;
        glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &bufferSize);
        if (bufferSize < lines.size() * sizeof(glm::vec3)) {
            glBufferData(GL_ARRAY_BUFFER, lines.size() * sizeof(glm::vec3), NULL, GL_DYNAMIC_DRAW);
        }

        glBufferSubData(GL_ARRAY_BUFFER, 0, lines.size() * sizeof(glm::vec3), lines.data());
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    glm::mat4 getViewMatrix() {
        glm::mat4 view = glm::mat4(1.0f);
        view = glm::translate(view, glm::vec3(xTrans, yTrans, zTrans));
        view = glm::rotate(view, glm::radians(xRot), glm::vec3(1.0f, 0.0f, 0.0f));
        view = glm::rotate(view, glm::radians(yRot), glm::vec3(0.0f, 1.0f, 0.0f));
        return view;
    }

    glm::mat4 getProjectionMatrix() {
        float aspect = window_width / window_height;
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
        return projection;
    }

private:

    std::unique_ptr<GeometryManager>& m_instance;

    // Shader program
    GLuint shaderProgram;

    // Uniform locations
    GLuint modelLoc, viewLoc, projLoc, normalMatrixLoc, lightPosLoc, viewPosLoc, colorLoc;

    // VAOs and VBOs
    GLuint vao_mesh, vbo_vertices, ebo_indices;
    GLuint vao_mesh_edges, ebo_edge_indices;
    GLuint vao_body, vbo_body_vertices, ebo_body_indices;
    GLuint vao_body_edges, ebo_body_edge_indices;
    GLuint vao_drag_points, vbo_drag_points;
    GLuint vao_drag_lines, vbo_drag_lines;
    GLuint vao_stitch_points, vbo_stitch_points;
    GLuint vao_stitch_lines, vbo_stitch_lines;
    GLuint vao_box, vbo_box_vertices, ebo_box_indices;

    size_t numIndices;
    size_t numEdgeIndices;
    size_t numBodyIndices;
    size_t numBodyEdgeIndices;
    size_t numBoxIndices;

    float xRot = 0.0f, yRot = 0.0f;
    float xTrans = 0.0f, yTrans = 0.0f, zTrans = 0.0f;
    int ox = 0, oy = 0;
    int buttonState = 0;
    bool drawsurface = true;
    bool drawline = true;
    bool drawbox = true;

public:

    float window_width = 1080.0f, window_height = 1080.0f;
    bool stopRender = true;

    glm::vec3 intersectionPoint; // 存储射线与网格的交点
    bool hasIntersection = false; // 是否有交点
    int selectedVertexIndex = -1; // -1 表示未选中任何顶点
    glm::vec3 dragTargetPosition; // 选中顶点的目标位置
    int dragTargetId; // 选中顶点的目标位置
    bool isDragging = false;  // 标记是否正在拖拽

    size_t active_stitch_pairs = 0; // 存储 active stitchpairs 的数量

private:
    // 颜色定义
    const glm::vec3 COLOR_CLOTH_SURFACE = glm::vec3(0.76f, 0.87f, 0.96f); // 柔和的蓝色
    const glm::vec3 COLOR_CLOTH_EDGES = glm::vec3(0.5f, 0.5f, 0.5f); // 浅灰色
    const glm::vec3 COLOR_BODY_SURFACE = glm::vec3(0.98f, 0.77f, 0.64f); // 柔和的肤色
    const glm::vec3 COLOR_BODY_EDGES = glm::vec3(0.5f, 0.5f, 0.5f); // 浅灰色
    const glm::vec3 COLOR_SELECTED_VERTEX = glm::vec3(1.0f, 1.0f, 0.0f); // 黄色
    const glm::vec3 COLOR_TARGET_POSITION = glm::vec3(1.0f, 0.0f, 0.0f); // 红色
    const glm::vec3 COLOR_DRAG_LINE = glm::vec3(0.0f, 1.0f, 0.0f); // 绿色
    const glm::vec3 COLOR_STITCH_POINTS = glm::vec3(0.0f, 0.0f, 1.0f); // 蓝色
    const glm::vec3 COLOR_BOX = glm::vec3(0.8f, 0.8f, 0.1f); // 柔和的黄色

};
