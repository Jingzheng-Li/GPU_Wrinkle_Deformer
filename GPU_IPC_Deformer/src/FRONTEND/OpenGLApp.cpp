// OpenGLApp.cpp
#include "OpenGLApp.hpp"
#include <iostream>

// 初始化静态成员
OpenGLApp* OpenGLApp::current_instance = nullptr;

// 构造函数
OpenGLApp::OpenGLApp(SimulationContext& context, Simulator& simulator)
    : ctx(context), sim(simulator) {}

// 运行 OpenGL 主循环
void OpenGLApp::run(int argc, char** argv) {
    // 设置当前实例
    current_instance = this;


    // 初始化 OpenGLRender 对象
    if (!ctx.glRender_ptr) {
        ctx.glRender_ptr = std::make_unique<OpenGLRender>(ctx.instance);
    }

    // 初始化 GLUT
    glutInit(&argc, argv);
    glutSetOption(GLUT_MULTISAMPLE, 16);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
    glutInitWindowSize(static_cast<int>(ctx.glRender_ptr->window_width), static_cast<int>(ctx.glRender_ptr->window_height));
    glutInitWindowPosition(100, 100);
    glutCreateWindow("GPU IPC");

    // 初始化 GLEW
    GLenum err = glewInit();
    if (GLEW_OK != err) {
        std::cerr << "Error initializing GLEW: " << glewGetErrorString(err) << std::endl;
        return;
    }

    // 初始化 OpenGL 相关设置
    ctx.glRender_ptr->init_OpenGL();

    // 设置 GLUT 回调函数
    glutDisplayFunc(OpenGLApp::display_func);
    glutIdleFunc(OpenGLApp::idle_func);
    glutReshapeFunc(OpenGLApp::reshape_func);
    glutMouseFunc(OpenGLApp::mouse_func);
    glutMotionFunc(OpenGLApp::motion_func);
    glutKeyboardFunc(OpenGLApp::keyboard_func);

    // 启动 GLUT 主循环
    std::cout << "OpenGL Server is running.\n";
    glutMainLoop();
}

// 静态回调函数实现，调用当前实例的成员函数
void OpenGLApp::display_func() {
    if (current_instance) {
        current_instance->on_display();
    }
}

void OpenGLApp::idle_func() {
    if (current_instance) {
        current_instance->on_idle();
    }
}

void OpenGLApp::reshape_func(GLint width, GLint height) {
    if (current_instance) {
        current_instance->on_reshape(width, height);
    }
}

void OpenGLApp::mouse_func(int button, int state, int x, int y) {
    if (current_instance) {
        current_instance->on_mouse(button, state, x, y);
    }
}

void OpenGLApp::motion_func(int x, int y) {
    if (current_instance) {
        current_instance->on_motion(x, y);
    }
}

void OpenGLApp::keyboard_func(unsigned char key, int x, int y) {
    if (current_instance) {
        current_instance->on_keyboard(key, x, y);
    }
}

// 非静态成员函数实现

void OpenGLApp::on_display() {
    sim.display();
    glutSwapBuffers();
}

void OpenGLApp::on_idle() {
    ctx.glRender_ptr->idle_func();
    glutPostRedisplay(); // 请求重新绘制
}

void OpenGLApp::on_reshape(GLint width, GLint height) {
    ctx.glRender_ptr->reshape_func(width, height);
}

void OpenGLApp::on_mouse(int button, int state, int x, int y) {
    ctx.glRender_ptr->mouse_func(button, state, x, y);
}

void OpenGLApp::on_motion(int x, int y) {
    ctx.glRender_ptr->motion_func(x, y);
}

void OpenGLApp::on_keyboard(unsigned char key, int x, int y) {
#if defined(GPUIPC_DRAG)
    if (key >= '0' && key <= '5') {
        int index = key - '0';  // 将字符转换为整数索引
        ctx.restart_program(index);
    }
#endif
    ctx.glRender_ptr->keyboard_func(key, x, y);
}
