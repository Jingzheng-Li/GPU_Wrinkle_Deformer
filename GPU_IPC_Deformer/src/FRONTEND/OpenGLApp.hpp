// OpenGLApp.hpp
#ifndef OPENGLAPP_HPP
#define OPENGLAPP_HPP

#include "SimulationContext.hpp"
#include "Simulator.hpp"
#include <GL/glut.h>

// OpenGLApp 类封装了所有 OpenGL 相关的功能
class OpenGLApp {
public:
    // 构造函数，接收 SimulationContext 和 Simulator 的引用
    OpenGLApp(SimulationContext& context, Simulator& simulator);

    // 运行 OpenGL 主循环
    void run(int argc, char** argv);

private:
    SimulationContext& ctx;
    Simulator& sim;

    // 静态指针，指向当前的 OpenGLApp 实例
    static OpenGLApp* current_instance;

    // 静态回调函数，用于 GLUT
    static void display_func();
    static void idle_func();
    static void reshape_func(GLint width, GLint height);
    static void mouse_func(int button, int state, int x, int y);
    static void motion_func(int x, int y);
    static void keyboard_func(unsigned char key, int x, int y);

    // 非静态成员函数，实际处理回调逻辑
    void on_display();
    void on_idle();
    void on_reshape(GLint width, GLint height);
    void on_mouse(int button, int state, int x, int y);
    void on_motion(int x, int y);
    void on_keyboard(unsigned char key, int x, int y);
};

#endif // OPENGLAPP_HPP
