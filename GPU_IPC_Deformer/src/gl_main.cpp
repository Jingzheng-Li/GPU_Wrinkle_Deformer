
#include "SimulationContext.hpp"
#include "Simulator.hpp"
#include "HTTPServer.hpp"
#include "OpenGLApp.hpp"
#include "Deformer.hpp"

void parse_arguments(int argc, char** argv, SimulationContext& ctx) {

#if defined(GPUIPC_ANIMATION)

    ctx.do_OpenGLRender = false;

    ctx.assets_dir_clothmesh = std::filesystem::path("../Assets/_clothmesh_fine/");
    ctx.assets_dir_bodymesh = std::filesystem::path("../Assets/_bodymesh1_mapped/");
    ctx.assets_dir_staticmesh = std::filesystem::path("../Assets/_staticmesh/");

    ctx.clothmeshname = "dress_one_shoulder";
    ctx.bodymeshname = "body_";
    ctx.bodytposename = "body_t_pose";
    // ctx.staticmeshname = "body_t_pose";
    
#endif



#if defined(GPUIPC_DEFORMER)
    ctx.assets_dir_clothmesh = std::filesystem::path("../Assets/");
    ctx.clothmeshname = "tubemesh";
#endif



#if defined(GPUIPC_DRAG)

    std::vector<std::string> _clothmesh_drag_vec = {
        "drag_majia",
        "drag_lianti",
        "drag_lianti_majia",
        "drag_dress",
        "drag_middle_dress",
        "drag_long_dress",
    };

    // 检查命令行参数中是否有索引
    if (argc > 1) {
        int index = std::atoi(argv[1]);
        if (index >= 0 && index < static_cast<int>(_clothmesh_drag_vec.size())) {
            ctx.clothmeshname = _clothmesh_drag_vec[index];
        } else {
            std::cerr << "无效的索引，使用默认的 clothmeshname。" << std::endl;
        }
        if (index == 0) {
            ctx.do_addStitchPairs = true;
        }
    }

#endif

}

int main(int argc, char** argv) {

    init_CUDA();

    SimulationContext ctx;
    Simulator sim(ctx);

    parse_arguments(argc, argv, ctx);

#if defined(GPUIPC_ANIMATION)

    if (ctx.do_OpenGLRender) {

        ctx.init_Geometry_ptr();
        sim.init_Geometry_mesh();
        OpenGLApp app(ctx, sim);
        app.run(argc, argv);
        ctx.reset_ptr();

    } else {

        ctx.init_Geometry_ptr();
        sim.init_Geometry_mesh();
        while (true) {
            sim.display_without_opengl_animation();
            if (ctx.simMesh_ptr->simulation_finished) break;
        }
        ctx.reset_ptr();
    }

#endif


#if defined(GPUIPC_DEFORMER)
    Deformer deformer(ctx, sim);
    ctx.init_Geometry_ptr();
    sim.init_Geometry_mesh();

    deformer.DeformerPipeline();

    ctx.reset_ptr();
#endif



#if defined(GPUIPC_HTTP)

    // 创建并运行 HTTP 服务器
    HTTPServer server(ctx, sim);
    server.solve_Http();

#endif

    return 0;
}
