// SimulationContext.cpp

#include "SimulationContext.hpp"



SimulationContext::SimulationContext()
    : collision_detection_buff_scale(1),
      interpolation_frames(10),
      animation_motion_rate(4),
      do_OpenGLRender(true),
      do_addSoftTargets(false),
      do_addStitchPairs(false),
      assets_dir_clothmesh(""),
      assets_dir_clothmesh_save(""),
      assets_dir_bodymesh(""),
      assets_dir_simjson(""),
      clothmeshname(""),
      bodymeshname(""),
      bodytposename(""),
      assets_dir_input_simjson(""),
      assets_dir_output_clothmesh_json("") 
      { }


SimulationContext::~SimulationContext() {
    reset_ptr();
}




std::string SimulationContext::getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm;
    localtime_r(&now_time, &local_tm);
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y%m%d%H%M");
    return oss.str();
}


void SimulationContext::reset_ptr() {
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    LBVH_CD_ptr.reset();
    PCG_ptr.reset();
    BH_ptr.reset();
    simMesh_ptr.reset();
    glRender_ptr.reset();
    instance.reset();
    CUDA_SAFE_CALL(cudaDeviceReset());
}


void SimulationContext::restart_program(int index) {
    reset_ptr();

    char self_path[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", self_path, PATH_MAX);
    if (count == -1) {
        perror("读取可执行文件路径失败");
        exit(EXIT_FAILURE);
    }
    self_path[count] = '\0';

    // 将索引转换为字符串
    char index_str[10];
    snprintf(index_str, sizeof(index_str), "%d", index);

    // 准备参数，传递索引作为命令行参数
    char* const args[] = { self_path, index_str, NULL };

    // 重新启动程序
    execv(self_path, args);

    // 如果execv返回，说明出现错误
    perror("重新启动程序失败");
    exit(EXIT_FAILURE);
}




void SimulationContext::init_Geometry_ptr() {

    if (!instance) {
        instance = std::make_unique<GeometryManager>();
    }
    CHECK_ERROR(instance, "init instance not initialize");

    if (!LBVH_CD_ptr) {
        LBVH_CD_ptr = std::make_unique<LBVHCollisionDetector>();
    }
    CHECK_ERROR(LBVH_CD_ptr, "init LBVH_CD_ptr not initialize");

    if (!PCG_ptr) {
        PCG_ptr = std::make_unique<PCGSolver>();
    }
    CHECK_ERROR(PCG_ptr, "init PCG_ptr not initialize");

    if (!BH_ptr) {
        BH_ptr = std::make_unique<BlockHessian>();
    }
    CHECK_ERROR(BH_ptr, "init BH_ptr not initialize");

    if (!simMesh_ptr) {
        simMesh_ptr = std::make_unique<SIMMesh>();
    }
    CHECK_ERROR(simMesh_ptr, "init simMesh_ptr not initialize");

}



