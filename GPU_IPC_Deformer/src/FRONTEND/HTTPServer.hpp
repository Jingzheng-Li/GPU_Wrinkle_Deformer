#ifndef HTTPSERVER_HPP
#define HTTPSERVER_HPP

#include "SimulationContext.hpp"
#include "Simulator.hpp"
#include <httplib.h>
#include <memory>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <nlohmann/json.hpp>
#include <map>
#include <string>

// 用于表示任务的状态信息
struct TaskInfo {
    std::string status;          // "queued", "running", "completed", "error"
    double progress;             // 0.0 to 100.0
    nlohmann::json result_json;  // 用于存储计算结果的JSON
    std::string error_message;   // 用于存储错误信息（如果有）
};

// 服务器参数结构体
struct HTTPServerParam {
    std::atomic<bool> is_simulation_running;
    std::atomic<bool> is_simulation_completed;
    Scalar sim_frame_id;
    Scalar sim_frame_range;
    nlohmann::json current_task;
};

class HTTPServer {
public:
    HTTPServer(SimulationContext& context, Simulator& simulator);
    ~HTTPServer();

    void solve_Http();

private:
    SimulationContext& ctx;
    Simulator& sim;

    std::unique_ptr<HTTPServerParam> params;

    std::queue<std::string> task_queue; // 存储task_id队列
    std::map<std::string, nlohmann::json> tasks_data; // task_id -> 任务原始json数据
    std::map<std::string, TaskInfo> tasks_status_map;  // task_id -> 任务状态

    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::thread worker_thread;
    bool stop_worker;

    // 生成基于当前时间戳的task_id
    std::string generate_timestamp();

    // 保存JSON到本地文件
    void save_json_to_file(const nlohmann::json& j, const std::string& filename);

    // 处理任务的工作线程函数
    void worker();

    // 处理/mesh请求的辅助函数
    void handle_mesh_request(const httplib::Request& req, httplib::Response& res);

    // 处理/poll请求的辅助函数
    void handle_poll_request(const httplib::Request& req, httplib::Response& res);
};

#endif // HTTPSERVER_HPP
