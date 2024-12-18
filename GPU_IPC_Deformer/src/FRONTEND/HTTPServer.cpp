#include "HTTPServer.hpp"
#include <thread>
#include <chrono>
#include <atomic>
#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <iomanip>

// 生成YYYYMMDDHHMMSSmmm格式时间戳（精确到毫秒）
std::string HTTPServer::generate_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto in_time_t = system_clock::to_time_t(now);
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &in_time_t);
#else
    localtime_r(&in_time_t, &tm);
#endif

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d%H%M%S")
       << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}

// 保存JSON到本地文件
void HTTPServer::save_json_to_file(const nlohmann::json& j, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(4);
        file.close();
    } else {
        std::cerr << "Failed to open file for writing JSON: " << filename << std::endl;
    }
}

// 构造函数
HTTPServer::HTTPServer(SimulationContext& context, Simulator& simulator)
    : ctx(context), sim(simulator), params(std::make_unique<HTTPServerParam>()), stop_worker(false)
{
    params->is_simulation_running = false;
    params->is_simulation_completed = false;
    params->sim_frame_id = 0.0;
    params->sim_frame_range = 0.0;

    // 启动工作线程
    worker_thread = std::thread(&HTTPServer::worker, this);
}

// 析构函数
HTTPServer::~HTTPServer() {
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        stop_worker = true;
    }
    queue_cv.notify_all();
    if (worker_thread.joinable()) {
        worker_thread.join();
    }
}

// 处理HTTP请求
void HTTPServer::solve_Http() {
    httplib::Server svr;

    // 定义根路径的GET请求处理器
    svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("Hello, World!", "text/plain");
    });

    // 定义/echo路径的POST请求处理器
    svr.Post("/echo", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content(req.body, "text/plain");
    });

    // 定义/mesh路径的POST请求处理器
    svr.Post("/mesh", [this](const httplib::Request& req, httplib::Response& res) {
        handle_mesh_request(req, res);
    });

    // 定义/poll路径的GET请求处理器
    svr.Get("/poll", [this](const httplib::Request& req, httplib::Response& res) {
        handle_poll_request(req, res);
    });

    std::cout << "Server is running on http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);
}

// 处理/mesh请求的辅助函数
void HTTPServer::handle_mesh_request(const httplib::Request& req, httplib::Response& res) {
    
    try {
        // 解析请求体中的JSON数据
        nlohmann::json task = nlohmann::json::parse(req.body);

        // 生成基于当前时间戳的task_id
        std::string task_id = generate_timestamp();

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            // 将task_id写入任务JSON
            task["task_id"] = task_id;

            // 存储任务数据
            tasks_data[task_id] = task;

            // 保存输入JSON到本地文件
            save_json_to_file(task, task_id + ".json");

            // 设置任务初始状态
            TaskInfo info;
            info.status = "queued";
            info.progress = 0.0;
            tasks_status_map[task_id] = info;

            // 将task_id入队
            task_queue.push(task_id);
        }

        // 通知工作线程有新任务
        queue_cv.notify_one();

        // 构建响应JSON
        nlohmann::json response;
        response["task_status"] = tasks_status_map[task_id].status; // "queued"
        response["task_id"] = task_id;  // 返回task_id给客户端

        // 发送响应给客户端
        res.set_content(response.dump(), "application/json");
    } catch (const std::exception& e) {
        // 处理异常并返回错误响应
        nlohmann::json response;
        response["task_status"] = "error";
        response["error_message"] = e.what();
        res.status = 400; // Bad Request
        res.set_content(response.dump(), "application/json");
    }
}

// 处理/poll请求的辅助函数
void HTTPServer::handle_poll_request(const httplib::Request& req, httplib::Response& res) {
    
    try {

        nlohmann::json poll_response;

        // 获取查询参数中的task_id（如果有）
        std::string query_task_id;
        if (req.has_param("task_id")) {
            query_task_id = req.get_param_value("task_id");
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if (!query_task_id.empty()) {
                // 返回指定task_id的状态
                if (tasks_status_map.find(query_task_id) != tasks_status_map.end()) {
                    const TaskInfo& info = tasks_status_map[query_task_id];
                    nlohmann::json task_status;
                    task_status["status"] = info.status;
                    task_status["progress"] = info.progress;
                    if (info.status == "completed") {
                        task_status["result"] = info.result_json;
                    }

                    poll_response["task_id"] = query_task_id;
                    poll_response["task_status"] = task_status;
                } else {
                    // 未找到该task_id，返回错误响应
                    nlohmann::json task_status;
                    task_status["status"] = "error";
                    task_status["error_message"] = "task_id not found";

                    poll_response["task_status"] = task_status;
                    res.status = 404; // Not Found
                }
            } else {
                // 返回所有任务的状态
                for (const auto& kv : tasks_status_map) {
                    const std::string& id = kv.first;
                    const TaskInfo& info = kv.second;
                    nlohmann::json task_status;
                    task_status["status"] = info.status;
                    task_status["progress"] = info.progress;
                    if (info.status == "completed") {
                        task_status["result"] = info.result_json;
                    }
                    task_status["error_message"] = info.error_message; // 如果有错误信息

                    poll_response[id] = task_status;
                }
            }
        }

        // 发送响应给客户端
        res.set_content(poll_response.dump(), "application/json");
    } catch (const std::exception& e) {
        // 处理异常并返回错误响应
        nlohmann::json response;
        response["task_status"] = "error";
        response["error_message"] = e.what();
        res.status = 500; // Internal Server Error
        res.set_content(response.dump(), "application/json");
    }
}

// 工作线程函数，处理任务队列中的任务
void HTTPServer::worker() {
    while (true) {
        std::string current_task_id;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            // 等待任务队列非空或接收到停止信号
            queue_cv.wait(lock, [this] { return !task_queue.empty() || stop_worker; });

            // 如果接收到停止信号且任务队列为空，退出线程
            if (stop_worker && task_queue.empty())
                break;

            // 从队列中取出一个任务
            current_task_id = task_queue.front();
            task_queue.pop();
        }

        // 获取任务的JSON数据
        nlohmann::json task_json;
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            task_json = tasks_data[current_task_id];
            tasks_status_map[current_task_id].status = "running";
            tasks_status_map[current_task_id].progress = 0.0;
            tasks_status_map[current_task_id].error_message = ""; // 清空之前的错误信息
        }

        // 更新服务器参数
        params->is_simulation_running = true;
        params->is_simulation_completed = false;
        params->sim_frame_id = 0.0;
        params->sim_frame_range = 0.0;
        params->current_task = task_json;

        try {
            // 初始化仿真环境
            ctx.init_Geometry_ptr();
            ctx.simMesh_ptr->httpjson = task_json;
            ctx.simMesh_ptr->output_httpjson = nlohmann::json();

            // 开始仿真计算
            sim.init_Geometry_mesh();
            params->sim_frame_range = ctx.instance->getHostSimulationFrameRange();

            while (true) {
                sim.display_without_opengl_animation();
                params->sim_frame_id = ctx.instance->getHostSimulationFrameId();

                double curr_progress = 0.0;
                if (params->sim_frame_range != 0) {
                    curr_progress = (params->sim_frame_id / (params->sim_frame_range + Scalar(ctx.interpolation_frames))) * 100.0;
                }

                {
                    std::lock_guard<std::mutex> lock(queue_mutex);
                    if (tasks_status_map.find(current_task_id) != tasks_status_map.end()) {
                        tasks_status_map[current_task_id].progress = curr_progress;
                    }
                }

                if (ctx.simMesh_ptr->simulation_finished) {
                    params->is_simulation_completed = true;
                    break;
                }
            }

            // 保存结果JSON到本地文件
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                tasks_status_map[current_task_id].status = "completed";
                tasks_status_map[current_task_id].progress = 100.0;
                tasks_status_map[current_task_id].result_json = ctx.simMesh_ptr->output_httpjson;

                save_json_to_file(ctx.simMesh_ptr->output_httpjson, current_task_id + "_result.json");
            }
        } catch (const std::exception& e) {
            // 处理仿真过程中发生的异常
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                tasks_status_map[current_task_id].status = "error";
                tasks_status_map[current_task_id].error_message = e.what();
            }

            // 记录错误日志
            std::cerr << "Error processing task " << current_task_id << ": " << e.what() << std::endl;
        }

        // 更新服务器参数
        params->is_simulation_running = false;
        ctx.reset_ptr();
    }
}
