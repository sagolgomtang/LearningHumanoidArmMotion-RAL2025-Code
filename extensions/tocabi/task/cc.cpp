#include "cc.h"

#include <ros/package.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>

using namespace TOCABI;

namespace {
// Legacy single-map (interleaved) for reference:
// constexpr int kLegJointMap[CustomController::num_actuator_action] = {
//     0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};
constexpr int kLegJointMapAction[CustomController::num_actuator_action] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    // 0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};
    // 0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11};


constexpr int kLegJointMapObs[CustomController::num_actuator_action] = {
    // 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};

constexpr std::array<std::array<double, 2>, CustomController::num_actuator_action> kLegJointPosLimits = {{
    {-0.3, 0.3},
    {-0.3, 0.3},	
    {-0.5, 0.5},
    {-0.5, 0.5},
    {-1.0, 0.5},
    {-1.0, 0.5},
    {-0.3, 1.2},
    {-0.3, 1.2},
    {-0.8, 0.5},
    {-0.8, 0.5},
    {-0.6, 0.6},
    {-0.6, 0.6},
}};


// Arm order (elbow before armlink) for future action/obs mapping.
constexpr int kArmJointMapUserOrder[16] = {
    15, 16, 17, 19, 18, 20, 21, 22,  // L_Shoulder1,2,3, Elbow, Armlink, Forearm, Wrist1, Wrist2
    25, 26, 27, 29, 28, 30, 31, 32   // R_Shoulder1,2,3, Elbow, Armlink, Forearm, Wrist1, Wrist2
};

}

CustomController::CustomController(RobotData &rd) 
    :   rd_(rd), //, wbc_(dc.wbc_)
        env(ORT_LOGGING_LEVEL_WARNING, "tocabi"),
        memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
        session(nullptr)
{    
    ControlVal_.setZero();

    nh_.getParam("/tocabi_cc/weight_dir", weight_dir_);
    nh_.param("/tocabi_cc/phase_period", phase_period_s_, 1.2);
    // nh_.param("/tocabi_cc/phase_offset", phase_offset_s_, 0.5);
    nh_.param("/tocabi_cc/phase_offset", phase_offset_s_, 0.0);
    nh_.param("/tocabi_cc/enable_value_stop", enable_value_stop_, false);
    nh_.param("/tocabi_cc/cmd_scale_x", cmd_scale_x_, 0.5);
    nh_.param("/tocabi_cc/cmd_scale_y", cmd_scale_y_, 0.4);
    nh_.param("/tocabi_cc/cmd_scale_yaw", cmd_scale_yaw_, 0.6);
    nh_.param("/tocabi_cc/cmd_vis_scale", cmd_vis_scale_, 1.0);
    nh_.param("/tocabi_cc/q_limit_scale", q_limit_scale_, 1.0);

    const std::string pkg_path = ros::package::getPath("tocabi_cc");
    if (pkg_path.empty())
    {
        ROS_ERROR("tocabi_cc package path not found. Check your ROS setup.");
    }
    
    if (is_write_file_)
    {
        std::string file_tag = weight_dir_;
        std::replace(file_tag.begin(), file_tag.end(), '/', '_');
        if (is_on_robot_)
        {
            writeFile.open(pkg_path + "/result/" + file_tag + ".csv",
                           std::ofstream::out | std::ofstream::app);
        }
        else
        {
            writeFile.open(pkg_path + "/result/" + file_tag + "data.csv",
                           std::ofstream::out | std::ofstream::trunc);
        }
        writeFile << std::fixed << std::setprecision(8);
    }
    initVariable();
    loadJointLimits();
    // Override action mapping if provided
    std::vector<double> action_scale_param;
    if (nh_.getParam("/tocabi_cc/action_scale", action_scale_param) &&
        action_scale_param.size() == static_cast<size_t>(num_actuator_action))
    {
        for (int i = 0; i < num_actuator_action; i++)
        {
            action_scale_(i, i) = action_scale_param[i];
        }
    }
    std::vector<double> action_offset_param;
    if (nh_.getParam("/tocabi_cc/action_offset", action_offset_param) &&
        action_offset_param.size() == static_cast<size_t>(num_actuator_action))
    {
        for (int i = 0; i < num_actuator_action; i++)
        {
            action_offset_(i, i) = action_offset_param[i];
        }
    }
    loadOnnX();

    joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy_gui", 10, &CustomController::joyCallback, this);
    // xbox_joy_sub_ = nh_.subscribe<sensor_msgs::Joy>("/joy", 10, &CustomController::xBoxJoyCallback, this);
    sim_command_pub_ = nh_.advertise<std_msgs::String>("/mujoco_ros_interface/sim_command_con2sim", 1);
    sim_time_sub_ = nh_.subscribe<std_msgs::Float32>("/mujoco_ros_interface/sim_time", 1, &CustomController::simTimeCallback, this);
    cmd_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/tocabi_cc/cmd_marker", 1);
    cmd_vec_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/tocabi_cc/cmd_vel", 1);
    cam_cmd_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("/tocabi_cc/cam_cmd", 1);
    gui_send_pub_ = nh_.advertise<std_msgs::Empty>("/tocabi_gui/send_position_command", 1);
    gui_send_sub_ = nh_.subscribe<std_msgs::Empty>("/tocabi_gui/send_position_command", 1, &CustomController::guiSendCallback, this);
    task_cmd_pub_ = nh_.advertise<tocabi_msgs::TaskCommand>("/tocabi/taskcommand", 1);
    pos_cmd_pub_ = nh_.advertise<tocabi_msgs::positionCommand>("/tocabi/positioncommand", 1);
}

void CustomController::initVariable()
{    
    rl_action_.resize(num_action, 1);
    rl_action_pre_.resize(num_action, 1);
    torq_diff_.resize(num_action, 1);
    energy.resize(num_action, 1);
    rl_action_.setZero();
    rl_action_pre_.setZero();
    torq_diff_.setZero();
    energy.setZero();

    state_cur_.resize(num_cur_state, 1);
    state_buffer_.resize(num_cur_state*num_state_skip*num_state_hist, 1);
    std::fill(state_cur_.begin(), state_cur_.end(), 0.0f);
    std::fill(state_buffer_.begin(), state_buffer_.end(), 0.0f);

    if (is_hist_encoder_) { 
        state_long_hist_.resize(num_hist_state * num_cur_state, 1); 
        state_long_hist_buffer_.resize(num_long_hist_len * num_cur_state, 1);
    }

    q_dot_lpf_.setZero();

    torque_bound_ << 333, 232, 263, 289, 222, 166,
                    333, 232, 263, 289, 222, 166,
                    303, 303, 303, 
                    64, 64, 64, 64, 23, 23, 10, 10,
                    10, 10,
                    64, 64, 64, 64, 23, 23, 10, 10;  
                    
    q_init_ << 0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, -0.24, 0.6, -0.36, 0.0,
                0.0, 0.0, 0.0,
                0.3, 0.3, 1.5, -1.27, -1.0, 0.0, -1.0, 0.0,
                0.0, 0.0,
                -0.3, -0.3, -1.5, 1.27, 1.0, 0.0, 1.0, 0.0;
    q_init_mode7_ = q_init_;

    pace_direction_ << 1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0,
                       1.0, 1.0, 1.0, 1.0;
    pace_bias_ << 0.0, 0.4, 0.8,
                  0.0, 0.4, 0.8,
                  0.0, 0.4, 0.8,
                  0.0, 0.4, 0.8;
    pace_scale_ << 0.5, 0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5, 0.5,
                   0.5, 0.5, 0.5, 0.5;
    pace_hold_q_ = q_init_;

    kp_.setZero();
    kv_.setZero();
    kp_.diagonal() <<   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0,
                        6000.0, 10000.0, 10000.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0,
                        100.0, 100.0,
                        400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0;
    kv_.diagonal() << 15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        15.0, 50.0, 20.0, 25.0, 24.0, 24.0,
                        200.0, 100.0, 100.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
                        2.0, 2.0,
                        10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0;

    action_offset_.setZero();
    action_scale_.setIdentity();
    for (int i = 0; i < num_actuator_action; i++)
    {
        action_offset_(i, i) = q_init_(kLegJointMapAction[i]);
        action_scale_(i, i) = 1.0;
    }
}

void CustomController::loadJointLimits()
{
    q_min_.setConstant(-1.0e9);
    q_max_.setConstant(1.0e9);

    std::string urdf_path;
    std::string robot_desc;
    urdf::Model model;
    bool ok = false;

    if (!ros::param::get("/tocabi_controller/urdf_path", urdf_path) || urdf_path.empty())
    {
        urdf_path = "/home/user/tocabi_mujoco_ws/src/dyros_tocabi_v2/tocabi_description/robots/dyros_tocabi.urdf";
        ros::param::set("/tocabi_controller/urdf_path", urdf_path);
    }

    if (!urdf_path.empty())
    {
        ok = model.initFile(urdf_path);
    }
    else if (ros::param::get("/robot_description", robot_desc) && !robot_desc.empty())
    {
        ok = model.initString(robot_desc);
    }

    if (!ok)
    {
        ROS_WARN("Failed to load URDF for joint limits. q_des clipping disabled.");
        return;
    }

    for (int i = 0; i < MODEL_DOF; i++)
    {
        const auto joint = model.getJoint(TOCABI::JOINT_NAME[i]);
        if (!joint || !joint->limits)
        {
            continue;
        }
        q_min_(i) = joint->limits->lower * q_limit_scale_;
        q_max_(i) = joint->limits->upper * q_limit_scale_;
    }

    // override leg joint limits for action scaling (lower body only)
    for (int i = 0; i < num_actuator_action; i++)
    {
        int joint_idx = kLegJointMapAction[i];
        q_min_(joint_idx) = kLegJointPosLimits[i][0] * q_limit_scale_;
        q_max_(joint_idx) = kLegJointPosLimits[i][1] * q_limit_scale_;
    }

    has_joint_limits_ = true;
}


void CustomController::loadOnnX()
{
    const std::string pkg_path = ros::package::getPath("tocabi_cc");
    if (pkg_path.empty())
    {
        ROS_ERROR("tocabi_cc package path not found. Check your ROS setup.");
        return;
    }

    std::string cur_path = weight_dir_;
    if (!cur_path.empty() && cur_path.front() != '/')
    {
        cur_path = pkg_path + "/" + cur_path;
    }

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.AddConfigEntry("session.use_deterministic_compute", "1");

    session = Ort::Session(env, cur_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    input_number = session.GetInputCount();
    output_number = session.GetOutputCount();

    input_names.resize(input_number);
    output_names.resize(output_number);

    input_names_char.resize(input_names.size());
    output_names_char.resize(output_names.size());

    for (size_t i = 0; i < input_number; i++) {
        Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(i, allocator);
        input_names[i] = input_name.get();
    }
    for (size_t i = 0; i < output_number; i++) {
        Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(i, allocator);
        output_names[i] = output_name.get();
    }

    // Print input/output names
    std::ostringstream in_names_ss;
    for (size_t i = 0; i < input_names.size(); ++i)
    {
        if (i > 0)
        {
            in_names_ss << " ";
        }
        in_names_ss << input_names[i];
    }
    std::ostringstream out_names_ss;
    for (size_t i = 0; i < output_names.size(); ++i)
    {
        if (i > 0)
        {
            out_names_ss << " ";
        }
        out_names_ss << output_names[i];
    }
    ROS_INFO_STREAM("[ONNX] Input names: " << in_names_ss.str());
    ROS_INFO_STREAM("[ONNX] Output names: " << out_names_ss.str());

    bool obs_found = false;
    for (size_t i = 0; i < input_names.size(); ++i) { 
        input_names_char[i] = input_names[i].c_str();
        if (input_names[i] == "obs")
        {
            input_obs_idx_ = static_cast<int>(i);
            obs_found = true;
        }
    }
    if (!obs_found)
    {
        ROS_WARN_STREAM("ONNX input 'obs' not found. Using input index 0.");
        input_obs_idx_ = 0;
    }
    for (size_t i = 0; i < output_names.size(); ++i) { output_names_char[i] = output_names[i].c_str();}

    // Initialize input tensors
    for (size_t i = 0; i < input_number; ++i) {
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_shape = tensor_info.GetShape();
        std::ostringstream shape_ss;
        shape_ss << "[";
        for (size_t j = 0; j < input_shape.size(); ++j)
        {
            if (j > 0)
            {
                shape_ss << ", ";
            }
            shape_ss << input_shape[j];
        }
        shape_ss << "]";
        ROS_INFO_STREAM("[ONNX] Input " << i << " shape: " << shape_ss.str()
                                        << " elem_count=" << tensor_info.GetElementCount());
        std::vector<float> input_tensor_values(tensor_info.GetElementCount(), 0.0);
        input_states_buffer.push_back(std::move(input_tensor_values));

        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_states_buffer.back().data(),
            input_states_buffer.back().size(),
            input_shape.data(),
            input_shape.size()));
    }
        
}

void CustomController::processNoise()
{
    time_cur_ = rd_cc_.control_time_us_ / 1e6;
    q_vel_noise_pre_ = q_vel_noise_;
    rl_action_pre_ = rl_action_;
    if (is_on_robot_)
    {
        q_vel_noise_ = rd_cc_.q_dot_virtual_.segment(6,MODEL_DOF);
        q_noise_= rd_cc_.q_virtual_.segment(6,MODEL_DOF);
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_dot_lpf_ = q_dot_lpf_;
        }
    }
    else
    {
        std::random_device rd;  
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-0.00001, 0.00001);
        for (int i = 0; i < MODEL_DOF; i++) {
            q_noise_(i) = rd_cc_.q_virtual_(6+i) + dis(gen);
        }
        if (time_cur_ - time_pre_ > 0.0)
        {
            q_vel_noise_ = (q_noise_ - q_noise_pre_) / (time_cur_ - time_pre_);
            q_dot_lpf_ = DyrosMath::lpf<MODEL_DOF>(q_vel_noise_, q_dot_lpf_, 1/(time_cur_ - time_pre_), 4.0);
        }
        else
        {
            q_vel_noise_ = q_vel_noise_;
            q_dot_lpf_ = q_dot_lpf_;
        }
        q_noise_pre_ = q_noise_;
    }
    time_pre_ = time_cur_;
}

void CustomController::processObservation()
{
    ROS_INFO_STREAM("[DBG] before obs last_action=" << rl_action_.transpose());

    int data_idx = 0;
    static int obs_log_counter = 0;
    static int cmd_log_counter = 0;
    static int base_vel_log_counter = 0;
    // if (debug_log_this_step_)
    // {
    //     ROS_INFO("[OBS_STEP] entry");
    // }
    
    Eigen::Quaterniond q;
    q.x() = rd_cc_.q_virtual_(3);
    q.y() = rd_cc_.q_virtual_(4);
    q.z() = rd_cc_.q_virtual_(5);
    q.w() = rd_cc_.q_virtual_(MODEL_DOF_QVIRTUAL-1);

    euler_angle_ = DyrosMath::rot2Euler_tf(q.toRotationMatrix());

    // base_lin_vel (body frame)
    Eigen::Vector3d base_lin_vel_bf = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(0, 3));
    state_cur_[data_idx++] = base_lin_vel_bf(0);
    state_cur_[data_idx++] = base_lin_vel_bf(1);
    state_cur_[data_idx++] = base_lin_vel_bf(2);
    if ((base_vel_log_counter++ % 20) == 0)
    {
        ROS_INFO_STREAM(
            "[BASE_VEL] qdot_lin="
            << rd_cc_.q_dot_virtual_(0) << " "
            << rd_cc_.q_dot_virtual_(1) << " "
            << rd_cc_.q_dot_virtual_(2)
            << " qdot_ang="
            << rd_cc_.q_dot_virtual_(3) << " "
            << rd_cc_.q_dot_virtual_(4) << " "
            << rd_cc_.q_dot_virtual_(5));
    }

    // base_heading (world yaw)
    state_cur_[data_idx++] = euler_angle_(2);

    // base_ang_vel (body frame)
    Eigen::Vector3d base_ang_vel_bf = quatRotateInverse(q, rd_cc_.q_dot_virtual_.segment(3, 3));
    state_cur_[data_idx++] = base_ang_vel_bf(0);
    state_cur_[data_idx++] = base_ang_vel_bf(1);
    state_cur_[data_idx++] = base_ang_vel_bf(2);

    // projected_gravity (body frame)
    Eigen::Vector3d gravity_bf = quatRotateInverse(q, Eigen::Vector3d(0.0, 0.0, -1.0));
    state_cur_[data_idx++] = gravity_bf(0);
    state_cur_[data_idx++] = gravity_bf(1);
    state_cur_[data_idx++] = gravity_bf(2);

    // velocity_commands
    state_cur_[data_idx++] = target_vel_x_;
    state_cur_[data_idx++] = target_vel_y_;
    state_cur_[data_idx++] = target_vel_yaw_;
    publishCommandMarker();
    publishCommandVector();
    // if ((cmd_log_counter++ % 50) == 0)
    // {
    //     ROS_INFO_STREAM("[JOY_CMD] x=" << target_vel_x_
    //                                    << " y=" << target_vel_y_
    //                                    << " yaw=" << target_vel_yaw_);
    // }

    // phase
    // if (phase_period_s_ > 0.0)
    // {
    //     if (sim_time_received_)
    //     {
    //         const double since_last_msg = (ros::Time::now() - last_sim_time_msg_).toSec();
    //         sim_paused_ = (since_last_msg > 0.2);
    //     }
    //     const double two_pi = 6.283185307179586;
    //     double t = 0.0;
    //     if (phase_started_)
    //     {
    //         if (sim_time_received_)
    //         {
    //             if (!sim_paused_)
    //             {
    //                 double dt = 0.0;
    //                 if (sim_time_prev_s_ > 0.0)
    //                 {
    //                     dt = sim_time_s_ - sim_time_prev_s_;
    //                 }
    //                 // Fallback to controller time if sim_time stalls or jumps.
    //                 if (!(dt > 0.0 && dt < 1.0) && phase_last_update_us_ > 0)
    //                 {
    //                     dt = (rd_cc_.control_time_us_ - phase_last_update_us_) / 1.0e6;
    //                 }
    //                 if (dt > 0.0 && dt < 1.0)
    //                 {
    //                     phase_time_s_ += dt;
    //                 }
    //             }
    //             sim_time_prev_s_ = sim_time_s_;
    //             phase_last_update_us_ = rd_cc_.control_time_us_;
    //         }
    //         else
    //         {
    //             if (!sim_paused_ && phase_last_update_us_ > 0)
    //             {
    //                 const double dt = (rd_cc_.control_time_us_ - phase_last_update_us_) / 1.0e6;
    //                 if (dt > 0.0 && dt < 1.0)
    //                 {
    //                     phase_time_s_ += dt;
    //                 }
    //             }
    //             phase_last_update_us_ = rd_cc_.control_time_us_;
    //         }
    //         t = phase_time_s_ + phase_offset_s_;
    //     }
    //     double phase = two_pi * (t / phase_period_s_);
    //     state_cur_[data_idx++] = std::sin(phase);
    //     state_cur_[data_idx++] = std::cos(phase);
    // }
    // else
    // {
    //     state_cur_[data_idx++] = 0.0;
    //     state_cur_[data_idx++] = 1.0;
    // }
    // phase
    if (phase_period_s_ > 0.0)
    {
        const double two_pi = 6.283185307179586;

        if (phase_started_)
        {
            // "obs가 계산되는 시점"에서만 이 함수가 호출되므로
            // dt는 control_time 기반으로 잡는 게 가장 안전함
            if (phase_last_update_us_ > 0)
            {
                double dt = (rd_cc_.control_time_us_ - phase_last_update_us_) / 1.0e6;
                if (dt > 0.0 && dt < 1.0)  // 안전 클램프
                {
                    phase_time_s_ += dt;
                }
            }
            phase_last_update_us_ = rd_cc_.control_time_us_;
        }

        double t = phase_started_ ? (phase_time_s_ + phase_offset_s_) : phase_offset_s_;
        double phase = two_pi * (t / phase_period_s_);
        state_cur_[data_idx++] = std::sin(phase);
        state_cur_[data_idx++] = std::cos(phase);
    }
    else
    {
        state_cur_[data_idx++] = 0.0;
        state_cur_[data_idx++] = 1.0;
    }


    // joint_pos (legs)
    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_[data_idx++] = q_noise_(kLegJointMapObs[i]);
    }

    // joint_vel (legs)
    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_[data_idx++] = q_vel_noise_(kLegJointMapObs[i]);
    }

    // last_leg_action
    for (int i = 0; i < num_actuator_action; i++)
    {
        state_cur_[data_idx++] = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
    }

    // CAM (centroidal angular momentum, mixed frame) - match tocabi_vanilla CM/CM_bf
    Eigen::Vector6d cm = rd_cc_.CMM * rd_cc_.q_dot_virtual_;
    Eigen::Matrix3d Rwb = q.toRotationMatrix();
    Eigen::Matrix3d Rbw = Rwb.transpose();
    Eigen::Vector6d cm_bf;
    cm_bf.head<3>() = Rbw * cm.head<3>();
    cm_bf.tail<3>() = Rbw * cm.tail<3>();
    state_cur_[data_idx++] = cm_bf(3);
    state_cur_[data_idx++] = cm_bf(4);
    state_cur_[data_idx++] = cm(5);

    size_t buffer_size = num_cur_state * num_state_skip * num_state_hist;
    std::copy(state_buffer_.begin() + num_cur_state, state_buffer_.end(), state_buffer_.begin());
    std::copy(state_cur_.begin(), state_cur_.end(), state_buffer_.begin() + buffer_size - num_cur_state);

    size_t obs_size = input_states_buffer[input_obs_idx_].size();
    // if (debug_log_this_step_)
    // {
    //     ROS_INFO_STREAM("[OBS_META] obs_size=" << obs_size
    //                                           << " num_cur_state=" << num_cur_state
    //                                           << " num_state_hist=" << num_state_hist
    //                                           << " num_state_skip=" << num_state_skip);
    // }
    if (obs_size == static_cast<size_t>(num_cur_state))
    {
        std::copy(state_cur_.begin(), state_cur_.end(), input_states_buffer[input_obs_idx_].begin());
        // if (debug_log_this_step_)
        // {
        //     ROS_INFO("[OBS_STEP] branch=cur");
        // }
        // bool log_due = debug_log_this_step_ || ((obs_log_counter++ % 100) == 0);
        // if (log_due)
        // {
        //     std::ostringstream oss;
        //     oss << std::fixed << std::setprecision(3);
        //     oss << "[OBS] obs[0:" << num_cur_state << "]=";
        //     for (int i = 0; i < num_cur_state; i++)
        //     {
        //         if (i > 0)
        //         {
        //             oss << ", ";
        //         }
        //         oss << state_cur_[i];
        //     }
        //     ROS_INFO_STREAM(oss.str());
        // }
        return;
    }
    if (obs_size == static_cast<size_t>(num_cur_state * num_state_hist))
    {
        for (size_t i = 0; i < num_state_hist; ++i)
        {
            std::copy(state_buffer_.begin() + num_cur_state * (num_state_skip * (i + 1) - 1),
                      state_buffer_.begin() + num_cur_state * (num_state_skip * (i + 1)),
                      input_states_buffer[input_obs_idx_].begin() + num_cur_state * i);
        }
        // if (debug_log_this_step_)
        // {
        //     ROS_INFO("[OBS_STEP] branch=hist");
        // }
        // bool log_due = debug_log_this_step_ || ((obs_log_counter++ % 100) == 0);
        // if (log_due)
        // {
        //     std::ostringstream oss;
        //     oss << std::fixed << std::setprecision(3);
        //     oss << "[OBS] obs[0:" << num_cur_state << "]=";
        //     for (int i = 0; i < num_cur_state; i++)
        //     {
        //         if (i > 0)
        //         {
        //             oss << ", ";
        //         }
        //         oss << input_states_buffer[input_obs_idx_][i];
        //     }
        //     ROS_INFO_STREAM(oss.str());
        // }
        return;
    }

    // Fallback: fill what we can in order.
    size_t copy_n = std::min(obs_size, static_cast<size_t>(num_cur_state));
    std::copy(state_cur_.begin(), state_cur_.begin() + copy_n, input_states_buffer[input_obs_idx_].begin());
    // if (debug_log_this_step_)
    // {
    //     ROS_INFO("[OBS_STEP] branch=fallback");
    //     std::ostringstream oss;
    //     oss << std::fixed << std::setprecision(3);
    //     oss << "[OBS_FALLBACK] obs[0:" << copy_n << "]=";
    //     for (size_t i = 0; i < copy_n; i++)
    //     {
    //         if (i > 0)
    //         {
    //             oss << ", ";
    //         }
    //         oss << state_cur_[i];
    //     }
    //     ROS_INFO_STREAM(oss.str());
    // }

}

void CustomController::publishCommandMarker()
{
    if (cmd_marker_pub_.getNumSubscribers() == 0)
    {
        return;
    }
    visualization_msgs::Marker marker;
    marker.header.stamp = ros::Time::now();
    marker.header.frame_id = "world";
    marker.ns = "tocabi_cmd";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.action = visualization_msgs::Marker::ADD;
    marker.lifetime = ros::Duration(0.2);

    geometry_msgs::Point p0;
    p0.x = rd_cc_.q_virtual_(0);
    p0.y = rd_cc_.q_virtual_(1);
    p0.z = rd_cc_.q_virtual_(2);

    geometry_msgs::Point p1;
    p1.x = p0.x + target_vel_x_ * cmd_vis_scale_;
    p1.y = p0.y + target_vel_y_ * cmd_vis_scale_;
    p1.z = p0.z;

    marker.points.push_back(p0);
    marker.points.push_back(p1);
    marker.scale.x = 0.05;  // shaft diameter
    marker.scale.y = 0.1;   // head diameter
    marker.scale.z = 0.1;   // head length
    marker.color.a = 0.9;
    marker.color.r = 0.1;
    marker.color.g = 0.8;
    marker.color.b = 0.2;

    cmd_marker_pub_.publish(marker);
}

void CustomController::publishCommandVector()
{
    if (cmd_vec_pub_.getNumSubscribers() == 0)
    {
        return;
    }
    std_msgs::Float32MultiArray msg;
    msg.data.resize(3);
    msg.data[0] = static_cast<float>(target_vel_x_);
    msg.data[1] = static_cast<float>(target_vel_y_);
    msg.data[2] = static_cast<float>(target_vel_yaw_);
    cmd_vec_pub_.publish(msg);
}

void CustomController::feedforwardPolicy()
{
    // std::fill(input_states_buffer[0].begin(), input_states_buffer[0].end(), 0.0);
    // std::fill(input_states_buffer[1].begin(), input_states_buffer[1].end(), 0.0);
    static int action_log_counter = 0;
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(), input_number, output_names_char.data(), output_number);

    if (output_tensors.empty())
    {
        ROS_ERROR("ONNX output_tensors is empty.");
        return;
    }

    for (size_t i = 0; i < output_tensors.size(); i++) {
        if (!output_tensors[i].IsTensor()) {
            std::cerr << "Output " << i << " is not a valid tensor." << std::endl;
            continue;
        }
    }

    // output tensor to rl_action_
    const float *action_data = output_tensors[0].GetTensorMutableData<float>();
    Ort::TypeInfo action_info = output_tensors[0].GetTypeInfo();
    auto action_tensor_info = action_info.GetTensorTypeAndShapeInfo();
    size_t action_count = action_tensor_info.GetElementCount();
    if (action_count < num_action)
    {
        ROS_ERROR("ONNX action output has %zu elements; expected at least %d.", action_count, num_action);
        return;
    }
    for (size_t i = 0; i < num_action; i++) {
        rl_action_(i) = DyrosMath::minmax_cut(action_data[i], -1.0, 1.0) * 1.0;
    }
    // bool log_due = debug_log_this_step_ || ((action_log_counter++ % 100) == 0);
    // if (log_due)
    // {
    //     std::ostringstream oss_raw;
    //     std::ostringstream oss_clip;
    //     oss_raw << std::fixed << std::setprecision(3);
    //     oss_clip << std::fixed << std::setprecision(3);
    //     oss_raw << "[ACTION_RAW] action[0:" << num_action << "]=";
    //     oss_clip << "[ACTION_CLIP] action[0:" << num_action << "]=";
    //     for (int i = 0; i < num_action; i++)
    //     {
    //         if (i > 0)
    //         {
    //             oss_raw << ", ";
    //             oss_clip << ", ";
    //         }
    //         oss_raw << action_data[i];
    //         oss_clip << rl_action_(i);
    //     }
    //     ROS_INFO_STREAM(oss_raw.str());
    //     ROS_INFO_STREAM(oss_clip.str());
    // }
    // cout << "RL Action: " << rl_action_.transpose() << endl;
    // output tensor to value_
    if (output_tensors.size() > 2)
    {
        value_ = output_tensors[2].GetTensorMutableData<float>()[0];
        value_valid_ = true;
    }
    else
    {
        value_ = 0.0;
        value_valid_ = false;
    }
    ROS_INFO_STREAM("[DBG] after infer new_action=" << rl_action_.transpose());

}

void CustomController::computeSlow()
{
    copyRobotData(rd_);
    const int tc_mode = rd_cc_.tc_.mode;
    if (debug_log_steps_remaining_ > 0)
    {
        debug_log_this_step_ = true;
        debug_log_steps_remaining_--;
    }
    else
    {
        debug_log_this_step_ = false;
    }
    if (tc_mode == 6)
    {
        if (rd_cc_.tc_init)
        {
            rd_.tc_init = false;
        }
        const bool mode6_rising = (prev_tc_mode_ != 6) || rd_cc_.tc_init;
        if (mode6_rising)
        {
            mode6_done_ = false;
            if (mode6_active_)
            {
                mode6_active_ = false;
                if (test_act_file_.is_open())
                {
                    test_act_file_.close();
                }
                if (test_act_mapped_file_.is_open())
                {
                    test_act_mapped_file_.close();
                }
            }
        }
        if (mode6_done_)
        {
            rd_.torque_desired.setZero();
            rd_.pc_mode = false;
            rd_.pc_gravity = false;
            prev_tc_mode_ = tc_mode;
            return;
        }
        if (!mode6_active_ && mode6_rising)
        {
            rd_.tc_run = true;
            mode6_active_ = true;
            sim_paused_ = false;
            mode6_logged_ = false;
            test_log_step_ = 0;
            test_act_buffer_.clear();
            start_time_ = rd_cc_.control_time_us_;
            phase_started_ = true;
            phase_start_time_s_ = start_time_ / 1.0e6;
            phase_time_s_ = 0.0;
            phase_last_update_us_ = rd_cc_.control_time_us_;
            sim_time_prev_s_ = sim_time_s_;
            q_init_mode7_ = rd_cc_.q_virtual_.segment(6, MODEL_DOF);
            q_noise_pre_ = q_noise_ = q_init_mode7_;
            time_cur_ = start_time_ / 1e6;
            time_pre_ = time_cur_ - 0.005;
            time_inference_pre_ = rd_cc_.control_time_us_ - (1 / 249.9) * 1e6;
            torque_init_ = rd_cc_.torque_desired;
            torque_rl_ = torque_init_;
            torque_spline_ = torque_init_;
            if (test_act_file_.is_open())
            {
                test_act_file_.close();
            }
            test_act_file_.open(test_log_dir_ + "/" + test_act_runtime_name_,
                                std::ofstream::out | std::ofstream::trunc);
            if (test_act_mapped_file_.is_open())
            {
                test_act_mapped_file_.close();
            }
            test_act_mapped_file_.open(test_log_dir_ + "/" + test_act_mapped_name_,
                                       std::ofstream::out | std::ofstream::trunc);
            if (!test_act_file_)
            {
                ROS_WARN_STREAM("[TEST_LOG] Failed to open files in " << test_log_dir_);
            }
            else
            {
                test_act_file_ << std::fixed << std::setprecision(6);
                ROS_INFO_STREAM("[TEST_LOG] Recording mode6 to " << test_log_dir_);
            }
            if (test_act_mapped_file_)
            {
                test_act_mapped_file_ << std::fixed << std::setprecision(6);
            }
            test_obs_buffer_.clear();
            test_obs_idx_ = 0;
            const std::string obs_path = test_log_dir_ + "/" + test_obs_input_name_;
            std::ifstream obs_in(obs_path);
            if (!obs_in)
            {
                ROS_WARN_STREAM("[TEST_LOG] Failed to read " << obs_path);
            }
            else
            {
                std::string line;
                while (std::getline(obs_in, line))
                {
                    if (line.empty())
                    {
                        continue;
                    }
                    std::istringstream iss(line);
                    double step_idx = 0.0;
                    if (!(iss >> step_idx))
                    {
                        continue;
                    }
                    std::vector<float> obs;
                    obs.reserve(num_cur_state);
                    double val = 0.0;
                    while (iss >> val)
                    {
                        obs.push_back(static_cast<float>(val));
                    }
                    if (obs.size() >= static_cast<size_t>(num_cur_state))
                    {
                        if (obs.size() > static_cast<size_t>(num_cur_state))
                        {
                            obs.resize(static_cast<size_t>(num_cur_state));
                        }
                        test_obs_buffer_.push_back(std::move(obs));
                    }
                }
                ROS_INFO_STREAM("[TEST_LOG] Loaded " << test_obs_buffer_.size()
                                                    << " obs rows from " << obs_path);
            }
        }

        processNoise();
        bool updated = false;
        const bool use_logged_obs = true;
        if (!use_logged_obs || test_obs_idx_ < test_obs_buffer_.size())
        {
            if ((rd_cc_.control_time_us_ - time_inference_pre_) / 1.0e6 >= 1 / 100.0)
            {
                if (use_logged_obs)
                {
                    const std::vector<float> &obs_row = test_obs_buffer_[test_obs_idx_];
                    if (input_obs_idx_ >= 0 && input_obs_idx_ < static_cast<int>(input_states_buffer.size()))
                    {
                        std::vector<float> &input_obs = input_states_buffer[input_obs_idx_];
                        if (input_obs.size() == obs_row.size())
                        {
                            std::copy(obs_row.begin(), obs_row.end(), input_obs.begin());
                        }
                        else if (input_obs.size() == static_cast<size_t>(num_cur_state * num_state_hist))
                        {
                            std::fill(input_obs.begin(), input_obs.end(), 0.0f);
                            std::copy(obs_row.begin(), obs_row.end(), input_obs.begin());
                        }
                    }
                }
                else
                {
                    processObservation();
                }
                feedforwardPolicy();
            if (test_act_file_ || test_act_mapped_file_)
            {
                if (test_act_file_)
                {
                    test_act_file_ << test_log_step_ << "\t";
                    for (int i = 0; i < num_action; i++)
                    {
                        test_act_file_ << rl_action_(i);
                        if (i + 1 < num_action)
                        {
                            test_act_file_ << " ";
                        }
                    }
                    test_act_file_ << "\n";
                }
                if (test_act_mapped_file_)
                {
                    std::array<double, CustomController::num_actuator_action> mapped{};
                    for (int i = 0; i < num_actuator_action; i++)
                    {
                        int joint_idx = kLegJointMapAction[i];
                        mapped[joint_idx] = rl_action_(i);
                    }
                    test_act_mapped_file_ << test_log_step_ << "\t";
                    for (int i = 0; i < num_actuator_action; i++)
                    {
                        test_act_mapped_file_ << mapped[i];
                        if (i + 1 < num_actuator_action)
                        {
                            test_act_mapped_file_ << " ";
                        }
                    }
                    test_act_mapped_file_ << "\n";
                }
                test_log_step_++;
            }
                if (use_logged_obs)
                {
                    test_obs_idx_++;
                }

                Vector12d target_pos;
                for (int i = 0; i < num_actuator_action; i++)
                {
                    double a = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
                    const double lo = kLegJointPosLimits[i][0] * q_limit_scale_;
                    const double hi = kLegJointPosLimits[i][1] * q_limit_scale_;
                    double target = 0.5 * (a + 1.0) * (hi - lo) + lo;
                if (has_joint_limits_)
                {
                    int joint_idx = kLegJointMapAction[i];
                    target = DyrosMath::minmax_cut(target, q_min_(joint_idx), q_max_(joint_idx));
                }
                    target_pos(i) = target;
                }
                for (int i = 0; i < num_actuator_action; i++)
                {
                    int joint_idx = kLegJointMapAction[i];
                    torque_rl_(joint_idx) = kp_(joint_idx, joint_idx) / 9.0 *
                                                (target_pos(i) - q_noise_(joint_idx)) -
                                            kv_(joint_idx, joint_idx) / 3.0 * q_vel_noise_(joint_idx);
                }
                for (int i = num_actuator_action; i < MODEL_DOF; i++)
                {
                    torque_rl_(i) = kp_(i, i) * (q_init_mode7_(i) - q_noise_(i)) -
                                    kv_(i, i) * q_vel_noise_(i);
                }

                if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
                {
                    for (int i = 0; i < MODEL_DOF; i++)
                    {
                        torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_,
                                                            start_time_,
                                                            start_time_ + 0.1e6,
                                                            torque_init_(i),
                                                            torque_rl_(i),
                                                            0.0,
                                                            0.0);
                    }
                    rd_.torque_desired = torque_spline_;
                }
                else
                {
                    rd_.torque_desired = torque_rl_;
                }
                for (int i = 0; i < MODEL_DOF; i++)
                {
                    rd_.torque_desired(i) = DyrosMath::minmax_cut(
                        rd_.torque_desired(i), -torque_bound_(i), torque_bound_(i));
                }
                updated = true;
                rd_.pc_mode = false;
                rd_.pc_gravity = false;
                time_inference_pre_ = rd_cc_.control_time_us_;
            }
        }
        else if (use_logged_obs)
        {
            mode6_active_ = false;
            mode6_done_ = true;
            if (test_act_file_.is_open())
            {
                test_act_file_.close();
            }
            ROS_INFO("[TEST_LOG] Mode6 finished.");
        }
        if (!updated)
        {
            if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
            {
                rd_.torque_desired = torque_spline_;
            }
            else
            {
                rd_.torque_desired = torque_rl_;
            }
            for (int i = 0; i < MODEL_DOF; i++)
            {
                rd_.torque_desired(i) = DyrosMath::minmax_cut(
                    rd_.torque_desired(i), -torque_bound_(i), torque_bound_(i));
            }
            rd_.pc_mode = false;
            rd_.pc_gravity = false;
        }
        prev_tc_mode_ = tc_mode;
        return;
    }
    if (mode6_active_)
    {
        mode6_active_ = false;
        if (test_obs_file_.is_open())
        {
            test_obs_file_.close();
        }
        if (test_act_file_.is_open())
        {
            test_act_file_.close();
        }
        ROS_INFO("[TEST_LOG] Mode6 stopped.");
    }
        if (tc_mode == 7)
        {
            static int obs_sim_tick = 0;
            auto write_obs_sim = [&]() {
                if (test_obs_sim_done_ || !test_obs_sim_file_)
                {
                    return;
                }
            test_obs_sim_file_ << test_obs_sim_step_ << "\t";
            for (int i = 0; i < num_cur_state; i++)
            {
                test_obs_sim_file_ << state_cur_[i];
                if (i + 1 < num_cur_state)
                {
                    test_obs_sim_file_ << " ";
                }
            }
            test_obs_sim_file_ << "\n";
                test_obs_sim_file_.flush();
                test_obs_sim_step_++;
                if ((obs_sim_tick++ % 10) == 0)
                {
                    ROS_INFO_STREAM("[TEST_LOG] obs_sim_step=" << test_obs_sim_step_
                                                             << " done=" << (test_obs_sim_done_ ? 1 : 0));
                }
                if (test_obs_sim_step_ >= test_obs_sim_max_steps_)
                {
                    test_obs_sim_done_ = true;
                    test_obs_sim_file_.close();
                ROS_INFO_STREAM("[TEST_LOG] Saved " << test_obs_sim_step_
                                                   << " obs samples to " << test_log_dir_
                                                   << "/" << test_obs_sim_name_);
            }
        };
        const bool mode7_rising = (prev_tc_mode_ != 7);
        if (mode7_rising)
        {
            test_log_step_ = 0;
            sim_paused_ = false;
            if (test_act_file_.is_open())
            {
                test_act_file_.close();
            }
            test_act_file_.open(test_log_dir_ + "/" + test_act_runtime_name_,
                                std::ofstream::out | std::ofstream::trunc);
            if (test_act_mapped_file_.is_open())
            {
                test_act_mapped_file_.close();
            }
            test_act_mapped_file_.open(test_log_dir_ + "/" + test_act_mapped_name_,
                                       std::ofstream::out | std::ofstream::trunc);
            if (!test_act_file_)
            {
                ROS_WARN_STREAM("[TEST_LOG] Failed to open " << test_act_runtime_name_ << " in " << test_log_dir_);
            }
            else
            {
                test_act_file_ << std::fixed << std::setprecision(6);
                ROS_INFO_STREAM("[TEST_LOG] Recording actions to " << test_log_dir_ << "/" << test_act_runtime_name_);
            }
            if (test_act_mapped_file_)
            {
                test_act_mapped_file_ << std::fixed << std::setprecision(6);
            }
            phase_started_ = false;
            phase_time_s_ = 0.0;
            phase_last_update_us_ = 0;
            sim_time_prev_s_ = sim_time_s_;
            // Treat mode-7 entry as a one-shot "send" so phase starts even with GUI-only control.
            mode7_send_triggered_ = true;
            {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(3);
                oss << "[GAIN][mode7] kp/kv leg: ";
                for (int i = 0; i < num_actuator_action; i++)
                {
                    int joint_idx = kLegJointMapAction[i];
                    if (i > 0)
                    {
                        oss << " | ";
                    }
                    oss << "j" << joint_idx << "(" << kp_(joint_idx, joint_idx)
                        << "," << kv_(joint_idx, joint_idx) << ")";
                }
                ROS_INFO_STREAM(oss.str());
            }
        }

        if (rd_cc_.tc_init || !mode7_active_)
        {
            // Initialize settings for Task Control.
            start_time_ = rd_cc_.control_time_us_;
            q_init_mode7_ = rd_cc_.q_virtual_.segment(6, MODEL_DOF);
            q_noise_pre_ = q_noise_ = q_init_mode7_;
            time_cur_ = start_time_ / 1e6;
            time_pre_ = time_cur_ - 0.005;
            time_inference_pre_ = rd_cc_.control_time_us_ - (1 / 249.9) * 1e6;

            rd_.tc_init = false;
            ROS_INFO("[CC] mode 7 init");
            debug_log_steps_remaining_ = 4;
            debug_log_this_step_ = true;
            // ROS_INFO("[OBS] force log 5 steps at mode7 init.");
            torque_init_ = rd_cc_.torque_desired;
            test_obs_sim_step_ = 0;
            test_obs_sim_done_ = false;
            if (test_obs_sim_file_.is_open())
            {
                test_obs_sim_file_.close();
            }
            test_obs_sim_file_.open(test_log_dir_ + "/" + test_obs_sim_name_,
                                    std::ofstream::out | std::ofstream::trunc);
            if (!test_obs_sim_file_)
            {
                ROS_WARN_STREAM("[TEST_LOG] Failed to open " << test_obs_sim_name_ << " in " << test_log_dir_);
            }
            else
            {
                test_obs_sim_file_ << std::fixed << std::setprecision(6);
                ROS_INFO_STREAM("[TEST_LOG] Recording obs to " << test_log_dir_ << "/" << test_obs_sim_name_);
            }

            if (test_act_file_.is_open())
            {
                test_act_file_.close();
            }
            test_act_file_.open(test_log_dir_ + "/" + test_act_runtime_name_,
                                std::ofstream::out | std::ofstream::trunc);
            if (test_act_mapped_file_.is_open())
            {
                test_act_mapped_file_.close();
            }
            test_act_mapped_file_.open(test_log_dir_ + "/" + test_act_mapped_name_,
                                       std::ofstream::out | std::ofstream::trunc);
            if (!test_act_file_)
            {
                ROS_WARN_STREAM("[TEST_LOG] Failed to open " << test_act_runtime_name_ << " in " << test_log_dir_);
            }
            else
            {
                test_act_file_ << std::fixed << std::setprecision(6);
                ROS_INFO_STREAM("[TEST_LOG] Recording actions to " << test_log_dir_ << "/" << test_act_runtime_name_);
            }
            if (test_act_mapped_file_)
            {
                test_act_mapped_file_ << std::fixed << std::setprecision(6);
            }

            processNoise();
            processObservation();
            write_obs_sim();
            // if (debug_log_this_step_)
            // {
            //     std::ostringstream oss;
            //     oss << std::fixed << std::setprecision(3);
            //     oss << "[OBS_SEND] obs[0:" << num_cur_state << "]=";
            //     for (int i = 0; i < num_cur_state; i++)
            //     {
            //         if (i > 0)
            //         {
            //             oss << ", ";
            //         }
            //         oss << state_cur_[i];
            //     }
            //     ROS_INFO_STREAM(oss.str());
            // }
            feedforwardPolicy();
            for (int i = 0; i < num_state_skip * num_state_hist; i++)
            {
                std::fill(state_buffer_.begin() + num_cur_state * i,
                          state_buffer_.begin() + num_cur_state * (i + 1),
                          0.0);
            }
            if (is_hist_encoder_)
            {
                for (size_t i = 0; i < num_hist_state; ++i)
                {
                    std::fill(state_long_hist_.begin() + num_cur_state * i,
                              state_long_hist_.begin() + num_cur_state * (i + 1),
                              0.0);
                }
            }
        }

        mode7_active_ = true;
        if (mode7_send_triggered_)
        {
            // Reset phase and last action on send so the next obs reflects the reset.
            rl_action_.setZero();
            rl_action_pre_.setZero();
            phase_started_ = true;
            phase_start_time_s_ = rd_cc_.control_time_us_ / 1.0e6;
            phase_time_s_ = 0.0;
            phase_last_update_us_ = rd_cc_.control_time_us_;
            sim_time_prev_s_ = sim_time_s_;
            mode7_send_triggered_ = false;
        }
        processNoise();

        if ((rd_cc_.control_time_us_ - time_inference_pre_) / 1.0e6 >= 1 / 100.0)
        {
            processObservation();
            static int dbg_tick = 0;
            if ((dbg_tick++ % 20) == 0)  // 100Hz 기준 0.2초에 1번
            {
                ROS_INFO_STREAM("[DBG-A] obs_step=" << test_obs_sim_step_
                                << " act_step=" << test_log_step_
                                << " prev_raw=" << rl_action_.transpose());
            }
            write_obs_sim();
            // if (debug_log_this_step_)
            // {
            //     std::ostringstream oss;
            //     oss << std::fixed << std::setprecision(3);
            //     oss << "[OBS_SEND] obs[0:" << num_cur_state << "]=";
            //     for (int i = 0; i < num_cur_state; i++)
            //     {
            //         if (i > 0)
            //         {
            //             oss << ", ";
            //         }
            //         oss << state_cur_[i];
            //     }
            //     ROS_INFO_STREAM(oss.str());
            // }
            feedforwardPolicy();
            static int dbg_tick2 = 0;
            if ((dbg_tick2++ % 20) == 0)
            {
                ROS_INFO_STREAM("[DBG-B] obs_step=" << test_obs_sim_step_
                                << " act_step=" << test_log_step_
                                << " new_raw=" << rl_action_.transpose());
            }
            if (test_act_file_ || test_act_mapped_file_)
            {
                if (test_act_file_)
                {
                    test_act_file_ << test_log_step_ << "\t";
                    for (int i = 0; i < num_action; i++)
                    {
                        test_act_file_ << rl_action_(i);
                        if (i + 1 < num_action)
                        {
                            test_act_file_ << " ";
                        }
                    }
                    test_act_file_ << "\n";
                }
                if (test_act_mapped_file_)
                {
                    std::array<double, CustomController::num_actuator_action> mapped{};
                    for (int i = 0; i < num_actuator_action; i++)
                    {
                        int joint_idx = kLegJointMapAction[i];
                        mapped[joint_idx] = rl_action_(i);
                    }
                    test_act_mapped_file_ << test_log_step_ << "\t";
                    for (int i = 0; i < num_actuator_action; i++)
                    {
                        test_act_mapped_file_ << mapped[i];
                        if (i + 1 < num_actuator_action)
                        {
                            test_act_mapped_file_ << " ";
                        }
                    }
                    test_act_mapped_file_ << "\n";
                }
                test_log_step_++;
            }

            if (enable_value_stop_ && value_valid_ && value_ < 100.0)
            {
                cout << "Value: " << value_ << endl;
                if (stop_by_value_thres_ == false)
                {
                    stop_by_value_thres_ = true;
                    stop_start_time_ = rd_cc_.control_time_us_;
                    q_stop_ = q_noise_;
                    std::cout << "Stop by Value Function" << std::endl;
                }
            }
            if (is_write_file_)
            {
                for (int i = 0; i < num_actuator_action; i++)
                {
                    torq_diff_(i) = (rl_action_(i) - rl_action_pre_(i)) * torque_bound_(i);
                    energy(i) = rl_action_(i) * torque_bound_(i) * q_vel_noise_(i);
                }
                writeFile << rd_cc_.control_time_ << "\t";
                writeFile << torq_diff_.norm() << "\t";
                writeFile << (q_vel_noise_ - q_vel_noise_pre_).norm() << "\t";
                writeFile << q_vel_noise_.norm() << "\t";
                writeFile << energy.sum() << "\t";
                writeFile << std::pow((desired_vel_x - rd_cc_.q_dot_virtual_(0)), 2) +
                                 std::pow((desired_vel_yaw - rd_cc_.q_dot_virtual_(5)), 2);
                writeFile << std::endl;
            }

            time_inference_pre_ = rd_cc_.control_time_us_;
        }

        Vector12d target_pos;
        static int target_log_counter = 0;
        for (int i = 0; i < num_actuator_action; i++)
        {
            double a = DyrosMath::minmax_cut(rl_action_(i), -1.0, 1.0);
            const double lo = kLegJointPosLimits[i][0] * q_limit_scale_;
            const double hi = kLegJointPosLimits[i][1] * q_limit_scale_;
            double target = 0.5 * (a + 1.0) * (hi - lo) + lo;
            if (has_joint_limits_)
            {
                int joint_idx = kLegJointMapAction[i];
                target = DyrosMath::minmax_cut(target, q_min_(joint_idx), q_max_(joint_idx));
            }
            target_pos(i) = target;
        }
        rd_.q_desired = q_init_mode7_;
        for (int i = 0; i < num_actuator_action; i++)
        {
            const int joint_idx = kLegJointMapAction[i];
            rd_.q_desired(joint_idx) = target_pos(i);
        }
        // bool log_due = debug_log_this_step_ || ((target_log_counter++ % 100) == 0);
        // if (log_due)
        // {
        //     std::ostringstream oss;
        //     oss << std::fixed << std::setprecision(3);
        //     oss << "[TARGET_POS] target_pos[0:" << num_actuator_action << "]=";
        //     for (int i = 0; i < num_actuator_action; i++)
        //     {
        //         if (i > 0)
        //         {
        //             oss << ", ";
        //         }
        //         oss << target_pos(i);
        //     }
        //     ROS_INFO_STREAM(oss.str());
        // }
        for (int i = 0; i < num_actuator_action; i++)
        {
            int joint_idx = kLegJointMapAction[i];
            torque_rl_(joint_idx) = kp_(joint_idx, joint_idx) / 9.0 *
                                        (target_pos(i) - q_noise_(joint_idx)) -
                                    kv_(joint_idx, joint_idx) / 3.0 * q_vel_noise_(joint_idx);
        }
        for (int i = num_actuator_action; i < MODEL_DOF; i++)
        {
            torque_rl_(i) = kp_(i, i) * (q_init_mode7_(i) - q_noise_(i)) -
                            kv_(i, i) * q_vel_noise_(i);
        }

        if (rd_cc_.control_time_us_ < start_time_ + 0.1e6)
        {
            for (int i = 0; i < MODEL_DOF; i++)
            {
                torque_spline_(i) = DyrosMath::cubic(rd_cc_.control_time_us_,
                                                    start_time_,
                                                    start_time_ + 0.1e6,
                                                    torque_init_(i),
                                                    torque_rl_(i),
                                                    0.0,
                                                    0.0);
            }
            rd_.torque_desired = torque_spline_;
        }
        else
        {
            rd_.torque_desired = torque_rl_;
        }

        if (enable_value_stop_ && stop_by_value_thres_)
        {
            rd_.torque_desired = kp_ * (q_stop_ - q_noise_) - kv_ * q_vel_noise_;
        }

        for (int i = 0; i < MODEL_DOF; i++)
        {
            rd_.torque_desired(i) = DyrosMath::minmax_cut(
                rd_.torque_desired(i), -torque_bound_(i), torque_bound_(i));
        }
        prev_tc_mode_ = tc_mode;
        return;
    }
    prev_tc_mode_ = tc_mode;
    mode7_active_ = false;
    LF_CF_FT_pre = rd_cc_.LF_CF_FT;
    RF_CF_FT_pre = rd_cc_.RF_CF_FT;
}

void CustomController::updatePace()
{
    static bool pace_autostarted = false;
    if (!pace_autostarted)
    {
        pace_autostarted = true;
    }
    if (rd_.tc_.mode == 8 && rd_.tc_init)
    {
        pace_trigger_ = true;
        rd_.tc_init = false;
        ROS_INFO("[PACE] Auto trigger from mode 8.");
    }
    if (!pace_active_ && !pace_trigger_)
    {
        return;
    }
    const double now_s = rd_.control_time_;
    if (pace_trigger_)
    {
        pace_trigger_ = false;
        pace_active_ = true;
        pace_start_time_s_ = now_s;
        pace_last_pub_time_s_ = 0.0;
        pace_align_active_ = true;
        pace_align_start_s_ = now_s;
        pace_hold_q_ = rd_.q_;
        pace_prev_des_ = pace_hold_q_;
        pace_has_prev_des_ = true;
        if (pace_collect_enabled_)
        {
            pace_collect_saved_ = false;
            pace_collect_start_time_s_ = -1.0;
            pace_collect_time_.clear();
            pace_collect_q_.clear();
            pace_collect_des_.clear();
        }
        if (!pace_gains_saved_ && rd_.pos_kp_v.size() == MODEL_DOF && rd_.pos_kv_v.size() == MODEL_DOF)
        {
            pace_pos_kp_backup_ = rd_.pos_kp_v;
            pace_pos_kv_backup_ = rd_.pos_kv_v;
            pace_gains_saved_ = true;
        }
        if (!pace_pc_gravity_saved_)
        {
            pace_pc_gravity_backup_ = rd_.pc_gravity;
            pace_pc_gravity_saved_ = true;
        }
        if (pos_cmd_pub_.getNumSubscribers() == 0)
        {
            ROS_WARN("[PACE] positionCommand has no subscribers.");
        }
        // notify mujoco_ros to start high-rate chirp logging
        ros::param::set("/mujoco_ros/chirp_record", true);
        ros::param::set("/mujoco_ros/chirp_record_duration", pace_duration_s_);
        ros::param::set("/mujoco_ros/chirp_record_delay", pace_align_duration_s_);
        ros::param::set("/mujoco_ros/chirp_record_prefix", "chirp_data");
        ros::param::set("/mujoco_ros/chirp_record_dir",
                        std::string("/home/user/other_ws/src/pace-sim2real/data/tocabi_mujoco"));
    }
    if (!pace_active_)
    {
        return;
    }
    if (!pace_active_)
    {
        return;
    }
    double t = now_s - pace_start_time_s_;
    const double total_duration_s = pace_duration_s_ + pace_align_duration_s_;
    bool pace_finished = t >= total_duration_s;
    if (pace_finished && !pace_hold_after_)
    {
        pace_active_ = false;
        if (pace_gains_saved_)
        {
            rd_.pos_kp_v = pace_pos_kp_backup_;
            rd_.pos_kv_v = pace_pos_kv_backup_;
            pace_gains_saved_ = false;
        }
        ros::param::set("/mujoco_ros/chirp_record", false);
        if (pace_pc_gravity_saved_)
        {
            rd_.pc_gravity = pace_pc_gravity_backup_;
            pace_pc_gravity_saved_ = false;
        }
        return;
    }
    if (pace_finished)
    {
        t = total_duration_s;
        pace_align_active_ = false;
        if (pace_gains_saved_)
        {
            rd_.pos_kp_v = pace_pos_kp_backup_;
            rd_.pos_kv_v = pace_pos_kv_backup_;
            pace_gains_saved_ = false;
        }
        ros::param::set("/mujoco_ros/chirp_record", false);
        if (pace_pc_gravity_saved_)
        {
            rd_.pc_gravity = pace_pc_gravity_backup_;
            pace_pc_gravity_saved_ = false;
        }
    }
    if ((now_s - pace_last_pub_time_s_) < 0.01)
    {
        return;
    }

    if (pace_gains_saved_)
    {
        rd_.pos_kp_v = pace_pos_kp_backup_;
        rd_.pos_kv_v = pace_pos_kv_backup_;
        for (int i = 0; i < MODEL_DOF; i++)
        {
            if (pace_align_active_)
            {
                rd_.pos_kp_v[i] *= pace_align_gain_;
                rd_.pos_kv_v[i] *= pace_align_gain_;
            }
        }
    }

    const double two_pi = 6.283185307179586;
    double chirp = 0.0;
    if (!pace_finished && pace_align_active_ && (now_s - pace_align_start_s_) < pace_align_duration_s_)
    {
        chirp = 0.0;
    }
    else
    {
        pace_align_active_ = false;
        if (pace_finished)
        {
            chirp = 0.0;
        }
        else
        {
            double t_chirp = t - pace_align_duration_s_;
            if (t_chirp < 0.0)
            {
                t_chirp = 0.0;
            }
            double phase = two_pi * (pace_min_freq_ * t_chirp
                                     + ((pace_max_freq_ - pace_min_freq_) / (2.0 * pace_duration_s_)) * t_chirp * t_chirp);
            chirp = std::sin(phase);
        }
    }

    tocabi_msgs::positionCommand msg;
    msg.traj_time = 0.01;
    msg.gravity = false;
    msg.relative = false;
    auto is_body_head_joint = [](int idx) -> bool {
        // MODEL order indices for waist/upperbody/neck/head.
        return (idx >= 12 && idx <= 14) || (idx >= 23 && idx <= 24);
    };
    for (int i = 0; i < MODEL_DOF; i++)
    {
        double scale = is_body_head_joint(i) ? 0.0 : 0.5;
        msg.position[i] = q_init_(i) + chirp * scale;
    }
    // Force position control locally to avoid missing ROS delivery.
    rd_.pc_mode = true;
    rd_.pc_gravity = false;
    rd_.positionHoldSwitch = false;
    if (pace_align_active_ && (now_s - pace_align_start_s_) < pace_align_duration_s_)
    {
        rd_.pc_time_ = pace_align_start_s_;
        rd_.pc_traj_time_ = pace_align_duration_s_;
        rd_.pc_pos_init = pace_hold_q_;
        rd_.pc_vel_init.setZero();
    }
    else
    {
        rd_.pc_time_ = now_s;
        rd_.pc_traj_time_ = 0.01;
        rd_.pc_pos_init = pace_has_prev_des_ ? pace_prev_des_ : rd_.q_desired;
        rd_.pc_vel_init.setZero();
    }
    rd_.pc_pos_des = pace_hold_q_;
    // Ensure task control does not override position control during pace.
    rd_.tc_run = false;
    rd_.tc_init = false;
    for (int i = 0; i < MODEL_DOF; i++)
    {
        rd_.pc_pos_des(i) = msg.position[i];
    }
    pace_prev_des_ = rd_.pc_pos_des;
    if (pace_collect_enabled_ && !pace_finished && !pace_align_active_)
    {
        if (pace_collect_start_time_s_ < 0.0)
        {
            pace_collect_start_time_s_ = now_s;
        }
        const double t_sample = now_s - pace_collect_start_time_s_;
        pace_collect_time_.push_back(t_sample);
        pace_collect_q_.push_back(rd_.q_);
        pace_collect_des_.push_back(rd_.pc_pos_des);
    }
    pos_cmd_pub_.publish(msg);
    pace_last_pub_time_s_ = now_s;

    if (pace_collect_enabled_ && pace_finished && !pace_collect_saved_)
    {
        const std::string time_path = pace_collect_output_dir_ + "/" + pace_collect_output_prefix_ + "_time.txt";
        const std::string q_path = pace_collect_output_dir_ + "/" + pace_collect_output_prefix_ + "_dof_pos.txt";
        const std::string des_path = pace_collect_output_dir_ + "/" + pace_collect_output_prefix_ + "_des_dof_pos.txt";
        std::ofstream time_out(time_path);
        std::ofstream q_out(q_path);
        std::ofstream des_out(des_path);
        if (!time_out || !q_out || !des_out)
        {
            ROS_WARN_STREAM("[PACE] Failed to open output files in " << pace_collect_output_dir_);
        }
        else
        {
            time_out << std::fixed << std::setprecision(6);
            q_out << std::fixed << std::setprecision(6);
            des_out << std::fixed << std::setprecision(6);
            for (size_t i = 0; i < pace_collect_time_.size(); i++)
            {
                time_out << pace_collect_time_[i] << "\n";
                for (int j = 0; j < MODEL_DOF; j++)
                {
                    q_out << pace_collect_q_[i](j);
                    des_out << pace_collect_des_[i](j);
                    if (j + 1 < MODEL_DOF)
                    {
                        q_out << " ";
                        des_out << " ";
                    }
                }
                q_out << "\n";
                des_out << "\n";
            }
            ROS_INFO_STREAM("[PACE] Saved " << pace_collect_time_.size() << " samples to " << pace_collect_output_dir_);
        }
        pace_collect_saved_ = true;
    }

    static int pace_log_counter = 0;
    if ((pace_log_counter++ % 50) == 0)
    {
        struct LogJoint
        {
            const char* tag;
            int idx;
        };
        const LogJoint joints[] = {
            {"PACE_L_SHOULDER1", 15},
            {"PACE_L_SHOULDER2", 16},
            {"PACE_L_SHOULDER3", 17},
            {"PACE_L_ARMLINK", 18},
            {"PACE_L_ELBOW", 19},
            {"PACE_L_FOREARM", 20},
            {"PACE_L_WRIST1", 21},
            {"PACE_L_WRIST2", 22},
            {"PACE_R_SHOULDER1", 25},
            {"PACE_R_SHOULDER2", 26},
            {"PACE_R_SHOULDER3", 27},
            {"PACE_R_ARMLINK", 28},
            {"PACE_R_ELBOW", 29},
            {"PACE_R_FOREARM", 30},
            {"PACE_R_WRIST1", 31},
            {"PACE_R_WRIST2", 32},
            {"PACE_WAIST1", 12},
            {"PACE_WAIST2", 13},
            {"PACE_UPPERBODY", 14},
        };
        for (const auto& joint : joints)
        {
            const int j = joint.idx;
            double kp = (j < static_cast<int>(rd_.pos_kp_v.size())) ? rd_.pos_kp_v[j] / 9.0 : 0.0;
            double kd = (j < static_cast<int>(rd_.pos_kv_v.size())) ? rd_.pos_kv_v[j] / 3.0 : 0.0;
            double q_des = rd_.pc_pos_des(j);
            double q = rd_.q_(j);
            double qd = rd_.q_dot_(j);
            double torque_raw = kp * (q_des - q) + kd * (0.0 - qd);
            double torque_clip = DyrosMath::minmax_cut(torque_raw, -rd_.torque_limit[j], rd_.torque_limit[j]);
            ROS_INFO_STREAM("[" << joint.tag << "] q_des=" << q_des
                                << " q=" << q
                                << " qd=" << qd
                                << " kp=" << kp
                                << " kd=" << kd
                                << " raw=" << torque_raw
                                << " clip=" << torque_clip
                                << " lim=" << rd_.torque_limit[j]);
        }
    }
}

void CustomController::computeFast()
{
    // if (tc.mode == 10)
    // {
    // }
    // else if (tc.mode == 11)
    // {
    // }
}

void CustomController::computePlanner()
{
}

void CustomController::copyRobotData(RobotData &rd_l)
{
    std::memcpy(&rd_cc_, &rd_l, sizeof(RobotData));
}

void CustomController::guiSendCallback(const std_msgs::Empty::ConstPtr& msg)
{
    mode7_send_triggered_ = true;
}

void CustomController::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[1] * cmd_scale_x_, -cmd_scale_x_, cmd_scale_x_);
    target_vel_y_ = DyrosMath::minmax_cut(joy->axes[0] * cmd_scale_y_, -cmd_scale_y_, cmd_scale_y_);
    target_vel_yaw_ = DyrosMath::minmax_cut(joy->axes[3] * cmd_scale_yaw_, -cmd_scale_yaw_, cmd_scale_yaw_);

    int yaw_dir = 0;
    if (joy->buttons.size() > 4 && joy->buttons[4]) yaw_dir += 1;
    if (joy->buttons.size() > 5 && joy->buttons[5]) yaw_dir -= 1;
    int zoom_dir = 0;
    if (joy->buttons.size() > 6 && joy->buttons[6]) zoom_dir += 1;
    if (joy->buttons.size() > 7 && joy->buttons[7]) zoom_dir -= 1;
    float elev_axis = 0.0f;
    if (joy->axes.size() > 8)
        elev_axis = -static_cast<float>(joy->axes[8]);
    else if (joy->axes.size() > 7)
        elev_axis = -static_cast<float>(joy->axes[7]);
    if (yaw_dir != 0 || zoom_dir != 0 || elev_axis != 0.0f)
    {
        std_msgs::Float32MultiArray cam_msg;
        cam_msg.data.resize(3);
        cam_msg.data[0] = static_cast<float>(yaw_dir);
        cam_msg.data[1] = static_cast<float>(zoom_dir);
        cam_msg.data[2] = elev_axis;
        cam_cmd_pub_.publish(cam_msg);
    }

    bool btn0 = (joy->buttons.size() > 0) ? (joy->buttons[0] != 0) : false;
    if (btn0 && !prev_btn0_)
    {
        std_msgs::String msg;
        msg.data = "pause";
        sim_command_pub_.publish(msg);
        sim_paused_ = !sim_paused_;
    }
    prev_btn0_ = btn0;

    bool btn1 = (joy->buttons.size() > 1) ? (joy->buttons[1] != 0) : false;
    if (btn1 && !prev_btn1_)
    {
        tocabi_msgs::TaskCommand msg;
        msg.mode = 7;
        task_cmd_pub_.publish(msg);
        debug_log_steps_remaining_ = 5;
        std_msgs::Empty send_msg;
        // gui_send_pub_.publish(send_msg);
    }
    prev_btn1_ = btn1;

    bool btn8 = (joy->buttons.size() > 8) ? (joy->buttons[8] != 0) : false;
    if (btn8 && !prev_btn8_)
    {
        pace_trigger_ = true;
        ROS_INFO("[PACE] Triggered 20s chirp command.");
    }
    prev_btn8_ = btn8;

    bool btn9 = (joy->buttons.size() > 9) ? (joy->buttons[9] != 0) : false;
    if (btn9 && !prev_btn9_)
    {
        std_msgs::String msg;
        msg.data = "mjreset";
        sim_command_pub_.publish(msg);
    }
    prev_btn9_ = btn9;

    bool btn10 = (joy->buttons.size() > 10) ? (joy->buttons[10] != 0) : false;
    if (btn10 && !prev_btn10_)
    {
        std_msgs::String msg;
        msg.data = "quit";
        sim_command_pub_.publish(msg);
    }
    prev_btn10_ = btn10;
}

void CustomController::simTimeCallback(const std_msgs::Float32ConstPtr& msg)
{
    sim_time_s_ = msg->data;
    last_sim_time_msg_ = ros::Time::now();
    sim_time_received_ = true;
}

void CustomController::xBoxJoyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
    target_vel_x_ = DyrosMath::minmax_cut(joy->axes[1] * cmd_scale_x_, -cmd_scale_x_, cmd_scale_x_);
    target_vel_y_ = DyrosMath::minmax_cut(joy->axes[0] * cmd_scale_y_, -cmd_scale_y_, cmd_scale_y_);
    target_vel_yaw_ = DyrosMath::minmax_cut(joy->axes[3] * cmd_scale_yaw_, -cmd_scale_yaw_, cmd_scale_yaw_);
}

void CustomController::quatToTanNorm(const Eigen::Quaterniond& quaternion, Eigen::Vector3d& tangent, Eigen::Vector3d& normal) {
    // Reference direction and normal vectors
    Eigen::Vector3d refDirection(1, 0, 0); // Tangent vector reference
    Eigen::Vector3d refNormal(0, 0, 1);    // Normal vector reference

    // Rotate the reference vectors
    tangent = quaternion * refDirection;
    normal = quaternion * refNormal;

    // Normalize the vectors
    tangent.normalize();
    normal.normalize();
}

Eigen::Vector3d CustomController::quatRotateInverse(const Eigen::Quaterniond& q, const Eigen::Vector3d& v) {

    Eigen::Vector3d q_vec = q.vec();
    double q_w = q.w();

    Eigen::Vector3d a = v * (2.0 * q_w * q_w - 1.0);
    Eigen::Vector3d b = 2.0 * q_w * q_vec.cross(v);
    Eigen::Vector3d c = 2.0 * q_vec * q_vec.dot(v);

    return a - b + c;
}

Eigen::Vector3d CustomController::mat2euler(Eigen::Matrix3d mat)
{
    Eigen::Vector3d euler;

    double cy = std::sqrt(mat(2, 2) * mat(2, 2) + mat(1, 2) * mat(1, 2));
    if (cy > std::numeric_limits<double>::epsilon())
    {
        euler(2) = -atan2(mat(0, 1), mat(0, 0));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = -atan2(mat(1, 2), mat(2, 2));
    }
    else
    {
        euler(2) = -atan2(-mat(1, 0), mat(1, 1));
        euler(1) =  -atan2(-mat(0, 2), cy);
        euler(0) = 0.0;
    }
    return euler;
}

Eigen::VectorQd CustomController::getControl()
{
    return ControlVal_;
}
