#include <multi_arm_controllers/grav_comp_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

namespace multi_arm_controllers {

constexpr double GravCompController::kDeltaTauMax;

bool GravCompController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
    std::string arm_id;
    if (!node_handle.getParam("arm_id", arm_id)) {
        ROS_ERROR("GravCompController: Could not read parameter arm_id");
        return false;
    }
   
    std::vector<std::string> joint_names;
    if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
        ROS_ERROR(
            "GravCompController: Invalid or no joint_names parameters provided, aborting "
            "controller init!");
        return false;
    }

    auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
    if (model_interface == nullptr) {
        ROS_ERROR_STREAM(
            "GravCompController: Error getting model interface from hardware");
        return false;
    }
    try {
        model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
            model_interface->getHandle(arm_id + "_model"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "GravCompController: Exception getting model handle from interface: "
            << ex.what());
        return false;
    }

    auto* cartesian_pose_interface = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
    if (cartesian_pose_interface == nullptr) {
        ROS_ERROR_STREAM(
            "GravCompController: Error getting cartesian pose interface from hardware");
        return false;
    }
    try {
        cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
            cartesian_pose_interface->getHandle(arm_id + "_robot"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "GravCompController: Exception getting cartesian pose handle from interface: "
            << ex.what());
        return false;
    }

    auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
    if (effort_joint_interface == nullptr) {
        ROS_ERROR_STREAM(
            "GravCompController: Error getting effort joint interface from hardware");
        return false;
    }
    for (size_t i = 0; i < 7; ++i) {
        try {
        joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
        } catch (const hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "GravCompController: Exception getting joint handles: " << ex.what());
        return false;
        }
    }

    initial_pose_.fill(0.0);

    return true;
}

void GravCompController::starting(const ros::Time& /*time*/) {
    initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
    cartesian_pose_handle_->setCommand(initial_pose_);
}

void GravCompController::update(const ros::Time& time, const ros::Duration& period) {

    cartesian_pose_handle_->setCommand(initial_pose_);
    
    franka::RobotState robot_state = cartesian_pose_handle_->getRobotState();
    std::array<double, 7> coriolis = model_handle_->getCoriolis();
    std::array<double, 7> gravity = model_handle_->getGravity();

    std::array<double, 7> tau_d_calculated;
    for (size_t i = 0; i < 7; ++i) {
        tau_d_calculated[i] = coriolis[i];
    }

    std::array<double, 7> tau_d_saturated = saturateTorqueRate(tau_d_calculated, robot_state.tau_J_d);

    for (size_t i = 0; i < 7; ++i) {
        joint_handles_[i].setCommand(tau_d_saturated[i]);
    }
}

std::array<double, 7> GravCompController::saturateTorqueRate(const std::array<double, 7>& tau_d_calculated, const std::array<double, 7>& tau_J_d) {  
    std::array<double, 7> tau_d_saturated{};
    for (size_t i = 0; i < 7; i++) {
        double difference = tau_d_calculated[i] - tau_J_d[i];
        tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, kDeltaTauMax), -kDeltaTauMax);
    }
    return tau_d_saturated;
}

}  // namespace multi_arm_controllers

PLUGINLIB_EXPORT_CLASS(multi_arm_controllers::GravCompController,
                       controller_interface::ControllerBase)
