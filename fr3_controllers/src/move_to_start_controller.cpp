#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <fr3_controllers/move_to_start_controller.h>

namespace fr3_controllers {

bool MoveToStartController::init(hardware_interface::RobotHW* robot_hardware,
                                 ros::NodeHandle& node_handle) {
  if (!robot_hardware) {
    ROS_ERROR("MoveToStartController: robot_hardware is null");
    return false;
  }

  // Get the position joint interface from RobotHW via MultiInterfaceController
  position_joint_interface_ =
      robot_hardware->get<hardware_interface::PositionJointInterface>();
  if (!position_joint_interface_) {
    ROS_ERROR("MoveToStartController: PositionJointInterface not available");
    return false;
  }

  // Required: joint_names
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR("MoveToStartController: Provide exactly 7 joint_names");
    return false;
  }

  // Optional: home_positions
  std::vector<double> home_vec;
  if (node_handle.getParam("home_positions", home_vec)) {
    if (home_vec.size() != 7) {
      ROS_ERROR("MoveToStartController: home_positions must have 7 elements");
      return false;
    }
    for (size_t i = 0; i < 7; ++i) home_positions_[i] = home_vec[i];
  }

  node_handle.param("move_duration", move_duration_, 3.0);
  node_handle.param("goal_tolerance", goal_tolerance_, 1e-3);
  if (move_duration_ <= 0.0) {
    ROS_ERROR("MoveToStartController: move_duration must be > 0");
    return false;
  }

  // Joint handles
  try {
    joint_handles_.reserve(7);
    for (const auto& name : joint_names) {
      joint_handles_.push_back(position_joint_interface_->getHandle(name));
    }
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM("MoveToStartController: getHandle failed: " << e.what());
    return false;
  }

  return true;
}

void MoveToStartController::starting(const ros::Time&) {
  // Capture start pose
  for (size_t i = 0; i < 7; ++i) q0_[i] = joint_handles_[i].getPosition();
  // Deltas to home
  for (size_t i = 0; i < 7; ++i) dq_[i] = home_positions_[i] - q0_[i];

  // Check goal
  at_goal_ = true;
  for (size_t i = 0; i < 7; ++i) {
    if (std::fabs(dq_[i]) > goal_tolerance_) { at_goal_ = false; break; }
  }

  t_ = 0.0;
}

void MoveToStartController::update(const ros::Time&, const ros::Duration& period) {
  if (at_goal_) {
    for (size_t i = 0; i < 7; ++i) joint_handles_[i].setCommand(home_positions_[i]);
    return;
  }

  t_ += period.toSec();
  double tau = std::min(std::max(t_ / move_duration_, 0.0), 1.0);
  // Quintic minimum-jerk scalar
  double s = 10.0 * std::pow(tau, 3) - 15.0 * std::pow(tau, 4) + 6.0 * std::pow(tau, 5);

  for (size_t i = 0; i < 7; ++i) {
    double q_cmd = q0_[i] + dq_[i] * s;
    joint_handles_[i].setCommand(q_cmd);
  }

  if (tau >= 1.0) {
    at_goal_ = true;
  }
}

void MoveToStartController::stopping(const ros::Time&) {
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(at_goal_ ? home_positions_[i]
                                          : joint_handles_[i].getPosition());
  }
}

}  // namespace fr3_controllers

// plugin export
PLUGINLIB_EXPORT_CLASS(fr3_controllers::MoveToStartController,
                       controller_interface::ControllerBase)
