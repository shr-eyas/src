#include <multi_arm_controllers/mirror_impedance_controller.h>

#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka/robot_state.h>

namespace multi_arm_controllers {

constexpr double MirrorImpedanceController::kDeltaTauMax;

bool MirrorImpedanceController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR("MirrorImpedanceController: Could not read parameter arm_id");
    return false;
  }

  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "MirrorImpedanceController: Invalid or no joint_names parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("k_gains", k_gains_) || k_gains_.size() != 7) {
    ROS_ERROR(
        "MirrorImpedanceController:  Invalid or no k_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  if (!node_handle.getParam("d_gains", d_gains_) || d_gains_.size() != 7) {
    ROS_ERROR(
        "MirrorImpedanceController:  Invalid or no d_gain parameters provided, aborting "
        "controller init!");
    return false;
  }

  node_handle.param("coriolis_factor", coriolis_factor_, coriolis_factor_);
  node_handle.param("source_topic", source_topic_, source_topic_);
  node_handle.param("stale_timeout", stale_timeout_, stale_timeout_);

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MirrorImpedanceController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "MirrorImpedanceController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* cartesian_pose_interface = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MirrorImpedanceController: Error getting cartesian pose interface from hardware");
    return false;
  }
  try {
    cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
        cartesian_pose_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "MirrorImpedanceController: Exception getting cartesian pose handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MirrorImpedanceController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "MirrorImpedanceController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  sub_ = node_handle.subscribe<sensor_msgs::JointState>(
      source_topic_, 1, &MirrorImpedanceController::jsCb, this,
      ros::TransportHints().tcpNoDelay());

  initial_pose_.fill(0.0);
  last_msg_stamp_ = ros::Time(0);
  return true;
}

void MirrorImpedanceController::starting(const ros::Time& /*time*/) {
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  cartesian_pose_handle_->setCommand(initial_pose_);
  dq_filtered_.fill(0.0);
}

void MirrorImpedanceController::jsCb(const sensor_msgs::JointStateConstPtr& msg) {
  static bool mapped = false;
  if (!mapped) {
    std::array<int,7> idx{};
    for (int j=0;j<7;j++) {
      const std::string expected = "fr3_joint" + std::to_string(j+1); 
      auto it = std::find(msg->name.begin(), msg->name.end(), expected);
      if (it == msg->name.end()) return;
      idx[j] = static_cast<int>(std::distance(msg->name.begin(), it));
    }
    name_to_idx_ = idx;
    mapped = true;
  }

  std::array<double,7> qd{}, dqd{};
  for (int j=0;j<7;j++) {
    int k = name_to_idx_[j];
    qd[j]  = (k < static_cast<int>(msg->position.size())) ? msg->position[k] : 0.0;
    dqd[j] = (k < static_cast<int>(msg->velocity.size())) ? msg->velocity[k] : 0.0;
  }
  qd_buf_.writeFromNonRT(qd);
  dqd_buf_.writeFromNonRT(dqd);
  last_msg_stamp_ = msg->header.stamp.isZero() ? ros::Time::now() : msg->header.stamp;
}

void MirrorImpedanceController::update(const ros::Time&, const ros::Duration& period) {

  cartesian_pose_handle_->setCommand(initial_pose_);
  franka::RobotState robot_state = cartesian_pose_handle_->getRobotState();

  constexpr double alpha = 0.99;
  for (size_t i=0; i<7; i++) dq_filtered_[i] = (1.0 - alpha)*dq_filtered_[i] + alpha*robot_state.dq[i];

  auto qd_ptr  = qd_buf_.readFromRT();
  auto dqd_ptr = dqd_buf_.readFromRT();
  std::array<double,7> qd{}, dqd{};
  if (qd_ptr)  qd  = *qd_ptr;  else qd  = robot_state.q;
  if (dqd_ptr) dqd = *dqd_ptr; else dqd.fill(0.0);

  const bool fresh = (ros::Time::now() - last_msg_stamp_) <= ros::Duration(stale_timeout_);
  if (!fresh) { qd = robot_state.q; dqd.fill(0.0); }

  const std::array<double,7> c = model_handle_->getCoriolis();

  std::array<double,7> tau{};
  for (size_t i=0; i<7; i++) {
    const double e  = qd[i]  - robot_state.q[i];
    const double de = dqd[i] - dq_filtered_[i];
    tau[i] = coriolis_factor_*c[i] + k_gains_[i]*e + d_gains_[i]*de;
  }

  const std::array<double,7> tau_cmd = saturateTorqueRate(tau, robot_state.tau_J_d);
  for (size_t i=0; i<7; i++) joint_handles_[i].setCommand(tau_cmd[i]);

}

std::array<double, 7> MirrorImpedanceController::saturateTorqueRate(
    const std::array<double, 7>& tau_d_calculated,
    const std::array<double, 7>& tau_J_d) {  
  std::array<double, 7> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_J_d[i] + std::max(std::min(difference, kDeltaTauMax), -kDeltaTauMax);
  }
  return tau_d_saturated;
}

}  // namespace multi_arm_controllers

PLUGINLIB_EXPORT_CLASS(multi_arm_controllers::MirrorImpedanceController,
                       controller_interface::ControllerBase)
