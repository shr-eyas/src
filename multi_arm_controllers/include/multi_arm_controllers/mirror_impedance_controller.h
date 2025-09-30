#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <realtime_tools/realtime_buffer.h>
#include <ros/node_handle.h>
#include <ros/subscriber.h>
#include <ros/time.h>
#include <sensor_msgs/JointState.h>

#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_model_interface.h>


namespace multi_arm_controllers {

class MirrorImpedanceController : public controller_interface::MultiInterfaceController<
                                            franka_hw::FrankaModelInterface,
                                            hardware_interface::EffortJointInterface,
                                            franka_hw::FrankaPoseCartesianInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  static constexpr double kDeltaTauMax{1.0};
  std::array<double, 7> saturateTorqueRate(
      const std::array<double, 7>& tau_d_calculated,
      const std::array<double, 7>& tau_J_d); 

  std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  std::vector<double> k_gains_;
  std::vector<double> d_gains_;
  double coriolis_factor_{1.0};
  std::string source_topic_{"/fr3_right/joint_states"};
  double stale_timeout_{0.05};

  // state
  std::array<double,16> initial_pose_{};
  std::array<double,7> dq_filtered_{{0,0,0,0,0,0,0}};
  std::array<int,7> name_to_idx_{{0,1,2,3,4,5,6}};
  ros::Time last_msg_stamp_;
  std::string source_arm_id_{"fr3"};

  // RT input buffers
  realtime_tools::RealtimeBuffer<std::array<double,7>> qd_buf_;
  realtime_tools::RealtimeBuffer<std::array<double,7>> dqd_buf_;

  // ROS
  ros::Subscriber sub_;

  // subscriber callback (non-RT)
  void jsCb(const sensor_msgs::JointStateConstPtr& msg);

};

}  // namespace multi_arm_controllers
