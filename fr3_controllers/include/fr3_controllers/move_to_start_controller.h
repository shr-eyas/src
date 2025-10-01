#pragma once

#include <array>
#include <string>
#include <vector>
#include <cmath>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

namespace fr3_controllers {

class MoveToStartController
    : public controller_interface::MultiInterfaceController<
          hardware_interface::PositionJointInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hardware,
            ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;
  void stopping(const ros::Time&) override;

 private:
  // Interfaces and handles
  hardware_interface::PositionJointInterface* position_joint_interface_{nullptr};
  std::vector<hardware_interface::JointHandle> joint_handles_;

  // Parameters
  std::array<double, 7> home_positions_{
      {0.0, -M_PI_4, 0.0, -3.0 * M_PI_4, 0.0, M_PI_2, M_PI_4}};
  double move_duration_{3.0};
  double goal_tolerance_{1e-3};

  // Trajectory state
  std::array<double, 7> q0_{{0,0,0,0,0,0,0}};
  std::array<double, 7> dq_{{0,0,0,0,0,0,0}};
  double t_{0.0};
  bool at_goal_{false};
};

}  // namespace fr3_controllers
