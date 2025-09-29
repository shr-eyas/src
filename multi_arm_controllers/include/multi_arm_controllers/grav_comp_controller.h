#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <franka_hw/franka_cartesian_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/trigger_rate.h>

namespace multi_arm_controllers {

class GravCompController 
    : public controller_interface::MultiInterfaceController<
        franka_hw::FrankaModelInterface,
        hardware_interface::EffortJointInterface,
        franka_hw::FrankaPoseCartesianInterface> {
    public:
        bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
        void starting(const ros::Time&) override;
        void update(const ros::Time&, const ros::Duration& period) override;

    private:
        // Saturation
        std::array<double, 7> saturateTorqueRate(
            const std::array<double, 7>& tau_d,
            const std::array<double, 7>& tau_prev);  
        
        // Handles
        std::unique_ptr<franka_hw::FrankaCartesianPoseHandle> cartesian_pose_handle_;
        std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
        std::vector<hardware_interface::JointHandle> joint_handles_;
        
        // Params
        static constexpr double kDeltaTauMax{1.0};
        std::array<double, 16> initial_pose_{};
};

}  // namespace multi_arm_controllers
