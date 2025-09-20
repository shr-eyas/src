#pragma once

#include <array>
#include <memory>
#include <vector>

#include <ros/ros.h>
#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <actionlib/server/simple_action_server.h>
#include <realtime_tools/realtime_buffer.h>
#include <fr3_controllers/ExecuteScheduleAction.h>

namespace fr3_controllers {

struct PreparedSchedule {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double dt{0.001};
  uint32_t T{0};
  ros::Time t0;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> x;    
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> xd;   
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> xdd;  
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> stiffness;    
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>> damping;    
  Eigen::Matrix3d inertia = Eigen::Matrix3d::Identity();
  double alpha{0.5};               
  bool use_nullspace{true};
  Eigen::Matrix<double,7,1> q_home = Eigen::Matrix<double,7,1>::Zero();
};

class Scheduler : public controller_interface::MultiInterfaceController<
                            franka_hw::FrankaModelInterface,
                            hardware_interface::EffortJointInterface,
                            franka_hw::FrankaStateInterface> {
public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

private:
  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  // Action server (non-RT thread)
  std::unique_ptr<actionlib::SimpleActionServer<fr3_controllers::ExecuteScheduleAction>> as_;
  void goalCallBack();     // accept and load new schedule
  void preemptCallBack();  // cancel current schedule

  // Communicate data between RT and non-RT threads.
  realtime_tools::RealtimeBuffer<std::shared_ptr<const PreparedSchedule>> sched_buf_;

  // Rollout logs (pre-allocated per goal)
  std::vector<float> x_meas_log_;   // 3*T
  std::vector<float> xd_meas_log_;  // 3*T
  std::vector<float> tau_log_;      // 7*T
  std::vector<float> q_meas_log_;   // 7*T  
  std::vector<float> dq_meas_log_;  // 7*T  
  double t0_sec_{0.0};           

  // Low-rate feedback timer (non-RT)
  ros::Timer fb_timer_;
  void feedbackCallBack(const ros::TimerEvent&);

  // RT state (only touched in update())
  Eigen::Matrix<double,3,7> previous_jacobian_{Eigen::Matrix<double,3,7>::Zero()};    // for Jdot*qdot FD
  Eigen::Matrix<double,7,1> previous_tau_{Eigen::Matrix<double,7,1>::Zero()};         // for torque-rate limit
  double delta_tau_max_{1.0};                                                         // Nm per cycle
  bool running_{false};
  uint32_t time_index_{0};
  double kp_ns_{15.0};
  double kd_ns_{3.0};

  Eigen::Matrix<double, 7, 1> dq_filtered_;
  Eigen::Matrix<double, 7, 1> q_d;    // desired joint position
  Eigen::Matrix<double, 7, 1> dq_d;   // desired joint velocity

  // RT helpers
  inline Eigen::Matrix<double,7,1> saturateTorqueRate(
      const Eigen::Matrix<double,7,1>& tau_des,
      const Eigen::Matrix<double,7,1>& tau_last) const;

  // NEW
  double filter_params_{0.005};
  double nullspace_stiffness_{20.0};
  double nullspace_stiffness_target_{20.0};
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  std::mutex position_and_orientation_d_target_mutex_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;
  // END NEW

  inline bool spd_floor(Eigen::Matrix3d& M, double eps) const;                        // clamp to SPD with floor
  inline uint32_t timeIndex(const ros::Time& now, const PreparedSchedule& S) const;   // t→k
};

} // namespace fr3_controllers
