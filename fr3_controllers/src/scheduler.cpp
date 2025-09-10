"""
Operational Space Impedance Controller with Trajectory Scheduling for the Franka Research 3

Copyright (C) 2025 Shreyas Kumar, HiRo Lab, Indian Institute of Science.
Email: shreyaskumar@iisc.ac.in

This module is licensed under the MIT License.
For more information, visit: https://opensource.org/licenses/MIT
"""

#include <fr3_controllers/scheduler.h>

#include <sstream>
#include <algorithm>

#include <pluginlib/class_list_macros.h>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <Eigen/StdVector>

namespace fr3_controllers {

bool Scheduler::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) {

    std::string arm_id;
    if (!node_handle.getParam("arm_id", arm_id)) {
        ROS_ERROR_STREAM("CartesianImpedanceExampleController: Could not read parameter arm_id");
        return false;
    }

    std::vector<std::string> joint_names;
    if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
        ROS_ERROR(
            "CartesianImpedanceExampleController: Invalid or no joint_names parameters provided, "
            "aborting controller init!");
        return false;
    }

    auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
    if (model_interface == nullptr) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Error getting model interface from hardware");
        return false;
    }
    try {
        model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
            model_interface->getHandle(arm_id + "_model"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Exception getting model handle from interface: "
            << ex.what());
        return false;
    }

    auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
    if (state_interface == nullptr) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Error getting state interface from hardware");
        return false;
    }
    try {
        state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
            state_interface->getHandle(arm_id + "_robot"));
    } catch (hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Exception getting state handle from interface: "
            << ex.what());
        return false;
    }

    auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
    if (effort_joint_interface == nullptr) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Error getting effort joint interface from hardware");
        return false;
    }
    for (size_t i = 0; i < 7; ++i) {
        try {
        joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
        } catch (const hardware_interface::HardwareInterfaceException& ex) {
        ROS_ERROR_STREAM(
            "CartesianImpedanceExampleController: Exception getting joint handles: " << ex.what());
        return false;
        }
    }

    // Initialize realtime buffers
    sched_buf_.writeFromNonRT(std::shared_ptr<const PreparedSchedule>());

    // Allocate action server object to manage goals of type ExecuteScheduleAction
    as_.reset(new actionlib::SimpleActionServer<fr3_controllers::ExecuteScheduleAction>(
        node_handle, "execute_schedule", false));
    // Call the callback to construct PreparedSchedule when a new goal is received
    as_->registerGoalCallback(boost::bind(&Scheduler::goalCallBack, this));
    // Call the callback to preempt (cancel) the current goal
    as_->registerPreemptCallback(boost::bind(&Scheduler::preemptCallBack, this));
    // Start the action server
    as_->start();

    // Create a timer to periodically publish non-RT feedback 
    fb_timer_ = node_handle.createTimer(ros::Duration(0.05), &Scheduler::feedbackCallBack, this);  // 20 Hz

    return true;
}

void Scheduler::starting(const ros::Time& /*time*/) {
    const franka::RobotState robot_state = state_handle_->getRobotState();
    q_d = Eigen::Map<const Eigen::Matrix<double, 7, 1>>(robot_state.q.data());
    dq_d = Eigen::Matrix<double, 7, 1>::Zero();
    previous_jacobian_.setZero();
    previous_tau_.setZero();
    running_ = false;
    time_index_ = 0;
}

void Scheduler::update(const ros::Time& now, const ros::Duration& period) {
    // Current schedule pointer (nullptr means idle)
    auto schedule_ptr = sched_buf_.readFromRT();
    const PreparedSchedule* schedule = schedule_ptr ? schedule_ptr->get() : nullptr;

    double dt_est = std::max(1e-6, period.toSec());

    // Robot state and model terms
    const franka::RobotState robot_state = state_handle_->getRobotState();

    const std::array<double, 49> mass_array         = model_handle_->getMass();
    const std::array<double,  7> coriolis_array     = model_handle_->getCoriolis();
    const std::array<double,  7> gravity_array      = model_handle_->getGravity();             
    const std::array<double, 42> jacobian_array     = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

    Eigen::Map<const Eigen::Matrix<double, 7, 7>> M(mass_array.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> C(coriolis_array.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> G(gravity_array.data());
    Eigen::Map<const Eigen::Matrix<double, 6, 7>> J_(jacobian_array.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
    Eigen::Map<const Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());

    const Eigen::Matrix<double, 3, 7> J = J_.topRows<3>();
    const Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
    const Eigen::Vector3d x(transform.translation());
    const Eigen::Vector3d xd = J * dq;

    // double alpha = 0.99;
    // for (size_t i = 0; i < 7; i++) {
    //     dq_filtered_[i] = (1 - alpha) * dq_filtered_[i] + alpha * robot_state.dq[i];
    // }

    // Eigen::Matrix<double,7,7> K, D;
    // K.setZero(); 
    // D.setZero();
    // K(0,0)=600; K(1,1)=600; K(2,2)=600; K(3,3)=600; K(4,4)=250; K(5,5)=150; K(6,6)=50;
    // D(0,0)=50;  D(1,1)=50;  D(2,2)=50;  D(3,3)=20;  D(4,4)=20;  D(5,5)=20;  D(6,6)=10;
  
    // Eigen::Matrix<double, 7, 1> tau_cmd = C + K*(q_d - q) + D*(dq_d - dq_filtered_); 

    Eigen::Matrix<double, 7, 1> tau_cmd = C; 
    running_ = false;

    if(schedule) {
        const uint32_t k = timeIndex(now, *schedule);
        time_index_ = k;

        if (k < schedule->T) {
            const Eigen::Vector3d x_d     = schedule->x[k];
            const Eigen::Vector3d xd_d    = schedule->xd[k];
            const Eigen::Vector3d xdd_d   = schedule->xdd[k];
            Eigen::Matrix3d K             = schedule->stiffness[k];
            Eigen::Matrix3d D             = schedule->damping[k];
            Eigen::Matrix3d H             = schedule->inertia;

            // Ensure K, D, H are SPD
            spd_floor(K, 1e-4);
            spd_floor(D, 1e-4);
            spd_floor(H, 1e-4);

            // Tracking errors
            const Eigen::Vector3d e  = x  - x_d;
            const Eigen::Vector3d ed = xd - xd_d;

            // Task-space acceleration command Eq. (5)
            const Eigen::Vector3d xdd_cmd = xdd_d - H.inverse() * (K * e + D * ed);

            Eigen::Vector3d jdot_qdot = Eigen::Vector3d::Zero();
            if (!previous_jacobian_.isZero(0)) {
                jdot_qdot = ((J - previous_jacobian_) / dt_est) * dq;
            }
            previous_jacobian_ = J;

            Eigen::LDLT<Eigen::Matrix<double,7,7>> ldlt(M);
            Eigen::Matrix<double,7,3> X = ldlt.solve(J.transpose());                // X = M^{-1} J^T
            Eigen::Matrix3d JJ = J * X;                                             // X = M^{-1} J^T
            JJ = 0.5 * (JJ + JJ.transpose()) + 1e-8 * Eigen::Matrix3d::Identity();  // Tikhonov

            Eigen::LLT<Eigen::Matrix3d> llt(JJ);
            Eigen::Matrix3d Lambda = llt.solve(Eigen::Matrix3d::Identity());

            const Eigen::Matrix<double,7,3> Jsharp = X * Lambda;                    // J# = M^{-1} J^T Λ

            // joint-space acceleration
            Eigen::Matrix<double, 7, 1> qdd_task = Jsharp * (xdd_cmd - jdot_qdot);
            Eigen::Matrix<double, 7, 1> qdd = qdd_task; 

            if (schedule->use_nullspace) {
                const Eigen::Matrix<double, 7, 7> I = Eigen::Matrix<double, 7, 7>::Identity();
                const Eigen::Matrix<double, 7, 7> N = I - Jsharp * J;
                const Eigen::Matrix<double, 7, 1> q_err = schedule->q_home - q;
                const Eigen::Matrix<double, 7, 1> qdd_ns = kp_ns_ * q_err - kd_ns_ * dq;
                qdd = qdd_task + N * qdd_ns;
            }

            // full dynamics torque
            tau_cmd = M * qdd + C;
            running_ = true;

            // torque-rate limit and command
            tau_cmd = saturateTorqueRate(tau_cmd, previous_tau_);
            previous_tau_ = tau_cmd;
            for (size_t i = 0; i < 7; ++i) joint_handles_[i].setCommand(tau_cmd(i));

            // log rollout
            x_meas_log_[3*k+0] = static_cast<float>(x(0));
            x_meas_log_[3*k+1] = static_cast<float>(x(1));
            x_meas_log_[3*k+2] = static_cast<float>(x(2));
            xd_meas_log_[3*k+0] = static_cast<float>(xd(0));
            xd_meas_log_[3*k+1] = static_cast<float>(xd(1));
            xd_meas_log_[3*k+2] = static_cast<float>(xd(2));
            for (int j = 0; j < 7; ++j) {
                tau_log_[7*k + j] = static_cast<float>(tau_cmd(j));
                q_meas_log_[7*k + j]  = static_cast<float>(q(j));
                dq_meas_log_[7*k + j] = static_cast<float>(dq(j));
            }
            return;
        }
    }

    // torque-rate limit and command
    tau_cmd = saturateTorqueRate(tau_cmd, previous_tau_);
    previous_tau_ = tau_cmd;
    for (size_t i = 0; i < 7; ++i) joint_handles_[i].setCommand(tau_cmd(i));
}

Eigen::Matrix<double,7,1> Scheduler::saturateTorqueRate(const Eigen::Matrix<double,7,1>& tau_des,
                                                        const Eigen::Matrix<double,7,1>& tau_last) const {
    Eigen::Matrix<double,7,1> out;
    for (int i = 0; i < 7; ++i) {
        const double d = tau_des(i) - tau_last(i);
        const double s = std::max(std::min(d, delta_tau_max_), -delta_tau_max_);
        out(i) = tau_last(i) + s;
    }
    return out;
}

bool Scheduler::spd_floor(Eigen::Matrix3d& mat, double eps) const {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(mat);
    if (es.info() != Eigen::Success) { mat = eps * Eigen::Matrix3d::Identity(); return false; }
    Eigen::Vector3d eigvals = es.eigenvalues().cwiseMax(eps);
    Eigen::Matrix3d eigvecs = es.eigenvectors();
    mat = eigvecs * eigvals.asDiagonal() * eigvecs.transpose();
    return true;
}

uint32_t Scheduler::timeIndex(const ros::Time& now, const PreparedSchedule& schedule) const {
    if (now < schedule.t0) return 0;
    const double elapsed_s = (now - schedule.t0).toSec();
    const double kf = std::round(elapsed_s / schedule.dt);
    if (kf < 0.0) return 0;
    if (kf >= static_cast<double>(schedule.T)) return schedule.T;
    return static_cast<uint32_t>(kf);
}

void Scheduler::goalCallBack() {
    auto goal = as_->acceptNewGoal();

    // basic checks
    auto bad = [&](bool cond, const char* why)->bool{
        if (!cond) return false;
        fr3_controllers::ExecuteScheduleResult res; 
        res.success=false; 
        res.reason=why; 
        res.T_exec=0;
        as_->setAborted(res); 
        return true;
    };

    if (bad(goal->dt <= 0.0, "dt <= 0")) return;

    const uint32_t T = goal->T;
    
    if (bad(goal->x_seq.size()   != 3ull*T, "len x_seq"))   return;
    if (bad(goal->xd_seq.size()  != 3ull*T, "len xd_seq"))  return;
    if (bad(goal->xdd_seq.size() != 3ull*T, "len xdd_seq")) return;
    if (bad(goal->K_seq.size()   != 9ull*T, "len K_seq"))   return;
    if (bad(goal->D_seq.size()   != 9ull*T, "len D_seq"))   return;
    if (bad(goal->H.size()       != 9ull,   "len H"))       return;
    if (bad(goal->q_home.size()  != 7ull,   "len q_home"))  return;
    
    // build schedule
    std::shared_ptr<PreparedSchedule> schedule(new PreparedSchedule());
    schedule->dt            = goal->dt;
    schedule->T             = T;
    schedule->alpha         = goal->alpha_kb;
    schedule->use_nullspace = goal->use_nullspace;
    schedule->inertia       = Eigen::Map<const Eigen::Matrix<double,3,3>>(goal->H.data());
    schedule->q_home        = Eigen::Map<const Eigen::Matrix<double,7,1>>(goal->q_home.data());

    schedule->x.resize(T);
    schedule->xd.resize(T);
    schedule->xdd.resize(T);
    schedule->stiffness.resize(T);
    schedule->damping.resize(T);

    for (uint32_t k = 0; k < T; ++k) {
        schedule->x[k]   = Eigen::Map<const Eigen::Matrix<double,3,1>>(&goal->x_seq[3*k]);
        schedule->xd[k] = Eigen::Map<const Eigen::Matrix<double,3,1>>(&goal->xd_seq[3*k]);
        schedule->xdd[k]  = Eigen::Map<const Eigen::Matrix<double,3,1>>(&goal->xdd_seq[3*k]);
        schedule->stiffness[k]  = Eigen::Map<const Eigen::Matrix<double,3,3>>(&goal->K_seq[9*k]);
        schedule->damping[k]    = Eigen::Map<const Eigen::Matrix<double,3,3>>(&goal->D_seq[9*k]);
    }

    // start slightly in the future for clean k=0 alignment
    schedule->t0 = ros::Time::now() + ros::Duration(0.10);

    // pre-allocate logs
    t0_sec_ = schedule->t0.toSec();     
    x_meas_log_.assign(3*T, 0.f);
    xd_meas_log_.assign(3*T, 0.f);
    tau_log_.assign(7*T, 0.f);
    q_meas_log_.assign(7*T, 0.f);          
    dq_meas_log_.assign(7*T, 0.f);        

    // arm schedule and reset RT state
    sched_buf_.writeFromNonRT(schedule);
    previous_jacobian_.setZero();
    previous_tau_.setZero();
    running_ = false;
    time_index_ = 0;
}

void Scheduler::preemptCallBack() {
    auto sp = sched_buf_.readFromNonRT();
    const PreparedSchedule* S = (sp && sp->get()) ? sp->get() : nullptr;
    sched_buf_.writeFromNonRT(std::shared_ptr<const PreparedSchedule>());

    fr3_controllers::ExecuteScheduleResult res;
    res.success = true;
    res.reason  = "preempted";
    res.T_exec  = time_index_;
    // return whatever logged so far (if any schedule existed)
    if (S) {
        const uint32_t T = S->T;
        const uint32_t N = std::min<uint32_t>(time_index_, T);
        res.x_meas_seq.assign(x_meas_log_.begin(), x_meas_log_.begin() + 3*N);
        res.xd_meas_seq.assign(xd_meas_log_.begin(), xd_meas_log_.begin() + 3*N);
        res.tau_seq.assign(tau_log_.begin(), tau_log_.begin() + 7*N);
        res.q_meas_seq.assign(q_meas_log_.begin(),  q_meas_log_.begin()  + 7*N);
        res.dq_meas_seq.assign(dq_meas_log_.begin(), dq_meas_log_.begin() + 7*N);
        res.t0_sec = t0_sec_;
    }
    as_->setPreempted(res);
}

void Scheduler::feedbackCallBack(const ros::TimerEvent&) {
    if (!as_->isActive()) return;
    auto sp = sched_buf_.readFromNonRT();
    if (!sp || !sp->get()) return;
    const PreparedSchedule& S = **sp;

    const uint32_t k_now = timeIndex(ros::Time::now(), S);
    if (k_now >= S.T) {
        fr3_controllers::ExecuteScheduleResult res;
        res.success = true;
        res.reason  = "completed";
        res.T_exec  = S.T;
        res.x_meas_seq  = x_meas_log_;
        res.xd_meas_seq = xd_meas_log_;
        res.tau_seq     = tau_log_;
        res.q_meas_seq  = q_meas_log_;
        res.dq_meas_seq = dq_meas_log_;
        res.t0_sec = t0_sec_;
        as_->setSucceeded(res);

        // clear schedule
        sched_buf_.writeFromNonRT(std::shared_ptr<const PreparedSchedule>());
    }
}

} // namespace fr3_controllers

PLUGINLIB_EXPORT_CLASS(fr3_controllers::Scheduler, controller_interface::ControllerBase)