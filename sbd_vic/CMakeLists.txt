cmake_minimum_required(VERSION 3.0.2)
project(sbd_vic)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  actionlib
  actionlib_msgs
  fr3_controllers
  franka_msgs
)
catkin_package()

catkin_install_python(PROGRAMS
  scripts/test_constant_schedule.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
