cmake_minimum_required(VERSION 3.0.2)
project(data_collection)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  actionlib
  actionlib_msgs
  fr3_controllers
  franka_msgs
  franka_buttons
  franka_gripper
  geometry_msgs
  rospy
  sensor_msgs
  tf
  tf2_ros
  message_filters 
  cv_bridge
  tf2_geometry_msgs  
  tf_conversions 
)
catkin_package()

catkin_install_python(PROGRAMS
    scripts/button_gripper_map.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
