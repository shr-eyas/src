cmake_minimum_required(VERSION 3.0.2)
project(multi_arm_controllers)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  controller_interface
  hardware_interface
  pluginlib
  franka_hw
  franka_msgs
  realtime_tools
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES multi_arm_controllers
  CATKIN_DEPENDS
    roscpp
    controller_interface
    hardware_interface
    pluginlib
    franka_hw
    franka_msgs
    realtime_tools
    sensor_msgs
)

include_directories(include ${catkin_INCLUDE_DIRS})

add_library(multi_arm_controllers
  src/grav_comp_controller.cpp
  src/mirror_impedance_controller.cpp
)

target_link_libraries(multi_arm_controllers ${catkin_LIBRARIES})
add_dependencies(multi_arm_controllers ${catkin_EXPORTED_TARGETS})

install(TARGETS multi_arm_controllers
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(FILES multi_arm_controller_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)