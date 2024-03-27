cmake_minimum_required(VERSION 3.0.2)
project(threeds_grasp)

find_package(
  catkin REQUIRED
  COMPONENTS sensor_msgs
             cv_bridge
             rospy
             message_generation
)

add_service_files(FILES CompleteCloud.srv)

generate_messages(DEPENDENCIES sensor_msgs)

catkin_package(
  #  INCLUDE_DIRS include
  #  LIBRARIES yolact_ros
  CATKIN_DEPENDS
    sensor_msgs
    cv_bridge
    rospy
    message_runtime
  #  DEPENDS system_lib
)

install(
  PROGRAMS scripts/completion.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
