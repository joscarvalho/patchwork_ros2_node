cmake_minimum_required(VERSION 3.8)
project(patchwork)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)

find_package(alfa_node REQUIRED)
find_package(alfa_msg REQUIRED)

find_package(sensor_msgs REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(Boost 1.54 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

add_executable(patchwork src/patchwork.cpp src/patchwork.hpp src/utils.hpp)
target_link_libraries(patchwork ${PCL_LIBRARIES})
ament_target_dependencies(patchwork rclcpp alfa_node sensor_msgs alfa_msg pcl_conversions pcl_ros tf2 tf2_ros)

install(TARGETS
patchwork
  DESTINATION lib/${PROJECT_NAME}/
)

ament_package()