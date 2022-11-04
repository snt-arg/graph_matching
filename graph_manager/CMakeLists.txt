cmake_minimum_required(VERSION 3.8)
project(graph_manager)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Edge.msg"
  "msg/Node.msg"
  "msg/SubgraphMatching.msg"
  "srv/SubgraphMatching.srv"
  DEPENDENCIES
)

ament_package()
