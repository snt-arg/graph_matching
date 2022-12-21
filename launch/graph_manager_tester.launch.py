import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            LogInfo, RegisterEventHandler, TimerAction, IncludeLaunchDescription)
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)
from launch.events import Shutdown
from launch.substitutions import (EnvironmentVariable, FindExecutable,
                                LaunchConfiguration, LocalSubstitution,
                                PythonExpression, PathJoinSubstitution, TextSubstitution)

from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # launch_tester_node_ns = LaunchConfiguration('launch_tester_node_ns')

    # launch_tester_node_arg = DeclareLaunchArgument(
    #     'launch_tester_node_ns',
    #     default_value=False
    # )


    graph_manager_node = Node(
        package='graph_manager',
        executable='graph_manager',
        namespace='graph_manager',
    )

    graph_manager_tester_node = Node(
        package='graph_manager',
        executable='graph_manager_tester',
        namespace='graph_manager',
    )

    launch_tester_node_with_timer = TimerAction(
        period=1.0,
        actions=[graph_manager_tester_node],
    )

    return LaunchDescription([
        graph_manager_node,
        launch_tester_node_with_timer
    ])