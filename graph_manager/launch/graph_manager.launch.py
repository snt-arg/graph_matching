import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            LogInfo, RegisterEventHandler, TimerAction)
from launch.event_handlers import (OnExecutionComplete, OnProcessExit,
                                OnProcessIO, OnProcessStart, OnShutdown)
from launch.events import Shutdown
from launch.substitutions import (EnvironmentVariable, FindExecutable,
                                LaunchConfiguration, LocalSubstitution,
                                PythonExpression)

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        graph_manager = Node(
            package='graph_manager',
            executable='graph_manager',
            namespace='graph_manager',
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=graph_manager,
                on_exit=[
                    Node(
                        package='graph_manager',
                        executable='graph_manager_tester',
                        namespace='graph_manager_tester',
                    ),
                ]
            )
        ),
    ])