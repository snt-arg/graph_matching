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
    # launch_tester_node_ns = LaunchConfiguration('launch_tester_node_ns')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    
    DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )

    config = os.path.join(
        get_package_share_directory('graph_matching'),
        'config',
        'params.yaml'
    )

    graph_matching_node = Node(
        package='graph_matching',
        executable='graph_matching',
        # namespace='graph_matching',
        parameters = [config,
                      {'use_sim_time': use_sim_time}],
        remappings=[
            # ('graph_matching/graphs', '/s_graphs/graph_structure'),
            ('graph_matching/graphs', '/s_graphs/enriched_graph_structure'),
        ],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        graph_matching_node,
    ])
