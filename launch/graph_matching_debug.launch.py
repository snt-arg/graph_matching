import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation clock'
    )

    graph_topic_arg = DeclareLaunchArgument(
        'graph_topic',
        default_value='/s_graphs/graph_structure',
        description='Topic name for the graph input'
    )

    debug_csv_file_arg = DeclareLaunchArgument(
        'debug_csv_file',
        default_value='',
        description='Path to the csv file for debug'
    )

    config = os.path.join(
        get_package_share_directory('graph_matching'),
        'config',
        'params.yaml'
    )

    graph_matching_node = Node(
        package='graph_matching',
        executable='graph_matching',
        parameters=[
            config,
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
            {'debug_csv_file': LaunchConfiguration('debug_csv_file')},
        ],
        remappings=[
            ('graph_matching/graphs', LaunchConfiguration('graph_topic')),
        ],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        use_sim_time_arg,
        graph_topic_arg,
        debug_csv_file_arg,
        graph_matching_node,
    ])
