import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory("gmmslam")
    config_file = LaunchConfiguration("config_file").perform(context)
    use_rviz = LaunchConfiguration("rviz").perform(context).lower() == "true"
    publish_sensor_tf = (
        LaunchConfiguration("publish_sensor_tf").perform(context).lower() == "true"
    )
    use_ext_odom_publisher = LaunchConfiguration("ext_odom_publisher").perform(context).lower() == "true"

    lidar_topic = LaunchConfiguration("lidar_topic").perform(context)
    sensor_frame = LaunchConfiguration("sensor_frame").perform(context)
    odom_frame = LaunchConfiguration("odom_frame").perform(context)
    map_frame = LaunchConfiguration("map_frame").perform(context)
    base_frame = LaunchConfiguration("base_frame").perform(context)
    gt_topic = LaunchConfiguration("gt_topic").perform(context)
    odometry_input = LaunchConfiguration("odometry_input").perform(context)

    common_overrides = {
        "config_file": config_file,
        "lidar_topic": lidar_topic,
        "sensor_frame": sensor_frame,
        "odom_frame": odom_frame,
        "map_frame": map_frame,
        "base_frame": base_frame,
        "gt_topic": gt_topic,
        "odometry_input": odometry_input,
    }

    nodes = []
    if publish_sensor_tf:
        nodes.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="depth_camera_tf",
                arguments=[
                    "0.083",
                    "0",
                    "0.0315",
                    "0",
                    "0",
                    "0",
                    base_frame,
                    sensor_frame,
                ],
            )
        )

    nodes.append(
        Node(
            package="gmmslam",
            executable="gmmslam_node",
            name="gmmslam_node",
            output="screen",
            parameters=[common_overrides],
        )
    )
    nodes.append(
        Node(
            package="gmmslam",
            executable="d2d_registration_node",
            name="d2d_registration_node",
            output="screen",
            parameters=[
                {
                    "config_file": config_file,
                    "request_topic": "/gmmslam_node/registration/request",
                    "result_topic": "/gmmslam_node/registration/result",
                    "num_workers": ParameterValue(
                        LaunchConfiguration("registration_workers"),
                        value_type=int,
                    ),
                    "suppress_backend_output": ParameterValue(
                        LaunchConfiguration("suppress_registration_backend_output"),
                        value_type=bool,
                    ),
                    "drop_stale_keyframe_age_s": ParameterValue(
                        LaunchConfiguration("drop_stale_keyframe_registration_age_s"),
                        value_type=float,
                    ),
                },
            ],
        )
    )

    if use_ext_odom_publisher:
        nodes.append(
            Node(
                package="gmmslam",
                executable="ext_odom_publisher_node",
                name="ext_odom_publisher_node",
                output="screen",
                parameters=[
                    {
                        "config_file": config_file,
                        "gt_topic": gt_topic,
                        "odom_frame": odom_frame,
                        "pose_topic": odometry_input,
                        "path_topic": "/gmmslam_node/ext_odom_path",
                    },
                ],
            )
        )

    if use_rviz:
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", os.path.join(pkg_share, "config", "gmmslam.rviz")],
            )
        )

    return nodes


def generate_launch_description():
    pkg_share = get_package_share_directory("gmmslam")
    default_config = os.path.join(pkg_share, "config", "params.yaml")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("lidar_topic", default_value="/m500_1/mpa/depth/points"),
            DeclareLaunchArgument("sensor_frame", default_value="depth_camera_1"),
            DeclareLaunchArgument("odom_frame", default_value="world"),
            DeclareLaunchArgument("map_frame", default_value="map"),
            DeclareLaunchArgument("base_frame", default_value="m500_1_base_link"),
            DeclareLaunchArgument(
                "gt_topic", default_value="/m500_1/mavros/local_position/pose"
            ),
            DeclareLaunchArgument(
                "odometry_input", default_value="/gmmslam_node/ext_odom_pose"
            ),
            DeclareLaunchArgument("registration_workers", default_value="1"),
            DeclareLaunchArgument(
                "suppress_registration_backend_output", default_value="true"
            ),
            DeclareLaunchArgument(
                "drop_stale_keyframe_registration_age_s", default_value="2.0"
            ),
            DeclareLaunchArgument("publish_sensor_tf", default_value="true"),
            DeclareLaunchArgument("rviz", default_value="false"),
            DeclareLaunchArgument("ext_odom_publisher", default_value="false"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
