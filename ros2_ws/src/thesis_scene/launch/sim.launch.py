from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PythonExpression, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Let Gazebo resolve package://thesis_scene and ur_description meshes
    share_roots = {os.path.dirname(get_package_share_directory(p)) for p in ('thesis_scene', 'ur_description')}
    gz_res = SetEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', os.pathsep.join(sorted(share_roots)))

    gui = LaunchConfiguration("gazebo_gui")
    moveit = LaunchConfiguration("moveit")
    scene = FindPackageShare("thesis_scene")

    urdf = Command([FindExecutable(name="xacro"), " ",
                    PathJoinSubstitution([scene, "urdf", "scene.urdf.xacro"]), " ur_type:=ur5"])

    rsp = Node(package="robot_state_publisher", executable="robot_state_publisher",
               parameters=[{"robot_description": ParameterValue(urdf, value_type=str), "use_sim_time": True}])

    gz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
        launch_arguments={"gz_args": [
            PythonExpression(["'-s' if '", gui, "' == 'false' else ''"]),
            " -r -v 3 ", PathJoinSubstitution([scene, "worlds", "scene.sdf"])]}.items())
    spawn = Node(package="ros_gz_sim", executable="create", output="screen",
                 arguments=["-topic", "robot_description", "-name", "ur5_scene"])

    bridge = Node(package="ros_gz_bridge", executable="parameter_bridge", output="screen",
                  arguments=["/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock",
                             "/ceiling_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image"])

    def spawner(name):
        return Node(package="controller_manager", executable="spawner", arguments=[name, "-c", "/controller_manager"])

    jsb = spawner("joint_state_broadcaster")
    jtc = spawner("joint_trajectory_controller")
    chain = [
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb])),
        RegisterEventHandler(OnProcessExit(target_action=jsb, on_exit=[jtc])),
    ]

    # sim_time=true makes ur_moveit_config use joint_trajectory_controller
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ur_moveit_config"), "/launch/ur_moveit.launch.py"]),
        launch_arguments={"ur_type": "ur5", "use_sim_time": "true", "launch_rviz": LaunchConfiguration("launch_rviz"),
                          "description_package": "thesis_scene", "description_file": "scene.urdf.xacro"}.items(),
        condition=IfCondition(moveit))

    return LaunchDescription([
        DeclareLaunchArgument("gazebo_gui", default_value="true"),
        DeclareLaunchArgument("moveit", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        gz_res, gz, rsp, spawn, bridge, *chain, moveit_launch,
    ])
