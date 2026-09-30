# ROS 2 Humble + UR5 + Gazebo Fortress + MoveIt 2 — cheatsheet

Workspace: `~/Documents/thesis/ros2_ws`, package `thesis_scene` (UR5 + table + ceiling camera).
Gazebo is **Ignition Fortress** (`ign gazebo`, not `gz sim`); ROS bridge is `ros_gz`, ros2_control plugin is `ign_ros2_control`.

## 0. Environment (every new terminal)
```bash
export PATH=/usr/bin:$PATH          # miniforge python breaks colcon (catkin_pkg missing)
source /opt/ros/humble/setup.bash
source ~/Documents/thesis/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=42             # isolate sim from the URSim/real-robot session on domain 0
```
Check: `printenv | grep -E "ROS|AMENT"`, `which python3` (must be /usr/bin/python3).
Domains are separate networks: terminals with different `ROS_DOMAIN_ID` cannot see each other.

## 1. Build
```bash
cd ~/Documents/thesis/ros2_ws
colcon build --symlink-install                 # all packages
colcon build --packages-select thesis_scene    # one package
rm -rf build install log                       # clean rebuild (needed after switching python/cmake cache)
```
- `--symlink-install`: edits to urdf/yaml/launch/py take effect without rebuild, but **new files** need a rebuild.
- After any build: `source install/setup.bash` again in open terminals.

## 2. Run the sim (this project)
```bash
ros2 launch thesis_scene sim.launch.py                          # Gazebo GUI + MoveIt + RViz
ros2 launch thesis_scene sim.launch.py gazebo_gui:=false        # Gazebo server only (no window)
ros2 launch thesis_scene sim.launch.py moveit:=false            # Gazebo + controllers only
ros2 launch thesis_scene sim.launch.py launch_rviz:=false       # no RViz (e.g. no DISPLAY / ssh)
ros2 launch thesis_scene sim.launch.py --show-args              # list launch arguments
```
Ready when the log prints `You can start planning now!`.
In RViz: MotionPlanning panel → drag the orange interactive marker at the tip → **Plan** → **Execute**
(or **Plan & Execute**). Planning Group = `ur_manipulator`.
Stop: Ctrl-C in the launch terminal; leftover Gazebo: `pkill -f "[i]gn gazebo"`.

## 3. Inspect the system
```bash
ros2 node list ; ros2 node info /move_group
ros2 topic list [-t]                       # -t shows types
ros2 topic echo /joint_states --once
ros2 topic echo /joint_states --once --field position
ros2 topic hz /ceiling_camera/image        # camera ~15 Hz
ros2 topic info /joint_states -v           # who publishes/subscribes, QoS
ros2 service list ; ros2 service call <srv> <type> "<yaml>"
ros2 action list ; ros2 action info /joint_trajectory_controller/follow_joint_trajectory
ros2 param list /move_group ; ros2 param get <node> <param> ; ros2 param set <node> <param> <val>
ros2 interface show control_msgs/action/FollowJointTrajectory
ros2 pkg prefix thesis_scene ; ros2 pkg list | grep ur_
ros2 doctor --report                       # env / middleware sanity check
```
Camera image: `ros2 run rqt_image_view rqt_image_view` (topic `/ceiling_camera/image`) or RViz → Add → Image.

## 4. ros2_control
```bash
ros2 control list_controllers
ros2 control list_hardware_interfaces      # command/state interfaces, claimed or not
ros2 control list_hardware_components
ros2 control switch_controllers --activate scaled_joint_trajectory_controller --deactivate joint_trajectory_controller
ros2 run controller_manager spawner joint_trajectory_controller -c /controller_manager
ros2 run controller_manager spawner <name> -c /controller_manager --inactive
```
Sim uses `joint_trajectory_controller` (+ `joint_state_broadcaster`). The real robot driver uses
`scaled_joint_trajectory_controller`; `ur_moveit.launch.py use_sim_time:=true` switches MoveIt to the former.

## 5. Send a motion without MoveIt
```bash
ros2 action send_goal /joint_trajectory_controller/follow_joint_trajectory \
  control_msgs/action/FollowJointTrajectory \
  "{trajectory: {joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint],
     points: [{positions: [0.5,-1.2,1.0,-1.57,0.0,0.0], time_from_start: {sec: 3}}]}}" --feedback
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
  "{joint_names: [shoulder_pan_joint, shoulder_lift_joint, elbow_joint, wrist_1_joint, wrist_2_joint, wrist_3_joint],
    points: [{positions: [0,-1.57,0,-1.57,0,0], time_from_start: {sec: 3}}]}"
```
Home pose from `ur_description/config/initial_positions.yaml`: `[0, -1.57, 0, -1.57, 0, 0]`.

## 6. TF / URDF
```bash
ros2 run tf2_tools view_frames                       # writes frames.pdf
ros2 run tf2_ros tf2_echo world tool0                # TCP pose in world
xacro src/thesis_scene/urdf/scene.urdf.xacro ur_type:=ur5 > /tmp/scene.urdf
check_urdf /tmp/scene.urdf                           # parse + link tree
urdf_to_graphviz /tmp/scene.urdf                     # link/joint graph pdf
ign sdf -p /tmp/scene.urdf                           # how Gazebo will see it (URDF->SDF)
ros2 launch ur_description view_ur.launch.py ur_type:=ur5   # bare UR5 in RViz + joint sliders
```
Gazebo silently **drops URDF links without `<inertial>`** — every fixed link in `scene.urdf.xacro` carries one.

## 7. Gazebo Fortress (`ign`)
```bash
ign gazebo -r -v 3 world.sdf          # -r run, -s server only, -v verbosity (0-4)
ign gazebo -g                          # GUI only (attach to a running server)
ign topic -l ; ign topic -e -t /world/scene/stats
ign service -l
ign model --list
ign gazebo --versions
```
Resource path (meshes `package://`): the launch file sets `IGN_GAZEBO_RESOURCE_PATH` to `install/share` and
`/opt/ros/humble/share`. For manual runs: `export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:$HOME/Documents/thesis/ros2_ws/install/share:/opt/ros/humble/share`.
Bridge (ROS ↔ Ignition): `ros2 run ros_gz_bridge parameter_bridge /topic@ros_msg_type[ign_msg_type`
(`[` = Ignition→ROS, `]` = ROS→Ignition, `@` = both). Already bridged: `/clock`, `/ceiling_camera/image`.
Headless/ssh: Fortress GUI needs a display; use `gazebo_gui:=false launch_rviz:=false`.

## 8. MoveIt 2
```bash
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 use_sim_time:=true \
    description_package:=thesis_scene description_file:=scene.urdf.xacro   # (sim.launch.py already does this)
ros2 topic echo /monitored_planning_scene --once        # scene incl. table + camera collision boxes
ros2 action list | grep -E "move_action|execute_trajectory"
ros2 service call /compute_fk moveit_msgs/srv/GetPositionFK ...
ros2 service call /compute_ik moveit_msgs/srv/GetPositionIK ...
```
Add obstacles at runtime: RViz → MotionPlanning → Scene Objects, or publish `moveit_msgs/msg/CollisionObject`
on `/collision_object`. Programmatic control: `moveit_py` (not in Humble apt) / C++ `MoveGroupInterface` /
call the `/move_action` action directly. Planners: OMPL (default); `pilz_industrial_motion_planner` for LIN/PTP if installed.

## 9. Real robot / URSim (later; **not** used by the sim)
```bash
ros2 run ur_client_library start_ursim.sh -m ur5                      # URSim docker, robot_ip 192.168.56.101
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=<IP> launch_rviz:=false
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5 robot_ip:=yyy use_fake_hardware:=true   # no robot, mock
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_rviz:=true   # against the driver above
ros2 service call /dashboard_client/play std_srvs/srv/Trigger   # also: pause, stop, power_on, brake_release
ros2 run ur_calibration calibration_correction robot_ip:=<IP> target_filename:="$HOME/my_ur5_cal.yaml"
```
Real robot needs the *External Control* URCap program running on the pendant (or `headless_mode:=true`).
A UR driver session may already be running on domain 0 (`pgrep -fa ur_robot_driver`) — keep the sim on another domain.
Planned "sim-first safety check": send the same trajectory to the Gazebo controller first, check
`error_code == 0` and no collisions, then forward to the real driver's `scaled_joint_trajectory_controller`.

## 10. Recording / debugging
```bash
ros2 bag record -o run1 /joint_states /joint_trajectory_controller/state /ceiling_camera/image
ros2 bag info run1 ; ros2 bag play run1 --clock
rqt_graph                                      # node/topic graph
rqt                                            # plugin hub (image view, joint trajectory controller GUI, ...)
ros2 launch thesis_scene sim.launch.py 2>&1 | tee sim.log
ros2 run --prefix 'gdb -ex run --args' <pkg> <exe>       # debugger
ros2 daemon stop ; ros2 daemon start                     # fixes stale `ros2 topic list` results
```
Log files: `~/.ros/log/`. Use `--ros-args --log-level debug` for one node.

## 11. Common pitfalls (seen in this project)
| Symptom | Cause / fix |
|---|---|
| `No module named catkin_pkg` at build | miniforge python first in PATH → `export PATH=/usr/bin:$PATH`, then `rm -rf build install log` |
| `file 'sim.launch.py' was not found in share directory` | launch/ dir installed before file existed → `colcon build` again |
| `list_controllers` shows io_and_status_controller etc. | you are looking at the UR driver on domain 0, not the sim → set `ROS_DOMAIN_ID=42` |
| Gazebo: `attached_to name[camera_link] ... does not match` | URDF link without `<inertial>` was dropped → add inertial |
| Gazebo: `Unable to find file model://thesis_scene/meshes/...` | `IGN_GAZEBO_RESOURCE_PATH` missing install/share |
| xacro `Undefined substitution argument` | `$(arg x)` used before `<xacro:arg name="x">`; the `<robot name>` attribute can't use args |
| Controller `inactive` not `active` | spawn with `ros2 run controller_manager spawner <name> -c /controller_manager`, check YAML joint names |
| Robot doesn't move, no error | check `use_sim_time` consistent (`/clock` bridged) and the controller is `active` |

---
# Appendix: general ROS 2 Humble reference

## A. Create packages / nodes
```bash
cd src
ros2 pkg create --build-type ament_python my_pkg --dependencies rclpy std_msgs sensor_msgs
ros2 pkg create --build-type ament_cmake  my_cpp --dependencies rclcpp moveit_ros_planning_interface
ros2 pkg create --build-type ament_python my_pkg --node-name my_node      # with a node stub
ros2 run my_pkg my_node --ros-args -p speed:=0.5 -r old_topic:=new_topic --log-level debug
ros2 run my_pkg my_node --ros-args --params-file params.yaml -r __ns:=/robot1 -r __node:=alt_name
```
ament_python: entry points go in `setup.py` → `entry_points={'console_scripts': ['my_node = my_pkg.my_node:main']}`;
launch/config/urdf need `data_files` entries (or use ament_cmake + `install(DIRECTORY ...)` like `thesis_scene`).
Build rules: python edits need `--symlink-install` to skip rebuilds; C++ always rebuilds; new files always rebuild.

## B. Minimal Python node (rclpy)
```python
import rclpy; from rclpy.node import Node; from std_msgs.msg import String
class N(Node):
    def __init__(self):
        super().__init__('n'); self.declare_parameter('rate', 10.0)
        self.pub = self.create_publisher(String, 'chatter', 10)
        self.sub = self.create_subscription(String, 'chatter', lambda m: self.get_logger().info(m.data), 10)
        self.create_timer(1.0 / self.get_parameter('rate').value, lambda: self.pub.publish(String(data='hi')))
rclpy.init(); rclpy.spin(N()); rclpy.shutdown()
```
Action client: `from rclpy.action import ActionClient` → `ActionClient(node, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')`,
`send_goal_async(goal)`. Sim time: `ros2 run ... --ros-args -p use_sim_time:=true`.
Use the system python for ROS: `/usr/bin/python3 my_node.py` (the miniforge python has no `rclpy`).

## C. colcon
```bash
colcon build --symlink-install --packages-up-to my_pkg      # pkg + its deps
colcon build --packages-select my_pkg --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon build --event-handlers console_direct+                # live compiler output
colcon test --packages-select my_pkg && colcon test-result --verbose
colcon list ; colcon graph
rosdep install --from-paths src --ignore-src -r -y           # install missing deps (needs sudo, rosdep init once)
```

## D. Launch files (Python)
```bash
ros2 launch <pkg> <file>.launch.py arg:=value ; ros2 launch <pkg> <file> --show-args ; ros2 launch -p <pkg> <file>   # -p = print description
ros2 launch /abs/path/file.launch.py
```
Building blocks: `Node`, `IncludeLaunchDescription`, `DeclareLaunchArgument`, `LaunchConfiguration`, `PathJoinSubstitution`,
`FindPackageShare`, `Command([FindExecutable(name='xacro'), ' ', path])`, `RegisterEventHandler(OnProcessExit(...))`,
`SetEnvironmentVariable`, `TimerAction(period=5.0, actions=[...])`, `GroupAction`, `IfCondition/UnlessCondition`, `OpaqueFunction`.
Example in this repo: `src/thesis_scene/launch/sim.launch.py`.

## E. Topic / service / action / param — full verb list
```bash
ros2 topic   list|echo|hz|bw|delay|info|type|find|pub  (pub -r 10 = 10 Hz, --once, -1)
ros2 service list|type|find|call
ros2 action  list|info|send_goal|type
ros2 param   list|get|set|dump|load|describe|delete
ros2 node    list|info
ros2 interface list|show|package|packages|proto        # msg/srv/action definitions
ros2 lifecycle nodes|get|set <node> configure|activate|deactivate|cleanup
ros2 component list|load|unload|types
ros2 wtf   (= ros2 doctor)
ros2 multicast receive|send                           # check DDS multicast between machines
ros2 param dump /move_group > mg.yaml                 # save params of a running node
ros2 topic echo /tf_static --qos-durability transient_local --qos-reliability reliable
ros2 topic pub /cmd std_msgs/msg/Float64 "{data: 0.1}" -r 10
```
QoS mismatch (no data although topic exists): compare `ros2 topic info -v`; sensors usually `best_effort`, latched topics `transient_local`.

## F. Networking / DDS
```bash
export ROS_DOMAIN_ID=42                          # 0-101; separates independent systems
export ROS_LOCALHOST_ONLY=1                      # never leave this machine
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp     # or rmw_fastrtps_cpp (default); needs ros-humble-rmw-cyclonedds-cpp
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET      # (newer distros; Humble uses ROS_LOCALHOST_ONLY)
ros2 daemon stop; ros2 daemon start              # refresh graph cache
```
Two machines: same domain ID, same subnet, multicast allowed, firewall open (UDP 7400+). Docker: `--network host`.

## G. TF2
```bash
ros2 run tf2_ros tf2_echo <parent> <child>
ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 1 --roll 0 --pitch 0 --yaw 0 --frame-id world --child-frame-id cam
ros2 run tf2_tools view_frames
ros2 topic echo /tf --once ; ros2 topic echo /tf_static --once
ros2 run tf2_ros tf2_monitor                     # delays / rates per frame
```
Python: `tf2_ros.Buffer` + `TransformListener(buf, node)` → `buf.lookup_transform('world','tool0', rclpy.time.Time())`.

## H. URDF / xacro / robot_state_publisher
```bash
xacro file.urdf.xacro arg:=val > out.urdf ; check_urdf out.urdf ; urdf_to_graphviz out.urdf
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:="$(xacro file.urdf.xacro)"
ros2 run joint_state_publisher_gui joint_state_publisher_gui        # sliders (sudo apt install ros-humble-joint-state-publisher-gui)
ros2 launch ur_description view_ur.launch.py ur_type:=ur5
rviz2 -d config.rviz                                                  # RViz: Fixed Frame = world; add RobotModel, TF, MotionPlanning
```
xacro tips: `<xacro:arg>` = command-line arg (declare before use), `<xacro:property>`, `${expr}`, `<xacro:macro name params>`,
`$(find pkg)`, `xacro.load_yaml(path)`. URDF needs `<inertial>` for every link you want Gazebo to keep.

## I. Gazebo Fortress ↔ ROS
```bash
ros2 run ros_gz_sim create -name box -string "<sdf ...>"      # or -file x.sdf / -topic robot_description, -x -y -z
ros2 run ros_gz_sim remove -name box   (not in all versions: use `ign service -s /world/<w>/remove ...`)
ros2 run ros_gz_bridge parameter_bridge /scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan
ros2 run ros_gz_image image_bridge /ceiling_camera/image          # image transport bridge
ign topic -l ; ign topic -e -t <topic> ; ign service -l ; ign gazebo -r world.sdf
```
Fortress uses `ignition.msgs.*` types; Harmonic (Jazzy) would use `gz.msgs.*` — don't mix examples from Jazzy docs.

## J. MoveIt 2 extras
```bash
sudo apt install ros-humble-moveit-setup-assistant ; ros2 launch moveit_setup_assistant setup_assistant.launch.py   # generate a config pkg
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5 launch_servo:=true
ros2 topic pub /servo_node/delta_twist_cmds geometry_msgs/msg/TwistStamped ...            # MoveIt Servo (needs `ros2 service call /servo_node/start_servo std_srvs/srv/Trigger`)
ros2 service call /move_group/get_planning_scene ...           # or /get_planning_scene moveit_msgs/srv/GetPlanningScene
```
Config files (in `ur_moveit_config/config/`): `kinematics.yaml` (IK solver), `ompl_planning.yaml`, `joint_limits.yaml`,
`controllers.yaml`, `ur_servo.yaml`. SRDF: planning groups, named poses, ACM (`disable_collisions`). Copy the package to customize.

## K. Bags
```bash
ros2 bag record -a | -o name /topic1 /topic2 ; ros2 bag record --max-bag-duration 60 -s mcap ...
ros2 bag info name ; ros2 bag play name -r 0.5 --loop --clock --topics /joint_states
ros2 bag convert -i in -o out.yaml                       # change storage/filter
```

## L. Debug / tooling
```bash
rqt ; rqt_graph ; rqt_console ; rqt_plot /joint_states/position[0] ; ros2 run rqt_reconfigure rqt_reconfigure
ros2 run rqt_joint_trajectory_controller rqt_joint_trajectory_controller
ros2 run plotjuggler plotjuggler                         # sudo apt install ros-humble-plotjuggler-ros
ros2 topic hz /topic --window 100 ; ros2 topic bw /topic
gdb/valgrind: ros2 run --prefix 'gdb -ex run --args' pkg exe ; --prefix 'valgrind --leak-check=full'
ros2 launch ... --debug ; export RCUTILS_COLORIZED_OUTPUT=1 ; export RCUTILS_CONSOLE_OUTPUT_FORMAT='[{severity}] [{name}]: {message}'
printenv | grep -E "ROS|AMENT|RMW|IGN|LD_LIBRARY"
apt list --installed 2>/dev/null | grep ros-humble-<name>  ; apt-cache search ros-humble | grep <kw>
sudo apt install ros-humble-<pkg-with-dashes>            # ROS pkg `foo_bar` → `ros-humble-foo-bar`
```

## M. Shell conveniences (`~/.bashrc`)
```bash
alias rs='export PATH=/usr/bin:$PATH; source /opt/ros/humble/setup.bash; source ~/Documents/thesis/ros2_ws/install/setup.bash; export ROS_DOMAIN_ID=42'
alias cb='cd ~/Documents/thesis/ros2_ws && colcon build --symlink-install && source install/setup.bash'
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash    # tab-completion for colcon
eval "$(register-python-argcomplete3 ros2)"                           # tab-completion for ros2 (if not already)
```
Put the aliases in `~/.bashrc` only — don't auto-source ROS there if you rely on the miniforge python in other terminals.

## N. Where things live
| What | Path |
|---|---|
| ROS install (apt) | `/opt/ros/humble/{bin,lib,share}` |
| UR description / meshes / configs | `/opt/ros/humble/share/ur_description/` |
| UR MoveIt config | `/opt/ros/humble/share/ur_moveit_config/` |
| UR driver launch files | `/opt/ros/humble/share/ur_robot_driver/launch/` |
| This workspace | `~/Documents/thesis/ros2_ws/{src,build,install,log}` |
| Scene package | `ros2_ws/src/thesis_scene/{urdf,worlds,config,launch,meshes}` |
| Node logs | `~/.ros/log/` |
