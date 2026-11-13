"""
Perceção + reação (sem o driver da câmara): para quando a câmara já está a
publicar noutro processo/contentor, ou para correr sobre um rosbag.

  ros2 launch va_percecao percecao.launch.py
  ros2 launch va_percecao percecao.launch.py modelo:=yolo11n.pt imgsz:=640 rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("va_percecao")
    cfg = os.path.join(share, "config", "percecao.yaml")
    ns = LaunchConfiguration("namespace")
    arg = DeclareLaunchArgument
    return LaunchDescription([
        arg("namespace", default_value="/atlascar2/camera_tras"),
        arg("modelo", default_value="yolo11m.pt", description="sem GPU: yolo11n.pt"),
        arg("imgsz", default_value="1280", description="sem GPU: 640"),
        arg("dispositivo", default_value="", description="'' auto | cpu | 0"),
        arg("use_sim_time", default_value="false", description="true ao reproduzir um bag com --clock"),
        arg("rviz", default_value="false"),
        Node(package="va_percecao", executable="percecao", name="percecao", namespace=ns,
             parameters=[cfg, {"modelo": ParameterValue(LaunchConfiguration("modelo"), value_type=str),
                               "imgsz": ParameterValue(LaunchConfiguration("imgsz"), value_type=int),
                               "dispositivo": ParameterValue(LaunchConfiguration("dispositivo"), value_type=str),
                               "use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
        Node(package="va_percecao", executable="reacao", name="reacao", namespace=ns,
             parameters=[cfg, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(share, "rviz", "percecao.rviz")],
             condition=IfCondition(LaunchConfiguration("rviz")), output="log"),
    ])
