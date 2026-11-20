"""
Sistema COMPLETO num só comando (fora do compose, p. ex. no devcontainer):
  driver da câmara + URDF -> perceção (+ grelha local) -> reação

  ros2 launch va_g06 sistema.launch.py
  ros2 launch va_g06 sistema.launch.py rviz:=true modelo:=yolo11n.pt imgsz:=640

O planeador comum corre à parte:  ros2 run va_planeamento local_astar --ros-args -r __ns:=/g06 ...
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory("va_g06")
    lanc = lambda f: PythonLaunchDescriptionSource(os.path.join(share, "launch", f))
    return LaunchDescription([
        DeclareLaunchArgument("namespace", default_value="/g06"),
        DeclareLaunchArgument("modelo", default_value="modelos/yolo11m.pt"),
        DeclareLaunchArgument("imgsz", default_value="1280"),
        DeclareLaunchArgument("rviz", default_value="false"),
        IncludeLaunchDescription(lanc("camera_tras.launch.py")),
        IncludeLaunchDescription(lanc("percecao.launch.py"), launch_arguments={
            "namespace": LaunchConfiguration("namespace"),
            "modelo": LaunchConfiguration("modelo"), "imgsz": LaunchConfiguration("imgsz"),
            "rviz": LaunchConfiguration("rviz")}.items()),
    ])
