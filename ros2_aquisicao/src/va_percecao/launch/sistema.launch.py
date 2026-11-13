"""
Sistema COMPLETO ao vivo, no carro:
  driver da câmara (usb_cam) -> perceção -> reação  (+ URDF da câmara, + rviz opcional)

  ros2 launch va_percecao sistema.launch.py
  ros2 launch va_percecao sistema.launch.py rviz:=true modelo:=yolo11n.pt imgsz:=640

A configuração do driver é a do contentor de aquisição (config/usb_cam.yaml).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("va_percecao")
    env = lambda k, d: os.environ.get(k, d)
    arg = DeclareLaunchArgument
    camara = Node(
        package="usb_cam", executable="usb_cam_node_exe", name="camera_tras",
        namespace="/atlascar2/camera_tras",
        parameters=[LaunchConfiguration("usb_cam_config")], output="screen")
    args = " ".join(f"{a}:={env(k, d)}" for a, k, d in (
        ("x", "CAM_X", "-0.50"), ("y", "CAM_Y", "0.00"), ("z", "CAM_Z", "1.10"),
        ("roll", "CAM_ROLL", "0.0"), ("pitch", "CAM_PITCH", "0.0"), ("yaw", "CAM_YAW", "3.14159")))
    urdf = Node(   # fragmento URDF da câmara -> /tf_static (base_link -> camera_tras_link)
        package="robot_state_publisher", executable="robot_state_publisher",
        name="urdf_camera_tras", namespace="/atlascar2/camera_tras",
        parameters=[{"robot_description": ParameterValue(
            Command(["xacro ", LaunchConfiguration("urdf"), " " + args]), value_type=str)}],
        output="screen")
    percecao = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "percecao.launch.py")),
        launch_arguments={"modelo": LaunchConfiguration("modelo"),
                          "imgsz": LaunchConfiguration("imgsz"),
                          "rviz": LaunchConfiguration("rviz")}.items())
    return LaunchDescription([
        arg("usb_cam_config", default_value="/ws/config/usb_cam.yaml"),
        arg("urdf", default_value="/ws/urdf/camera_tras_isolada.urdf.xacro"),
        arg("modelo", default_value="yolo11m.pt"),
        arg("imgsz", default_value="1280"),
        arg("rviz", default_value="false"),
        camara, urdf, percecao,
    ])
