"""
Perceção + reação sobre uma GRAVAÇÃO (rosbag MCAP), como se fosse ao vivo:
o bag publica a imagem gravada e os nós reagem a ela.

  ros2 launch va_g06 bag.launch.py bag:=/data/<pasta_do_bag>
  ros2 launch va_g06 bag.launch.py bag:=... rate:=0.5 rviz:=true
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    share = get_package_share_directory("va_g06")
    arg = DeclareLaunchArgument
    play = ExecuteProcess(
        cmd=["ros2", "bag", "play", LaunchConfiguration("bag"), "--clock",
             "--rate", LaunchConfiguration("rate")],
        output="screen")
    percecao = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(share, "launch", "percecao.launch.py")),
        launch_arguments={"modelo": LaunchConfiguration("modelo"),
                          "imgsz": LaunchConfiguration("imgsz"),
                          "rviz": LaunchConfiguration("rviz"),
                          "use_sim_time": "true"}.items())
    return LaunchDescription([
        arg("bag", description="pasta (ou .mcap) do rosbag gravado"),
        arg("rate", default_value="1.0"),
        arg("modelo", default_value="modelos/yolo11m.pt"),
        arg("imgsz", default_value="1280"),
        arg("rviz", default_value="false"),
        percecao, play,
    ])
