"""
Perceção (+ grelha local), visualização e reação, no namespace do grupo (/g06). É o
serviço 'percecao' do compose: a câmara (real, simulada ou um bag) publica noutro sítio.

  ros2 launch va_g06 percecao.launch.py
  ros2 launch va_g06 percecao.launch.py modelo:=yolo11n.pt imgsz:=640 rviz:=true
  ros2 launch va_g06 percecao.launch.py image_topic:=/outro/image_raw comprimida:=false
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
    share = get_package_share_directory("va_g06")
    cfg = os.path.join(share, "config", "params.yaml")
    ns = LaunchConfiguration("namespace")
    arg = DeclareLaunchArgument
    # a imagem que a perceção e a visualização subscrevem (a mesma para as duas)
    imagem = {"image_topic": ParameterValue(LaunchConfiguration("image_topic"), value_type=str),
              "comprimida": ParameterValue(LaunchConfiguration("comprimida"), value_type=bool)}
    return LaunchDescription([
        arg("namespace", default_value="/g06"),
        arg("modelo", default_value="modelos/yolo11m.pt", description="sem GPU: modelos/yolo11n.pt"),
        arg("imgsz", default_value="1280", description="sem GPU: 640"),
        arg("dispositivo", default_value="", description="'' auto | cpu | 0"),
        arg("use_sim_time", default_value="false", description="true ao reproduzir um bag com --clock"),
        arg("rviz", default_value="false"),
        arg("image_topic", default_value="/atlascar2/camera_tras/image_raw/compressed",
            description="tópico da imagem da câmara (sensor_msgs/CompressedImage ou Image)"),
        arg("comprimida", default_value="true", description="false -> sensor_msgs/Image (raw)"),
        Node(package="va_g06", executable="percecao", name="percecao", namespace=ns,
             parameters=[cfg, imagem, {"modelo": ParameterValue(LaunchConfiguration("modelo"), value_type=str),
                               "imgsz": ParameterValue(LaunchConfiguration("imgsz"), value_type=int),
                               "dispositivo": ParameterValue(LaunchConfiguration("dispositivo"), value_type=str),
                               "use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
        Node(package="va_g06", executable="visualizador", name="visualizador", namespace=ns,
             parameters=[cfg, imagem, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
        Node(package="va_g06", executable="reacao", name="reacao", namespace=ns,
             parameters=[cfg, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
             output="screen", emulate_tty=True),
        Node(package="rviz2", executable="rviz2", name="rviz2",
             arguments=["-d", os.path.join(share, "rviz", "percecao.rviz")],
             condition=IfCondition(LaunchConfiguration("rviz")), output="log"),
    ])
