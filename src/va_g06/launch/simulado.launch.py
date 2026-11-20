"""
Câmara traseira SIMULADA (sem carro nem bag) + URDF da câmara.

  ros2 launch va_g06 simulado.launch.py                       # estrada sintética
  ros2 launch va_g06 simulado.launch.py video:=/data/videos_teste/x.mp4

Publica nos MESMOS tópicos e com o mesmo frame_id do driver real, por isso a
perceção, a reação e o planeador funcionam sem mudar nada.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from va_g06.urdf_camara import descricao_robo   # o mesmo URDF do driver real


def generate_launch_description():
    share = get_package_share_directory('va_g06')
    return LaunchDescription([
        DeclareLaunchArgument('video', default_value='',
                              description="vídeo em /data (vazio = estrada sintética)"),
        DeclareLaunchArgument('fps', default_value='15.0'),
        DeclareLaunchArgument('metade', default_value='',
                              description="'esquerda' = só a metade esquerda (vídeo lado a lado)"),
        Node(package='va_g06', executable='camara_simulada', name='camara_simulada',
             namespace='/atlascar2/camera_tras',
             parameters=[{'video': ParameterValue(LaunchConfiguration('video'), value_type=str),
                          'fps': ParameterValue(LaunchConfiguration('fps'), value_type=float),
                          'metade': ParameterValue(LaunchConfiguration('metade'), value_type=str)}],
             output='screen'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             name='urdf_camera_tras', namespace='/atlascar2/camera_tras',
             parameters=[{'robot_description': descricao_robo(share)}], output='screen'),
    ])
