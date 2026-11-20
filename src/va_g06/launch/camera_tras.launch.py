"""
Driver da câmara traseira (usb_cam, fluxo de cor) + URDF da câmara (robot_state_publisher).

  ros2 launch va_g06 camera_tras.launch.py

Tópicos publicados:
  /atlascar2/camera_tras/image_raw              (imagem a cores)
  /atlascar2/camera_tras/image_raw/compressed   (imagem comprimida, a que se grava)
  /atlascar2/camera_tras/camera_info            (parâmetros da câmara)
  /tf_static  base_link -> camera_tras_link -> camera_tras_optical_frame

A pose da câmara no carro vem das variáveis CAM_X, CAM_Y, CAM_Z, CAM_ROLL, CAM_PITCH,
CAM_YAW (ficheiro .env da raiz do repositório) e é aplicada ao URDF da câmara.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

from va_g06.urdf_camara import descricao_robo


def generate_launch_description():
    share = get_package_share_directory('va_g06')
    usb_cam = Node(
        package='usb_cam', executable='usb_cam_node_exe', name='camera_tras',
        namespace='/atlascar2/camera_tras',
        parameters=[os.path.join(share, 'config', 'usb_cam.yaml'),
                    {'camera_info_url': 'file://' + os.path.join(share, 'config', 'camera_info.yaml')}],
        output='screen')
    urdf = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='urdf_camera_tras', namespace='/atlascar2/camera_tras',
        parameters=[{'robot_description': descricao_robo(share)}], output='screen')
    return LaunchDescription([usb_cam, urdf])
