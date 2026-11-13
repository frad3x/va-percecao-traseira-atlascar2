# Launch da câmara traseira: driver UVC (usb_cam) + URDF da câmara
# (robot_state_publisher). Publica no namespace /atlascar2/camera_tras.
#
# Tópicos publicados:
#   /atlascar2/camera_tras/image_raw              (imagem a cores)
#   /atlascar2/camera_tras/image_raw/compressed   (imagem comprimida, para gravar)
#   /atlascar2/camera_tras/camera_info            (parametros da camara)
#   /tf_static  base_link -> camera_tras_link -> camera_tras_optical_frame
#   /atlascar2/camera_tras/robot_description (URDF da câmara)
#
# A pose da câmara no carro vem do .env (CAM_X, CAM_Y, CAM_Z, CAM_ROLL, CAM_PITCH,
# CAM_YAW) e é aplicada ao fragmento URDF urdf/camera_tras.urdf.xacro.
import os

from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

URDF = '/ws/urdf/camera_tras_isolada.urdf.xacro'


def descricao_robo():
    env = lambda k, d: os.environ.get(k, d)   # lê a montagem do .env
    args = ' '.join(f'{a}:={env(k, d)}' for a, k, d in (
        ('x', 'CAM_X', '-0.50'), ('y', 'CAM_Y', '0.00'), ('z', 'CAM_Z', '1.10'),
        ('roll', 'CAM_ROLL', '0.0'), ('pitch', 'CAM_PITCH', '0.0'), ('yaw', 'CAM_YAW', '3.14159')))
    return ParameterValue(Command(f'xacro {URDF} {args}'), value_type=str)


def generate_launch_description():
    cfg = '/ws/config/usb_cam.yaml'

    # Driver da câmara (fluxo de cor, tratado como câmara monocular)
    usb_cam = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera_tras',
        namespace='/atlascar2/camera_tras',
        parameters=[cfg],
        output='screen',
    )

    # Pose da câmara no carro (fragmento URDF) -> /tf_static
    urdf = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='urdf_camera_tras',
        namespace='/atlascar2/camera_tras',
        parameters=[{'robot_description': descricao_robo()}],
        output='screen',
    )

    return LaunchDescription([usb_cam, urdf])
