# Launch da câmara traseira em RGB-D (COR + PROFUNDIDADE) com o driver Orbbec.
#
# A Astra Pro dá a COR como webcam UVC (é o que o usb_cam capta), mas a
# PROFUNDIDADE só sai pelo driver próprio da Orbbec. Este launch arranca esse
# driver (OrbbecSDK_ROS2) + a transformação estática base_link -> camera_tras_link.
#
# >>> CONFIRMAR com o dispositivo real: o nome do pacote/launch do driver varia
#     conforme a versão. Ajusta pelas variáveis de ambiente (no .env):
#       ORBBEC_PKG    (default 'orbbec_camera')
#       ORBBEC_LAUNCH (default 'astra.launch.py')   # p.ex. 'astra_pro.launch.py'
#     Lista o que tens instalado com:  ros2 pkg list | grep -i orbbec
#     e:  ros2 launch orbbec_camera <TAB>
#
# Tópicos esperados (namespace do driver), a gravar no bag:
#   .../color/image_raw        + .../color/camera_info
#   .../depth/image_raw        + .../depth/camera_info     (profundidade, p/ a nuvem)
#   .../depth/points           (nuvem de pontos, se o driver a publicar)
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    env = lambda k, d: os.environ.get(k, d)
    pkg = env('ORBBEC_PKG', 'orbbec_camera')
    lf  = env('ORBBEC_LAUNCH', 'astra.launch.py')

    # Inclui o launch do driver Orbbec (profundidade + cor + nuvem de pontos).
    # Falha de propósito se o pacote não estiver instalado (constrói a imagem RGB-D).
    driver = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory(pkg), 'launch', lf)),
        launch_arguments={
            'enable_point_cloud': 'true',
            'depth_registration': 'true',
        }.items()
    )

    # Pose da câmara no carro (igual ao launch de cor) — base_link -> camera_tras_link
    tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_camera_tras',
        arguments=[
            '--x', env('CAM_X', '-0.50'),
            '--y', env('CAM_Y', '0.00'),
            '--z', env('CAM_Z', '1.10'),
            '--yaw', env('CAM_YAW', '3.14159'),
            '--pitch', env('CAM_PITCH', '0.0'),
            '--roll', env('CAM_ROLL', '0.0'),
            '--frame-id', 'base_link',
            '--child-frame-id', 'camera_tras_link',
        ],
        output='screen',
    )

    return LaunchDescription([driver, tf])
