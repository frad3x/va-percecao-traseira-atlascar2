# Launch da câmara traseira: driver UVC (usb_cam) + transformação estática
# base_link -> camera_tras_link. Publica no namespace /atlascar2/camera_tras.
#
# Tópicos publicados:
#   /atlascar2/camera_tras/image_raw              (imagem a cores)
#   /atlascar2/camera_tras/image_raw/compressed   (imagem comprimida, para gravar)
#   /atlascar2/camera_tras/camera_info            (parametros da camara)
# e ainda /tf_static com base_link -> camera_tras_link.
import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = '/ws/config/usb_cam.yaml'
    env = lambda k, d: os.environ.get(k, d)   # lê a montagem do .env

    # Driver da câmara (fluxo de cor, tratado como câmara monocular)
    usb_cam = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='camera_tras',
        namespace='/atlascar2/camera_tras',
        parameters=[cfg],
        output='screen',
    )

    # Pose da câmara no carro (base_link -> camera_tras_link), em /tf_static
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

    return LaunchDescription([usb_cam, tf])
