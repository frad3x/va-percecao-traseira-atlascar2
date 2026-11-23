"""robot_description da câmara traseira, partilhado pelos launch (driver real e simulado)."""
import os

POSE_OMISSAO = (('x', 'CAM_X', '-0.50'), ('y', 'CAM_Y', '0.00'), ('z', 'CAM_Z', '1.75'),
                ('roll', 'CAM_ROLL', '0.0'), ('pitch', 'CAM_PITCH', '0.0'),
                ('yaw', 'CAM_YAW', '3.14159'))


def descricao_robo(share):
    """xacro do URDF isolado da câmara com a pose do .env (CAM_X ... CAM_YAW)."""
    from launch.substitutions import Command
    from launch_ros.parameter_descriptions import ParameterValue
    args = ' '.join(f'{a}:={os.environ.get(k, d)}' for a, k, d in POSE_OMISSAO)
    urdf = os.path.join(share, 'urdf', 'camera_tras_isolada.urdf.xacro')
    return ParameterValue(Command(f'xacro {urdf} {args}'), value_type=str)
