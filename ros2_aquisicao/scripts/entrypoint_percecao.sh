#!/bin/bash
# Prepara o ambiente ROS 2 (base + pacotes do grupo) e corre o comando dado.
set -e
source /opt/ros/jazzy/setup.bash
source /ws/ros/install/setup.bash
exec "$@"
