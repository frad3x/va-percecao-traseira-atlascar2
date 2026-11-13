#!/usr/bin/env bash
# Confirma que a câmara está mesmo a publicar (lista tópicos + taxa da imagem).
# Corre isto NOUTRO terminal, com a câmara já ligada (docker compose up).
set -e
cd "$(dirname "$0")/.."

docker compose exec camera bash -lc "
  source /opt/ros/jazzy/setup.bash &&
  echo '================ Tópicos disponíveis ================' &&
  ros2 topic list &&
  echo &&
  echo '========= Taxa da imagem comprimida (6 s) =========' &&
  echo '(deve mostrar ~30 Hz; Ctrl+C se demorar)' &&
  timeout 6 ros2 topic hz /atlascar2/camera_tras/image_raw/compressed || true &&
  echo &&
  echo '========= Transformações (/tf_static) =========' &&
  timeout 3 ros2 topic echo --once /tf_static || true
"
