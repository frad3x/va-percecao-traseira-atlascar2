#!/usr/bin/env bash
# Abre o rviz2 para VERES a imagem da câmara antes/depois de gravar.
# A câmara tem de estar a correr (docker compose up). No rviz: Add -> By topic
# -> /atlascar2/camera_tras/image_raw -> Image.
set -e
cd "$(dirname "$0")/.."

# Autoriza o contentor a abrir janelas no teu ecrã (X11 / XWayland).
xhost +local:root >/dev/null 2>&1 || true
xhost +local: >/dev/null 2>&1 || true

docker compose exec \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e QT_X11_NO_MITSHM=1 \
  camera bash -lc "source /opt/ros/jazzy/setup.bash && rviz2"
