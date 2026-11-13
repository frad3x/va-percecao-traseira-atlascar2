#!/usr/bin/env bash
# Abre UMA JANELA com o vídeo da câmara ao vivo (o que está a ser publicado).
# É a forma mais simples e fiável de VERES a câmara. A câmara tem de estar a
# correr (docker compose up). Corre isto NOUTRO terminal.
#
# Uso:   ./scripts/ver.sh          # abre a janela (Ctrl+C ou fechar a janela para sair)
set -e
cd "$(dirname "$0")/.."

# Autoriza o contentor a abrir janelas no teu ecrã (X11 e XWayland).
xhost +local:root  >/dev/null 2>&1 || true
xhost +local:      >/dev/null 2>&1 || true
xhost +SI:localuser:root >/dev/null 2>&1 || true

echo "A abrir a janela do vídeo (DISPLAY=${DISPLAY:-:0})..."
echo "Se NÃO aparecer janela, lê a mensagem de erro aqui em baixo."
echo

# -e DISPLAY/QT_X11_NO_MITSHM garantem que a janela abre mesmo em Wayland.
# Corre em primeiro plano para VERES qualquer erro (ao contrário do -d).
docker compose exec \
  -e DISPLAY="${DISPLAY:-:0}" \
  -e QT_X11_NO_MITSHM=1 \
  camera bash -lc "
    source /opt/ros/jazzy/setup.bash &&
    ros2 run image_view image_view --ros-args \
      -r image:=/atlascar2/camera_tras/image_raw
  "
