#!/usr/bin/env bash
# Grava um percurso em MCAP e ABRE UMA JANELA com o vídeo ao vivo durante a
# gravação (para veres como está a ficar). A câmara tem de estar a correr
# (docker compose up).
#
# Uso:   ./scripts/gravar.sh [nome_do_percurso] [sem-janela]
#        Ctrl+C para TERMINAR a gravação (a janela fecha sozinha).
#        2.º argumento "sem-janela" grava sem abrir a pré-visualização.
set -e
cd "$(dirname "$0")/.."

PERCURSO="${1:-percurso1}"
JANELA="${2:-com-janela}"
NOME="g02_camtras_$(date +%Y%m%d_%H%M%S)_${PERCURSO}"

# Abre a janela de pré-visualização (image_view), em segundo plano, dentro do
# contentor. Fecha-a automaticamente quando a gravação terminar.
abrir_janela() {
  xhost +local:root >/dev/null 2>&1 || true
  xhost +local: >/dev/null 2>&1 || true
  docker compose exec -d \
    -e DISPLAY="${DISPLAY:-:0}" \
    -e QT_X11_NO_MITSHM=1 \
    camera bash -lc "
    source /opt/ros/jazzy/setup.bash &&
    ros2 run image_view image_view --ros-args \
      -r image:=/atlascar2/camera_tras/image_raw
  " || echo "(aviso: não consegui abrir a janela; usa ./scripts/ver.sh noutro terminal)"
}
fechar_janela() {
  docker compose exec camera bash -lc "pkill -f image_view" >/dev/null 2>&1 || true
}

if [ "$JANELA" != "sem-janela" ]; then
  abrir_janela
  trap fechar_janela EXIT     # garante que a janela fecha ao sair (Ctrl+C)
fi

echo "=============================================================="
echo " A gravar:   ${NOME}"
if [ "$JANELA" != "sem-janela" ]; then
  echo " Janela de pré-visualização aberta (vídeo ao vivo)."
fi
echo " Carrega Ctrl+C quando quiseres PARAR a gravação."
echo "=============================================================="

docker compose exec camera bash -lc "
  source /opt/ros/jazzy/setup.bash &&
  cd /data &&
  ros2 bag record -s mcap -o '${NOME}' \
    /atlascar2/camera_tras/image_raw/compressed \
    /atlascar2/camera_tras/camera_info \
    /tf /tf_static
"

echo
echo "Gravação terminada. A pasta '${NOME}' está em ./bags/ (ou no DATA_DIR do .env)."
echo "É essa pasta que levas para o MATLAB (dados/bags/)."
