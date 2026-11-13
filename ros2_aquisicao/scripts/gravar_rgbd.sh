#!/usr/bin/env bash
# Grava um percurso em RGB-D (COR + PROFUNDIDADE + nuvem de pontos) a partir do
# driver Orbbec. A câmara RGB-D tem de estar a correr:
#   docker compose --profile rgbd up
#
# Uso:   ./scripts/gravar_rgbd.sh [nome_do_percurso]
#        Ctrl+C para TERMINAR a gravação.
#
# Grava TODOS os tópicos (-a) para não perder a profundidade (os nomes variam
# conforme o driver). A pasta gravada vai para ./bags (ou DATA_DIR do .env) e é
# a que levas para o MATLAB: a profundidade converte-se em nuvem com
#   python ../python/profundidade_para_nuvem.py --fonte <pasta> --saida_dir ../dados/nuvens
# e depois analisas com percecao3D.m.
set -e
cd "$(dirname "$0")/.."

PERCURSO="${1:-percurso1}"
NOME="g02_camtras_rgbd_$(date +%Y%m%d_%H%M%S)_${PERCURSO}"

echo "=============================================================="
echo " A gravar RGB-D:   ${NOME}"
echo " (cor + profundidade + nuvem + tf — todos os tópicos)"
echo " Carrega Ctrl+C quando quiseres PARAR a gravação."
echo "=============================================================="

docker compose exec camera_rgbd bash -lc "
  source /opt/ros/jazzy/setup.bash &&
  source /ws/orbbec_ws/install/setup.bash &&
  cd /data &&
  ros2 bag record -s mcap -o '${NOME}' -a
"

echo
echo "Gravação RGB-D terminada. Pasta '${NOME}' em ./bags/ (ou DATA_DIR)."
