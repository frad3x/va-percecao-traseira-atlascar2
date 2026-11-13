"""
Pacote de perceção (Tema 2 — câmara traseira do ATLASCAR2), versão Python.

Núcleo partilhado pela aplicação off-line (../main.py, python -m percecao) e pelo
nó ROS 2 (ros2_aquisicao/src/va_percecao). Módulos:
  pipeline      — processa UMA imagem de ponta a ponta (classe Pipeline)
  app           — a aplicação off-line (fonte -> player, CSV, gráficos)
  seguranca     — distância, faixa e avisos de segurança
  fontes        — abrir vídeo / pasta de imagens / rosbag MCAP (com acesso aleatório)
  detecao       — YOLO (Ultralytics) com rastreio (IDs) -> veículos/peões/sinais
  linhas        — linhas da faixa (clássica, sem calibração, com suavização temporal)
  sinais        — deteção clássica de sinais por cor (HSV) + forma (OpenCV)
  aproximacao   — tempo-para-colisão (TTC) dos veículos que seguem atrás
  desenho       — overlay das deteções sobre a imagem
  player        — player interativo (play/pausa, recuar, velocidade, seek)
  exportar      — CSV das deteções
  avaliacao     — métricas (precisão/recall/F1) vs ground truth
  percecao3d    — perceção 3D (Open3D): chão (RANSAC) + obstáculos + estacionamento
  profundidade  — profundidade do bag -> nuvem de pontos
"""
__all__ = [
    "fontes", "detecao", "linhas", "sinais", "aproximacao", "desenho",
    "player", "exportar", "avaliacao", "percecao3d", "profundidade",
]
