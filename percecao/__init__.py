"""
Pacote de perceção da câmara traseira do ATLASCAR2 (Projeto 2).

Núcleo partilhado pela aplicação off-line (../main.py, python -m percecao) e pelo
nó ROS 2 (src/va_g06). Módulos:
  pipeline      — processa UMA imagem de ponta a ponta (classe Pipeline)
  estabilizar   — caixas estáveis no tempo (para o desenho)
  app           — a aplicação off-line (fonte -> player, CSV, gráficos)
  gui           — a aplicação de janela (python -m percecao.gui)
  seguranca     — distância, faixa e avisos de segurança
  fontes        — abrir vídeo / pasta de imagens / rosbag MCAP (com acesso aleatório)
  detecao       — YOLO (Ultralytics) com rastreio (IDs) -> veículos/peões/sinais
  linhas        — linhas da faixa (clássica, sem calibração, com suavização temporal)
  sinais        — deteção clássica de sinais por cor (HSV) + forma (OpenCV)
  sinais_treinados — sinais com os modelos treinados (GTSDB, GTSRB, sinais PT)
  aproximacao   — tempo-para-colisão (TTC) dos veículos que seguem atrás
  desenho       — overlay das deteções sobre a imagem
  player        — player interativo (play/pausa, recuar, velocidade, seek)
  exportar      — CSV das deteções
  avaliacao     — métricas (precisão/recall/F1) vs ground truth
  percecao3d    — perceção 3D (Open3D): chão (RANSAC) + obstáculos + estacionamento
  profundidade  — profundidade do bag -> nuvem de pontos
"""
__all__ = [
    "pipeline", "estabilizar", "app", "gui", "seguranca", "fontes", "detecao", "linhas", "sinais",
    "sinais_treinados", "aproximacao", "desenho", "player", "exportar", "avaliacao",
    "percecao3d", "profundidade",
]
