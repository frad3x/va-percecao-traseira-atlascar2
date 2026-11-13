#!/usr/bin/env python3
"""
main.py — Perceção com câmara traseira (ATLASCAR2 / Tema 2). Versão SÓ Python.

Basta dar Run (ou: python main.py). Faz, numa só passagem sobre a fonte:
  - deteção + rastreio (YOLO) de veículos / duas rodas / peões / sinais
  - marcações no pavimento: linhas da faixa (+ tipo), passadeiras, linhas de paragem
  - sinais (modelos treinados, 43 tipos; ou clássico cor + forma)
  - tempo-para-colisão (TTC) dos veículos que seguem atrás
  - distância (monocular) a cada objeto e AVISOS: distância de segurança,
    mudança de faixa (faixa ao lado livre? linha contínua?), ângulo morto,
    peão atrás, espaço livre atrás (estacionamento)
  - estatísticas, gráficos e exportação para CSV
  - player interativo no fim (recuar, avançar, velocidade, seek)
Para o apoio ao estacionamento (3D/RGB-D) ver percecao/percecao3d.py.

Edita o bloco CONFIG abaixo e corre. (Também aceita variáveis de ambiente
VA_FONTE, VA_MAX_FRAMES, VA_MOSTRAR, VA_GRAFICOS — úteis para testes rápidos.)
"""
import os

# ========================= CONFIG (edita e dá Run) =========================
# >>> METE AQUI o caminho do que queres analisar (entre as aspas, depois do r):
FONTE = r"C:\Users\tomas\OneDrive\Documentos\Universidade\5º ano\VA\Projeto\dados\bags\vídeos teste\videoplayback.mp4"
#   -> vídeo .avi/.mp4, pasta de imagens, OU pasta/.mcap de um rosbag (converte sozinho)
#   Exemplos (Windows):
#     FONTE = r"C:\Users\tomas\Desktop\o_meu_video.mp4"
#     FONTE = r"dados\bags\g02_camtras_20261104_160818_percurso1"   # pasta do bag
FONTE = os.environ.get("VA_FONTE", FONTE)   # (permite também definir por variável de ambiente)
MODELO = "yolo11m.pt"      # modelo YOLO (descarregado na 1.ª vez). 'm' = bom equilíbrio
                           #   precisão/velocidade com GPU. Sem GPU: "yolo11n.pt" + IMGSZ=640.
IMGSZ = 1280               # resolução de inferência: 1280 apanha objetos PEQUENOS
                           #   (sinais ao longe) muito melhor que 640 — testado: 16x mais STOP
CONF = 0.25                # confiança mínima do YOLO
USAR_TRACK = True          # rastreio com IDs (ByteTrack) — preciso para o TTC
LINHAS = True              # deteção das linhas da faixa (método da aula 5, sem calibração)
# Afinar as linhas para ESTE vídeo (opcional; ver PARAMS em percecao/linhas.py). Ex.:
#   {"v_base": 0.82}     -> a câmara apanha o carro/capô em baixo (ex. Argoverse)
#   {"horizonte": 0.50}  -> fixar o horizonte em vez de o estimar
LINHAS_PARAMS = {}
SINAIS_CLASSICO = True     # YOLO (stop) + detetor clássico (restantes sinais PT)
# Sinais TREINADOS (gerados por treinar_sinais.py). Se existirem, substituem o
# detetor clássico e reconhecem 43 tipos (limite 50, cedência, rotunda, obras...).
SINAIS_DET = "modelos/sinais_det.pt"
SINAIS_CLS = "modelos/sinais_cls.pt"
ANALISAR_APROX = True      # TTC dos veículos atrás
LIMIAR_RISCO = 4.0         # TTC (s) abaixo do qual se assinala RISCO
SEGURANCA = True           # distâncias + avisos (ver percecao/seguranca.py)
SEGURANCA_PARAMS = {}      # ex.: {"vel_ego_kmh": 50} -> regra dos 2 s; {"traseira": False}
                           #      para vídeos de câmara FRONTAL (lados não espelhados);
                           #      {"dist_seguranca": 15}; LINHAS_PARAMS {"altura": 1.0}
                           #      (altura da câmara, em m) acerta a escala das distâncias
ESTACIONAMENTO = False     # guias de distância no chão (também: tecla G no player)
MOSTRAR = os.environ.get("VA_MOSTRAR", "1") != "0"     # abrir o player no fim
GRAFICOS = os.environ.get("VA_GRAFICOS", "1") != "0"   # mostrar gráficos
GRAVAR_VIDEO = os.environ.get("VA_GRAVAR_VIDEO", "0") != "0"  # gravar vídeo anotado (mp4)
FICH_VIDEO = "dados/resultado_percecao.mp4"
MAX_FRAMES = int(os.environ.get("VA_MAX_FRAMES", "0")) # 0 = todas (ex. 50 p/ testar)
FICH_GT = ""               # ground truth .json (opcional) -> avaliação de métricas
# ==========================================================================


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))   # corre a partir do Projeto/
    from percecao.app import executar
    from percecao.pipeline import Config
    executar(FONTE,
             Config(modelo=MODELO, imgsz=IMGSZ, conf=CONF, usar_track=USAR_TRACK,
                    linhas=LINHAS, linhas_params=LINHAS_PARAMS,
                    sinais_det=SINAIS_DET, sinais_cls=SINAIS_CLS,
                    sinais_classico=SINAIS_CLASSICO, ttc=ANALISAR_APROX,
                    limiar_risco=LIMIAR_RISCO, seguranca=SEGURANCA,
                    seguranca_params=SEGURANCA_PARAMS),
             mostrar=MOSTRAR, graficos=GRAFICOS, gravar_video=GRAVAR_VIDEO,
             fich_video=FICH_VIDEO, max_frames=MAX_FRAMES, fich_gt=FICH_GT,
             estacionamento=ESTACIONAMENTO)


if __name__ == "__main__":
    main()
