"""
Aplicação de processamento off-line, pela linha de comandos.

  python -m percecao <fonte> [opções]        (ou 'va-percecao' depois de 'pip install .')

<fonte> = vídeo (.mp4/.avi), pasta de imagens ou rosbag MCAP (pasta ou .mcap).
Exemplos:
  python -m percecao dados/bags/g02_camtras_20261104_160818_percurso1
  python -m percecao video.mp4 --modelo modelos/yolo11n.pt --imgsz 640 --max-frames 300
  python -m percecao video.mp4 --sem-player --gravar-video --vel-kmh 50
"""
import argparse
import json
import os

from .app import executar
from .pipeline import Config

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # pasta do projeto
FILTRO = "Gravações (*.mcap *.mp4 *.avi *.mov *.mkv);;Todos os ficheiros (*)"


def escolher_fonte():
    """Janela para escolher a gravação (ficheiro .mcap/vídeo). Para um rosbag, escolhe o
    ficheiro .mcap dentro da pasta. Cancelar na 1.ª janela abre a escolha de PASTA
    (pasta de imagens ou pasta do bag)."""
    inicio = os.path.join(RAIZ, "dados", "bags")
    inicio = inicio if os.path.isdir(inicio) else RAIZ
    try:
        from PySide6 import QtWidgets
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
        f, _ = QtWidgets.QFileDialog.getOpenFileName(None, "Escolhe a gravação a analisar",
                                                     inicio, FILTRO)
        if not f:
            f = QtWidgets.QFileDialog.getExistingDirectory(
                None, "…ou escolhe uma PASTA (bag ou imagens)", inicio)
        return f or None
    except ImportError:
        import tkinter as tk
        from tkinter import filedialog
        tk.Tk().withdraw()
        f = filedialog.askopenfilename(title="Escolhe a gravação a analisar", initialdir=inicio,
                                       filetypes=[("Gravações", "*.mcap *.mp4 *.avi *.mov *.mkv"),
                                                  ("Todos", "*.*")])
        return f or filedialog.askdirectory(title="…ou escolhe uma PASTA", initialdir=inicio) or None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="va-percecao", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fonte", nargs="?", default=None,
                    help="vídeo, pasta de imagens ou rosbag MCAP "
                         "(se faltar, abre uma janela para escolher)")
    ap.add_argument("--modelo", default="modelos/yolo11m.pt",
                    help="pesos YOLO (sem GPU: modelos/yolo11n.pt)")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--dispositivo", default=None, help="'cpu', '0' (GPU 0)... omissão: automático")
    ap.add_argument("--sinais-det", default="modelos/sinais_det.pt")
    ap.add_argument("--sinais-cls", default="modelos/sinais_cls.pt")
    ap.add_argument("--sem-linhas", action="store_true")
    ap.add_argument("--linhas-params", default="{}", help='JSON, ex. \'{"v_base": 0.82}\'')
    ap.add_argument("--limiar-risco", type=float, default=4.0, help="TTC de risco (s)")
    ap.add_argument("--vel-kmh", type=float, default=None,
                    help="velocidade do carro -> distância de segurança pela regra dos 2 s")
    ap.add_argument("--frontal", action="store_true",
                    help="vídeo de câmara FRONTAL (os lados não estão espelhados)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--sem-player", action="store_true")
    ap.add_argument("--sem-graficos", action="store_true")
    ap.add_argument("--gravar-video", action="store_true")
    ap.add_argument("--video-saida", default=None,
                    help="omissão: resultados/<gravação>_<data>/video_anotado.mp4")
    ap.add_argument("--estacionamento", action="store_true", help="guias de estacionamento")
    ap.add_argument("--gt", default="", help="ground truth .json -> métricas")
    a = ap.parse_args(argv)

    fonte = a.fonte or escolher_fonte()
    if not fonte:
        print("Nenhuma gravação escolhida — nada a fazer.")
        return
    fonte = os.path.abspath(fonte)
    os.chdir(RAIZ)        # modelos/, dados/ e os CSV ficam sempre na pasta do projeto

    seg = {"traseira": not a.frontal}
    if a.vel_kmh is not None:
        seg["vel_ego_kmh"] = a.vel_kmh
    cfg = Config(modelo=a.modelo, imgsz=a.imgsz, conf=a.conf, dispositivo=a.dispositivo,
                 linhas=not a.sem_linhas, linhas_params=json.loads(a.linhas_params),
                 sinais_det=a.sinais_det, sinais_cls=a.sinais_cls,
                 limiar_risco=a.limiar_risco, seguranca_params=seg)
    executar(fonte, cfg, mostrar=not a.sem_player, graficos=not a.sem_graficos,
             gravar_video=a.gravar_video, fich_video=a.video_saida, max_frames=a.max_frames,
             fich_gt=a.gt, estacionamento=a.estacionamento)


if __name__ == "__main__":
    main()
