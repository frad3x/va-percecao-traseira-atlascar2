"""Figuras da grelha local SEM ROS, a partir de um vídeo (complementa o figura_bag.py).

Corre o mesmo processamento do nó de perceção (pipeline + grelha + A* comum) sobre um
vídeo e grava, para alguns instantes, a imagem anotada ao lado da grelha (livre/
ocupado/desconhecido), da zona proibida, do caminho do A* e do objetivo de mudança de faixa.

    python tools/figura_grelha.py data/<video>.mp4 --instantes 300 900 1500
    python tools/figura_grelha.py data/<video>.mp4 --metade esquerda --saida docs/img/grelha.png
"""
import argparse
import math
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (RAIZ, os.path.join(RAIZ, "src", "va_g06"), os.path.join(RAIZ, "src", "va_planeamento")):
    sys.path.insert(0, p)

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--instantes", type=int, nargs="+", default=[300, 900, 1500])
    ap.add_argument("--metade", default="", help="'esquerda' = só a metade esquerda (vídeo lado a lado)")
    ap.add_argument("--objetivo", type=float, nargs=2, default=[-20.0, 3.5],
                    help="objetivo do A* no base_link (m): faixa da esquerda, 20 m atrás")
    ap.add_argument("--modelo", default="modelos/yolo11m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--saida", default="docs/img/grelha_instantes.png")
    ap.add_argument("--aquecimento", type=int, default=150,
                    help="começa N imagens antes do 1.º instante (o rastreio precisa de algumas)")
    a = ap.parse_args()
    os.chdir(RAIZ)

    from percecao import desenho
    from percecao.pipeline import Config, Pipeline
    from va_g06.grelha import construir
    from va_planeamento.astar import ParametrosPlaneador, planear

    pipe = Pipeline(Config(modelo=a.modelo, imgsz=a.imgsz), log=lambda m: None)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    alvo = set(a.instantes)
    paineis = []
    k = max(0, min(alvo) - a.aquecimento)       # não é preciso processar o vídeo desde o início
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    while len(paineis) < len(alvo):
        ok, fr = cap.read()
        if not ok:
            break
        if a.metade in ("esquerda", "direita"):
            m = fr.shape[1] // 2
            fr = fr[:, :m] if a.metade == "esquerda" else fr[:, m:]
        res = pipe.processar(fr, k / fps)          # todas as imagens: o rastreio precisa delas
        if k in alvo:
            g = construir(res)
            r = planear(g, (0.0, 0.0), tuple(a.objetivo),        # os mesmos de params.yaml
                        ParametrosPlaneador(raio_inflacao=1.0, desconhecido_livre=False,
                                            corredor_y_min=-5.25, corredor_y_max=5.25))
            chega = bool(r.caminho_xy) and math.dist(r.caminho_xy[-1], a.objetivo) < 0.5
            paineis.append((k, desenho.desenhar(fr, res)[:, :, ::-1], g, r, chega))
        k += 1

    fig, eixos = plt.subplots(len(paineis), 2, figsize=(11, 3.6 * len(paineis)),
                              gridspec_kw={"width_ratios": [1.35, 1]}, squeeze=False)
    cores = ListedColormap(["#9e9e9e", "#ffffff", "#000000"])          # -1, 0, 100
    for (k, img, g, r, chega), (e1, e2) in zip(paineis, eixos):
        e1.imshow(img)
        e1.set_title(f"imagem {k} (t = {k / fps:.1f} s)", fontsize=9)
        e1.axis("off")
        val = np.select([g.data == 100, g.data == 0], [2, 1], 0)
        ext = [g.x_min, g.x_max, g.y_min, g.y_max]
        e2.imshow(val, origin="lower", extent=ext, cmap=cores, vmin=0, vmax=2, interpolation="nearest")
        if r.proibido is not None:
            proib = np.ma.masked_where(~r.proibido | (g.data == 100), r.proibido)
            e2.imshow(proib, origin="lower", extent=ext, cmap=ListedColormap(["#e53935"]),
                      alpha=0.45, interpolation="nearest")
        if r.caminho_xy:
            xs, ys = zip(*r.caminho_xy)
            e2.plot(xs, ys, color="#43a047", lw=2)
        e2.plot(*a.objetivo, "x", color="#e53935", ms=9, mew=2)
        e2.plot(0, 0, "s", color="#1e88e5", ms=6)
        e2.set_title(f"grelha local (base_link) — mudança p/ a esquerda: "
                     f"{'possível' if chega else 'insegura'}  [{r.estado}]", fontsize=9)
        e2.set_xlabel("x (m)  <- atrás do carro")
        e2.set_ylabel("y (m)  esquerda ->")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.saida) or ".", exist_ok=True)
    fig.savefig(a.saida, dpi=110)
    print("figura:", a.saida)


if __name__ == "__main__":
    main()
