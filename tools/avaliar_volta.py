#!/usr/bin/env python3
"""
Métricas de deteção na volta real (veículos, duas rodas, peões).

Escolhe N imagens espalhadas pela volta toda e compara as deteções do sistema (o
pipeline completo, com o rastreio e a estabilização, aquecido com as imagens anteriores)
com uma REFERÊNCIA:
  - por omissão, um modelo maior e independente (YOLO11x a 1280 px, confiança >= 0,5):
    é "concordância com um modelo de referência", NÃO anotação manual;
  - com --gt, uma anotação MANUAL (feita com tools/anotar.py) — essa é a métrica a sério.

Só contam objetos com pelo menos --alt_min píxeis de altura (ao longe nem um humano
os anota com segurança). Emparelhamento por IoU >= 0,5 (percecao/avaliacao.py).

  python tools/avaliar_volta.py data/passeio_20261120_091854/rgb_camara_traseira.mp4
  python tools/avaliar_volta.py <video> --gt resultados/avaliacao_volta/gt_manual.json

Saída em resultados/avaliacao_volta/: imagens/ (as N imagens), gt_referencia.json,
metricas.json e erros.jpg (exemplos de falsos positivos e falhas).
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from percecao import avaliacao  # noqa: E402
from percecao.pipeline import Config, Pipeline  # noqa: E402

GRUPOS = ("veic", "duas_rodas", "peoes")
COCO = {"car": "veic", "truck": "veic", "bus": "veic", "bicycle": "duas_rodas",
        "motorcycle": "duas_rodas", "person": "peoes"}


def referencia(modelo, img, conf, alt_min):
    r = modelo.predict(img, imgsz=1280, conf=conf, verbose=False)[0]
    bb, cl = [], []
    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        g = COCO.get(modelo.names[int(c)])
        if g and y2 - y1 >= alt_min:
            bb.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
            cl.append(g)
    return {"bb": bb, "classe": cl}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--aquecer", type=int, default=10, help="imagens (a 15 fps) antes de cada uma")
    ap.add_argument("--alt_min", type=float, default=20.0)
    ap.add_argument("--ref", default="modelos/yolo11x.pt")
    ap.add_argument("--ref_conf", type=float, default=0.5)
    ap.add_argument("--gt", default="", help="anotação manual (tools/anotar.py) em vez da referência")
    ap.add_argument("--saida", default="resultados/avaliacao_volta")
    ap.add_argument("--sem_estabilizar", action="store_true", help="caixas cruas do YOLO+ByteTrack")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.saida, "imagens"), exist_ok=True)
    cap = cv2.VideoCapture(a.video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    alvos = np.linspace(a.aquecer * 2 + 1, total - 1, a.n).astype(int)

    pred, imgs = [], []
    for k, f in enumerate(alvos):
        pipe = Pipeline(Config(modelo="modelos/yolo11m.pt", imgsz=1280,
                               estabilizar=not a.sem_estabilizar), log=lambda *x: None)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f) - 2 * a.aquecer)
        for j in range(a.aquecer + 1):        # 1 em cada 2 (15 fps, como no carro)
            ok, fr = cap.read()
            if j < a.aquecer:
                cap.read()
            res = pipe.processar(fr, j / 15.0)
        nome = f"{k:03d}_f{f}.jpg"
        if not os.path.isfile(os.path.join(a.saida, "imagens", nome)):
            cv2.imwrite(os.path.join(a.saida, "imagens", nome), fr)
        imgs.append(nome)
        pred.append({g: {"bb": [b for b in res[g]["bb"] if b[3] >= a.alt_min]} for g in GRUPOS})
        print(f"[avaliar] {k + 1}/{len(alvos)}", end="\r", flush=True)
    print()

    if a.gt:
        gt = json.load(open(a.gt, encoding="utf-8"))
        gt = [gt.get(n, {"bb": [], "classe": []}) if isinstance(gt, dict) else gt[i]
              for i, n in enumerate(imgs)]
        fonte = f"anotação manual ({a.gt})"
    else:
        from ultralytics import YOLO
        ref = YOLO(a.ref)
        gt = [referencia(ref, cv2.imread(os.path.join(a.saida, "imagens", n)), a.ref_conf, a.alt_min)
              for n in imgs]
        json.dump(dict(zip(imgs, gt)), open(os.path.join(a.saida, "gt_referencia.json"), "w"),
                  indent=1)
        fonte = f"referência {os.path.basename(a.ref)} (conf >= {a.ref_conf})"

    print(f"Referência: {fonte}; {len(imgs)} imagens; objetos com >= {a.alt_min:.0f} px de altura")
    metr = avaliacao.avaliar(pred, gt)
    metr = {g: v for g, v in metr.items() if g in GRUPOS or g in ("iou", "n_frames")}
    metr["referencia"] = fonte
    metr["alt_min_px"] = a.alt_min
    sufixo = "_sem_estabilizar" if a.sem_estabilizar else ""
    json.dump(metr, open(os.path.join(a.saida, f"metricas{sufixo}.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    folha_erros(a.saida, imgs, pred, gt)


def folha_erros(saida, imgs, pred, gt, n=12):
    """Imagens com mais discordâncias: verde = acerto, vermelho = falso positivo do sistema,
    amarelo = objeto da referência que o sistema não deu."""
    from percecao.sinais import iou
    pont = []
    for k, nome in enumerate(imgs):
        P = [(b, g) for g in GRUPOS for b in pred[k][g]["bb"]]
        G = list(zip(gt[k]["bb"], gt[k]["classe"]))
        usados, marcas = set(), []
        for b, g in P:
            j = max(range(len(G)), key=lambda j: iou(b, G[j][0]) if G[j][1] == g else -1, default=None)
            if j is not None and j not in usados and G[j][1] == g and iou(b, G[j][0]) >= 0.5:
                usados.add(j); marcas.append((b, (60, 200, 60)))
            else:
                marcas.append((b, (40, 40, 230)))
        marcas += [(G[j][0], (0, 220, 255)) for j in range(len(G)) if j not in usados]
        erros = sum(1 for _, c in marcas if c != (60, 200, 60))
        pont.append((erros, nome, marcas))
    pont.sort(key=lambda t: -t[0])
    tiles = []
    for erros, nome, marcas in pont[:n]:
        img = cv2.imread(os.path.join(saida, "imagens", nome))
        for (x, y, w, h), c in marcas:
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
        cv2.putText(img, nome, (6, img.shape[0] - 8), 0, 0.5, (255, 255, 255), 1)
        tiles.append(cv2.resize(img, (480, 360)))
    while len(tiles) % 3:
        tiles.append(np.zeros((360, 480, 3), np.uint8))
    cv2.imwrite(os.path.join(saida, "erros.jpg"),
                np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))


if __name__ == "__main__":
    main()
