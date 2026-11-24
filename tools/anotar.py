#!/usr/bin/env python3
"""
Anotação MANUAL (ground truth) das imagens escolhidas por tools/avaliar_volta.py.

Abre cada imagem com as caixas da referência já desenhadas (se existirem), para só ser
preciso CORRIGIR: apagar as erradas, acrescentar as que faltam, mudar a classe.

  python tools/anotar.py                       # resultados/avaliacao_volta/
  python tools/avaliar_volta.py <video> --gt resultados/avaliacao_volta/gt_manual.json

Teclas:  arrastar com o rato = nova caixa da classe atual
         1 veículo · 2 duas rodas · 3 peão   (classe atual; com o rato sobre uma caixa, muda-a)
         clique direito sobre uma caixa = apagar
         N / espaço = seguinte · P = anterior · S = gravar · Q / Esc = gravar e sair
Grava em <pasta>/gt_manual.json (retoma onde ficou).
"""
import argparse
import json
import os

import cv2

CLASSES = {ord("1"): "veic", ord("2"): "duas_rodas", ord("3"): "peoes"}
COR = {"veic": (255, 160, 30), "duas_rodas": (0, 128, 255), "peoes": (0, 230, 255)}
ESC = 2                                     # imagens 640x480 mostradas a 2x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasta", default="resultados/avaliacao_volta")
    a = ap.parse_args()
    pasta_img = os.path.join(a.pasta, "imagens")
    nomes = sorted(os.listdir(pasta_img))
    f_gt = os.path.join(a.pasta, "gt_manual.json")
    f_ref = os.path.join(a.pasta, "gt_referencia.json")
    if os.path.isfile(f_gt):
        gt = json.load(open(f_gt, encoding="utf-8"))
    elif os.path.isfile(f_ref):
        gt = json.load(open(f_ref, encoding="utf-8"))   # ponto de partida: a referência
    else:
        gt = {}
    for n in nomes:
        gt.setdefault(n, {"bb": [], "classe": []})

    est = {"k": 0, "classe": "veic", "ini": None, "rato": (0, 0)}

    def caixa_em(x, y):
        g = gt[nomes[est["k"]]]
        for i, (bx, by, bw, bh) in enumerate(g["bb"]):
            if bx <= x <= bx + bw and by <= y <= by + bh:
                return i
        return None

    def rato(ev, x, y, flags, _):
        x, y = x / ESC, y / ESC
        est["rato"] = (x, y)
        g = gt[nomes[est["k"]]]
        if ev == cv2.EVENT_LBUTTONDOWN:
            est["ini"] = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and est["ini"]:
            x0, y0 = est["ini"]
            est["ini"] = None
            if abs(x - x0) > 3 and abs(y - y0) > 3:
                g["bb"].append([min(x0, x), min(y0, y), abs(x - x0), abs(y - y0)])
                g["classe"].append(est["classe"])
        elif ev == cv2.EVENT_RBUTTONDOWN:
            i = caixa_em(x, y)
            if i is not None:
                del g["bb"][i], g["classe"][i]

    def gravar():
        json.dump(gt, open(f_gt, "w", encoding="utf-8"), indent=1)
        print(f"gravado {f_gt}")

    cv2.namedWindow("anotar")
    cv2.setMouseCallback("anotar", rato)
    while True:
        n = nomes[est["k"]]
        img = cv2.imread(os.path.join(pasta_img, n))
        img = cv2.resize(img, None, fx=ESC, fy=ESC)
        for (x, y, w, h), c in zip(gt[n]["bb"], gt[n]["classe"]):
            cv2.rectangle(img, (int(x * ESC), int(y * ESC)), (int((x + w) * ESC), int((y + h) * ESC)),
                          COR[c], 2)
        if est["ini"]:
            x0, y0 = est["ini"]
            x1, y1 = est["rato"]
            cv2.rectangle(img, (int(x0 * ESC), int(y0 * ESC)), (int(x1 * ESC), int(y1 * ESC)),
                          COR[est["classe"]], 1)
        cv2.putText(img, f"{est['k'] + 1}/{len(nomes)} {n}  classe: {est['classe']}  "
                    "[1/2/3 classe, dir. apaga, N/P, S, Q]", (10, 28), 0, 0.7, (255, 255, 255), 2)
        cv2.imshow("anotar", img)
        t = cv2.waitKey(30) & 0xFF
        if t in CLASSES:
            i = caixa_em(*est["rato"])
            if i is not None:
                gt[n]["classe"][i] = CLASSES[t]
            else:
                est["classe"] = CLASSES[t]
        elif t in (ord("n"), ord(" ")):
            est["k"] = min(est["k"] + 1, len(nomes) - 1)
        elif t == ord("p"):
            est["k"] = max(est["k"] - 1, 0)
        elif t == ord("s"):
            gravar()
        elif t in (ord("q"), 27):
            gravar()
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
