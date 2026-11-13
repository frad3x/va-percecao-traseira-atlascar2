"""
sinais.py — deteção clássica de sinalética por COR (HSV) + FORMA (OpenCV).

Porte do detetarSinais.m. O YOLO (COCO) só conhece 'stop sign'; este módulo
apanha os restantes sinais PT (perigo, proibição, obrigação, obras) por cor
dominante (vermelho/azul/amarelo) + forma (triângulo/círculo/octógono).

detetar(frame_BGR) -> {'bb':[[x,y,w,h],...], 'sc':[...], 'tipo':[...]}

AFINAÇÃO: os limiares estão todos em PARAMS (fácil de ajustar com os dados reais).
Afinado para POUCOS falsos positivos; numa estrada real, se perderes sinais,
baixa sat_min/val_min e sobe area_rel_max.
"""
import cv2
import numpy as np

PARAMS = dict(
    # cor em HSV do OpenCV: H em [0,179], S e V em [0,255]
    red_h_lo=8, red_h_hi=172,      # vermelho = H perto de 0 ou de 179
    blue_lo=100, blue_hi=126,      # azul
    yel_lo=22, yel_hi=33,          # amarelo
    sat_min=150, val_min=100,      # saturação/brilho mínimos (cortam pastéis/metálicos)
    ar_lo=0.70, ar_hi=1.45,        # razão de aspeto (placa ~quadrada/redonda)
    area_rel_min=6e-4, area_rel_max=0.04,
    sol_min=0.90, ext_min=0.58,    # solidez e preenchimento mínimos
    ecc_max=0.88,                  # excentricidade máxima
    circ_min=0.82,                 # circularidade mínima p/ "circular"
    morph=3,                       # raio do fecho morfológico
)


def detetar(frame, par=None):
    if par is None:
        par = PARAMS
    h, w = frame.shape[:2]
    area_img = h * w
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    sv = (S >= par["sat_min"]) & (V >= par["val_min"])

    m_vermelho = ((H <= par["red_h_lo"]) | (H >= par["red_h_hi"])) & sv
    m_azul = (H >= par["blue_lo"]) & (H <= par["blue_hi"]) & sv
    m_amarelo = (H >= par["yel_lo"]) & (H <= par["yel_hi"]) & sv

    out = {"bb": [], "sc": [], "tipo": []}
    for mask, cor in ((m_vermelho, "vermelho"), (m_azul, "azul"), (m_amarelo, "amarelo")):
        _candidatos(mask.astype(np.uint8) * 255, cor, area_img, par, out)

    # supressão de não-máximos entre candidatos sobrepostos
    _nms_inplace(out, 0.3)
    return out


def _candidatos(mask, cor, area_img, par, out):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (par["morph"] * 2 + 1,) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = _preencher_buracos(mask)

    cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        a_rel = area / area_img
        if a_rel < par["area_rel_min"] or a_rel > par["area_rel_max"]:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ar = bw / max(bh, 1)
        if ar < par["ar_lo"] or ar > par["ar_hi"]:
            continue
        hull = cv2.convexHull(c)
        sol = area / max(cv2.contourArea(hull), 1e-6)       # solidez
        ext = area / max(bw * bh, 1e-6)                      # preenchimento
        if sol < par["sol_min"] or ext < par["ext_min"]:
            continue
        ecc = _excentricidade(c)
        if ecc > par["ecc_max"]:
            continue
        forma = _forma(c, area)
        sinal = "stop" if (cor == "vermelho" and forma == "octogonal") else f"{cor}-{forma}"
        conf = min(1.0, sol * ext * 1.1)
        out["bb"].append([float(x), float(y), float(bw), float(bh)])
        out["sc"].append(float(conf))
        out["tipo"].append(sinal)


def _forma(c, area):
    per = cv2.arcLength(c, True)
    circ = 4 * np.pi * area / max(per * per, 1e-6)
    if circ >= 0.82:
        return "circular"
    aprox = cv2.approxPolyDP(c, 0.03 * per, True)
    nv = len(aprox)
    if nv <= 3:
        return "triangular"
    if nv >= 7:
        return "octogonal"
    return "poligonal"


def _excentricidade(c):
    if len(c) < 5:
        return 0.0
    (_, _), (ma, mi), _ = cv2.fitEllipse(c)
    a, b = max(ma, mi), min(ma, mi)
    if a <= 0:
        return 0.0
    return float(np.sqrt(max(1 - (b / a) ** 2, 0)))


def _preencher_buracos(mask):
    ff = mask.copy()
    hh, ww = mask.shape
    m2 = np.zeros((hh + 2, ww + 2), np.uint8)
    cv2.floodFill(ff, m2, (0, 0), 255)
    return mask | cv2.bitwise_not(ff)


def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    uni = a[2] * a[3] + b[2] * b[3] - inter
    return inter / uni if uni > 0 else 0.0


def _nms_inplace(out, thr):
    n = len(out["bb"])
    if n <= 1:
        return
    ordem = sorted(range(n), key=lambda i: out["sc"][i], reverse=True)
    manter = []
    for i in ordem:
        if all(iou(out["bb"][i], out["bb"][j]) < thr for j in manter):
            manter.append(i)
    for chave in ("bb", "sc", "tipo"):
        out[chave] = [out[chave][i] for i in manter]


def combinar(yolo_sinais, classico, thr=0.4):
    """Junta os sinais do YOLO (stop) com os do clássico, removendo duplicados (NMS)."""
    comb = {"bb": list(yolo_sinais.get("bb", [])) + list(classico["bb"]),
            "sc": list(yolo_sinais.get("sc", [])) + list(classico["sc"]),
            "tipo": list(yolo_sinais.get("tipo", [])) + list(classico["tipo"])}
    _nms_inplace(comb, thr)
    return comb
