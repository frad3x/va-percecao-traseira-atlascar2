"""
avaliacao.py — métricas de deteção (precisão/recall/F1 + MAE) vs ground truth.

Porte de avaliarDeteccoes.m. Emparelhamento greedy por IoU, por grupo.

gt: lista {1xN}; gt[k] = {'bb':[[x,y,w,h],...], 'classe':['veic'|'peoes'|'sinais',...]}
(obtém-se do CVAT/Roboflow/labelImg — ver README).
"""
from .sinais import iou

GRUPOS = ("veic", "peoes", "sinais")


def _bb_grupo_pred(res, g):
    return list(res.get(g, {}).get("bb", []))


def _bb_grupo_gt(gtk, g):
    if not gtk or "bb" not in gtk:
        return []
    bbs = gtk["bb"]
    cls = gtk.get("classe")
    if cls is None:
        return list(bbs)
    return [bb for bb, c in zip(bbs, cls) if c == g]


def _emparelhar(P, G, iou_min):
    nP, nG = len(P), len(G)
    if nP == 0:
        return 0, 0, nG
    if nG == 0:
        return 0, nP, 0
    pares = sorted(((iou(P[i], G[j]), i, j) for i in range(nP) for j in range(nG)),
                   reverse=True)
    usP, usG = set(), set()
    tp = 0
    for v, i, j in pares:
        if v < iou_min:
            break
        if i not in usP and j not in usG:
            usP.add(i); usG.add(j); tp += 1
    return tp, nP - tp, nG - tp


def avaliar(resultados, gt, iou_min=0.5):
    n = min(len(resultados), len(gt))
    if n == 0:
        print("Sem frames para avaliar."); return {}
    acc = {g: [0, 0, 0, 0] for g in GRUPOS}   # TP, FP, FN, somaErroContagem
    for k in range(n):
        for g in GRUPOS:
            P = _bb_grupo_pred(resultados[k], g)
            G = _bb_grupo_gt(gt[k], g)
            tp, fp, fn = _emparelhar(P, G, iou_min)
            acc[g][0] += tp; acc[g][1] += fp; acc[g][2] += fn
            acc[g][3] += abs(len(P) - len(G))

    metr = {"iou": iou_min, "n_frames": n}
    print(f"{'grupo':8} {'TP':>5} {'FP':>5} {'FN':>5} {'precis':>8} {'recall':>8} {'F1':>6} {'MAE/f':>7}")
    for g in GRUPOS:
        tp, fp, fn, se = acc[g]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        mae = se / n
        metr[g] = dict(TP=tp, FP=fp, FN=fn, precisao=prec, recall=rec, F1=f1, MAE=mae)
        print(f"{g:8} {tp:5d} {fp:5d} {fn:5d} {prec:8.3f} {rec:8.3f} {f1:6.3f} {mae:7.2f}")
    return metr
