"""
desenho.py — overlay das deteções sobre a imagem (OpenCV).

Linhas da faixa (como na aula 5 / insertLaneBoundary):
  esquerda a VERMELHO, direita a VERDE; linha tracejada desenhada a tracejado;
  o tipo ("continua"/"tracejada") escrito junto a cada linha; faixa do carro
  preenchida; outras linhas fortes a branco fino.
  Dashboard (aula 5, ex. 4b): triângulo AMARELO se só há 1 linha, VERMELHO se 0.
  Mini VISTA DE CIMA no canto superior direito, com as linhas detetadas.
Veículos: verde (vermelho se TTC em risco / score baixo), com TTC.
Peões: amarelo. Sinais: magenta com o nome.
"""
import unicodedata

import cv2
import numpy as np

VERDE = (0, 200, 0)
VERMELHO = (0, 0, 255)
AMARELO = (0, 255, 255)
MAGENTA = (255, 0, 255)
BRANCO = (255, 255, 255)

LINHA_ESQ = (40, 40, 235)      # vermelho (BGR)
LINHA_DIR = (40, 200, 40)      # verde
VERDE_FAIXA = (80, 220, 80)    # preenchimento da faixa


def desenhar(frame, res, ttc_risco=4.0, lim_veic=0.5, vista_cima=True):
    out = frame.copy()
    # --- linhas da faixa (por baixo das caixas) ---
    _linhas(out, res.get("linhas"))
    # --- veículos (com TTC) ---
    v = res.get("veic", {})
    ttc = v.get("ttc", [None] * len(v.get("bb", [])))
    for i, bb in enumerate(v.get("bb", [])):
        sc = v["sc"][i]
        etiq = f"{sc:.2f}"
        cor = VERDE if sc >= lim_veic else VERMELHO
        t = ttc[i] if i < len(ttc) else None
        if t is not None and t != float("inf"):
            etiq += f" | TTC {t:.1f}s"
            if t < ttc_risco:
                cor = VERMELHO
        _caixa(out, bb, cor, etiq)
    # --- peões ---
    for i, bb in enumerate(res.get("peoes", {}).get("bb", [])):
        _caixa(out, bb, AMARELO, f"{res['peoes']['sc'][i]:.2f}")
    # --- sinais ---
    s = res.get("sinais", {})
    for i, bb in enumerate(s.get("bb", [])):
        rot = s["tipo"][i] if i < len(s.get("tipo", [])) else "sinal"
        _caixa(out, bb, MAGENTA, str(rot))
    # --- dashboard e vista de cima (por cima de tudo) ---
    lin = res.get("linhas")
    if lin:
        _alerta_linhas(out, lin.get("n", 0))
        if vista_cima and lin.get("chave"):
            _vista_de_cima(out, frame, lin)
    return out


# ------------------------------- linhas -------------------------------
def _linhas(img, lin):
    if not lin:
        return
    esq, dir_ = lin.get("esq"), lin.get("dir")
    W = img.shape[1]
    esp = max(2, W // 300)
    # faixa do carro (segue as curvas)
    if esq and dir_:
        poli = np.array(esq["pts"] + dir_["pts"][::-1], np.int32)
        camada = img.copy()
        cv2.fillPoly(camada, [poli], VERDE_FAIXA)
        cv2.addWeighted(camada, 0.25, img, 0.75, 0, dst=img)
    # outras linhas fortes (não são da faixa do carro)
    for o in lin.get("outras", []):
        _polilinha(img, o["pts"], BRANCO, max(1, esp // 2), o.get("tipo") == "tracejada")
    # linhas da faixa do carro + tipo escrito junto
    for seg, cor in ((esq, LINHA_ESQ), (dir_, LINHA_DIR)):
        if seg:
            _polilinha(img, seg["pts"], cor, esp + 1, seg.get("tipo") == "tracejada")
            u, v = seg["pts"][-1]                      # ponto mais perto do carro
            _rotulo(img, seg.get("tipo", ""), (int(u), int(v) - 12), cor)


def _polilinha(img, pts, cor, esp, tracejada=False):
    p = np.asarray(pts, np.float32)
    if len(p) < 2:
        return
    if not tracejada:
        cv2.polylines(img, [p.astype(np.int32)], False, cor, esp, cv2.LINE_AA)
        return
    W = img.shape[1]
    traco, falha = max(10, W // 45), max(7, W // 70)
    d = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    a = 0.0
    while a < s[-1]:
        b = min(a + traco, s[-1])
        t = np.linspace(a, b, 6)
        seg = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
        cv2.polylines(img, [seg.astype(np.int32)], False, cor, esp, cv2.LINE_AA)
        a = b + falha


def _alerta_linhas(img, n):
    """Aula 5, ex. 4b: triângulo amarelo (só 1 linha) ou vermelho (nenhuma)."""
    if n >= 2:
        return
    H, W = img.shape[:2]
    cor = AMARELO if n == 1 else VERMELHO
    t = max(18, H // 18)
    cx, cy = W // 2, int(0.08 * H) + t
    tri = np.array([[cx, cy - t], [cx - t, cy + int(0.7 * t)], [cx + t, cy + int(0.7 * t)]], np.int32)
    cv2.fillPoly(img, [tri], cor)
    cv2.polylines(img, [tri], True, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "!", (cx - t // 7, cy + int(0.55 * t)), cv2.FONT_HERSHEY_DUPLEX,
                t / 22.0, (0, 0, 0), max(2, t // 10), cv2.LINE_AA)


def _vista_de_cima(img, frame, lin):
    """Mini vista de cima (bird's-eye) com as linhas, no canto superior direito."""
    from .linhas import geometria
    try:
        g, mx, my, _ = geometria(tuple(lin["chave"]))
    except Exception:
        return
    bev = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bh, bw = bev.shape[:2]
    for o in lin.get("outras", []):
        _curva_bev(bev, o, BRANCO, 1)
    for chave_lado, cor in (("esq", LINHA_ESQ), ("dir", LINHA_DIR)):
        if lin.get(chave_lado):
            _curva_bev(bev, lin[chave_lado], cor, 2)
    cv2.line(bev, (bw // 2, bh - 1), (bw // 2, bh - 12), AMARELO, 2)   # eixo da câmara
    H, W = img.shape[:2]
    alt = int(0.36 * H)
    larg = int(alt * bw / bh)
    mini = cv2.resize(bev, (larg, alt), interpolation=cv2.INTER_AREA)
    x0, y0 = W - larg - 10, 10
    img[y0:y0 + alt, x0:x0 + larg] = mini
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + larg, y0 + alt), BRANCO, 1)
    _rotulo(img, "vista de cima", (x0 + 4, y0 + alt - 6), (60, 60, 60))


def _curva_bev(bev, seg, cor, esp):
    bh = bev.shape[0]
    rn = np.linspace(max(seg.get("rn_top", 0.0), 0.0), 1.0, 30)
    col = np.polyval(seg["coef"], rn)
    pts = np.stack([col, rn * (bh - 1)], 1).astype(np.int32)
    cv2.polylines(bev, [pts], False, cor, esp, cv2.LINE_AA)


# ------------------------------- caixas/texto -------------------------------
def _ascii(txt):
    """O putText do OpenCV não desenha acentos: 'cedência' -> 'cedencia'."""
    return unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()


def _rotulo(img, txt, org, cor):
    txt = _ascii(txt)
    if not txt:
        return
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    x, y = max(0, min(org[0] - tw // 2, img.shape[1] - tw - 4)), max(th + 4, org[1])
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), cor, -1)
    cv2.putText(img, txt, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BRANCO, 1, cv2.LINE_AA)


def _caixa(img, bb, cor, etiqueta):
    x, y, w, h = [int(round(v)) for v in bb]
    cv2.rectangle(img, (x, y), (x + w, y + h), cor, 2)
    etiqueta = _ascii(etiqueta)
    if etiqueta:
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        yb = max(y, th + 4)
        cv2.rectangle(img, (x, yb - th - 4), (x + tw + 4, yb), cor, -1)
        cv2.putText(img, etiqueta, (x + 2, yb - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, BRANCO, 1, cv2.LINE_AA)
