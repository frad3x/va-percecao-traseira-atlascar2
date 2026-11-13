"""
desenho.py — overlay das deteções e dos avisos sobre a imagem (OpenCV).

Camadas (de baixo para cima):
  1) Marcações no pavimento: faixa do carro preenchida; linha esquerda a VERMELHO e
     direita a VERDE (como o insertLaneBoundary da aula 5), tracejadas desenhadas a
     tracejado com o tipo escrito; outras linhas a branco; PASSADEIRAS (ciano) e
     LINHAS DE PARAGEM (vermelho/branco) com a distância.
  2) Guias de ESTACIONAMENTO no chão (opcional): corredor do carro com marcas a
     0.5 / 1 / 2 / 3 m (vermelho -> amarelo -> verde), como nas câmaras de marcha-atrás.
  3) Objetos, cada grupo com a SUA cor (legenda no painel):
       veículos AZUL · duas rodas LARANJA · peões AMARELO · sinais MAGENTA
     etiqueta: tipo, #ID, distância (m) e TTC; o SEGUIDOR (veículo atrás na nossa
     faixa) leva marcador no chão e linha de distância; caixa VERMELHA se há risco.
  4) Painel (HUD) no canto superior esquerdo: linhas e tipo (lado do CARRO: numa
     câmara traseira a esquerda da imagem é a direita do carro), contagens por grupo,
     veículo de trás (distância/TTC/velocidade de aproximação), mudança de faixa
     (esq./dir. DO CARRO), espaço livre atrás. Triângulo da aula 5 (amarelo: 1 linha;
     vermelho: nenhuma).
  5) Faixa de AVISO no topo (risco de colisão, peão atrás, ângulo morto...) e moldura
     vermelha em caso de perigo.
  6) Mini VISTA DE CIMA (canto superior direito) com as linhas E os objetos.
  7) Tira com os últimos SINAIS reconhecidos (recorte + nome), no canto inferior esquerdo.
"""
import os
import unicodedata
from collections import OrderedDict
from functools import lru_cache

import cv2
import numpy as np

# ---------------------------- cores (BGR) ----------------------------
BRANCO = (255, 255, 255)
PRETO = (0, 0, 0)
VERMELHO = (50, 50, 240)
AMARELO = (0, 220, 255)
AMBAR = (0, 170, 255)
VERDE = (90, 200, 60)
CIANO = (230, 210, 0)

COR_GRUPO = {"veic": (255, 160, 30),        # azul
             "duas_rodas": (0, 128, 255),   # laranja
             "peoes": (0, 230, 255),        # amarelo
             "sinais": (220, 60, 220)}      # magenta
NOME_GRUPO = {"veic": "veiculos", "duas_rodas": "2 rodas", "peoes": "peoes", "sinais": "sinais"}
COR_ESTADO = {"perigo": VERMELHO, "atencao": AMBAR, "ok": VERDE, "livre": (170, 170, 170)}

LINHA_ESQ = (40, 40, 235)      # vermelho
LINHA_DIR = (40, 200, 40)      # verde
VERDE_FAIXA = (80, 220, 80)    # preenchimento da faixa do carro

NOME_PT = {"car": "carro", "truck": "camiao", "bus": "autocarro", "bicycle": "bicicleta",
           "motorcycle": "mota", "person": "peao", "stop sign": "STOP"}

FONTE = cv2.FONT_HERSHEY_SIMPLEX


def desenhar(frame, res, ttc_risco=4.0, lim_veic=0.5, vista_cima=True, estacionamento=False):
    out = frame.copy()
    H, W = out.shape[:2]
    e = _escala(H)
    seg = res.get("seg") or {}
    lin = res.get("linhas")

    _linhas(out, lin, e)
    _marcacoes(out, (lin or {}).get("marcacoes", []), e)
    if estacionamento and seg.get("chave"):
        _guias_estacionamento(out, seg, e)
    _objetos(out, res, seg, ttc_risco, e)
    _painel(out, res, seg, ttc_risco, e)
    _avisos(out, seg.get("alertas", []), e)
    if vista_cima and lin and lin.get("chave"):
        _vista_de_cima(out, frame, lin, res, e)
    _tira_sinais(out, res.get("sinais_vistos", []), e)
    return out


def _escala(H):
    return max(0.6, H / 720.0)


# ================================ texto ================================
def _ascii(txt):
    """O putText do OpenCV não desenha acentos: 'cedência' -> 'cedencia'."""
    return unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()


def _texto(img, txt, org, cor=BRANCO, fs=0.5, esp=1, contorno=True):
    txt = _ascii(txt)
    if contorno:
        cv2.putText(img, txt, org, FONTE, fs, PRETO, esp + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONTE, fs, cor, esp, cv2.LINE_AA)
    return cv2.getTextSize(txt, FONTE, fs, esp)[0][0]


def _rotulo(img, txt, org, cor, fs=0.5, centrado=True):
    """Texto branco sobre um retângulo da cor dada."""
    txt = _ascii(txt)
    if not txt:
        return
    esp = max(1, int(round(fs * 2)))
    (tw, th), base = cv2.getTextSize(txt, FONTE, fs, esp)
    x = org[0] - tw // 2 if centrado else org[0]
    x = max(0, min(x, img.shape[1] - tw - 6))
    y = max(th + 6, min(org[1], img.shape[0] - 2))
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), cor, -1)
    cv2.putText(img, txt, (x + 3, y - 3), FONTE, fs, BRANCO, esp, cv2.LINE_AA)


def _caixa_translucida(img, x0, y0, x1, y1, cor=(20, 20, 20), alfa=0.62):
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alfa) + np.array(cor, np.float32) * alfa).astype(np.uint8)


# ================================ linhas ================================
def _linhas(img, lin, e):
    if not lin:
        return
    esq, dir_ = lin.get("esq"), lin.get("dir")
    esp = max(2, int(round(3 * e)))
    if esq and dir_:                                  # faixa do carro (segue as curvas)
        poli = np.array(esq["pts"] + dir_["pts"][::-1], np.int32)
        camada = img.copy()
        cv2.fillPoly(camada, [poli], VERDE_FAIXA)
        cv2.addWeighted(camada, 0.22, img, 0.78, 0, dst=img)
    for o in lin.get("outras", []):
        _polilinha(img, o["pts"], BRANCO, max(1, esp // 2), o.get("tipo") == "tracejada")
    for seg, cor in ((esq, LINHA_ESQ), (dir_, LINHA_DIR)):
        if seg:
            _polilinha(img, seg["pts"], cor, esp + 1, seg.get("tipo") == "tracejada")
            u, v = seg["pts"][-1]                      # ponto mais perto do carro
            _rotulo(img, seg.get("tipo", ""), (int(u), int(v) - int(14 * e)), cor, 0.45 * e)


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


def _marcacoes(img, marcs, e):
    for m in marcs:
        poli = np.array(m["pts"], np.int32)
        camada = img.copy()
        if m["tipo"] == "passadeira":
            cv2.fillPoly(camada, [poli], CIANO)
            cv2.addWeighted(camada, 0.30, img, 0.70, 0, dst=img)
            cv2.polylines(img, [poli], True, CIANO, max(2, int(2 * e)), cv2.LINE_AA)
            nome, cor = "passadeira", (160, 140, 0)
        else:
            cv2.fillPoly(camada, [poli], VERMELHO)
            cv2.addWeighted(camada, 0.35, img, 0.65, 0, dst=img)
            cv2.polylines(img, [poli], True, BRANCO, max(2, int(2 * e)), cv2.LINE_AA)
            nome, cor = "linha de paragem", VERMELHO
        cx, cy = poli.mean(0).astype(int)
        _rotulo(img, f"{nome} {m.get('dist', 0):.0f} m", (int(cx), int(cy)), cor, 0.45 * e)


# =========================== guias de estacionamento ===========================
def _guias_estacionamento(img, seg, e):
    from .linhas import geometria, veic_para_img
    g = geometria(tuple(seg["chave"]))[0]
    H = img.shape[0]
    meia = 0.9                                        # metade da largura do carro (m)
    troços = [(0.3, 1.0, VERMELHO), (1.0, 2.0, AMARELO), (2.0, 3.0, VERDE)]
    esp = max(2, int(3 * e))
    for x0, x1, cor in troços:
        for lado in (meia, -meia):
            u0, v0 = veic_para_img(x0, lado, g)
            u1, v1 = veic_para_img(x1, lado, g)
            if min(v0, v1) < H and max(v0, v1) > g["y_h"]:
                cv2.line(img, (int(u0), int(v0)), (int(u1), int(v1)), cor, esp, cv2.LINE_AA)
        u0, v0 = veic_para_img(x1, meia, g)
        u1, v1 = veic_para_img(x1, -meia, g)
        if v0 < H:
            cv2.line(img, (int(u0), int(v0)), (int(u1), int(v1)), cor, max(1, esp - 1), cv2.LINE_AA)
            _texto(img, f"{x1:.0f} m", (int(u1) + 6, int(v1) + 4), cor, 0.45 * e)


# ================================ objetos ================================
def _objetos(img, res, seg, ttc_risco, e):
    fs = 0.45 * e
    seguidor = seg.get("seguidor") or {}
    for grupo in ("sinais", "peoes", "duas_rodas", "veic"):
        gr = res.get(grupo) or {}
        bbs = gr.get("bb", [])
        for i, bb in enumerate(bbs):
            cor = COR_GRUPO[grupo]
            tipo = _get(gr, "tipo", i, "")
            nome = NOME_PT.get(tipo, tipo)
            partes = [nome or NOME_GRUPO[grupo]]
            oid = _get(gr, "id", i, -1)
            if oid is not None and oid >= 0:
                partes[0] += f" #{oid}"
            d = _get(gr, "dist", i)
            if d is not None:
                partes.append(("<" if _get(gr, "cortada", i, False) else "") + f"{d:.0f} m")
            ttc = _get(gr, "ttc", i)
            risco = ttc is not None and np.isfinite(ttc) and ttc < ttc_risco
            if ttc is not None and np.isfinite(ttc):
                partes.append(f"TTC {ttc:.1f}s")
            if grupo == "sinais" and _get(gr, "sc", i) is not None:
                partes.append(f"{gr['sc'][i]:.2f}")
            e_seguidor = seguidor.get("grupo") == grupo and seguidor.get("i") == i
            if e_seguidor:
                _marcador_seguidor(img, bb, COR_ESTADO.get(seg.get("estado_dist"), cor), e)
            if risco:
                cor = VERMELHO
            _caixa(img, bb, cor, "  ".join(partes), fs, grossa=risco or e_seguidor)


def _get(gr, chave, i, padrao=None):
    lst = gr.get(chave)
    return lst[i] if lst is not None and i < len(lst) else padrao


def _caixa(img, bb, cor, etiqueta, fs, grossa=False):
    x, y, w, h = [int(round(v)) for v in bb]
    esp = 3 if grossa else 2
    cv2.rectangle(img, (x, y), (x + w, y + h), cor, 1, cv2.LINE_AA)
    c = max(6, min(w, h) // 4)                       # cantos reforçados
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(img, (px, py), (px + dx * c, py), cor, esp + 1, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * c), cor, esp + 1, cv2.LINE_AA)
    _rotulo(img, etiqueta, (x, y - 2), cor, fs, centrado=False)


def _marcador_seguidor(img, bb, cor, e):
    """Elipse no chão por baixo do veículo de trás + linha até ao nosso carro."""
    x, y, w, h = bb
    H, W = img.shape[:2]
    base = (int(x + w / 2), int(min(y + h, H - 2)))
    camada = img.copy()
    cv2.ellipse(camada, base, (int(w * 0.5), max(4, int(min(w, h) * 0.1))), 0, 0, 360, cor, -1,
                cv2.LINE_AA)
    cv2.addWeighted(camada, 0.35, img, 0.65, 0, dst=img)
    cv2.line(img, (W // 2, H - 1), base, cor, max(1, int(2 * e)), cv2.LINE_AA)


# ================================ painel (HUD) ================================
# Desenhado com o PIL (letra TrueType suavizada, cantos redondos) e depois misturado na
# imagem — o putText do OpenCV só tem letra "de pixel" e sem acentos.
UA_VERDE = (146, 212, 0)                 # cores em RGB (PIL)
P_AMBAR, P_VERMELHO, P_CINZA = (243, 178, 27), (240, 89, 42), (140, 140, 140)
P_BRANCO, P_TEXTO2 = (245, 245, 245), (165, 165, 165)
P_GRUPO = {"veic": (30, 160, 255), "duas_rodas": (255, 128, 0), "peoes": (255, 230, 0),
           "sinais": (220, 60, 220)}
P_ESTADO = {"ok": ("SEGURO", UA_VERDE), "atencao": ("PERTO", P_AMBAR),
            "perigo": ("PERIGO", P_VERMELHO), "livre": ("LIVRE", P_CINZA)}
P_MUDAR = {"livre": UA_VERDE, "proibido": P_AMBAR, "ocupado": P_VERMELHO}


@lru_cache(maxsize=32)
def _fonte(tam, negrito=False):
    from PIL import ImageFont
    nomes = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if negrito else
             ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
    for pasta in ("C:/Windows/Fonts", "/usr/share/fonts/truetype/dejavu", ""):
        for n in nomes:
            try:
                return ImageFont.truetype(os.path.join(pasta, n) if pasta else n, tam)
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=tam)
    except TypeError:
        return ImageFont.load_default()


def _pill(d, x, y, txt, cor, f, alt, cheia=False):
    """Pastilha arredondada: fundo da cor (translúcido) e texto na cor. Devolve a largura."""
    w = int(d.textlength(txt, font=f)) + alt
    fundo = cor + ((255,) if cheia else (60,))
    d.rounded_rectangle([x, y, x + w, y + alt], radius=alt // 2, fill=fundo,
                        outline=None if cheia else cor + (150,), width=1)
    d.text((x + w / 2, y + alt / 2), txt, font=f, fill=(15, 15, 15) if cheia else cor,
           anchor="mm")
    return w


def _triangulo_pil(d, cx, cy, t, cor):
    """Triângulo de aviso (aula 5, ex. 4b): amarelo = 1 linha, vermelho = nenhuma."""
    d.polygon([(cx, cy - t), (cx - t, cy + 0.8 * t), (cx + t, cy + 0.8 * t)], fill=cor)
    d.text((cx, cy + 0.15 * t), "!", font=_fonte(max(8, int(1.2 * t)), True),
           fill=(15, 15, 15), anchor="mm")


def _painel(img, res, seg, ttc_risco, e):
    from PIL import Image, ImageDraw
    H, W = img.shape[:2]
    k = max(e, 0.72)                              # escala (720p = 1); mínimo p/ ler em 360p
    pad, pw = int(14 * k), int(340 * k)
    f_tit, f_lab = _fonte(int(11 * k), True), _fonte(int(12 * k))
    f_val, f_big = _fonte(int(13 * k), True), _fonte(int(24 * k), True)
    f_pill = _fonte(int(11 * k), True)
    alt_p = int(20 * k)

    tela = Image.new("RGBA", (pw, int(320 * k)), (0, 0, 0, 0))
    d = ImageDraw.Draw(tela)
    x, y = pad + int(4 * k), pad

    # --- cabeçalho + estado das linhas (triângulo da aula 5) ---
    d.text((x, y), "RETAGUARDA", font=f_tit, fill=UA_VERDE)
    lin = res.get("linhas") or {}
    if lin:
        n = lin.get("n", 0)
        if n < 2:
            txt, cor = ("1 linha", P_AMBAR) if n == 1 else ("sem linhas", P_VERMELHO)
            wl = int(d.textlength(txt, font=f_pill)) + alt_p
            xr = pw - pad - wl
            _pill(d, xr, y - int(3 * k), txt, cor, f_pill, alt_p)
            _triangulo_pil(d, xr - int(14 * k), y + int(7 * k), int(8 * k), cor)
    y += int(24 * k)

    # --- linhas da faixa (lado do carro) ---
    if lin:
        tipos = seg.get("tipos_linha") or {
            "esquerda": (lin.get("esq") or {}).get("tipo"), "direita": (lin.get("dir") or {}).get("tipo")}
        d.text((x, y + alt_p / 2), "Linhas", font=f_lab, fill=P_TEXTO2, anchor="lm")
        xx = x + int(78 * k)
        for lado, ab in (("esquerda", "Esq"), ("direita", "Dir")):
            t = tipos.get(lado)
            if t:
                xx += _pill(d, xx, y, f"{ab} · {'contínua' if t == 'continua' else t}", P_BRANCO,
                            f_pill, alt_p) + int(6 * k)
            else:
                xx += _pill(d, xx, y, f"{ab} · —", P_CINZA, f_pill, alt_p) + int(6 * k)
        y += alt_p + int(10 * k)

    # --- contagens por grupo ---
    xx = x
    for g, nome in (("veic", "veíc."), ("duas_rodas", "2 rodas"), ("peoes", "peões"), ("sinais", "sinais")):
        nn = len((res.get(g) or {}).get("bb", []))
        cor = P_GRUPO[g] if nn else P_CINZA
        r = int(4 * k)
        d.ellipse([xx, y + int(8 * k) - r, xx + 2 * r, y + int(8 * k) + r], fill=cor)
        xx += 2 * r + int(5 * k)
        d.text((xx, y + int(8 * k)), str(nn), font=f_val, fill=P_BRANCO if nn else P_CINZA, anchor="lm")
        xx += int(d.textlength(str(nn), font=f_val)) + int(4 * k)
        d.text((xx, y + int(8 * k)), nome, font=f_lab, fill=P_TEXTO2, anchor="lm")
        xx += int(d.textlength(nome, font=f_lab)) + int(13 * k)
    y += int(26 * k)

    if seg:
        d.line([(x, y), (pw - pad, y)], fill=(255, 255, 255, 40), width=1)
        y += int(10 * k)
        # --- veículo atrás ---
        s = seg.get("seguidor")
        est = seg.get("estado_dist", "livre")
        rot, cor = P_ESTADO.get(est, P_ESTADO["livre"])
        d.text((x, y), "Veículo atrás", font=f_lab, fill=P_TEXTO2)
        _pill(d, pw - pad - int(d.textlength(rot, font=f_pill)) - alt_p, y - int(2 * k), rot, cor,
              f_pill, alt_p, cheia=True)
        y += int(20 * k)
        if s:
            dist = ("<" if s.get("cortada") else "") + f"{s['d']:.0f} m"
            d.text((x, y), dist, font=f_big, fill=cor)
            xx = x + int(d.textlength(dist, font=f_big)) + int(12 * k)
            extra = []
            if s.get("ttc") is not None and np.isfinite(s["ttc"]):
                extra.append(f"TTC {s['ttc']:.1f} s")
            if s.get("vrel") is not None and abs(s["vrel"]) > 0.5:
                extra.append(f"{'+' if s['vrel'] > 0 else '−'}{abs(3.6 * s['vrel']):.0f} km/h")
            if extra:
                d.text((xx, y + int(15 * k)), "  ·  ".join(extra), font=f_lab, fill=P_TEXTO2, anchor="lm")
            y += int(34 * k)
        else:
            d.text((x, y), "nenhum na nossa faixa", font=f_lab, fill=P_CINZA)
            y += int(22 * k)

        # --- mudança de faixa ---
        mud = seg.get("mudanca", {})
        if mud:
            d.text((x, y + alt_p / 2), "Mudar faixa", font=f_lab, fill=P_TEXTO2, anchor="lm")
            xx = x + int(78 * k)
            for lado, seta in (("esquerda", "← Esq"), ("direita", "Dir →")):
                st, _ = mud.get(lado, ("livre", ""))
                xx += _pill(d, xx, y, f"{seta} · {st}", P_MUDAR[st], f_pill, alt_p) + int(6 * k)
            y += alt_p + int(10 * k)

        # --- estacionamento ---
        if seg.get("livre_atras") is not None:
            d.text((x, y + alt_p / 2), "Espaço livre", font=f_lab, fill=P_TEXTO2, anchor="lm")
            d.text((x + int(78 * k), y + alt_p / 2), f"{seg['livre_atras']:.1f} m", font=f_val,
                   fill=P_BRANCO, anchor="lm")
            y += alt_p + int(6 * k)

    ph = min(y + pad - int(4 * k), tela.height)
    # cartão: fundo preto translúcido, cantos redondos, faixa verde UA à esquerda
    fundo = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    fd = ImageDraw.Draw(fundo)
    fd.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(12 * k), fill=(10, 10, 10, 200),
                         outline=(255, 255, 255, 30))
    fd.rounded_rectangle([0, 0, int(4 * k), ph - 1], radius=int(2 * k), fill=UA_VERDE + (255,))
    fundo.alpha_composite(tela.crop((0, 0, pw, ph)))

    x0, y0 = int(12 * k), int(12 * k)
    x1, y1 = min(W, x0 + pw), min(H, y0 + ph)
    rgba = np.asarray(fundo)[: y1 - y0, : x1 - x0].astype(np.float32)
    a = rgba[:, :, 3:4] / 255.0
    roi = img[y0:y1, x0:x1].astype(np.float32)
    roi[:] = roi * (1 - a) + rgba[:, :, 2::-1] * a          # RGB -> BGR
    img[y0:y1, x0:x1] = roi.astype(np.uint8)


def _avisos(img, alertas, e):
    if not alertas:
        return
    H, W = img.shape[:2]
    perigo = any(n == "perigo" for n, _ in alertas)
    if perigo:                                         # moldura vermelha
        cv2.rectangle(img, (0, 0), (W - 1, H - 1), VERMELHO, max(4, int(6 * e)))
    fs = 0.62 * e
    y = int(14 * e)
    for nivel, txt in alertas[:3]:
        cor = VERMELHO if nivel == "perigo" else AMBAR
        txt = "! " + _ascii(txt)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
        x = (W - tw) // 2
        _caixa_translucida(img, x - 12, y, x + tw + 12, y + th + int(14 * e), cor, 0.85)
        cv2.putText(img, txt, (x, y + th + int(7 * e)), cv2.FONT_HERSHEY_DUPLEX, fs, BRANCO, 2,
                    cv2.LINE_AA)
        y += th + int(20 * e)


# ================================ vista de cima ================================
def _vista_de_cima(img, frame, lin, res, e):
    """Mini vista de cima (bird's-eye) com as linhas e os objetos."""
    from .linhas import geometria
    try:
        g, mx, my, _ = geometria(tuple(lin["chave"]))
    except Exception:
        return
    bev = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bh, bw = bev.shape[:2]
    for m in lin.get("marcacoes", []):
        r0, r1, c0, c1 = m["bev"]
        cv2.rectangle(bev, (c0, r0), (c1, r1), CIANO if m["tipo"] == "passadeira" else VERMELHO, 2)
    for o in lin.get("outras", []):
        _curva_bev(bev, o, BRANCO, 1)
    for chave_lado, cor in (("esq", LINHA_ESQ), ("dir", LINHA_DIR)):
        if lin.get(chave_lado):
            _curva_bev(bev, lin[chave_lado], cor, 2)
    # objetos no chão (retângulo com a pegada aproximada)
    k = (bh - 1.0) / max(g["x_far"] - g["x_near"], 1e-6)
    for grupo, comp, larg in (("veic", 4.5, 1.8), ("duas_rodas", 2.0, 0.8), ("peoes", 0.6, 0.6)):
        gr = res.get(grupo) or {}
        for i in range(len(gr.get("bb", []))):
            d, lat = _get(gr, "dist", i), _get(gr, "lat", i)
            if d is None or lat is None:
                continue
            c = (g["lado"] - lat) * g["ppm"]
            r = (g["x_far"] - d) * k
            if not (-bw * 0.2 < c < bw * 1.2):
                continue
            cor = COR_GRUPO[grupo]
            if r < 0:                                  # mais longe do que a vista: seta no topo
                cc = int(min(max(c, 6), bw - 7))
                cv2.fillPoly(bev, [np.array([[cc, 1], [cc - 6, 11], [cc + 6, 11]], np.int32)], cor)
                continue
            hw = larg / 2 * g["ppm"]
            cv2.rectangle(bev, (int(c - hw), int(r - comp * k)), (int(c + hw), int(r)), cor, -1)
            cv2.rectangle(bev, (int(c - hw), int(r - comp * k)), (int(c + hw), int(r)), PRETO, 1)
    cv2.line(bev, (bw // 2, bh - 1), (bw // 2, bh - 12), AMARELO, 2)   # eixo da câmara
    H, W = img.shape[:2]
    alt = int(0.36 * H)
    larg = int(alt * bw / bh)
    mini = cv2.resize(bev, (larg, alt), interpolation=cv2.INTER_AREA)
    x0, y0 = W - larg - 10, 10
    img[y0:y0 + alt, x0:x0 + larg] = mini
    cv2.rectangle(img, (x0 - 1, y0 - 1), (x0 + larg, y0 + alt), BRANCO, 1)
    _texto(img, f"vista de cima  ~{g['x_far']:.0f} m", (x0 + 4, y0 + alt - 6), BRANCO, 0.4 * e)


def _curva_bev(bev, seg, cor, esp):
    bh = bev.shape[0]
    rn = np.linspace(max(seg.get("rn_top", 0.0), 0.0), 1.0, 30)
    col = np.polyval(seg["coef"], rn)
    pts = np.stack([col, rn * (bh - 1)], 1).astype(np.int32)
    cv2.polylines(bev, [pts], False, cor, esp, cv2.LINE_AA)


# ================================ sinais vistos ================================
def _tira_sinais(img, vistos, e):
    if not vistos:
        return
    H, W = img.shape[:2]
    lado = int(64 * e)
    x = int(10 * e)
    y = H - lado - int(34 * e)
    _texto(img, "SINAIS", (x, y - int(6 * e)), COR_GRUPO["sinais"], 0.45 * e)
    for s in vistos:
        rec = cv2.resize(s["recorte"], (lado, lado), interpolation=cv2.INTER_AREA)
        if x + lado > W:
            break
        img[y:y + lado, x:x + lado] = rec
        cv2.rectangle(img, (x, y), (x + lado, y + lado), COR_GRUPO["sinais"], 2)
        nome = _ascii(s["nome"])[:18]
        fs = 0.38 * e
        _caixa_translucida(img, x, y + lado, x + max(lado, int(len(nome) * 7.5 * e)), y + lado + int(20 * e))
        _texto(img, nome, (x + 2, y + lado + int(15 * e)), BRANCO, fs, contorno=False)
        x += max(lado, int(len(nome) * 7.5 * e)) + int(8 * e)


class MemoriaSinais:
    """Guarda os últimos sinais RECONHECIDOS (vistos em >= 3 frames), com um
    recorte, para os mostrar numa tira — o condutor vê o que passou atrás.
    Os recortes são partilhados entre frames (não ocupam memória por frame)."""

    def __init__(self, maximo=4, validade=8.0, min_vistas=3):
        self.maximo, self.validade, self.min_vistas = maximo, validade, min_vistas
        self._lista = OrderedDict()        # nome -> dict(nome, recorte, t, sc)
        self._contagem = {}                # nome -> (vezes, t_ultimo)

    def atualizar(self, frame, sinais, t):
        H, W = frame.shape[:2]
        for i, bb in enumerate(sinais.get("bb", [])):
            nome = NOME_PT.get(_get(sinais, "tipo", i, "sinal"), _get(sinais, "tipo", i, "sinal"))
            n, tu = self._contagem.get(nome, (0, t))
            n = n + 1 if t - tu < 1.0 else 1
            self._contagem[nome] = (n, t)
            if n < self.min_vistas:
                continue
            sc = _get(sinais, "sc", i, 0.0)
            ant = self._lista.get(nome)
            x, y, w, h = [int(round(v)) for v in bb]
            m = int(0.1 * max(w, h))
            rec = frame[max(0, y - m):min(H, y + h + m), max(0, x - m):min(W, x + w + m)]
            if rec.size == 0:
                continue
            if ant is None or w * h >= ant["area"] or t - ant["t"] > 1.0:
                self._lista[nome] = dict(nome=nome, recorte=rec.copy(), t=t, sc=sc, area=w * h)
            else:
                ant["t"] = t
            self._lista.move_to_end(nome)
        for nome in [k for k, v in self._lista.items() if t - v["t"] > self.validade]:
            del self._lista[nome]
        while len(self._lista) > self.maximo:
            self._lista.popitem(last=False)
        return list(self._lista.values())
