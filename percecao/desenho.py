"""
desenho.py — overlay das deteções e dos avisos sobre a imagem (OpenCV).

Vídeos pequenos (640x480) são anotados ampliados e com um realce suave (só no que se
mostra; as deteções usam a imagem original).

Camadas (de baixo para cima):
  1) Marcações no pavimento: faixa do carro preenchida; linha esquerda a VERMELHO e
     direita a VERDE (como o insertLaneBoundary da aula 5), como FITAS com a largura em
     perspetiva; das tracejadas só os traços que lá estão, com o tipo escrito;
     PASSADEIRAS (ciano) e LINHAS DE PARAGEM (vermelho/branco) com a distância.
  2) Guias de ESTACIONAMENTO no chão (opcional): corredor do carro com marcas a
     0.5 / 1 / 2 / 3 m (vermelho -> amarelo -> verde), como nas câmaras de marcha-atrás.
  3) Objetos, cada grupo com a SUA cor (legenda no painel):
       veículos AZUL · duas rodas LARANJA · peões AMARELO · sinais MAGENTA
     etiqueta: tipo, nº, distância (m) e TTC (só se <= 10 s), sem rótulos sobrepostos
     (têm prioridade os mais perto); caixas pequenas ao longe só com a caixa. O SEGUIDOR
     (veículo atrás na nossa faixa) leva marcador no chão; caixa VERMELHA se há risco.
     As caixas vêm suavizadas e confirmadas no tempo (estabilizar.py).
  4) Painel (HUD) no canto superior esquerdo: linhas e tipo (lado do CARRO: numa
     câmara traseira a esquerda da imagem é a direita do carro), contagens por grupo,
     veículo de trás (distância/TTC/velocidade de aproximação), mudança de faixa
     (esq./dir. DO CARRO), espaço livre atrás. Triângulo da aula 5 (amarelo: 1 linha;
     vermelho: nenhuma).
  5) Faixa de AVISO no topo (risco de colisão, peão atrás, ângulo morto...) e moldura
     vermelha em caso de perigo.
  6) VISTA DE CIMA esquemática (canto superior direito, estilo do painel): o carro, a
     estrada até 40 m, linhas, faixa do carro e agentes à escala.
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
DIVISORIA = (235, 235, 235)    # divisória ESTIMADA (sem linhas pintadas): branco
TTC_MOSTRAR = 10.0             # s: TTC maiores não se escrevem (não há risco; é só ruído)

NOME_PT = {"car": "carro", "truck": "camiao", "bus": "autocarro", "bicycle": "bicicleta",
           "motorcycle": "mota", "person": "peao", "stop sign": "STOP"}

FONTE = cv2.FONT_HERSHEY_SIMPLEX


def desenhar(frame, res, ttc_risco=1.0, lim_veic=0.5, vista_cima=True, estacionamento=False):
    # Vídeos pequenos (ex. 640x480): anota-se a imagem AMPLIADA, para o texto e o painel
    # ficarem nítidos quando se vê o vídeo em ecrã inteiro (desenhar pequeno e o leitor
    # esticar dá letra esborratada). As coordenadas do res são ampliadas na mesma medida.
    res = _com_estabilizadas(res)
    f = ampliacao(frame.shape[0])
    if f > 1:
        out = _realcar(cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC))
        res = _escalar_res(res, f)
    else:
        out = frame.copy()
    H, W = out.shape[:2]
    e = _escala(H)
    seg = res.get("seg") or {}
    lin = res.get("linhas")

    _faixa_do_carro(out, lin, seg, e, f)
    _linhas(out, lin, e)
    _marcacoes(out, (lin or {}).get("marcacoes", []), e)
    if estacionamento and seg.get("chave"):
        _guias_estacionamento(out, seg, e, f)
    _objetos(out, res, seg, ttc_risco, e)
    _painel(out, res, seg, ttc_risco, e)
    _avisos(out, seg.get("alertas", []), e)
    if vista_cima and lin and lin.get("chave"):
        _radar(out, res, seg, lin, e)
    _tira_sinais(out, res.get("sinais_vistos", []), e)
    return out


def _com_estabilizadas(res):
    """Desenha as caixas ESTABILIZADAS (res['estab'], de estabilizar.py) quando existem;
    o seguidor (veículo atrás) passa a apontar para a caixa estabilizada correspondente."""
    est = res.get("estab")
    if not est:
        return res
    out = dict(res)
    out.update(est)
    seg = res.get("seg") or {}
    s = seg.get("seguidor")
    if s and s.get("grupo") in est:
        ib = est[s["grupo"]].get("i_bruto", [])
        if s.get("i") in ib:
            out["seg"] = dict(seg, seguidor=dict(s, i=ib.index(s["i"])))
        else:
            out["seg"] = dict(seg, seguidor=dict(s, i=-1))
    return out


def _realcar(img):
    """Só para MOSTRAR um vídeo pequeno ampliado: contraste local suave (CLAHE na luminância),
    um pouco mais de cor e nitidez (unsharp). As deteções usam sempre a imagem original."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = _CLAHE_VIS.apply(lab[:, :, 0])
    lab[:, :, 1:] = np.clip((lab[:, :, 1:].astype(np.int16) - 128) * 1.12 + 128, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    desf = cv2.GaussianBlur(out, (0, 0), 1.4)
    return cv2.addWeighted(out, 1.4, desf, -0.4, 0)


_CLAHE_VIS = cv2.createCLAHE(clipLimit=1.25, tileGridSize=(6, 6))


def _faixa_do_carro(img, lin, seg, e, f=1):
    """A faixa do carro pintada no chão, com um degradê que se apaga ao longe. Com as duas
    linhas, entre elas; com uma só linha ou só com a divisória estimada, a outra margem fica
    a uma largura de faixa (medida quando houve duas linhas; 3,5 m por omissão) — essa margem
    não é desenhada como linha, só o preenchimento a esbater."""
    if not lin or not lin.get("chave"):
        return
    from .linhas import geometria, img_para_chao, lateral_da_linha, veic_para_img
    g = geometria(tuple(lin["chave"]))[0]
    esq, dir_, div = lin.get("esq"), lin.get("dir"), lin.get("divisoria")
    L = float((seg or {}).get("larg_faixa") or 3.5)
    x_fundo = img_para_chao(g["W"] / 2.0, g["H"] - 1.0, g)[0]     # o chão na última linha
    x0 = max(0.5, min(g["x_near"], x_fundo if np.isfinite(x_fundo) else g["x_near"]) - 0.3)
    xs = np.concatenate([[x0], np.linspace(max(g["x_near"], x0 + 0.5), 28.0, 28)])

    def lat(sg):
        return np.array([lateral_da_linha(sg, float(x), g) for x in xs])

    def alcance(sg):
        return g["x_far"] - max(sg.get("rn_top", 0.0), 0.0) * (g["x_far"] - g["x_near"])

    conf = 1.0
    if esq and dir_:
        ye, yd = lat(esq), lat(dir_)
        x_max = min(alcance(esq), alcance(dir_))
        a0 = 0.26
    elif esq or dir_:
        sg = esq or dir_
        y = lat(sg)
        ye, yd = (y, y - L) if esq else (y + L, y)
        x_max = alcance(sg)
        a0 = 0.22
    elif div:
        yd = lat(div)
        ye = yd + L
        x_max = 22.0
        a0 = 0.18
        conf = float(div.get("confianca", 1.0))
    else:
        return
    for sg in (esq, dir_):
        if sg:
            conf = min(conf, float(sg.get("confianca", 1.0)))
    ok = xs <= x_max
    if ok.sum() < 3:
        return
    xs_, ye, yd = xs[ok], ye[ok], yd[ok]
    H, W = img.shape[:2]
    mascara = np.zeros((H, W), np.uint8)
    n = len(xs_)
    for i in range(n - 1):            # bandas: perto opaco, longe transparente
        quad = [veic_para_img(xs_[i], ye[i], g), veic_para_img(xs_[i + 1], ye[i + 1], g),
                veic_para_img(xs_[i + 1], yd[i + 1], g), veic_para_img(xs_[i], yd[i], g)]
        quad = np.array([[u * f, v * f] for u, v in quad], np.float32)
        if not np.all(np.isfinite(quad)):
            continue
        a = a0 * conf * (1.0 - i / (n - 1)) ** 1.3
        cv2.fillPoly(mascara, [quad.astype(np.int32)], int(255 * a), cv2.LINE_AA)
    if not mascara.any():
        return
    m = cv2.GaussianBlur(mascara, (0, 0), 2.0).astype(np.float32)[:, :, None] / 255.0
    cor = np.array(VERDE_FAIXA, np.float32)
    img[:] = (img.astype(np.float32) * (1 - m) + cor * m).astype(np.uint8)


def ampliacao(H):
    """Fator inteiro para anotar vídeos com menos de 720 linhas (480 -> 2)."""
    return int(np.ceil(720.0 / H)) if H < 700 else 1


def _escalar_res(res, f):
    """Cópia do res com as coordenadas de IMAGEM multiplicadas por f (o resto partilha-se)."""
    out = dict(res)
    for g in ("veic", "duas_rodas", "peoes", "sinais"):
        if res.get(g):
            gr = dict(res[g])
            gr["bb"] = [[v * f for v in bb] for bb in gr.get("bb", [])]
            out[g] = gr
    esc = lambda pts: [[u * f, v * f] for u, v in pts]
    lin = res.get("linhas")
    if lin:
        lin = dict(lin)
        for lado in ("esq", "dir", "divisoria"):
            if lin.get(lado):
                d = dict(lin[lado])
                d["pts"] = esc(d["pts"])
                if d.get("tracos") is not None:
                    d["tracos"] = [esc(t) for t in d["tracos"]]
                lin[lado] = d
        lin["outras"] = [dict(o, pts=esc(o["pts"]),
                              tracos=None if o.get("tracos") is None else [esc(t) for t in o["tracos"]])
                         for o in lin.get("outras", [])]
        lin["marcacoes"] = [dict(m, pts=esc(m["pts"])) for m in lin.get("marcacoes", [])]
        if lin.get("horizonte") is not None:
            lin["horizonte"] = lin["horizonte"] * f
        out["linhas"] = lin
    return out


def _escala(H):
    return max(0.6, H / 720.0)


# ================================ texto ================================
def _ascii(txt):
    """O putText do OpenCV não desenha acentos: 'cedência' -> 'cedencia'."""
    return unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()


def _px(fs):
    """Tamanho da letra (px) equivalente à escala fs do putText."""
    return max(9, int(round(27 * fs)))


def _pil_texto(img, txt, x, y_base, cor_bgr, tam, negrito=False, contorno=0):
    """Escreve txt com letra TrueType suavizada (acentos incluídos) diretamente na imagem
    BGR, mexendo só no retângulo do texto. y_base = linha de base. Devolve a largura."""
    from PIL import Image, ImageDraw
    f = _fonte(tam, negrito)
    x0, y0, x1, y1 = f.getbbox(txt, stroke_width=contorno)
    w, h = x1 - x0 + 4, y1 - y0 + 4
    H, W = img.shape[:2]
    ax, ay = int(x) + x0 - 2, int(y_base) - (f.getmetrics()[0]) + y0 - 2
    cx0, cy0, cx1, cy1 = max(ax, 0), max(ay, 0), min(ax + w, W), min(ay + h, H)
    if cx1 <= cx0 or cy1 <= cy0:
        return w
    roi = Image.fromarray(img[cy0:cy1, cx0:cx1, ::-1].copy())
    d = ImageDraw.Draw(roi)
    d.text((ax - cx0 - x0 + 2, ay - cy0 - y0 + 2), txt, font=f, fill=tuple(int(c) for c in cor_bgr[::-1]),
           stroke_width=contorno, stroke_fill=(0, 0, 0))
    img[cy0:cy1, cx0:cx1] = np.asarray(roi)[:, :, ::-1]
    return w


def _texto(img, txt, org, cor=BRANCO, fs=0.5, esp=1, contorno=True):
    return _pil_texto(img, str(txt), org[0], org[1], cor, _px(fs), negrito=esp > 1,
                      contorno=1 if contorno else 0)


def _rotulo(img, txt, org, cor, fs=0.5, centrado=True, ocupados=None, alt=None):
    """Texto branco sobre um retângulo (cantos arredondados) da cor dada. Com 'ocupados'
    (lista de retângulos já usados) não se sobrepõe a outro rótulo: tenta a posição 'alt'
    e, se também estiver ocupada, não escreve. Devolve True se escreveu."""
    txt = str(txt)
    if not txt:
        return False
    tam = _px(fs)
    f = _fonte(tam, True)
    tw = int(f.getlength(txt))
    asc, desc = f.getmetrics()
    th = asc
    for o in (org, alt):
        if o is None:
            continue
        x = o[0] - tw // 2 if centrado else o[0]
        x = max(0, min(x, img.shape[1] - tw - 8))
        y = max(th + 6, min(o[1], img.shape[0] - 2))
        r = (x, y - th - 5, x + tw + 8, y + desc - 1)
        if ocupados is not None and any(r[0] < q[2] and q[0] < r[2] and r[1] < q[3] and q[1] < r[3]
                                        for q in ocupados):
            continue
        rr = max(2, (r[3] - r[1]) // 4)
        cv2.rectangle(img, (r[0] + rr, r[1]), (r[2] - rr, r[3]), cor, -1)
        cv2.rectangle(img, (r[0], r[1] + rr), (r[2], r[3] - rr), cor, -1)
        for cx_, cy_ in ((r[0] + rr, r[1] + rr), (r[2] - rr, r[1] + rr),
                         (r[0] + rr, r[3] - rr), (r[2] - rr, r[3] - rr)):
            cv2.circle(img, (cx_, cy_), rr, cor, -1, cv2.LINE_AA)
        _pil_texto(img, txt, x + 4, y - 2, BRANCO, tam, negrito=True)
        if ocupados is not None:
            ocupados.append(r)
        return True
    return False


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
    H = img.shape[0]
    y_h = float(lin.get("horizonte") or 0.4 * H)
    # espessura em PERSPETIVA: larga perto do carro, fina ao longe (como a tinta no chão).
    # As 'outras' linhas (das faixas vizinhas) não se desenham: piscam e enchem a imagem.
    esp = (max(1.5, 1.2 * e), max(4.0, 9.0 * e), y_h)
    for seg, cor in ((esq, LINHA_ESQ), (dir_, LINHA_DIR)):
        if seg and _desenha_linha(img, seg, cor, esp):
            u, v = seg["pts"][-1]                      # ponto mais perto do carro
            _rotulo(img, seg.get("tipo", ""), (int(u), int(v) - int(16 * e)), cor, 0.42 * e)
    div = lin.get("divisoria")
    if div and not (esq or dir_):                     # sem linhas pintadas: divisória estimada
        camada = img.copy()
        _fita(camada, div["pts"], DIVISORIA, esp)
        a = 0.75 * max(float(div.get("confianca", 1.0)), 0.2)
        cv2.addWeighted(camada, a, img, 1 - a, 0, dst=img)
        u, v = div["pts"][0]                           # ponto mais perto do carro
        _rotulo(img, "divisoria (estimada)", (int(u), int(v) - int(16 * e)), (90, 90, 90), 0.40 * e)


def _fita(img, pts, cor, esp, contorno=True):
    """Linha como uma FITA de largura variável com a perspetiva (esp = (fina, larga, y_h)),
    com um contorno escuro discreto para se destacar do asfalto e de chão claro."""
    p = np.asarray(pts, np.float32)
    if len(p) < 2:
        return
    d = np.hypot(*np.diff(p, axis=0).T)                # reamostra uniforme: curva suave
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 2:
        return
    t = np.linspace(0, s[-1], max(8, int(s[-1] / 6)))
    p = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
    fina, larga, y_h = esp
    H = img.shape[0]
    w = fina + (larga - fina) * np.clip((p[:, 1] - y_h) / max(H - y_h, 1.0), 0, 1)
    tg = np.gradient(p, axis=0)
    tg /= np.maximum(np.linalg.norm(tg, axis=1, keepdims=True), 1e-6)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    camadas = ([(1.6, PRETO)] if contorno else []) + [(0.0, cor)]
    for extra, c in camadas:
        ww = (w / 2 + extra)[:, None]
        poli = np.concatenate([p + nrm * ww, (p - nrm * ww)[::-1]]).astype(np.int32)
        if extra:
            camada = img.copy()
            cv2.fillPoly(camada, [poli], c, cv2.LINE_AA)
            cv2.addWeighted(camada, 0.35, img, 0.65, 0, dst=img)
        else:
            cv2.fillPoly(img, [poli], c, cv2.LINE_AA)


def _desenha_linha(img, seg, cor, esp):
    """Contínua: a curva toda. Tracejada: SÓ os traços que lá estão (nunca um padrão
    inventado). Uma linha só prevista (sem medida há algumas imagens) vai esbatendo.
    Devolve True se desenhou alguma coisa."""
    conf = float(seg.get("confianca", 1.0))
    if conf < 0.99:
        camada = img.copy()
        feito = _desenha_linha(camada, dict(seg, confianca=1.0), cor, esp)
        cv2.addWeighted(camada, max(conf, 0.15), img, 1 - max(conf, 0.15), 0, dst=img)
        return feito
    if not isinstance(esp, tuple):                    # espessura fixa
        esp = (float(esp), float(esp), 0.0)
    if seg.get("tipo") != "tracejada":
        _fita(img, seg["pts"], cor, esp)
        return True
    tracos = seg.get("tracos")
    if tracos is None:                                 # resultados antigos sem 'tracos'
        _polilinha(img, seg["pts"], cor, max(2, int(esp[1] / 2)), True)
        return True
    for t in tracos:
        _fita(img, t, cor, esp)
    return bool(tracos)


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
def _guias_estacionamento(img, seg, e, f=1):
    from .linhas import geometria, veic_para_img
    g = geometria(tuple(seg["chave"]))[0]
    H = img.shape[0]
    meia = 0.9                                        # metade da largura do carro (m)
    troços = [(0.3, 1.0, VERMELHO), (1.0, 2.0, AMARELO), (2.0, 3.0, VERDE)]
    esp = max(2, int(3 * e))
    vi = lambda x, y: tuple(c * f for c in veic_para_img(x, y, g))    # na imagem ampliada
    for x0, x1, cor in troços:
        for lado in (meia, -meia):
            u0, v0 = vi(x0, lado)
            u1, v1 = vi(x1, lado)
            if min(v0, v1) < H and max(v0, v1) > g["y_h"] * f:
                cv2.line(img, (int(u0), int(v0)), (int(u1), int(v1)), cor, esp, cv2.LINE_AA)
        u0, v0 = vi(x1, meia)
        u1, v1 = vi(x1, -meia)
        if v0 < H:
            cv2.line(img, (int(u0), int(v0)), (int(u1), int(v1)), cor, max(1, esp - 1), cv2.LINE_AA)
            _texto(img, f"{x1:.0f} m", (int(u1) + 6, int(v1) + 4), cor, 0.45 * e)


# ================================ objetos ================================
def _objetos(img, res, seg, ttc_risco, e):
    fs = 0.42 * e
    H = img.shape[0]
    seguidor = seg.get("seguidor") or {}
    # do mais PERTO (base da caixa mais abaixo) para o mais longe: os rótulos dos mais
    # perto têm prioridade; os de longe que chocariam com eles não se escrevem
    itens = []
    for grupo in ("sinais", "peoes", "duas_rodas", "veic"):
        gr = res.get(grupo) or {}
        for i, bb in enumerate(gr.get("bb", [])):
            itens.append((bb[1] + bb[3], grupo, i))
    itens.sort(key=lambda t: -t[0])
    ocupados = []
    caixas = []
    for _, grupo, i in itens:
        gr = res[grupo]
        bb = gr["bb"][i]
        cor = COR_GRUPO[grupo]
        tipo = _get(gr, "tipo", i, "")
        nome = NOME_PT.get(tipo, tipo)
        partes = [nome or NOME_GRUPO[grupo]]
        oid = _get(gr, "id_vis", i, None)
        if oid is None:
            oid = _get(gr, "id", i, -1)
        if oid is not None and oid >= 0 and grupo != "sinais":
            partes[0] += f" {oid}"
        d = _get(gr, "dist", i)
        if d is not None and np.isfinite(d):
            # ao longe a distância monocular é grosseira: de 5 em 5 m (não "salta")
            d = round(d / 5.0) * 5 if d >= 30 else round(d)
            partes.append(("<" if _get(gr, "cortada", i, False) else "") + f"{d:.0f} m")
        ttc = _get(gr, "ttc", i)
        risco = ttc is not None and np.isfinite(ttc) and ttc <= ttc_risco
        if ttc is not None and np.isfinite(ttc) and ttc <= TTC_MOSTRAR:
            partes.append(f"TTC {ttc:.1f}s")       # só os que se aproximam a sério
        if grupo == "sinais" and _get(gr, "sc", i) is not None:
            partes.append(f"{100 * gr['sc'][i]:.0f}%")
        e_seguidor = seguidor.get("grupo") == grupo and seguidor.get("i") == i
        if e_seguidor:
            _marcador_seguidor(img, bb, COR_ESTADO.get(seg.get("estado_dist"), cor), e)
        if risco:
            cor = VERMELHO
        # caixas pequenas (longe) sem rótulo, a não ser que importem
        importante = risco or e_seguidor or grupo in ("peoes", "duas_rodas", "sinais")
        # FOCO: veículos fora das faixas que interessam (estacionados ao lado da via,
        # do outro lado) ou muito longe ficam discretos, sem rótulo
        faixa = _get(gr, "faixa", i)
        vrel = _get(gr, "vrel", i)
        secundario = (grupo == "veic" and not importante and
                      (faixa == "fora" or (d is not None and d > 45) or
                       (vrel is not None and vrel < -1.5)))     # a afastar-se: não é ameaça
        etiqueta = "" if secundario else (
            "  ".join(partes) if (bb[3] >= 0.06 * H or importante) else "")
        caixas.append((bb, cor, etiqueta, risco or e_seguidor,
                       bool(_get(gr, "previsto", i, False)), secundario))
    for bb, cor, etiqueta, grossa, previsto, secundario in caixas:   # caixas, rótulos por cima
        if secundario:
            _caixa_discreta(img, bb, cor)
        else:
            _caixa(img, bb, cor, "", fs, grossa=grossa, tenue=previsto)
    for bb, cor, etiqueta, grossa, previsto, secundario in caixas:
        if etiqueta:
            x, y, w, h = bb
            _rotulo(img, etiqueta, (int(x), int(y) - 3), cor, fs, centrado=False,
                    ocupados=ocupados, alt=(int(x), int(y + h) + int(18 * e)))


def _get(gr, chave, i, padrao=None):
    lst = gr.get(chave)
    return lst[i] if lst is not None and i < len(lst) else padrao


def _caixa(img, bb, cor, etiqueta, fs, grossa=False, tenue=False):
    """Caixa fina com cantos reforçados; 'tenue' = objeto previsto (falhou esta imagem)."""
    alvo = img.copy() if tenue else img
    x, y, w, h = [int(round(v)) for v in bb]
    esp = 3 if grossa else 2
    camada = alvo.copy()
    cv2.rectangle(camada, (x, y), (x + w, y + h), cor, 1, cv2.LINE_AA)
    cv2.addWeighted(camada, 0.55, alvo, 0.45, 0, dst=alvo)      # contorno discreto
    c = max(6, min(w, h) // 4)                                   # cantos reforçados
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(alvo, (px, py), (px + dx * c, py), cor, esp + 1, cv2.LINE_AA)
        cv2.line(alvo, (px, py), (px, py + dy * c), cor, esp + 1, cv2.LINE_AA)
    if tenue:
        cv2.addWeighted(alvo, 0.5, img, 0.5, 0, dst=img)
    if etiqueta:
        _rotulo(img, etiqueta, (x, y - 2), cor, fs, centrado=False)


def _caixa_discreta(img, bb, cor):
    """Só os cantos, finos e meio transparentes (veículo que não interessa agora)."""
    x, y, w, h = [int(round(v)) for v in bb]
    camada = img.copy()
    c = max(5, min(w, h) // 5)
    for (px, py, dx, dy) in ((x, y, 1, 1), (x + w, y, -1, 1), (x, y + h, 1, -1), (x + w, y + h, -1, -1)):
        cv2.line(camada, (px, py), (px + dx * c, py), cor, 2, cv2.LINE_AA)
        cv2.line(camada, (px, py), (px, py + dy * c), cor, 2, cv2.LINE_AA)
    cv2.addWeighted(camada, 0.45, img, 0.55, 0, dst=img)


def _marcador_seguidor(img, bb, cor, e):
    """Elipse no chão por baixo do veículo de trás (o que segue na nossa faixa)."""
    x, y, w, h = bb
    H = img.shape[0]
    base = (int(x + w / 2), int(min(y + h, H - 2)))
    camada = img.copy()
    cv2.ellipse(camada, base, (int(w * 0.5), max(4, int(min(w, h) * 0.1))), 0, 0, 360, cor, -1,
                cv2.LINE_AA)
    cv2.addWeighted(camada, 0.35, img, 0.65, 0, dst=img)


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
            if s.get("ttc") is not None and np.isfinite(s["ttc"]) and s["ttc"] <= TTC_MOSTRAR:
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
    fd.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(12 * k), fill=(12, 12, 12, 252),
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
    # no espaço livre ENTRE o painel (esq.) e a vista de cima (dir.), para não os tapar
    k = max(e, 0.72)
    esq_livre, dir_livre = int(364 * k), W - int(254 * k)
    if dir_livre - esq_livre < 0.3 * W:                # imagem estreita: centrado
        esq_livre, dir_livre = 0, W
    tam = _px(0.58 * e)
    f = _fonte(tam, True)
    y = int(14 * e)
    for nivel, txt in alertas[:3]:
        cor = VERMELHO if nivel == "perigo" else AMBAR
        txt = "! " + str(txt)
        tw, th = int(f.getlength(txt)), f.getmetrics()[0]
        x = (esq_livre + dir_livre - tw) // 2
        x = max(12, min(x, W - tw - 12))
        _caixa_translucida(img, x - 12, y, x + tw + 12, y + th + int(14 * e), cor, 0.88)
        _pil_texto(img, txt, x, y + th + int(5 * e), BRANCO, tam, negrito=True)
        y += th + int(20 * e)


# ================================ vista de cima ================================
def _radar(img, res, seg, lin, e):
    """Vista de cima ESQUEMÁTICA (canto superior direito), no estilo do painel: o nosso
    carro em baixo, a estrada para trás até 40 m, as linhas (e a divisória estimada), a faixa
    do carro e os agentes à escala, com a cor do grupo (o seguidor com a cor do estado; a
    vermelho se há risco). Esquerda/direita como na imagem da câmara."""
    from PIL import Image, ImageDraw
    from .linhas import geometria, lateral_da_linha
    try:
        g = geometria(tuple(lin["chave"]))[0]
    except Exception:
        return
    H, W = img.shape[:2]
    k = max(e, 0.72)
    pw, ph = int(230 * k), int(300 * k)
    pad = int(12 * k)
    topo = int(30 * k)                                  # espaço do título
    alc, lado = 40.0, 6.0                               # m para trás, m para cada lado
    ax0, ax1 = pad, pw - pad
    ay0, ay1 = topo, ph - int(34 * k)                   # zona do desenho (ay1 = traseira do carro)
    sx = (ax1 - ax0) / (2 * lado)
    sy = (ay1 - ay0) / alc

    def P(x, y):                                        # (m atrás, m à esq. da imagem) -> px
        return ax0 + (lado - y) * sx, ay1 - x * sy

    tela = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    db = ImageDraw.Draw(tela)
    db.rounded_rectangle([0, 0, pw - 1, ph - 1], radius=int(12 * k), fill=(12, 12, 12, 245),
                         outline=(255, 255, 255, 30))
    db.rounded_rectangle([pw - int(4 * k) - 1, 0, pw - 1, ph - 1], radius=int(2 * k),
                         fill=UA_VERDE + (255,))
    camada = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))   # desenho por cima (misturado no fim)
    d = ImageDraw.Draw(camada)
    f_tit, f_pq = _fonte(int(11 * k), True), _fonte(int(10 * k))
    d.text((pad, int(10 * k)), "VISTA DE CIMA", font=f_tit, fill=UA_VERDE)
    # anéis de distância
    for dist in (10, 20, 30, 40):
        yy = P(dist, 0)[1]
        for xx in range(int(ax0), int(ax1), int(8 * k)):
            d.line([(xx, yy), (xx + int(4 * k), yy)], fill=(255, 255, 255, 38), width=1)
        d.text((ax1 - int(2 * k), yy - int(2 * k)), f"{dist} m", font=f_pq, fill=(150, 150, 150, 200),
               anchor="rb")
    # linhas e faixa
    xs = np.linspace(max(g["x_near"], 1.0), alc, 30)

    def lat(sg):
        return np.array([lateral_da_linha(sg, float(x), g) for x in xs])

    esq, dir_, div = lin.get("esq"), lin.get("dir"), lin.get("divisoria")
    L = float((seg or {}).get("larg_faixa") or 3.5)
    bordas = None
    if esq and dir_:
        bordas = (lat(esq), lat(dir_))
    elif esq:
        y = lat(esq); bordas = (y, y - L)
    elif dir_:
        y = lat(dir_); bordas = (y + L, y)
    elif div:
        y = lat(div); bordas = (y + L, y)
    if bordas is not None:
        poli = [P(x, y) for x, y in zip(xs, bordas[0])] + [P(x, y) for x, y in zip(xs[::-1], bordas[1][::-1])]
        d.polygon(poli, fill=(80, 220, 80, 70))
    for sg, cor in ((esq, (235, 40, 40)), (dir_, (40, 200, 40))):
        if not sg:
            continue
        pts = [P(x, y) for x, y in zip(xs, lat(sg))]
        if sg.get("tipo") == "tracejada":
            for i in range(0, len(pts) - 1, 2):
                d.line([pts[i], pts[i + 1]], fill=cor + (255,), width=max(2, int(3 * k)))
        else:
            d.line(pts, fill=cor + (255,), width=max(2, int(3 * k)), joint="curve")
    if div and not (esq or dir_):
        d.line([P(x, y) for x, y in zip(xs, lat(div))], fill=(235, 235, 235, 170), width=max(2, int(2 * k)))
    # agentes
    seguidor = (seg or {}).get("seguidor") or {}
    for grupo, comp, larg in (("veic", 4.5, 1.8), ("duas_rodas", 2.0, 0.8), ("peoes", 0.6, 0.6)):
        gr = res.get(grupo) or {}
        for i in range(len(gr.get("bb", []))):
            dd, yy = _get(gr, "dist", i), _get(gr, "lat", i)
            if dd is None or yy is None or not np.isfinite(dd) or abs(yy) > lado + 1:
                continue
            cor = P_GRUPO[grupo]
            ttc = _get(gr, "ttc", i)
            if seguidor.get("grupo") == grupo and seguidor.get("i") == i:
                cor = P_ESTADO.get((seg or {}).get("estado_dist"), ("", cor))[1]
            if ttc is not None and np.isfinite(ttc) and ttc <= 1.0:
                cor = P_VERMELHO
            if dd + comp > alc:                         # mais longe: seta no topo
                cx = P(0, yy)[0]
                d.polygon([(cx, ay0 - int(6 * k)), (cx - int(5 * k), ay0 + int(2 * k)),
                           (cx + int(5 * k), ay0 + int(2 * k))], fill=cor + (200,))
                continue
            x0_, y0_ = P(dd + comp, yy + larg / 2)
            x1_, y1_ = P(dd, yy - larg / 2)
            secundario = grupo == "veic" and _get(gr, "faixa", i) == "fora"
            d.rounded_rectangle([x0_, y0_, x1_, y1_], radius=max(2, int(3 * k)),
                                fill=cor + ((110,) if secundario else (235,)))
    # o nosso carro (visto de cima, traseira para cima)
    cw, cl = 1.8 * sx, 4.4 * sy
    cx, cy = (ax0 + ax1) / 2, ay1
    d.rounded_rectangle([cx - cw / 2, cy, cx + cw / 2, cy + cl], radius=int(5 * k), fill=(235, 235, 235, 255))
    d.rounded_rectangle([cx - cw / 2 + 2, cy + 0.12 * cl, cx + cw / 2 - 2, cy + 0.32 * cl],
                        radius=int(2 * k), fill=(60, 60, 60, 255))
    d.polygon([(cx, cy - int(5 * k)), (cx - int(5 * k), cy + int(1 * k)), (cx + int(5 * k), cy + int(1 * k))],
              fill=UA_VERDE + (255,))                  # a câmara (olha para trás = para cima)
    tela.alpha_composite(camada)
    x0, y0 = W - pw - int(12 * k), int(12 * k)
    x1, y1 = min(W, x0 + pw), min(H, y0 + ph)
    rgba = np.asarray(tela)[: y1 - y0, : x1 - x0].astype(np.float32)
    a = rgba[:, :, 3:4] / 255.0
    roi = img[y0:y1, x0:x1].astype(np.float32)
    roi[:] = roi * (1 - a) + rgba[:, :, 2::-1] * a
    img[y0:y1, x0:x1] = roi.astype(np.uint8)


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
