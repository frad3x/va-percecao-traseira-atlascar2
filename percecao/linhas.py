"""
linhas.py — deteção das linhas da faixa, seguindo a AULA 5 (exemplo MATLAB
"Visual Perception Using Monocular Camera"), mas SEM precisar de calibração.

Pipeline (com o nome da função MATLAB equivalente):
  1) VISTA DE CIMA por IPM com chão plano ............... birdsEyeView
  2) marcas por deteção de RIDGE (não Canny!) ........... segmentLaneMarkerRidge
     + contraste local (CLAHE, contra reflexos do sol e sombras)
  3) candidatas: histograma + janelas deslizantes
     + PARÁBOLAS por RANSAC (seguem curvas) ............ findParabolicLaneBoundaries
  4) rejeição: curvatura (validateBoundaryFcn), comprimento mínimo
     (isOfMinLength) e força >= 20% da maior (strongBoundaries usa 40%)
  5) tipo de linha: CONTÍNUA / TRACEJADA ................ classifyLaneTypes
  6) faixa do carro = linha mais próxima à esq. e à dir. . leftEgoBoundary/rightEgoBoundary
  7) seguimento + suavização + confirmação temporal: cada linha procura-se perto de
     onde estava; um salto só é aceite se se repetir (não tremem, piscam nem saltam)

Sem calibração: a vista de cima é definida RELATIVAMENTE À IMAGEM — a linha de
baixo cobre a largura da imagem e o topo fica a uma fração do caminho até ao
horizonte (só a estrada próxima, onde o chão é plano). Todos os limiares são
frações da largura da vista de cima, por isso funcionam com qualquer câmara.
O único parâmetro crítico é o HORIZONTE: com ele certo, as linhas ficam paralelas
na vista de cima. Nos vídeos 640x480 da câmara do ATLASCAR2 é fixo (medido, ver
horizonte_conhecido); nos outros é ESTIMADO pelo ponto de fuga das linhas (mediana
ao longo do tempo). Pode fixar-se em PARAMS['horizonte'].
(hfov/altura só dão uma escala aproximada em metros; não mudam a deteção.)

Sentido de esq./dir.: o da IMAGEM (numa câmara traseira, a esquerda da imagem é a
direita do carro).

detetar(frame, excluir=[bb...]) -> {
   'n': 0|1|2,
   'esq'/'dir': None ou {'pts': [[u,v],...] (imagem), 'tipo': 'continua'|'tracejada',
                         'coef': [a,b,c], 'rn_top': ..., 'cobertura': ...},
   'outras': [...linhas fortes que não são da faixa do carro...],
   'chave': (...) parâmetros da vista de cima (para desenhar a mini-vista),
   'horizonte': y_h }
"""
import math
from collections import deque
from functools import lru_cache

import cv2
import numpy as np

PARAMS = dict(
    # --- geometria da vista de cima ---
    hfov=66.0,            # campo de visão horizontal (graus). Astra Pro ~66°
    altura=1.75,          # altura da câmara (m) no ATLASCAR2 (~1,75 m): dá a escala em metros
                          #   (igual ao CAM_Z do .env e ao altura_camara do nó ROS)
    horizonte=None,       # linha do horizonte (fração da altura); None = automático: fixo se a
                          #   resolução estiver em horizonte_conhecido, senão pelo ponto de fuga
    # Câmara traseira do ATLASCAR2 (Orbbec Astra Pro, cor gravada a 640x480): está FIXA no
    # carro, por isso o horizonte é praticamente constante — medido na volta real, o ponto de
    # fuga da estrada fica a ~175 px de 480 em vários troços. O ponto de fuga automático
    # escolhia bermas e carros estacionados e dava ~250 px (distâncias e linhas erradas).
    horizonte_conhecido={(640, 480): 0.365},
    horizonte_ini=0.52,   # valor inicial, até haver estimativa pelo ponto de fuga
    v_base=0.92,          # linha mais baixa usada (fração da altura): corta capô / OSD
    frac_topo=0.20,       # topo da vista: 20% do caminho do horizonte até à base
    bev_w=280, bev_h=360, # tamanho da vista de cima (px)
    # --- marcas (ridge); larguras em fração da largura da vista de cima ---
    marca=0.022,          # ~ approxMarkerWidth do segmentLaneMarkerRidge
    ridge_min=30,         # contraste mínimo do ridge (0-255, após CLAHE)
    clahe=2.0,            # contraste local (contra reflexos do sol / sombras)
    area_min=12,          # px: remove manchas minúsculas
    # --- parábolas ---
    max_linhas=4,         # ~ maxLanes
    n_janelas=14,         # janelas deslizantes
    janela=0.05,          # meia-largura da janela (fração)
    tol_ransac=0.025,     # tolerância do RANSAC (fração)
    curv_max=0.35,        # desvio por curvatura DENTRO do troço visível / largura
                          #   (rejeita curvas absurdas) ~ validateBoundaryFcn
    decl_max=0.6,         # |inclinação junto ao carro| / largura (rejeita atravessadas)
    min_comp=0.30,        # extensão mínima (fração da altura) ~ isOfMinLength
    forca_rel=0.20,       # força >= 20% da maior. A aula usa 40% e diz que "pode ser
                          #   ajustado": com uma contínua forte ao lado, 40% elimina as
                          #   tracejadas (têm naturalmente menos pontos) — medido no dashcam
    min_pts=25,
    duplicado=0.06,       # linhas a menos disto (fração) = a mesma linha
    largura_faixa=(0.18, 1.05),  # largura plausível da faixa do carro (fração)
    # --- tipo de linha ---
    cobertura_continua=0.88,     # medido: contínuas ~1.00, tracejadas ~0.64 (p90 ~0.8)
    # --- é mesmo TINTA? (estradas reais: fissuras, bermas, passadeiras, lombas, reflexos) ---
    ridge_bruto_min=10,   # contraste mínimo SEM o CLAHE (o CLAHE realça texturas do asfalto)
    traco_min_m=1.2,      # um troço de tinta tem pelo menos isto de comprimento (m); os traços
                          #   pintados têm 1,5-3 m, os pedaços de lombas e fissuras menos
    estrutura_min=0.6,    # fração da cobertura em troços >= traco_min_m; abaixo é ruído
    cobertura_min=0.15,   # uma tracejada cobre pelo menos isto da sua extensão
    largura_max_m=0.35,   # uma linha pintada tem ~0,10-0,15 m (+ desfocagem e erro de escala);
                          #   as riscas das passadeiras têm ~0,5 m
    contraste_min=12,     # tinta - asfalto (o lado mais escuro, a 0,35 m), níveis 0-255, sem
                          #   CLAHE. Medido: linhas reais 32-117 (mediana); fissuras/terra 0-2
    contraste_claro_min=6,  # tinta - o lado mais CLARO (0,2-0,5 m): asfalto dos dois lados.
    simetria_min=0.65,      # (tinta - lado claro) / (tinta - lado escuro). Bordas de passeio,
                            #   de estacionamento e as diagonais das riscas das passadeiras têm
                            #   um lado claro (ratio < 0,6); linhas pintadas ~0,8-1. Medido na
                            #   volta real: corta quase todas as bermas e perde 1 linha em ~270
    # --- sem linhas: divisória ESTIMADA a meio da estrada (asfalto segmentado) ---
    divisoria=True,
    estrada_lado=7.0,     # m para cada lado na vista larga usada para achar as bermas
    estrada_longe=20.0,   # m: até onde se procura a estrada
    largura_estrada=(4.0, 14.0),   # m: largura plausível da estrada
    faixa_largura=3.5,    # m: largura de uma via (para a divisória quando o meio falha)
    divisoria_lado_min=1.2,  # m: a divisória fica pelo menos isto à ESQUERDA do carro
    textura_asfalto_max=9.0,       # desvio padrão local máx. do "asfalto" atrás do carro
                                   #   (medido: asfalto 2-6,5; calçada ~9,7)
    textura_chao_max=15.0,         # idem, medido na IMAGEM (sem a desfocagem da vista de
                                   #   cima), no chão logo atrás do carro. Medido: asfalto 3-5,5,
                                   #   chão da garagem ~2, calçada portuguesa ~50
    # --- tempo ---
    alpha=0.35,           # suavização (0 = nada muda, 1 = sem suavização)
    memoria=8,            # frames que uma linha "sobrevive" sem ser vista
    confirmar_n=6,        # um lado só aparece se foi visto em >= k das
    confirmar_k=3,        #   últimas n frames (corta "piscares")
    salto_max=0.12,       # salto maior que isto (fração) = posição nova "suspeita"
    confirmar_salto=5,    # ... só é aceite se se repetir em k frames seguidas (mudança de
                          #   faixa real); se não, ignora-se (era uma linha errada)
    seguir=0.08,          # procura a linha de cada lado a menos disto (fração) de onde
                          #   estava antes, antes de escolher pela proximidade ao centro
    # --- outras marcações no pavimento ---
    marcacoes=True,       # passadeiras e linhas de paragem (na vista de cima)
)

_RNG = np.random.default_rng(0)


# ======================= geometria: vista de cima sem calibração =======================
def chave_ipm(W, H, y_h, p):
    """Tupla com tudo o que define a vista de cima (serve de chave de cache)."""
    return (int(W), int(H), int(round(y_h)), float(p["hfov"]), float(p["altura"]),
            float(p["v_base"]), float(p["frac_topo"]), int(p["bev_w"]), int(p["bev_h"]))


@lru_cache(maxsize=16)
def geometria(chave):
    """Modelo pinhole com chão plano (como monoCamera + birdsEyeView do MATLAB).
    Devolve (g, mapx, mapy, valido): g = parâmetros; mapas para cv2.remap."""
    W, H, y_h, hfov, altura, v_base, frac_topo, bw, bh = chave
    f = (W / 2.0) / math.tan(math.radians(hfov) / 2.0)
    cx, cy = W / 2.0, H / 2.0
    teta = math.atan2(cy - y_h, f)            # inclinação (positivo = a olhar para baixo)
    c, s = math.cos(teta), math.sin(teta)

    def x_de_v(v):                            # distância ao chão vista na linha v
        t = (v - cy) / f
        den = t * c + s
        return altura * (c - t * s) / den if den > 1e-9 else 1e6

    v0 = v_base * H
    x_near = max(x_de_v(v0), 0.3)
    x_far = max(x_de_v(y_h + frac_topo * (v0 - y_h)), x_near * 1.5)
    lado = (W / 2.0) * (x_near * c + altura * s) / f   # a linha de baixo cobre a imagem
    xs = np.linspace(x_far, x_near, bh)        # linha 0 = longe, última = junto ao carro
    ys = np.linspace(lado, -lado, bw)          # coluna 0 = esquerda da imagem
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zc = xx * c + altura * s
    u = cx - f * yy / zc
    v = cy + f * (altura * c - xx * s) / zc
    valido = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    g = dict(W=W, H=H, f=f, cx=cx, cy=cy, c=c, s=s, h=altura, y_h=y_h,
             x_near=x_near, x_far=x_far, lado=lado, bw=bw, bh=bh,
             ppm=(bw - 1) / (2.0 * lado))
    return g, u.astype(np.float32), v.astype(np.float32), valido


def veic_para_img(x, y, g):
    """Ponto no chão (x à frente da câmara, y à esquerda) -> píxel (u, v)."""
    zc = x * g["c"] + g["h"] * g["s"]
    return g["cx"] - g["f"] * y / zc, g["cy"] + g["f"] * (g["h"] * g["c"] - x * g["s"]) / zc


def img_para_chao(u, v, g):
    """Inverso de veic_para_img: píxel (u, v) -> ponto no chão (x à frente, y à esq.).
    Só faz sentido abaixo do horizonte (v > y_h). Devolve (x, y) em metros."""
    t = (v - g["cy"]) / g["f"]
    den = t * g["c"] + g["s"]
    if den <= 1e-6:
        return float("inf"), 0.0
    x = g["h"] * (g["c"] - t * g["s"]) / den
    zc = x * g["c"] + g["h"] * g["s"]
    return float(x), float(-(u - g["cx"]) * zc / g["f"])


def lateral_da_linha(seg, x, g):
    """Posição lateral (m, + = esquerda da imagem) de uma linha detetada, à distância x."""
    rn = (g["x_far"] - x) / max(g["x_far"] - g["x_near"], 1e-6)
    rn = min(max(rn, 0.0), 1.0)                 # fora da vista de cima: prolonga a direito
    return g["lado"] - float(np.polyval(seg["coef"], rn)) / g["ppm"]


def bev_para_img(r, c, g):
    """Píxel (linha r, coluna c) da vista de cima -> píxel (u, v) da imagem."""
    x = g["x_far"] - (r / (g["bh"] - 1.0)) * (g["x_far"] - g["x_near"])
    y = g["lado"] - c / g["ppm"]
    return veic_para_img(x, y, g)


def curva_para_img(coef, rn_top, g, n=40):
    """Parábola da vista de cima (col = a·rn² + b·rn + c) -> pontos na imagem."""
    rn = np.linspace(max(rn_top, 0.0), 1.0, n)
    col = np.polyval(coef, rn)
    x = g["x_far"] - rn * (g["x_far"] - g["x_near"])
    y = g["lado"] - col / g["ppm"]
    u, v = veic_para_img(x, y, g)
    ok = (u > -g["W"]) & (u < 2 * g["W"]) & (v >= 0) & (v <= g["H"])
    return [[float(a), float(b)] for a, b in zip(u[ok], v[ok])]


# ================================ detetor ================================
class DetetorLinhas:
    def __init__(self, par=None):
        self.p = dict(PARAMS)
        if par:
            self.p.update(par)
        self._retas = _DetetorRetas()          # só para estimar o ponto de fuga
        self._amostras_h = deque(maxlen=120)
        self._k = 0
        self._estado = {"esq": None, "dir": None}    # (a, b, c, rn_top) suavizados
        self._falta = {"esq": 0, "dir": 0}
        self._pendente = {"esq": [], "dir": []}      # posições "suspeitas" seguidas
        n = self.p["confirmar_n"]
        self._vistos = {"esq": deque(maxlen=n), "dir": deque(maxlen=n)}
        self._tipo = {"esq": deque(maxlen=15), "dir": deque(maxlen=15)}
        self._tipo_atual = {"esq": None, "dir": None}
        self._mostrada = {"esq": False, "dir": False}   # histerese do aparecer/desaparecer
        self._rn_top = {"esq": None, "dir": None}    # extensão desenhada (suavizada à parte)
        self._clahe = cv2.createCLAHE(clipLimit=self.p["clahe"], tileGridSize=(4, 4))
        self._marcas = None
        self._hist_marc = {}
        self.debug = {}

    # ------------------------------------------------------------------
    def detetar(self, frame, excluir=()):
        p = self.p
        H, W = frame.shape[:2]
        y_h = self._horizonte(frame, excluir, H)
        chave = chave_ipm(W, H, y_h, p)
        g, mx, my, valido = geometria(chave)
        bw = p["bev_w"]

        # --- 1) vista de cima ---
        bev = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        val = valido.copy()
        if len(excluir):                       # carros/peões não são linhas
            m = np.zeros((H, W), np.uint8)
            for bb in excluir:
                x, y, bw_, bh_ = [int(v) for v in bb]
                cv2.rectangle(m, (x - 4, y - 4), (x + bw_ + 4, y + bh_ + 4), 255, -1)
            val &= cv2.remap(m, mx, my, cv2.INTER_NEAREST) == 0
        if val.sum() < 500:
            return self._saida(chave, g, {}, [], y_h)

        # --- 2) marcas por RIDGE (como o segmentLaneMarkerRidge) ---
        # Dois canais, cada um com contraste local (CLAHE):
        #  - BRANCAS em min(R,G,B): a tinta branca tem os 3 canais altos, mas o
        #    reflexo do sol no asfalto é quente (pouco azul) -> desaparece;
        #  - AMARELAS em (R+G)/2 - B ("amarelidão").
        B, G, R = (bev[:, :, i].astype(np.int16) for i in range(3))
        branco = self._clahe.apply(np.minimum(np.minimum(R, G), B).astype(np.uint8))
        amarelo = np.clip((R + G) // 2 - B, 0, 255).astype(np.uint8)
        tau = max(2, int(round(p["marca"] * bw)))
        r = np.maximum(_ridge(branco.astype(np.float32), tau),
                       _ridge(self._clahe.apply(amarelo).astype(np.float32), tau))
        dentro = cv2.erode(val.astype(np.uint8), np.ones((3, 2 * tau + 1), np.uint8)) > 0
        # contraste "a sério" (sem CLAHE): a tinta é mais clara que o asfalto à volta;
        # o CLAHE sozinho faz de fissuras, juntas e calçada "marcas"
        self._bruto = (np.minimum(np.minimum(R, G), B).astype(np.float32), amarelo.astype(np.float32))
        bruto = np.maximum(_ridge(self._bruto[0], tau), _ridge(self._bruto[1], tau))
        marcas = (r >= p["ridge_min"]) & (bruto >= p["ridge_bruto_min"]) & dentro
        _, lab, st, _ = cv2.connectedComponentsWithStats(marcas.astype(np.uint8), 8)
        manter = st[:, cv2.CC_STAT_AREA] >= p["area_min"]
        manter[0] = False
        marcas = manter[lab]
        # passadeiras, linhas de paragem (e lombas pintadas) NÃO são linhas da faixa:
        # as suas barras saem das marcas antes de procurar as linhas
        marc = self._marcacoes(branco, val, g) if p["marcacoes"] else []
        # (todas as vistas NESTA imagem, mesmo antes da confirmação temporal do desenho)
        ppm_r = (p["bev_h"] - 1.0) / max(g["x_far"] - g["x_near"], 1e-6)
        mr = int(0.6 * ppm_r)
        for _, r0, r1, _, _ in getattr(self, "_marc_brutas", []):
            marcas[max(0, r0 - mr):r1 + mr, :] = False
        self._marcas = marcas

        # --- 3) candidatas: histograma + janelas deslizantes + RANSAC ---
        cand = self._candidatas(marcas)

        # --- 4)+5) filtros da aula e tipo de linha ---
        fortes = _filtrar(cand, p)
        tol = max(2, int(0.02 * bw))
        marcas_dil = cv2.dilate(marcas.astype(np.uint8), np.ones((1, 2 * tol + 1), np.uint8)) > 0
        self._marcas_dil = marcas_dil
        for c in fortes:
            c["tipo"], c["cobertura"] = _tipo_linha(c, marcas_dil, marcas, g, p, self._bruto)
        fortes = [c for c in fortes if c["tipo"]]          # sem estrutura de tinta: não é linha

        # --- 6) faixa do carro: mais próxima à esquerda e à direita ---
        prev = {l: (float(np.polyval(e[:3], 1.0)) if e is not None else None)
                for l, e in self._estado.items()}
        ego = _faixa_do_carro(fortes, p, prev)
        outras = [c for c in fortes if c is not ego.get("esq") and c is not ego.get("dir")]
        self.debug = dict(n_marcas=int(marcas.sum()), n_cand=len(cand),
                          n_fortes=len(fortes), horizonte=y_h)
        out = self._saida(chave, g, ego, outras, y_h)
        # --- 8) outras marcações no pavimento: passadeiras e linhas de paragem ---
        if p["marcacoes"]:
            out["marcacoes"] = marc
        # --- 9) sem linhas: divisória estimada a meio da estrada ---
        out["divisoria"] = self._divisoria(frame, excluir, g, out["n"] == 0) if p["divisoria"] else None
        return out

    def _marcacoes(self, branco, val, g):
        """Passadeira (zebra) e linha de paragem, na vista de cima.
        Na vista de cima as escalas são métricas (aprox.), por isso as regras são em m:
          - passadeira: numa mesma linha, >= 3 barras claras de 0.3-0.9 m de largura,
            espaçadas com regularidade, ao longo de >= 1.2 m de profundidade;
          - linha de paragem: barra clara TRANSVERSAL >= 1.6 m de comprimento e
            0.15-0.7 m de espessura.
        Confirmação temporal (2 das últimas 4 frames) para não piscar."""
        bh, bw = branco.shape
        ppm = g["ppm"]                                             # px/m (lateral)
        ppm_r = (bh - 1.0) / max(g["x_far"] - g["x_near"], 1e-6)   # px/m (profundidade)
        dentro = cv2.erode(val.astype(np.uint8), np.ones((3, 5), np.uint8)) > 0
        vals = branco[dentro]
        achados = []
        if vals.size > 500:
            thr = max(150.0, float(np.percentile(vals, 85)), float(vals.mean() + 1.5 * vals.std()))
            claro = (branco >= thr) & dentro
            claro = cv2.morphologyEx(claro.astype(np.uint8), cv2.MORPH_OPEN,
                                     np.ones((3, 3), np.uint8)) > 0
            # ---- passadeira: barras longitudinais repetidas ----
            lmin, lmax = 0.3 * ppm, 0.9 * ppm
            linha_zebra = np.zeros(bh, bool)
            extremos = {}
            # troços claros de todas as linhas de uma vez (inícios e fins vêm pela mesma ordem)
            d = np.diff(np.pad(claro.astype(np.int8), ((0, 0), (1, 1))), axis=1)
            ri, ci = np.nonzero(d == 1)
            _, cf = np.nonzero(d == -1)
            ok = ((cf - ci) >= lmin) & ((cf - ci) <= lmax)
            ri, ci, cf = ri[ok], ci[ok], cf[ok]
            for r in np.flatnonzero(np.bincount(ri, minlength=bh) >= 3):
                sel = ri == r
                cen = (ci[sel] + cf[sel]) / 2.0
                gaps = np.diff(cen)
                if gaps.min() > 0.6 * ppm and gaps.std() < 0.35 * gaps.mean():
                    linha_zebra[r] = True
                    extremos[r] = (int(ci[sel].min()), int(cf[sel].max()))
            for r0, r1 in _corridas(linha_zebra, int(0.25 * ppm_r)):
                if r1 - r0 >= 1.2 * ppm_r:
                    cs = [extremos[r] for r in range(r0, r1) if r in extremos]
                    c0 = int(np.median([c[0] for c in cs]))
                    c1 = int(np.median([c[1] for c in cs]))
                    achados.append(("passadeira", r0, r1, c0, c1))
            # ---- linha de paragem: barra transversal ----
            comp = cv2.morphologyEx(claro.astype(np.uint8), cv2.MORPH_OPEN,
                                    np.ones((1, max(3, int(1.6 * ppm))), np.uint8)) > 0
            linhas_t = comp.sum(1) >= 1.6 * ppm
            for r0, r1 in _corridas(linhas_t, 1):
                esp = (r1 - r0) / ppm_r
                perto_zebra = any(a[0] == "passadeira" and a[1] - 5 <= r0 <= a[2] + 5
                                  for a in achados)
                if 0.15 <= esp <= 0.7 and not perto_zebra:
                    cols = np.flatnonzero(comp[r0:r1].any(0))
                    achados.append(("linha_paragem", r0, r1, int(cols.min()), int(cols.max())))
        # várias barras transversais seguidas = passadeira vista de lado (as barras de
        # uma zebra também são transversais quando a câmara olha na diagonal)
        barras = sorted([a for a in achados if a[0] == "linha_paragem"], key=lambda a: a[1])
        if len(barras) >= 2 and (barras[-1][1] - barras[0][2]) <= 4.0 * ppm_r:
            achados = [a for a in achados if a[0] != "linha_paragem"]
            achados.append(("passadeira", barras[0][1], barras[-1][2],
                            min(a[3] for a in barras), max(a[4] for a in barras)))
        self._marc_brutas = list(achados)
        saida = []
        for tipo in ("passadeira", "linha_paragem"):
            h = self._hist_marc.setdefault(tipo, deque(maxlen=4))
            atual = [a for a in achados if a[0] == tipo]
            h.append(bool(atual))
            if atual and sum(h) >= 2:
                for _, r0, r1, c0, c1 in atual:
                    cant = [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]
                    pts = [[float(v) for v in bev_para_img(r, c, g)] for r, c in cant]
                    dist = g["x_far"] - (r1 / (bh - 1.0)) * (g["x_far"] - g["x_near"])
                    saida.append(dict(tipo=tipo, pts=pts, dist=float(dist),
                                      bev=(int(r0), int(r1), int(c0), int(c1))))
        return saida

    # ------------------------------------------------------------------
    def _tracos(self, coef, rn_top, g):
        """Troços REAIS de uma tracejada (onde há tinta ao longo da curva), em pontos da
        imagem: o desenho mostra só os traços que lá estão, nunca um tracejado inventado."""
        m = getattr(self, "_marcas_dil", None)
        if m is None:
            return []
        bh, bw = m.shape
        rows = np.arange(int(max(rn_top, 0.0) * (bh - 1)), bh)
        cols = np.round(np.polyval(coef, rows / (bh - 1.0))).astype(int)
        ok = (cols >= 0) & (cols < bw)
        cob = np.zeros(rows.shape, bool)
        cob[ok] = m[rows[ok], cols[ok]]
        ppm_r = (bh - 1.0) / max(g["x_far"] - g["x_near"], 1e-6)
        out = []
        for a, b in _corridas(cob, int(0.25 * ppm_r)):
            if b - a >= 0.4 * ppm_r:
                rn = rows[a:b] / (bh - 1.0)
                out.append(_troco_para_img(coef, float(rn.min()), float(rn.max()), g))
        return out

    def _tipo_estavel(self, lado):
        """Tipo com histerese: só troca quando o outro tipo tem >= 70% das últimas 15
        imagens (uma tracejada com traços longos parece contínua numa ou noutra imagem)."""
        tipos = list(self._tipo[lado]) or ["continua"]
        atual = self._tipo_atual[lado]
        if atual is None:
            atual = max(set(tipos), key=tipos.count)
        else:
            outro = "continua" if atual == "tracejada" else "tracejada"
            if len(tipos) >= 6 and tipos.count(outro) >= 0.7 * len(tipos):
                atual = outro
        self._tipo_atual[lado] = atual
        return atual

    def _divisoria(self, frame, excluir, g, precisa):
        """Sem linhas pintadas: divisória ESTIMADA a meio da estrada.
        Numa vista de cima LARGA (±estrada_lado m) segmenta-se o asfalto pela cor e pela
        textura (a calçada, a relva e os passeios têm outra cor ou mais textura), a partir
        do asfalto logo atrás do carro; as bermas são os limites dessa região em cada
        distância; a divisória é o meio entre as duas (ou, só com a berma do nosso lado,
        uma largura de faixa para dentro). Suavizada e confirmada no tempo."""
        p = self.p
        st = getattr(self, "_div", None)
        if st is None:
            st = self._div = dict(y=None, vistos=deque(maxlen=6), falta=0, ativa=False)
        y_med = _bermas(frame, excluir, g, p)      # sempre (é barato): pronta quando faltarem linhas
        xs = _XS_DIV(g, p)
        st["vistos"].append(y_med is not None)
        if y_med is not None:
            st["y"] = y_med if st["y"] is None else 0.15 * y_med + 0.85 * st["y"]
            st["falta"] = 0
        else:
            st["falta"] += 1
            if st["falta"] > 8:
                st["y"] = None
        # histerese: aparece depois de vista em >= 3 das últimas 6 imagens e, a partir daí,
        # mantém-se (a esbater) enquanto houver estimativa — não liga e desliga
        if st["y"] is None:
            st["ativa"] = False
        elif sum(st["vistos"]) >= 3:
            st["ativa"] = True
        if not precisa or not st["ativa"]:
            return None
        # curva na imagem + coeficientes no referencial da vista de cima normal (para a
        # lateral_da_linha e para a faixa de cada agente)
        pts = [veic_para_img(float(x), float(y), g) for x, y in zip(xs, st["y"])]
        pts = [[float(u), float(v)] for u, v in pts if np.isfinite(u) and np.isfinite(v)]
        rn = (g["x_far"] - xs) / max(g["x_far"] - g["x_near"], 1e-6)
        col = (g["lado"] - st["y"]) * g["ppm"]
        coef = np.polyfit(rn, col, 2)
        if len(pts) < 2:
            return None
        return dict(pts=pts, tipo="estimada", coef=[float(v) for v in coef],
                    rn_top=float(max(rn.min(), 0.0)), estimada=True,
                    confianca=float(1.0 - st["falta"] / 9.0))

    # ------------------------------------------------------------------
    def _horizonte(self, frame, excluir, H):
        """Horizonte = mediana do ponto de fuga das linhas retas próximas."""
        p = self.p
        if p["horizonte"] is not None:
            return p["horizonte"] * H
        fixo = (p.get("horizonte_conhecido") or {}).get((frame.shape[1], H))
        if fixo is not None:
            return fixo * H
        self._k += 1
        passo = 3 if len(self._amostras_h) < 60 else 15
        if self._k % passo == 1:
            self._retas.detetar(frame, excluir)
            f = self._retas.debug.get("escolha", {})
            fe, fd = f.get("esq"), f.get("dir")
            if fe and fd and abs(fe[0] - fd[0]) > 1e-3:
                y = (fd[1] - fe[1]) / (fe[0] - fd[0])        # interseção das 2 retas
                if 0.25 * H < y < 0.72 * H:
                    self._amostras_h.append(y)
        if len(self._amostras_h) >= 8:
            y_h = float(np.median(self._amostras_h))
        else:
            y_h = p["horizonte_ini"] * H
        return round(y_h / 2.0) * 2.0                         # evita refazer mapas por 1 px

    def _candidatas(self, marcas):
        p = self.p
        bh, bw = marcas.shape
        ys, xs = np.nonzero(marcas)
        if len(xs) < p["min_pts"]:
            return []
        # sementes: picos do histograma de colunas (mais peso junto ao carro)
        hist = np.bincount(xs, weights=0.3 + 0.7 * ys / (bh - 1), minlength=bw)
        k = max(3, int(0.015 * bw) | 1)
        hist = cv2.GaussianBlur(hist.reshape(1, -1).astype(np.float32), (k, 1), 0).ravel()
        lim = max(4.0, 0.12 * float(hist.max()))
        picos = [i for i in range(1, bw - 1)
                 if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1] and hist[i] >= lim]
        picos.sort(key=lambda i: -hist[i])
        dist = p["duplicado"] * bw
        sementes = []
        for i in picos:
            if all(abs(i - j) > dist for j in sementes):
                sementes.append(i)
            if len(sementes) >= 2 * p["max_linhas"]:
                break
        for lado in ("esq", "dir"):              # seguimento: onde estava a linha antes
            e = self._estado[lado]
            if e is not None:
                col = float(np.polyval(e[:3], 1.0))
                if 0 <= col < bw and all(abs(col - j) > dist / 2 for j in sementes):
                    sementes.append(int(col))

        cand = []
        for s0 in sementes:
            rr, cc = _janelas(ys, xs, s0, bh, p["n_janelas"], p["janela"] * bw)
            if len(rr) < p["min_pts"]:
                continue
            res = _ransac_parabola(rr / (bh - 1.0), cc.astype(float), p["tol_ransac"] * bw)
            if res is None:
                continue
            coef, inl = res
            rn = rr[inl] / (bh - 1.0)
            cand.append(dict(coef=coef, forca=int(inl.sum()),
                             rn_min=float(rn.min()), rn_max=float(rn.max())))
        return cand

    def _saida(self, chave, g, ego, outras, y_h):
        """Suavização + confirmação temporal e pontos na imagem."""
        p = self.p
        bw = p["bev_w"]
        out = {"n": 0, "esq": None, "dir": None, "outras": [], "chave": chave,
               "horizonte": y_h}
        for lado in ("esq", "dir"):
            c = ego.get(lado)
            em_espera = False
            if c is not None:
                novo = np.array(list(c["coef"]) + [c["rn_min"]], float)
                ant = self._estado[lado]
                col = float(np.polyval(novo[:3], 1.0))
                salto = ant is not None and abs(col - np.polyval(ant[:3], 1.0)) > p["salto_max"] * bw
                if salto:
                    # posição nova longe da anterior: só se aceita se se mantiver
                    # (mudança de faixa real); uma deteção isolada é ignorada
                    pend = self._pendente[lado]
                    if pend and abs(col - pend[-1]) > p["salto_max"] * bw:
                        pend.clear()                  # nem sequer é consistente consigo
                    pend.append(col)
                    if len(pend) >= p["confirmar_salto"]:
                        pend.clear()
                        self._tipo[lado].clear()
                    else:
                        c, em_espera = None, True     # mantém a anterior por agora
                else:
                    self._pendente[lado].clear()
                    if ant is not None:
                        novo = p["alpha"] * novo + (1 - p["alpha"]) * ant
            self._vistos[lado].append(c is not None or em_espera)
            if c is not None:
                self._estado[lado] = novo
                self._falta[lado] = 0
                self._tipo[lado].append(c["tipo"])
            else:
                self._falta[lado] += 1
                if self._falta[lado] > p["memoria"]:
                    self._estado[lado] = None
                    self._tipo[lado].clear()
                    self._tipo_atual[lado] = None
                    self._rn_top[lado] = None
                    self._mostrada[lado] = False
            e = self._estado[lado]
            # aparece depois de confirmada (k de n imagens); a partir daí fica (a esbater)
            # enquanto houver estado, em vez de piscar quando falha umas imagens
            if e is not None and sum(self._vistos[lado]) >= p["confirmar_k"]:
                self._mostrada[lado] = True
            if e is not None and self._mostrada[lado]:
                tipo = self._tipo_estavel(lado)
                # extensão: cresce depressa (vê-se mais longe) e encolhe devagar
                rt = self._rn_top[lado]
                rt = e[3] if rt is None else (0.5 * e[3] + 0.5 * rt if e[3] < rt else 0.1 * e[3] + 0.9 * rt)
                self._rn_top[lado] = rt
                e = np.array(list(e[:3]) + [rt])
                pts = curva_para_img(e[:3], e[3] - 0.05, g)
                if len(pts) >= 2:
                    out[lado] = dict(pts=pts, tipo=tipo, coef=[float(v) for v in e[:3]],
                                     rn_top=float(e[3]),
                                     # sem medida há algumas imagens: a linha vai esbatendo
                                     confianca=float(1.0 - self._falta[lado] / (p["memoria"] + 1.0)),
                                     cobertura=float(c["cobertura"]) if c else None,
                                     tracos=self._tracos(e[:3], e[3], g) if tipo == "tracejada" else None)
        for c in outras:
            pts = curva_para_img(c["coef"], c["rn_min"], g, n=25)
            if len(pts) >= 2:
                out["outras"].append(dict(pts=pts, tipo=c["tipo"],
                                          coef=[float(v) for v in c["coef"]],
                                          rn_top=c["rn_min"],
                                          tracos=self._tracos(c["coef"], c["rn_min"], g)
                                          if c["tipo"] == "tracejada" else None))
        out["n"] = int(out["esq"] is not None) + int(out["dir"] is not None)
        return out


# ============================ funções auxiliares ============================
def _corridas(mask, tol):
    """Troços contíguos de True num vetor (tolera falhas até tol) -> [(ini, fim)]."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out, ini, ant = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - ant > tol + 1:
            out.append((int(ini), int(ant) + 1))
            ini = i
        ant = i
    out.append((int(ini), int(ant) + 1))
    return out


def _desloca(img, d):
    """img deslocada d colunas (d>0: valor do vizinho à esquerda)."""
    out = np.zeros_like(img)
    if d > 0:
        out[:, d:] = img[:, :-d]
    else:
        out[:, :d] = img[:, -d:]
    return out


def _ridge(img, tau):
    """Ridge horizontal simétrico: a marca é mais clara que AMBOS os lados à
    distância tau; o termo |esq-dir| anula as arestas em degrau (sombras, bermas)."""
    esq, dir_ = _desloca(img, tau), _desloca(img, -tau)
    return 2 * img - esq - dir_ - np.abs(esq - dir_)


def _janelas(ys, xs, c0, bh, n_jan, meia):
    """Segue uma linha de baixo para cima com janelas deslizantes."""
    alt = bh / n_jan
    c, deriva = float(c0), 0.0
    sel_total = np.zeros(len(xs), bool)
    for k in range(n_jan):
        r1 = bh - k * alt
        r0 = r1 - alt
        sel = (ys >= r0) & (ys < r1) & (np.abs(xs - c) < meia)
        if sel.sum() >= 4:
            novo = float(xs[sel].mean())
            deriva = 0.5 * deriva + 0.5 * (novo - c)
            c = novo
            sel_total |= sel
        else:
            c += deriva                      # continua na direção que trazia (tracejadas)
    return ys[sel_total], xs[sel_total]


def _ransac_parabola(rn, col, tol, it=60):
    """col = a·rn² + b·rn + c robusto a pontos errados. Amostragem estratificada
    (1 ponto de cada terço da altura) e TODAS as hipóteses avaliadas de uma vez."""
    if len(rn) < 6:
        return None
    tercos = np.array_split(np.argsort(rn), 3)
    if any(len(t) == 0 for t in tercos):
        return None
    idx = np.stack([t[_RNG.integers(len(t), size=it)] for t in tercos], 1)   # (it, 3)
    r3, c3 = rn[idx], col[idx]
    sep = np.min(np.abs(np.stack([r3[:, 0] - r3[:, 1], r3[:, 1] - r3[:, 2],
                                  r3[:, 0] - r3[:, 2]], 1)), 1)
    ok = sep >= 0.03                          # pontos quase à mesma altura: mal condicionado
    if not ok.any():
        return None
    r3, c3 = r3[ok], c3[ok]
    V = np.stack([r3 ** 2, r3, np.ones_like(r3)], 2)                          # (k, 3, 3)
    coefs = np.linalg.solve(V, c3[..., None])[..., 0]                        # (k, 3)
    pred = coefs[:, :1] * rn ** 2 + coefs[:, 1:2] * rn + coefs[:, 2:]        # (k, n)
    inl = np.abs(pred - col) < tol
    cont = inl.sum(1)
    j = int(np.argmax(cont))
    if cont[j] < 6:
        return None
    coef = np.polyfit(rn[inl[j]], col[inl[j]], 2)                            # refinar
    return coef, np.abs(np.polyval(coef, rn) - col) < tol


def _filtrar(cand, p):
    """Validação da aula: curvatura, comprimento mínimo, força >= 20% da maior,
    e remoção de duplicados (a mesma linha encontrada por duas sementes)."""
    bw = p["bev_w"]
    bons = []
    for c in cand:
        a, b, _ = c["coef"]
        ext = c["rn_max"] - c["rn_min"]
        # curvatura medida só no troço onde a linha existe (a·ext² = desvio
        # quadrático nesse troço); extrapolar para a altura toda exagerava-a
        if abs(a) * ext ** 2 > p["curv_max"] * bw or abs(2 * a + b) > p["decl_max"] * bw:
            continue
        if ext < p["min_comp"] or c["forca"] < p["min_pts"]:
            continue
        bons.append(c)
    if not bons:
        return []
    fmax = max(c["forca"] for c in bons)
    bons = [c for c in bons if c["forca"] >= p["forca_rel"] * fmax]
    bons.sort(key=lambda c: -c["forca"])
    rn = np.linspace(0.3, 1.0, 15)
    unicos = []
    for c in bons:
        if all(np.mean(np.abs(np.polyval(c["coef"], rn) - np.polyval(u["coef"], rn)))
               > p["duplicado"] * bw for u in unicos):
            unicos.append(c)
        if len(unicos) >= p["max_linhas"]:
            break
    return unicos


def _troco_para_img(coef, rn0, rn1, g, n=8):
    rn = np.linspace(rn0, rn1, n)
    col = np.polyval(coef, rn)
    x = g["x_far"] - rn * (g["x_far"] - g["x_near"])
    y = g["lado"] - col / g["ppm"]
    return [[float(u), float(v)] for u, v in (veic_para_img(a, b, g) for a, b in zip(x, y))]


def _XS_DIV(g, p):
    """Distâncias (m) a que se descreve a divisória estimada."""
    return np.linspace(g["x_near"], min(p["estrada_longe"], max(g["x_far"], g["x_near"] + 4)), 12)


@lru_cache(maxsize=8)
def _ipm_largo(chave, lado, longe, bw=160, bh=160):
    """Vista de cima LARGA (±lado m, até 'longe' m) com o mesmo modelo de câmara."""
    g = geometria(chave)[0]
    xs = np.linspace(longe, g["x_near"], bh)
    ys = np.linspace(lado, -lado, bw)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    zc = xx * g["c"] + g["h"] * g["s"]
    u = g["cx"] - g["f"] * yy / zc
    v = g["cy"] + g["f"] * (g["h"] * g["c"] - xx * g["s"]) / zc
    valido = (u >= 0) & (u <= g["W"] - 1) & (v >= 0) & (v <= g["H"] - 1) & (zc > 0)
    return u.astype(np.float32), v.astype(np.float32), valido, xs, ys


def _bermas(frame, excluir, g, p):
    """Lateral (m) da divisória estimada às distâncias _XS_DIV, ou None."""
    lado, longe = p["estrada_lado"], p["estrada_longe"]
    mx, my, valido, xs, ys = _ipm_largo(tuple(chave_ipm(g["W"], g["H"], g["y_h"], dict(
        hfov=math.degrees(2 * math.atan((g["W"] / 2.0) / g["f"])), altura=g["h"],
        v_base=p["v_base"], frac_topo=p["frac_topo"], bev_w=g["bw"], bev_h=g["bh"]))), lado, longe)
    bev = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bh, bw = valido.shape
    carros = np.zeros(valido.shape, bool)
    if len(excluir):
        m = np.zeros(frame.shape[:2], np.uint8)
        for bb in excluir:
            x, y, w_, h_ = [int(v) for v in bb]
            cv2.rectangle(m, (x - 4, y - 4), (x + w_ + 4, y + h_ + 4), 255, -1)
        carros = cv2.remap(m, mx, my, cv2.INTER_NEAREST) > 0
    # o chão logo atrás do carro, na imagem: a calçada tem juntas a cada ~10 cm, que a vista
    # de cima (9 cm/píxel, desfocada) apaga; aqui veem-se bem
    H = frame.shape[0]
    W = frame.shape[1]
    roi = cv2.cvtColor(frame[int(0.8 * H):, int(0.35 * W):int(0.65 * W)], cv2.COLOR_BGR2LAB)[:, :, 0]
    roi = roi.astype(np.float32)
    mu = cv2.blur(roi, (7, 7))
    if float(np.median(np.sqrt(np.maximum(cv2.blur(roi * roi, (7, 7)) - mu * mu, 0)))) > p["textura_chao_max"]:
        return None                         # o carro está em calçada / terra: não há estrada
    lab = cv2.cvtColor(cv2.GaussianBlur(bev, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, Bc = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    # asfalto de referência: logo atrás do carro, ao centro
    ref = np.zeros(valido.shape, bool)
    ref[int(0.8 * bh):, int(0.42 * bw):int(0.58 * bw)] = True
    ref &= valido & ~carros
    if ref.sum() < 80:
        return None
    Lr, Ar, Br = (float(np.median(c[ref])) for c in (L, A, Bc))
    Ls = max(6.0, 1.4826 * float(np.median(np.abs(L[ref] - Lr))))
    # textura local (desvio padrão de L numa janela 7x7): calçada e relva têm mais
    mu = cv2.blur(L, (7, 7))
    tex = np.sqrt(np.maximum(cv2.blur(L * L, (7, 7)) - mu * mu, 0))
    tex_r = float(np.median(tex[ref]))
    if tex_r > p["textura_asfalto_max"]:   # o que está atrás do carro não é asfalto liso
        return None                         #   (calçada, terra, relva): não se estima nada
    estrada = (np.abs(L - Lr) < max(22.0, 3.5 * Ls)) & (np.abs(A - Ar) < 7) & \
              (np.abs(Bc - Br) < 9) & (tex < max(9.0, 2.2 * tex_r)) & valido
    estrada = cv2.morphologyEx(estrada.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    estrada = cv2.morphologyEx(estrada, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    n, lbl = cv2.connectedComponents(estrada.astype(np.uint8), connectivity=4)
    if n < 2:
        return None
    ids, cont = np.unique(lbl[ref & estrada], return_counts=True)
    ids, cont = ids[ids > 0], cont[ids > 0]
    if ids.size == 0:
        return None
    reg = lbl == ids[np.argmax(cont)]
    pe, pd, pm = [], [], []             # (x, y) das bermas esq./dir. da imagem e do meio
    c0 = bw // 2
    for r in range(bh - 1, int(0.15 * bh), -2):
        lin = reg[r]
        if not lin[c0]:
            cols = np.flatnonzero(lin)
            if cols.size == 0:
                continue
            c0r = int(cols[np.argmin(np.abs(cols - c0))])
        else:
            c0r = c0
        a = c0r
        while a > 0 and lin[a - 1]:
            a -= 1
        b = c0r
        while b < bw - 1 and lin[b + 1]:
            b += 1
        # só conta como berma se do outro lado há chão VISTO e não um carro
        ok_e = a > 2 and valido[r, a - 3] and not carros[r, max(a - 4, 0):a].any()
        ok_d = b < bw - 3 and valido[r, b + 3] and not carros[r, b + 1:b + 5].any()
        if ok_e:
            pe.append((xs[r], ys[a]))
        if ok_d:
            pd.append((xs[r], ys[b]))
        if ok_e and ok_d and p["largura_estrada"][0] <= ys[a] - ys[b] <= p["largura_estrada"][1]:
            pm.append((xs[r], (ys[a] + ys[b]) / 2.0))      # o meio da estrada nesta distância
    xq = _XS_DIV(g, p)

    def ajusta(pts, a_max=0.25):
        if len(pts) < 10:
            return None
        P = np.array(pts)
        best = None
        rng = np.random.default_rng(1)
        for _ in range(40):                      # reta robusta y = a·x + b (RANSAC)
            i, j = rng.choice(len(P), 2, replace=False)
            if abs(P[i, 0] - P[j, 0]) < 1.0:
                continue
            a = (P[j, 1] - P[i, 1]) / (P[j, 0] - P[i, 0])
            if abs(a) > a_max:                   # segue (mais ou menos) a direção da estrada
                continue
            bb = P[i, 1] - a * P[i, 0]
            inl = np.abs(P[:, 1] - (a * P[:, 0] + bb)) < 0.35
            if best is None or inl.sum() > best[0]:
                best = (inl.sum(), inl)
        if best is None or best[0] < max(8, 0.5 * len(P)):
            return None
        a, bb = np.polyfit(P[best[1], 0], P[best[1], 1], 1)
        return (a * xq + bb) if abs(a) <= a_max else None

    # 1) as duas bermas vistas: o MEIO da estrada em cada distância. É robusto a um
    #    horizonte ligeiramente errado (as bermas abrem ou fecham por igual; o meio não)
    ym = ajusta(pm)
    if ym is not None:
        # Circula-se pela direita: a divisória fica à ESQUERDA do carro (y < 0 aqui, que é a
        # direita da imagem da câmara traseira). Se o "meio" cai debaixo do carro, a
        # segmentação parou antes da berma do outro lado (asfalto de outra cor, sombras,
        # carros) — então assume-se o carro a meio da sua via e põe-se a divisória meia via
        # à esquerda, com a direção medida.
        y6 = float(np.interp(6.0, xq, ym))
        if y6 > -p["divisoria_lado_min"]:
            ym = ym - (y6 + p["faixa_largura"] / 2.0)
        return ym
    # (só com uma berma não se estima: no vídeo real dava divisórias tortas)
    return None


def _tipo_linha(c, marcas_dil, marcas, g, p, bruto=None):
    """Contínua vs tracejada pela fração da extensão com marca (classifyLaneTypes).
    marcas_dil = máscara das marcas alargada na horizontal (tolerância)."""
    bh, bw = marcas_dil.shape
    rows = np.arange(int(c["rn_min"] * (bh - 1)), int(c["rn_max"] * (bh - 1)) + 1)
    cols = np.round(np.polyval(c["coef"], rows / (bh - 1.0))).astype(int)
    ok = (cols >= 0) & (cols < bw)
    if not ok.any():
        return None, 0.0
    rows, cols = rows[ok], cols[ok]
    v = marcas_dil[rows, cols]
    cob = float(v.mean())
    ppm_r = (bh - 1.0) / max(g["x_far"] - g["x_near"], 1e-6)
    # estrutura: a tinta aparece em TROÇOS com comprimento (traços de >= traco_min_m);
    # fissuras, calçada e reflexos dão pedacinhos soltos
    tracos = [b - a for a, b in _corridas(v, int(0.15 * ppm_r))]
    longos = sum(t for t in tracos if t >= p["traco_min_m"] * ppm_r)
    estrutura = longos / max(float(v.sum()), 1.0)
    # largura da TINTA: zona clara (acima de meia altura entre o asfalto e o pico) à volta da
    # linha, no canal sem CLAHE. Uma linha tem ~0,10-0,15 m; a borda de uma risca larga de
    # passadeira (~0,5 m) ou de uma lomba parece uma "marca" fina no ridge, mas a tinta é larga
    if bruto is not None:
        canal = bruto[0]
        larg = []
        lim = int(0.8 * g["ppm"])
        for r, c_ in zip(rows[v][::3], cols[v][::3]):
            a0, b0 = max(0, c_ - lim), min(bw, c_ + lim + 1)
            perfil = canal[r, a0:b0]
            k = int(np.argmax(perfil[max(0, c_ - a0 - 3):c_ - a0 + 4])) + max(0, c_ - a0 - 3)
            fundo = float(np.percentile(perfil, 20))
            meio = fundo + 0.5 * (float(perfil[k]) - fundo)
            if perfil[k] - fundo < 8:
                continue
            a, b = k, k
            while a > 0 and perfil[a - 1] >= meio:
                a -= 1
            while b < len(perfil) - 1 and perfil[b + 1] >= meio:
                b += 1
            larg.append(b - a + 1)
        if larg and np.median(larg) / g["ppm"] > p["largura_max_m"]:
            return None, cob
    # contraste real (sem CLAHE): a tinta é bem mais clara do que o asfalto dos dois lados
    if bruto is not None and v.any():
        off = max(2, int(0.35 * g["ppm"]))
        rr, cc = rows[v], cols[v]
        contr = -1e9
        for canal in bruto:
            dentro = canal[rr, cc]
            # contra o lado mais ESCURO: junto a um passeio ao sol (lado claro) a tinta
            # continua a destacar-se do asfalto; uma fissura ou terra não se destaca de nada
            lados = np.minimum(canal[rr, np.clip(cc - off, 0, bw - 1)],
                               canal[rr, np.clip(cc + off, 0, bw - 1)])
            contr = max(contr, float(np.median(dentro - lados)))
        if contr < p["contraste_min"]:
            return None, cob
        # dos DOIS lados há asfalto mais escuro: uma borda (passeio, calçada, estacionamento,
        # risca de passadeira) é clara de um lado e escura do outro
        offs = sorted({max(1, int(o * g["ppm"])) for o in (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)})
        simetrica = False
        for canal in bruto:
            dentro = canal[rr, cc]
            e = np.median([canal[rr, np.clip(cc - o, 0, bw - 1)] for o in offs], axis=0)
            d = np.median([canal[rr, np.clip(cc + o, 0, bw - 1)] for o in offs], axis=0)
            c_claro = float(np.median(dentro - np.maximum(e, d)))
            c_escuro = float(np.median(dentro - np.minimum(e, d)))
            if c_claro >= p["contraste_claro_min"] and c_claro >= p["simetria_min"] * c_escuro:
                simetrica = True
                break
        if not simetrica:
            return None, cob
    if cob >= p["cobertura_continua"]:
        return "continua", cob
    if cob >= p["cobertura_min"] and longos > 0 and estrutura >= p["estrutura_min"]:
        return "tracejada", cob
    return None, cob


def _faixa_do_carro(fortes, p, prev=None):
    """Linha forte mais próxima do eixo da câmara de cada lado (como a aula),
    exigindo uma largura de faixa plausível entre as duas.
    prev = {'esq': col, 'dir': col} onde estavam as linhas na frame anterior:
    primeiro procura-se cada linha PERTO de onde estava (seguimento) — assim, se a
    linha certa falhar numa frame, não se salta para a da faixa ao lado."""
    if not fortes:
        return {}
    bw = p["bev_w"]
    meio = (bw - 1) / 2.0
    for c in fortes:
        c["col_base"] = float(np.polyval(c["coef"], 1.0))     # junto ao carro
    lmin, lmax = p["largura_faixa"][0] * bw, p["largura_faixa"][1] * bw
    if prev:
        seg = {}
        for lado in ("esq", "dir"):
            if prev.get(lado) is None:
                continue
            perto = [c for c in fortes if abs(c["col_base"] - prev[lado]) <= p["seguir"] * bw]
            if perto:
                seg[lado] = min(perto, key=lambda c: abs(c["col_base"] - prev[lado]))
        if len(seg) == 2 and (seg["esq"] is seg["dir"] or
                              not lmin <= seg["dir"]["col_base"] - seg["esq"]["col_base"] <= lmax):
            seg = {}                                   # incoerente: escolhe de novo
        if len(seg) == 2:
            return seg
        if len(seg) == 1:                             # um lado seguido; o outro pela regra
            (lado, c0), = seg.items()
            if lado == "esq":
                outros = [d for d in fortes if lmin <= d["col_base"] - c0["col_base"] <= lmax]
                outros.sort(key=lambda d: d["col_base"])
                return {"esq": c0, "dir": outros[0]} if outros else seg
            outros = [e for e in fortes if lmin <= c0["col_base"] - e["col_base"] <= lmax]
            outros.sort(key=lambda e: -e["col_base"])
            return {"esq": outros[0], "dir": c0} if outros else seg
    esq = sorted([c for c in fortes if c["col_base"] < meio], key=lambda c: meio - c["col_base"])
    dir_ = sorted([c for c in fortes if c["col_base"] >= meio], key=lambda c: c["col_base"] - meio)
    for e in esq[:3]:                                  # par plausível mais interior
        for d in dir_[:3]:
            if lmin <= d["col_base"] - e["col_base"] <= lmax:
                return {"esq": e, "dir": d}
    # sem par plausível: fica só a mais forte das duas mais interiores
    lados = [x for x in (("esq", esq[0]) if esq else None,
                         ("dir", dir_[0]) if dir_ else None) if x]
    if not lados:
        return {}
    lado, c = max(lados, key=lambda x: x[1]["forca"])
    return {lado: c}


# =================== estimador do ponto de fuga (linhas retas) ===================
PARAMS_RETAS = dict(
    roi_topo=0.58, roi_base=0.93, roi_meia_topo=0.18,
    branco_l_min=50, branco_s_max=110, tophat_k=0.03, tophat_min=22,
    amar_h=(12, 38), amar_s_min=45, amar_l_min=30, tophat_amar_min=10,
    canny=(50, 150), hough_thr=18, hough_min_len=0.035, hough_max_gap=0.03,
    decl_min=0.30, decl_max=5.0, agrupar=0.06, suporte_min=0.05,
)


class _DetetorRetas:
    """Detetor simples de retas (esq./dir.) usado APENAS para estimar o ponto de
    fuga -> horizonte da vista de cima. (Era o detetor de linhas da versão anterior.)"""

    def __init__(self, par=None):
        self.p = dict(PARAMS_RETAS)
        if par:
            self.p.update(par)
        self.debug = {}

    def detetar(self, frame, excluir=()):
        p = self.p
        h, w = frame.shape[:2]
        y_topo, y_base = int(p["roi_topo"] * h), int(p["roi_base"] * h)
        roi = np.zeros((h, w), np.uint8)
        cx = w / 2
        cv2.fillPoly(roi, [np.array([[0, y_base], [w - 1, y_base],
                                     [cx + p["roi_meia_topo"] * w, y_topo],
                                     [cx - p["roi_meia_topo"] * w, y_topo]], np.int32)], 255)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        H, L, S = hls[:, :, 0], hls[:, :, 1], hls[:, :, 2]
        k = max(int(p["tophat_k"] * w) | 1, 5)
        tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        branco = (tophat >= p["tophat_min"]) & (L >= p["branco_l_min"]) & (S <= p["branco_s_max"])
        amarelo = ((H >= p["amar_h"][0]) & (H <= p["amar_h"][1]) & (S >= p["amar_s_min"]) &
                   (L >= p["amar_l_min"]) & (tophat >= p["tophat_amar_min"]))
        cor = cv2.dilate((branco | amarelo).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        for bb in excluir:
            x, y, bw, bh = [int(v) for v in bb]
            cv2.rectangle(roi, (x - 3, y - 3), (x + bw + 3, y + bh + 3), 0, -1)
        arestas = cv2.bitwise_and(cv2.Canny(cor, *p["canny"]), roi)
        segs = cv2.HoughLinesP(arestas, 1, np.pi / 180, p["hough_thr"],
                               minLineLength=max(int(p["hough_min_len"] * w), 8),
                               maxLineGap=max(int(p["hough_max_gap"] * w), 5))
        lados = {"esq": [], "dir": []}
        if segs is not None:
            for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):   # OpenCV 4 e 5
                if x2 == x1:
                    continue
                m = (y2 - y1) / (x2 - x1)
                if not (p["decl_min"] <= abs(m) <= p["decl_max"]):
                    continue
                xm, comp = (x1 + x2) / 2, np.hypot(x2 - x1, y2 - y1)
                if m < 0 and xm < cx + 0.1 * w:
                    lados["esq"].append((x1, y1, x2, y2, comp))
                elif m > 0 and xm > cx - 0.1 * w:
                    lados["dir"].append((x1, y1, x2, y2, comp))
        cand = {}
        for lado in ("esq", "dir"):
            grupos = _grupos_retas(lados[lado], y_base, cx, p["agrupar"] * w, p["suporte_min"] * w)
            cand[lado] = [f for f in (_ajustar_reta(gr) for gr in grupos[:3]) if f is not None]
        melhor = None
        for i, fe in enumerate(cand["esq"] + [None]):
            for j, fd in enumerate(cand["dir"] + [None]):
                if fe is None and fd is None:
                    continue
                if fe is not None and fd is not None and _retas_cruzam(fe, fd, y_topo, y_base):
                    continue
                chave = (-((fe is not None) + (fd is not None)),
                         (i if fe else 0) + (j if fd else 0),
                         -((fe[2] if fe else 0) + (fd[2] if fd else 0)))
                if melhor is None or chave < melhor[0]:
                    melhor = (chave, fe, fd)
        esc = {"esq": melhor[1], "dir": melhor[2]} if melhor else {"esq": None, "dir": None}
        self.debug = dict(escolha=esc)
        return esc


def _ajustar_reta(segs):
    if not segs:
        return None
    ys, xs, ws = [], [], []
    for x1, y1, x2, y2, c in segs:
        ys += [y1, y2]; xs += [x1, x2]; ws += [c, c]
    ys, xs, ws = np.array(ys, float), np.array(xs, float), np.array(ws, float)
    if np.ptp(ys) < 4:
        return None
    a, b = np.polyfit(ys, xs, 1, w=np.sqrt(ws))
    return float(a), float(b), float(ws.sum() / 2)


def _grupos_retas(segs, y_base, cx, dist_grupo, sup_min):
    if not segs:
        return []
    itens = sorted(((s[0] + (y_base - s[1]) / ((s[3] - s[1]) / (s[2] - s[0])), s) for s in segs),
                   key=lambda t: t[0])
    grupos, atual = [], [itens[0]]
    for it in itens[1:]:
        if abs(it[0] - atual[-1][0]) <= dist_grupo:
            atual.append(it)
        else:
            grupos.append(atual); atual = [it]
    grupos.append(atual)
    validos = [gr for gr in grupos if sum(t[1][4] for t in gr) >= sup_min]
    validos.sort(key=lambda gr: abs(np.median([t[0] for t in gr]) - cx))
    return [[t[1] for t in gr] for gr in validos]


def _retas_cruzam(fe, fd, y_topo, y_base, tol=2.0):
    xe_t, xd_t = fe[0] * y_topo + fe[1], fd[0] * y_topo + fd[1]
    xe_b, xd_b = fe[0] * y_base + fe[1], fd[0] * y_base + fd[1]
    return (xe_b >= xd_b) or (xe_t > xd_t + tol)
