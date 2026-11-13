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
     (isOfMinLength) e força >= 40% da maior (strongBoundaries)
  5) tipo de linha: CONTÍNUA / TRACEJADA ................ classifyLaneTypes
  6) faixa do carro = linha mais próxima à esq. e à dir. . leftEgoBoundary/rightEgoBoundary
  7) suavização + confirmação temporal (não tremem nem "piscam")

Sem calibração: a vista de cima é definida RELATIVAMENTE À IMAGEM — a linha de
baixo cobre a largura da imagem e o topo fica a uma fração do caminho até ao
horizonte (só a estrada próxima, onde o chão é plano). Todos os limiares são
frações da largura da vista de cima, por isso funcionam com qualquer câmara.
O único parâmetro crítico é o HORIZONTE: é ESTIMADO automaticamente pelo ponto
de fuga das linhas (mediana ao longo do tempo) — com ele certo, as linhas ficam
paralelas na vista de cima. Pode fixar-se em PARAMS['horizonte'].
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
    altura=1.2,           # altura da câmara (m): só dá a escala aproximada em metros
    horizonte=None,       # linha do horizonte (fração da altura); None = AUTOMÁTICO
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
    # --- tempo ---
    alpha=0.35,           # suavização (0 = nada muda, 1 = sem suavização)
    memoria=8,            # frames que uma linha "sobrevive" sem ser vista
    confirmar_n=6,        # um lado só aparece se foi visto em >= k das
    confirmar_k=3,        #   últimas n frames (corta "piscares")
    salto_max=0.12,       # salto maior que isto (fração) reinicia a suavização
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
        n = self.p["confirmar_n"]
        self._vistos = {"esq": deque(maxlen=n), "dir": deque(maxlen=n)}
        self._tipo = {"esq": deque(maxlen=9), "dir": deque(maxlen=9)}
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
        bh, bw = p["bev_h"], p["bev_w"]

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
        marcas = (r >= p["ridge_min"]) & dentro
        _, lab, st, _ = cv2.connectedComponentsWithStats(marcas.astype(np.uint8), 8)
        manter = st[:, cv2.CC_STAT_AREA] >= p["area_min"]
        manter[0] = False
        marcas = manter[lab]
        self._marcas = marcas

        # --- 3) candidatas: histograma + janelas deslizantes + RANSAC ---
        cand = self._candidatas(marcas)

        # --- 4)+5) filtros da aula e tipo de linha ---
        fortes = _filtrar(cand, p)
        tol = max(2, int(0.02 * bw))
        marcas_dil = cv2.dilate(marcas.astype(np.uint8), np.ones((1, 2 * tol + 1), np.uint8)) > 0
        for c in fortes:
            c["tipo"], c["cobertura"] = _tipo_linha(c, marcas_dil, p)

        # --- 6) faixa do carro: mais próxima à esquerda e à direita ---
        ego = _faixa_do_carro(fortes, p)
        outras = [c for c in fortes if c is not ego.get("esq") and c is not ego.get("dir")]
        self.debug = dict(n_marcas=int(marcas.sum()), n_cand=len(cand),
                          n_fortes=len(fortes), horizonte=y_h)
        out = self._saida(chave, g, ego, outras, y_h)
        # --- 8) outras marcações no pavimento: passadeiras e linhas de paragem ---
        if p["marcacoes"]:
            out["marcacoes"] = self._marcacoes(branco, val, g)
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
    def _horizonte(self, frame, excluir, H):
        """Horizonte = mediana do ponto de fuga das linhas retas próximas."""
        p = self.p
        if p["horizonte"] is not None:
            return p["horizonte"] * H
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
            self._vistos[lado].append(c is not None)
            if c is not None:
                novo = np.array(list(c["coef"]) + [c["rn_min"]], float)
                ant = self._estado[lado]
                salto = ant is not None and abs(np.polyval(novo[:3], 1.0) -
                                                np.polyval(ant[:3], 1.0)) > p["salto_max"] * bw
                if ant is not None and not salto:
                    novo = p["alpha"] * novo + (1 - p["alpha"]) * ant
                if salto:
                    self._tipo[lado].clear()
                self._estado[lado] = novo
                self._falta[lado] = 0
                self._tipo[lado].append(c["tipo"])
            else:
                self._falta[lado] += 1
                if self._falta[lado] > p["memoria"]:
                    self._estado[lado] = None
                    self._tipo[lado].clear()
            e = self._estado[lado]
            if e is not None and sum(self._vistos[lado]) >= p["confirmar_k"]:
                tipos = list(self._tipo[lado]) or ["continua"]
                tipo = max(set(tipos), key=tipos.count)      # voto: não troca a cada frame
                pts = curva_para_img(e[:3], e[3] - 0.05, g)
                if len(pts) >= 2:
                    out[lado] = dict(pts=pts, tipo=tipo, coef=[float(v) for v in e[:3]],
                                     rn_top=float(e[3]),
                                     cobertura=float(c["cobertura"]) if c else None)
        for c in outras:
            pts = curva_para_img(c["coef"], c["rn_min"], g, n=25)
            if len(pts) >= 2:
                out["outras"].append(dict(pts=pts, tipo=c["tipo"],
                                          coef=[float(v) for v in c["coef"]],
                                          rn_top=c["rn_min"]))
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
    """Validação da aula: curvatura, comprimento mínimo, força >= 40% da maior,
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


def _tipo_linha(c, marcas_dil, p):
    """Contínua vs tracejada pela fração da extensão com marca (classifyLaneTypes).
    marcas_dil = máscara das marcas alargada na horizontal (tolerância)."""
    bh, bw = marcas_dil.shape
    rows = np.arange(int(c["rn_min"] * (bh - 1)), int(c["rn_max"] * (bh - 1)) + 1, 2)
    cols = np.round(np.polyval(c["coef"], rows / (bh - 1.0))).astype(int)
    ok = (cols >= 0) & (cols < bw)
    cob = float(marcas_dil[rows[ok], cols[ok]].mean()) if ok.any() else 0.0
    return ("continua" if cob >= p["cobertura_continua"] else "tracejada"), cob


def _faixa_do_carro(fortes, p):
    """Linha forte mais próxima do eixo da câmara de cada lado (como a aula),
    exigindo uma largura de faixa plausível entre as duas."""
    if not fortes:
        return {}
    bw = p["bev_w"]
    meio = (bw - 1) / 2.0
    for c in fortes:
        c["col_base"] = float(np.polyval(c["coef"], 1.0))     # junto ao carro
    esq = sorted([c for c in fortes if c["col_base"] < meio], key=lambda c: meio - c["col_base"])
    dir_ = sorted([c for c in fortes if c["col_base"] >= meio], key=lambda c: c["col_base"] - meio)
    lmin, lmax = p["largura_faixa"][0] * bw, p["largura_faixa"][1] * bw
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
