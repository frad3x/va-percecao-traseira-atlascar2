"""grelha.py — perceção da câmara traseira -> grelha de ocupação local (base_link).

Sem ROS (só numpy + va_planeamento.grid), para se poder testar fora do contentor.
Contrato da aula 10: nav_msgs/OccupancyGrid no base_link, -1 desconhecido, 0 livre,
100 ocupado. Para a câmara traseira (grupo G6) a grelha cobre a zona ATRÁS do carro
(x negativo):

  OCUPADO      veículos, duas rodas e peões, como caixas no chão (a distância e a
               posição lateral vêm do modelo monocular de chão plano do pipeline);
               opcionalmente as linhas CONTÍNUAS (não se podem atravessar)
  LIVRE        o chão visível: dentro do campo de visão, desde a câmara até ao primeiro
               obstáculo em cada direção (ou até ao alcance máximo)
  DESCONHECIDO o que está atrás de um veículo, fora do campo de visão ou longe demais

Referenciais: o pipeline dá (d, lat) no chão, no referencial do CORPO da câmara
(camera_tras_link: x para onde a câmara olha, y para a esquerda da imagem). A pose
2D da câmara no carro (tx, ty, yaw) vem do /tf_static (URDF) — com yaw ≈ π a câmara
olha para trás e d vira x negativo no base_link.
"""
import math

import numpy as np

from va_planeamento.grid import LIVRE, LocalGrid

# tamanho aproximado no chão (comprimento, largura, altura) em m
TAMANHO = {"veic": (4.5, 1.8, 1.5), "duas_rodas": (2.0, 0.8, 1.5), "peoes": (0.6, 0.6, 1.75)}

PARAMS = dict(
    resolucao=0.2, x_min=-35.0, x_max=5.0, y_min=-10.0, y_max=10.0,
    hfov=66.0,                 # campo de visão horizontal da câmara (graus)
    alcance_livre=30.0,        # m: até onde o chão visível conta como livre
    n_raios=120,               # direções do contorno da zona visível
    linhas_continuas=True,     # marcar as linhas contínuas como ocupadas
)


def caixas_no_chao(res):
    """Lista de (d, lat, comprimento, largura) no referencial da câmara."""
    out = []
    for grupo, (comp, larg, _) in TAMANHO.items():
        gr = res.get(grupo) or {}
        for i in range(len(gr.get("bb", []))):
            d = (gr.get("dist") or [None] * (i + 1))[i]
            lat = (gr.get("lat") or [None] * (i + 1))[i]
            if d is None or lat is None or not (math.isfinite(d) and math.isfinite(lat)):
                continue
            out.append((float(d), float(lat), comp, larg))
    return out


def linhas_continuas_no_chao(res, passo=0.25):
    """Pontos (d, lat) no chão das linhas da faixa CONTÍNUAS (lado da imagem)."""
    from percecao.linhas import geometria, img_para_chao
    lin = res.get("linhas") or {}
    if not lin.get("chave"):
        return []
    g = geometria(tuple(lin["chave"]))[0]
    out = []
    for lado in ("esq", "dir"):
        l = lin.get(lado)
        if not l or l.get("tipo") != "continua":
            continue
        pts = [img_para_chao(u, v, g) for u, v in l["pts"]]
        pts = [(x, y) for x, y in pts if np.isfinite(x) and np.isfinite(y) and x > 0]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):        # densifica o segmento
            n = max(2, int(math.hypot(x1 - x0, y1 - y0) / passo))
            out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n)]
    return out


def construir(res, pose_camara=(-0.5, 0.0, math.pi), par=None):
    """res = resultado do Pipeline.processar; pose_camara = (tx, ty, yaw) da câmara no
    base_link. Devolve uma LocalGrid (va_planeamento) no base_link."""
    p = dict(PARAMS)
    if par:
        p.update(par)
    tx, ty, yaw = pose_camara
    c, s = math.cos(yaw), math.sin(yaw)

    def para_base(d, lat):                    # câmara (chão) -> base_link
        d, lat = np.asarray(d, float), np.asarray(lat, float)
        return tx + d * c - lat * s, ty + d * s + lat * c

    g = LocalGrid(p["resolucao"], p["x_min"], p["x_max"], p["y_min"], p["y_max"])
    caixas = caixas_no_chao(res)

    # --- LIVRE: raios no campo de visão até ao 1.º obstáculo dessa direção ---
    meio = math.radians(p["hfov"]) / 2.0
    ang = np.linspace(-meio, meio, int(p["n_raios"]))
    alc = np.full(ang.shape, float(p["alcance_livre"]))
    for d, lat, _, larg in caixas:
        a0, a1 = math.atan2(lat - larg / 2, d), math.atan2(lat + larg / 2, d)
        alc[(ang >= a0) & (ang <= a1)] = np.minimum(alc[(ang >= a0) & (ang <= a1)], d)
    # a zona visível é o polígono câmara -> pontas dos raios (mais rápido do que marcar
    # raio a raio, e sem falhas entre raios ao longe)
    xe, ye = para_base(alc * np.cos(ang), alc * np.sin(ang))
    g.marcar_poligono([(tx, ty)] + list(zip(xe, ye)), valor=LIVRE)

    # --- OCUPADO: caixas dos agentes no chão ---
    for d, lat, comp, larg in caixas:
        cantos = [(d, lat - larg / 2), (d + comp, lat - larg / 2),
                  (d + comp, lat + larg / 2), (d, lat + larg / 2)]
        xs, ys = para_base([q[0] for q in cantos], [q[1] for q in cantos])
        g.marcar_poligono(list(zip(xs, ys)))

    # --- OCUPADO (opcional): linhas contínuas — não se podem atravessar ---
    if p["linhas_continuas"]:
        pts = linhas_continuas_no_chao(res)
        if pts:
            xs, ys = para_base([q[0] for q in pts], [q[1] for q in pts])
            g.marcar_pontos(xs, ys)
    return g
