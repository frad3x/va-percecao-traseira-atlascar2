"""
reconstrucao.py — mensagens da perceção (Agentes, Pavimento, EstadoSeguranca) -> o
dicionário de resultados que o desenho (percecao/desenho.py) usa.

É o inverso de conversao.py e permite que a VISUALIZAÇÃO seja um nó à parte (nó
'visualizador'), que só conhece o que a perceção publica — como pede o enunciado
(processamento separado da representação visual). Não importa nada de ROS: funciona
com quaisquer objetos com os mesmos campos (e testa-se sem ROS).
"""
import math

GRUPO = {"veiculo": "veic", "duas_rodas": "duas_rodas", "peao": "peoes", "sinal": "sinais"}
CAMPOS = ("bb", "sc", "id", "tipo", "dist", "lat", "faixa", "cortada", "ttc", "vrel")


def _num(v):
    """NaN -> None (é assim que o desenho trata 'desconhecido')."""
    v = float(v)
    return None if math.isnan(v) else v


def _faixa_img(faixa_carro, traseira=True):
    """'nossa'/'esquerda'/'direita'/'fora' (lado do CARRO) -> 'ego'/'img_esq'/'img_dir'/'fora'.
    Numa câmara traseira a esquerda do carro aparece à DIREITA da imagem."""
    if faixa_carro == "nossa":
        return "ego"
    if faixa_carro in ("esquerda", "direita"):
        esq_img = (faixa_carro == "direita") if traseira else (faixa_carro == "esquerda")
        return "img_esq" if esq_img else "img_dir"
    return faixa_carro or None


def _lado_img(lado_carro, traseira=True):
    """Lado de uma LINHA: 'esquerda'/'direita' do carro -> 'esq'/'dir' da imagem."""
    esq_img = (lado_carro == "direita") if traseira else (lado_carro == "esquerda")
    return "esq" if esq_img else "dir"


def grupos(msg_agentes, traseira=True):
    res = {g: {c: [] for c in CAMPOS} for g in GRUPO.values()}
    for a in msg_agentes.agentes:
        g = res[GRUPO.get(a.categoria, "veic")]
        g["bb"].append([float(a.x), float(a.y), float(a.largura), float(a.altura)])
        g["sc"].append(float(a.confianca))
        g["id"].append(int(a.id))
        g["tipo"].append(a.classe)
        g["dist"].append(_num(a.distancia))
        g["lat"].append(_num(a.lateral))
        g["faixa"].append(_faixa_img(a.faixa, traseira))
        g["cortada"].append(bool(a.cortada))
        g["ttc"].append(float(a.ttc))
        g["vrel"].append(_num(a.vel_aproximacao))
    return res


def _pontos(lista):
    return [[float(p.x), float(p.y)] for p in lista]


def _tracos(linha):
    """Os traços reais de uma tracejada: os pontos de todos seguidos + quantos tem cada um."""
    if not getattr(linha, "tracos_n", None):
        return None
    pts, out, i = _pontos(linha.tracos), [], 0
    for n in linha.tracos_n:
        out.append(pts[i:i + int(n)])
        i += int(n)
    return out


def _coef(pts, g):
    """Parábola da linha na vista de cima (como no detetor) a partir dos pontos da imagem:
    é o que o desenho usa para a faixa do carro e para a vista de cima esquemática."""
    import numpy as np
    from percecao.linhas import img_para_chao
    rn, col = [], []
    for u, v in pts:
        x, y = img_para_chao(u, v, g)
        if math.isfinite(x) and x > 0:
            rn.append((g["x_far"] - x) / max(g["x_far"] - g["x_near"], 1e-6))
            col.append((g["lado"] - y) * g["ppm"])
    if len(rn) < 2:
        return None, None
    c = np.polyfit(rn, col, min(2, len(rn) - 1))
    c = [0.0] * (3 - len(c)) + [float(v) for v in c]
    return c, float(max(min(rn), 0.0))


def linhas(msg_pav, forma, params_linhas=None, traseira=True):
    from percecao.linhas import PARAMS, chave_ipm, geometria
    H, W = forma[:2]
    p = dict(PARAMS)
    p.update(params_linhas or {})
    y_h = _num(msg_pav.horizonte)
    if y_h is None:
        y_h = p["horizonte_ini"] * H
    chave = chave_ipm(W, H, y_h, p)
    g = geometria(tuple(chave))[0]
    lin = {"n": int(msg_pav.n_linhas), "esq": None, "dir": None, "divisoria": None,
           "outras": [], "marcacoes": [], "chave": chave, "horizonte": y_h}
    for L in msg_pav.linhas:
        pts = _pontos(L.pontos)
        if len(pts) < 2:
            continue
        coef, rn_top = _coef(pts, g)
        d = dict(pts=pts, tipo=L.tipo, tracos=_tracos(L), confianca=1.0)
        if coef is not None:
            d.update(coef=coef, rn_top=rn_top)
        if L.lado == "divisoria":
            d["estimada"] = True
            lin["divisoria"] = d if coef is not None else None
        elif L.lado in ("esquerda", "direita"):
            if coef is not None:
                lin[_lado_img(L.lado, traseira)] = d
        else:
            lin["outras"].append(d)
    for m in msg_pav.marcacoes:
        lin["marcacoes"].append(dict(tipo=m.tipo, dist=_num(m.distancia),
                                     pts=_pontos(m.contorno)))
    return lin


def seguranca(msg_seg, res, msg_pav=None):
    """EstadoSeguranca -> o 'seg' do desenho. O seguidor é o agente da NOSSA faixa cuja
    distância é a publicada como distância ao veículo de trás."""
    seg = {"estado_dist": msg_seg.estado_distancia or "livre",
           "dist_min": _num(msg_seg.distancia_minima),
           "mudanca": {"esquerda": (msg_seg.mudar_esquerda or "livre", ""),
                       "direita": (msg_seg.mudar_direita or "livre", "")},
           "alertas": [(a.nivel, a.texto) for a in msg_seg.avisos],
           "livre_atras": _num(msg_seg.espaco_livre_atras),
           "larg_faixa": 3.5, "seguidor": None}
    if msg_pav is not None:
        seg["tipos_linha"] = {L.lado: L.tipo for L in msg_pav.linhas
                              if L.lado in ("esquerda", "direita")}
    d_atras = _num(msg_seg.distancia_atras)
    if d_atras is not None:
        melhor = None
        for grupo in ("veic", "duas_rodas"):
            gr = res.get(grupo) or {}
            for i, (d, fx) in enumerate(zip(gr.get("dist", []), gr.get("faixa", []))):
                if fx == "ego" and d is not None and (melhor is None or abs(d - d_atras) < melhor[0]):
                    melhor = (abs(d - d_atras), grupo, i)
        if melhor is not None:
            _, grupo, i = melhor
            gr = res[grupo]
            seg["seguidor"] = dict(grupo=grupo, i=i, d=gr["dist"][i], ttc=gr["ttc"][i],
                                   vrel=gr["vrel"][i], cortada=gr["cortada"][i])
    return seg


def resultados(msg_agentes, msg_pav, msg_seg, forma, params_linhas=None, traseira=True):
    """Tudo junto: o 'res' que percecao.desenho.desenhar() recebe."""
    res = grupos(msg_agentes, traseira)
    if msg_pav is not None:
        res["linhas"] = linhas(msg_pav, forma, params_linhas, traseira)
    res["seg"] = seguranca(msg_seg, res, msg_pav) if msg_seg is not None else {}
    return res
