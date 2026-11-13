"""
conversao.py — resultado do Pipeline (dict Python) -> mensagens ROS 2.

Separado do nó para se poder testar sem a câmara. Os tipos de mensagem são
importados aqui; o nó só chama as funções.
"""
import math

from geometry_msgs.msg import Point, Point32, Vector3
from std_msgs.msg import ColorRGBA
from va_percecao_msgs.msg import (Agente, Agentes, Aviso, EstadoSeguranca, LinhaFaixa, Marcacao,
                                  Pavimento)
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray

NAN = float("nan")
INF = float("inf")

CATEGORIA = {"veic": Agente.VEICULO, "duas_rodas": Agente.DUAS_RODAS,
             "peoes": Agente.PEAO, "sinais": Agente.SINAL}
NOME_PT = {"car": "carro", "truck": "camiao", "bus": "autocarro", "bicycle": "bicicleta",
           "motorcycle": "mota", "person": "peao", "stop sign": "STOP"}
# cores dos marcadores no rviz (r, g, b) — as mesmas do overlay
COR = {"veic": (0.12, 0.63, 1.0), "duas_rodas": (1.0, 0.5, 0.0), "peoes": (1.0, 0.9, 0.0)}
# tamanho aproximado no chão (comprimento, largura, altura) em m
TAMANHO = {"veic": (4.5, 1.8, 1.5), "duas_rodas": (2.0, 0.8, 1.5), "peoes": (0.6, 0.6, 1.75)}


def _f(v, padrao=NAN):
    return padrao if v is None else float(v)


def _get(gr, chave, i, padrao=None):
    lst = gr.get(chave)
    return lst[i] if lst is not None and i < len(lst) else padrao


def _faixa_carro(faixa_img, lado_carro):
    """'ego'/'img_esq'/'img_dir'/'fora' -> 'nossa'/'esquerda'/'direita'/'fora' (lado do CARRO)."""
    if faixa_img is None:
        return ""
    if faixa_img == "ego":
        return "nossa"
    if faixa_img in ("img_esq", "img_dir"):
        return lado_carro(faixa_img)
    return faixa_img


# ---------------------------------------------------------------------------
def agentes(res, header, lado_carro):
    msg = Agentes(header=header)
    for grupo, cat in CATEGORIA.items():
        gr = res.get(grupo) or {}
        for i, (x, y, w, h) in enumerate(gr.get("bb", [])):
            tipo = _get(gr, "tipo", i, "")
            a = Agente()
            a.categoria = cat
            a.classe = NOME_PT.get(tipo, str(tipo))
            oid = _get(gr, "id", i, -1)
            a.id = int(oid if oid is not None else -1)
            a.confianca = _f(_get(gr, "sc", i), 0.0)
            a.x, a.y, a.largura, a.altura = float(x), float(y), float(w), float(h)
            a.distancia = _f(_get(gr, "dist", i))
            a.lateral = _f(_get(gr, "lat", i))
            a.faixa = _faixa_carro(_get(gr, "faixa", i), lado_carro)
            a.cortada = bool(_get(gr, "cortada", i, False))
            a.ttc = _f(_get(gr, "ttc", i), INF)
            a.vel_aproximacao = _f(_get(gr, "vrel", i))
            msg.agentes.append(a)
    return msg


def deteccoes(msg_agentes):
    """Versão normalizada (vision_msgs) — para quem só conhece tipos standard."""
    out = Detection2DArray(header=msg_agentes.header)
    for a in msg_agentes.agentes:
        d = Detection2D(header=msg_agentes.header)
        d.id = str(a.id) if a.id >= 0 else ""
        d.bbox.center.position.x = a.x + a.largura / 2.0
        d.bbox.center.position.y = a.y + a.altura / 2.0
        d.bbox.size_x, d.bbox.size_y = a.largura, a.altura
        hip = ObjectHypothesisWithPose()
        hip.hypothesis.class_id = f"{a.categoria}/{a.classe}"
        hip.hypothesis.score = a.confianca
        if math.isfinite(a.distancia):
            hip.pose.pose.position.x = a.distancia
            hip.pose.pose.position.y = a.lateral
        d.results.append(hip)
        out.detections.append(d)
    return out


def pavimento(res, header, lado_carro):
    lin = res.get("linhas") or {}
    msg = Pavimento(header=header)
    msg.n_linhas = int(lin.get("n", 0))
    msg.horizonte = _f(lin.get("horizonte"))
    for chave, lado_img in (("esq", "img_esq"), ("dir", "img_dir")):
        seg = lin.get(chave)
        if seg:
            msg.linhas.append(LinhaFaixa(lado=lado_carro(lado_img), tipo=seg.get("tipo", ""),
                                         pontos=[Point32(x=float(u), y=float(v), z=0.0)
                                                 for u, v in seg["pts"]]))
    for o in lin.get("outras", []):
        msg.linhas.append(LinhaFaixa(lado="outra", tipo=o.get("tipo", ""),
                                     pontos=[Point32(x=float(u), y=float(v), z=0.0)
                                             for u, v in o["pts"]]))
    for m in lin.get("marcacoes", []):
        msg.marcacoes.append(Marcacao(tipo=m["tipo"], distancia=_f(m.get("dist")),
                                      contorno=[Point32(x=float(u), y=float(v), z=0.0)
                                                for u, v in m["pts"]]))
    return msg


def seguranca(res, header):
    seg = res.get("seg") or {}
    s = seg.get("seguidor") or {}
    mud = seg.get("mudanca", {})
    msg = EstadoSeguranca(header=header)
    msg.estado_distancia = seg.get("estado_dist") or "livre"
    msg.distancia_atras = _f(s.get("d"))
    msg.distancia_minima = _f(seg.get("dist_min"))
    msg.ttc_atras = _f(s.get("ttc"), INF)
    msg.vel_aproximacao = _f(s.get("vrel"))
    msg.mudar_esquerda = mud.get("esquerda", ("livre", ""))[0]
    msg.mudar_direita = mud.get("direita", ("livre", ""))[0]
    msg.espaco_livre_atras = _f(seg.get("livre_atras"))
    msg.avisos = [Aviso(nivel=n, texto=t) for n, t in seg.get("alertas", [])]
    return msg


def marcadores(res, header, altura_camara, vida_s=0.3):
    """Agentes no chão (referencial da câmara: x para trás do carro, y à esquerda da
    imagem, z para cima; o chão está a -altura_camara) — para ver no rviz."""
    out = MarkerArray()
    apagar = Marker(header=header, action=Marker.DELETEALL)
    out.markers.append(apagar)
    k = 0
    for grupo in ("veic", "duas_rodas", "peoes"):
        gr = res.get(grupo) or {}
        comp, larg, alt = TAMANHO[grupo]
        r, g, b = COR[grupo]
        for i in range(len(gr.get("bb", []))):
            d, lat = _get(gr, "dist", i), _get(gr, "lat", i)
            if d is None or lat is None:
                continue
            ttc = _get(gr, "ttc", i)
            risco = ttc is not None and math.isfinite(ttc) and ttc < 4.0
            m = Marker(header=header, ns=grupo, id=k, type=Marker.CUBE, action=Marker.ADD)
            m.pose.position = Point(x=float(d) + comp / 2.0, y=float(lat),
                                    z=-altura_camara + alt / 2.0)
            m.pose.orientation.w = 1.0
            m.scale = Vector3(x=comp, y=larg, z=alt)
            m.color = ColorRGBA(r=1.0, g=0.15, b=0.15, a=0.85) if risco else \
                ColorRGBA(r=r, g=g, b=b, a=0.75)
            m.lifetime.nanosec = int(vida_s * 1e9)
            out.markers.append(m)
            k += 1
            t = Marker(header=header, ns=grupo + "_txt", id=k, type=Marker.TEXT_VIEW_FACING,
                       action=Marker.ADD)
            t.pose.position = Point(x=float(d) + comp / 2.0, y=float(lat),
                                    z=-altura_camara + alt + 0.4)
            t.pose.orientation.w = 1.0
            t.scale.z = 0.5
            t.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            txt = f"{d:.0f} m"
            if ttc is not None and math.isfinite(ttc):
                txt += f"  TTC {ttc:.1f}s"
            t.text = txt
            t.lifetime.nanosec = int(vida_s * 1e9)
            out.markers.append(t)
            k += 1
    return out
