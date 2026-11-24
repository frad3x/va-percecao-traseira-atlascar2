"""Mensagens da perceção -> dados do desenho (nó visualizador), sem ROS."""
import math
from types import SimpleNamespace as NS

import numpy as np

from va_g06 import reconstrucao as R

NAN, INF = float("nan"), float("inf")


def _agente(cat, classe, i, x, faixa, d=NAN, ttc=INF):
    return NS(categoria=cat, classe=classe, id=i, confianca=0.9, x=x, y=200.0, largura=60.0,
              altura=40.0, distancia=d, lateral=0.5, faixa=faixa, cortada=False, ttc=ttc,
              vel_aproximacao=NAN)


def _p(u, v):
    return NS(x=float(u), y=float(v), z=0.0)


def test_grupos_e_faixas():
    msg = NS(agentes=[_agente("veiculo", "carro", 7, 100.0, "nossa", d=12.0),
                      _agente("peao", "peao", 3, 300.0, "esquerda"),
                      _agente("sinal", "STOP", -1, 400.0, "")])
    res = R.grupos(msg, traseira=True)
    assert res["veic"]["bb"] == [[100.0, 200.0, 60.0, 40.0]]
    assert res["veic"]["faixa"] == ["ego"] and res["veic"]["dist"] == [12.0]
    # câmara traseira: a esquerda do CARRO aparece à direita da imagem
    assert res["peoes"]["faixa"] == ["img_dir"] and res["peoes"]["dist"] == [None]
    assert res["sinais"]["tipo"] == ["STOP"]


def test_linhas_tracos_e_seguidor():
    H, W = 480, 640
    # uma linha à direita do carro (esquerda da imagem): reta na metade de baixo
    pts = [_p(200 - 40 * k, 300 + 30 * k) for k in range(6)]
    tracos = pts[:2] + pts[3:5]
    linha = NS(lado="direita", tipo="tracejada", pontos=pts, tracos=tracos, tracos_n=[2, 2])
    div = NS(lado="divisoria", tipo="estimada", pontos=[_p(420 + 30 * k, 300 + 30 * k) for k in range(6)],
             tracos=[], tracos_n=[])
    pav = NS(n_linhas=1, horizonte=175.0, linhas=[linha, div], marcacoes=[])
    lin = R.linhas(pav, (H, W, 3))
    assert lin["esq"] is not None and lin["dir"] is None
    assert lin["esq"]["tracos"] == [[[200.0, 300.0], [160.0, 330.0]], [[80.0, 390.0], [40.0, 420.0]]]
    assert len(lin["esq"]["coef"]) == 3 and np.all(np.isfinite(lin["esq"]["coef"]))
    assert lin["divisoria"] is not None and lin["divisoria"]["estimada"]

    ag = NS(agentes=[_agente("veiculo", "carro", 1, 100.0, "nossa", d=20.0),
                     _agente("veiculo", "carro", 2, 300.0, "nossa", d=9.0)])
    seg = NS(estado_distancia="atencao", distancia_atras=9.0, distancia_minima=10.0,
             ttc_atras=INF, vel_aproximacao=NAN, mudar_esquerda="livre", mudar_direita="ocupado",
             espaco_livre_atras=NAN, avisos=[NS(nivel="atencao", texto="VEICULO ATRAS")])
    res = R.resultados(ag, pav, seg, (H, W, 3))
    s = res["seg"]
    assert s["seguidor"]["grupo"] == "veic" and s["seguidor"]["i"] == 1     # o de 9 m
    assert s["mudanca"]["direita"][0] == "ocupado" and s["livre_atras"] is None
    assert s["tipos_linha"] == {"direita": "tracejada"}
    assert math.isclose(res["linhas"]["horizonte"], 175.0)
