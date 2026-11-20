"""Testes da grelha local da câmara traseira (sem ROS): correm no CI (pytest)."""
import math

from va_planeamento.astar import ParametrosPlaneador, planear
from va_planeamento.grid import DESCONHECIDO, LIVRE, OCUPADO

from va_g06.grelha import construir

POSE = (-0.5, 0.0, math.pi)          # câmara 0,5 m atrás do base_link, a olhar para trás


def _res(veiculos=()):
    """Resultado mínimo do pipeline: veículos com (distância, lateral) no chão."""
    n = len(veiculos)
    vazio = {"bb": [], "dist": [], "lat": []}
    return {"veic": {"bb": [[0, 0, 1, 1]] * n, "dist": [v[0] for v in veiculos],
                     "lat": [v[1] for v in veiculos]},
            "duas_rodas": dict(vazio), "peoes": dict(vazio), "linhas": {}}


def _valor(g, x, y):
    ix, iy, ok = g.para_celula(x, y)
    assert ok
    return int(g.data[iy, ix])


def test_chao_visivel_livre_e_resto_desconhecido():
    g = construir(_res(), POSE)
    assert _valor(g, -10.0, 0.0) == LIVRE           # 9,5 m atrás, ao centro: visível
    assert _valor(g, 3.0, 0.0) == DESCONHECIDO      # à frente do carro: a câmara não vê
    assert _valor(g, -3.0, 9.0) == DESCONHECIDO     # fora do campo de visão


def test_veiculo_atras_ocupa_e_tapa_o_que_esta_atras_dele():
    # veículo a 10 m da câmara, 3,5 m à esquerda da IMAGEM = 3,5 m à DIREITA do carro
    g = construir(_res([(10.0, 3.5)]), POSE)
    assert _valor(g, -0.5 - 12.0, -3.5) == OCUPADO      # caixa: de 10 a 14,5 m
    assert _valor(g, -0.5 - 25.0, -3.5 * 2.5) == DESCONHECIDO or \
        _valor(g, -0.5 - 25.0, -8.75) != OCUPADO
    assert _valor(g, -0.5 - 6.0, -2.1) == LIVRE         # entre a câmara e o veículo


def _chega(r, objetivo, tol=0.5):
    return bool(r.caminho_xy) and math.dist(r.caminho_xy[-1], objetivo) <= tol


def test_mudanca_de_faixa_so_chega_ao_objetivo_com_a_faixa_livre():
    """Objetivo na faixa da direita, 20 m atrás. O A* comum procura a célula livre mais
    próxima do objetivo: com a faixa livre o caminho CHEGA lá; com um camião nessa faixa
    fica noutra célula (ou não há caminho = BLOQUEADO) -> mudança de faixa insegura."""
    p = ParametrosPlaneador(raio_inflacao=1.0, desconhecido_livre=True,
                            corredor_y_min=-5.25, corredor_y_max=5.25)   # como em params.yaml
    obj = (-20.0, -3.5)
    livre = planear(construir(_res(), POSE), (0.0, 0.0), obj, p)
    ocup = construir(_res([(d, 3.5) for d in (4.0, 8.5, 13.0, 17.5)]), POSE)
    com_camiao = planear(ocup, (0.0, 0.0), obj, p)
    assert livre.estado != "BLOQUEADO" and _chega(livre, obj)
    assert com_camiao.estado == "BLOQUEADO" or not _chega(com_camiao, obj)
