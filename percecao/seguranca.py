"""
seguranca.py — o que o sistema DIZ ao condutor a partir do que deteta.

Responde às colunas do plano de trabalho (grupo G6, Tema 2):
  - Ações de navegação:   DISTÂNCIA DE SEGURANÇA ao veículo de trás; ESTACIONAMENTO
                          (espaço livre atrás + guias de distância no chão)
  - Planeamento:          pode MUDAR DE FAIXA para a esquerda/direita? (faixa ao lado
                          livre? a linha é contínua?)
  - Info. de segurança:   avisos de risco de colisão traseira (TTC), peão atrás,
                          duas-rodas a aproximar-se, veículo no ângulo morto

Como se obtém a DISTÂNCIA só com uma câmara (monocular): o fundo da caixa de um
objeto toca no chão; com o modelo pinhole de chão plano das linhas (horizonte +
altura da câmara + campo de visão) a linha da imagem v corresponde a uma distância
x no chão (é o mesmo modelo do birdsEyeView). É APROXIMADA (depende da altura da
câmara e do horizonte): serve para níveis (perto/longe), não para metrologia.

Lados: numa câmara TRASEIRA a imagem está "espelhada" em relação ao condutor —
a faixa à ESQUERDA da imagem é a faixa à DIREITA do carro. Os avisos usam o lado
do CARRO; os campos internos usam o lado da imagem ('img_esq'/'img_dir').
"""
import numpy as np

from .linhas import PARAMS as PARAMS_LINHAS, chave_ipm, geometria, img_para_chao, lateral_da_linha

PARAMS = dict(
    largura_carro=1.8,      # m (ATLASCAR2 ~1.75 m)
    largura_faixa=3.5,      # m, quando não há linhas para a medir
    dist_seguranca=10.0,    # m: seguidor mais perto do que isto -> "a colar" (sem velocidade)
    vel_ego_kmh=None,       # velocidade do carro (km/h) se conhecida -> regra dos 2 s
    tempo_seguranca=2.0,    # s (regra dos 2 segundos)
    dist_peao=8.0,          # m: peão atrás, no corredor do carro
    dist_angulo_morto=12.0, # m: veículo na faixa ao lado mais perto do que isto = ocupada
    ttc_risco=4.0,          # s
    dist_max=80.0,          # m: além disto a distância monocular não vale nada
    janela=12,              # amostras para estimar a velocidade de aproximação
    traseira=True,          # câmara TRASEIRA (imagem espelhada). False p/ vídeos de câmara frontal
)

LADO_CARRO = {"img_esq": "direita", "img_dir": "esquerda"}


class AnaliseSeguranca:
    def __init__(self, par=None, linhas_params=None):
        self.p = dict(PARAMS)
        if par:
            self.p.update(par)
        self.lp = dict(PARAMS_LINHAS)
        if linhas_params:
            self.lp.update(linhas_params)
        self._hist = {}           # (grupo, id) -> (ts, ds)
        self._larg = None         # largura da faixa medida (EMA), m

    # ------------------------------------------------------------------
    def analisar(self, shape, res, t):
        """Acrescenta a cada grupo 'dist', 'lat', 'faixa', 'vrel', 'cortada'
        e devolve o resumo res['seg'] (também guardado em res)."""
        H, W = shape[:2]
        p = self.p
        lin = res.get("linhas") or {}
        if lin.get("chave"):
            g = geometria(tuple(lin["chave"]))[0]
        else:
            y_h = (self.lp["horizonte"] or self.lp["horizonte_ini"]) * H
            g = geometria(chave_ipm(W, H, y_h, self.lp))[0]
        esq, dir_ = lin.get("esq"), lin.get("dir")

        objs = []
        for grupo in ("veic", "duas_rodas", "peoes"):
            gr = res.get(grupo)
            if not gr:
                continue
            n = len(gr.get("bb", []))
            gr["dist"], gr["lat"], gr["faixa"], gr["vrel"], gr["cortada"] = \
                [None] * n, [None] * n, [None] * n, [None] * n, [False] * n
            for i, (x, y, w, h) in enumerate(gr["bb"]):
                v = y + h
                if v <= g["y_h"] + 3:                      # acima do horizonte: sem chão
                    continue
                X, Y = img_para_chao(x + w / 2.0, v, g)
                if not np.isfinite(X) or X > p["dist_max"] or X <= 0:
                    continue
                cortada = v >= H - 3                       # sai por baixo: está AINDA mais perto
                faixa = self._faixa(X, Y, esq, dir_, g)
                oid = gr["id"][i] if i < len(gr.get("id", [])) else -1
                vrel = self._vel(grupo, oid, t, X)
                gr["dist"][i], gr["lat"][i], gr["faixa"][i] = X, Y, faixa
                gr["vrel"][i], gr["cortada"][i] = vrel, cortada
                ttc = (gr.get("ttc") or [None] * n)[i] if grupo != "peoes" else None
                objs.append(dict(grupo=grupo, i=i, d=X, lat=Y, faixa=faixa, vrel=vrel,
                                 ttc=ttc, cortada=cortada, id=oid))
        self._limpar(t)
        seg = self._resumo(objs, esq, dir_)
        seg["chave"] = tuple(lin["chave"]) if lin.get("chave") else \
            chave_ipm(W, H, g["y_h"], self.lp)
        res["seg"] = seg
        return seg

    def lado_carro(self, lado_img):
        """'img_esq'/'img_dir' -> 'direita'/'esquerda' do CARRO (trocado se traseira)."""
        if self.p["traseira"]:
            return LADO_CARRO[lado_img]
        return {"img_esq": "esquerda", "img_dir": "direita"}[lado_img]

    # ------------------------------------------------------------------
    def _faixa(self, X, Y, esq, dir_, g):
        """Em que faixa está o objeto: 'ego' | 'img_esq' | 'img_dir' | 'fora'."""
        ye = lateral_da_linha(esq, X, g) if esq else None
        yd = lateral_da_linha(dir_, X, g) if dir_ else None
        if ye is not None and yd is not None and ye > yd:
            larg = ye - yd
            if 2.4 < larg < 5.0:                          # medição plausível -> aprende
                self._larg = larg if self._larg is None else 0.9 * self._larg + 0.1 * larg
        L = self._larg or self.p["largura_faixa"]
        if ye is None and yd is None:
            ye, yd = L / 2.0, -L / 2.0
        elif ye is None:
            ye = yd + L
        elif yd is None:
            yd = ye - L
        if yd <= Y <= ye:
            return "ego"
        if ye < Y <= ye + L:
            return "img_esq"
        if yd - L <= Y < yd:
            return "img_dir"
        return "fora"

    def _vel(self, grupo, oid, t, d):
        """Velocidade de aproximação (m/s, >0 = a aproximar-se), por regressão."""
        if oid < 0:
            return None
        ts, ds = self._hist.setdefault((grupo, oid), ([], []))
        ts.append(t); ds.append(d)
        del ts[:-self.p["janela"]], ds[:-self.p["janela"]]
        if len(ts) < 5 or ts[-1] - ts[0] <= 0:
            return None
        return float(-np.polyfit(np.array(ts) - ts[0], np.array(ds), 1)[0])

    def _limpar(self, t):
        for k in [k for k, (ts, _) in self._hist.items() if t - ts[-1] > 5.0]:
            del self._hist[k]

    # ------------------------------------------------------------------
    def _resumo(self, objs, esq, dir_):
        p = self.p
        alertas = []
        veics = [o for o in objs if o["grupo"] in ("veic", "duas_rodas")]

        # --- distância de segurança: o seguidor = o mais perto na NOSSA faixa ---
        seguidores = sorted((o for o in veics if o["faixa"] == "ego"), key=lambda o: o["d"])
        seg = seguidores[0] if seguidores else None
        estado = "livre"
        dist_min = p["dist_seguranca"]
        if seg is not None:
            if p["vel_ego_kmh"] is not None:               # regra dos 2 s (vel. do seguidor)
                v_seg = max(0.0, p["vel_ego_kmh"] / 3.6 + (seg["vrel"] or 0.0))
                dist_min = max(3.0, p["tempo_seguranca"] * v_seg)
            ttc = seg["ttc"]
            if ttc is not None and np.isfinite(ttc) and ttc < p["ttc_risco"]:
                estado = "perigo"
                alertas.append(("perigo", f"RISCO DE COLISAO TRASEIRA  TTC {ttc:.1f} s"))
            elif seg["d"] < dist_min or seg["cortada"]:
                estado = "atencao"
                alertas.append(("atencao", f"VEICULO ATRAS A {seg['d']:.0f} m (< {dist_min:.0f} m)"))
            else:
                estado = "ok"

        # --- mudança de faixa (lado do CARRO) ---
        mudanca = {}
        for lado_img, linha in (("img_esq", esq), ("img_dir", dir_)):
            lado = self.lado_carro(lado_img)
            ocup = [o for o in veics if o["faixa"] == lado_img and (
                o["d"] < p["dist_angulo_morto"] or o["cortada"] or
                (o["ttc"] is not None and np.isfinite(o["ttc"]) and o["ttc"] < 2 * p["ttc_risco"]))]
            if ocup:
                o = min(ocup, key=lambda o: o["d"])
                mudanca[lado] = ("ocupado", f"veiculo a {o['d']:.0f} m")
                if o["d"] < 0.6 * p["dist_angulo_morto"] or o["cortada"]:
                    alertas.append(("atencao", f"VEICULO NO ANGULO MORTO ({lado.upper()})"))
            elif linha is not None and linha.get("tipo") == "continua":
                mudanca[lado] = ("proibido", "linha continua")
            else:
                mudanca[lado] = ("livre", "")

        # --- duas rodas a aproximar-se (vulneráveis, difíceis de ver nos espelhos) ---
        for o in objs:
            if o["grupo"] == "duas_rodas" and o["d"] < 20 and (o["vrel"] or 0) > 1.0:
                alertas.append(("atencao", f"DUAS RODAS A APROXIMAR-SE ({o['d']:.0f} m)"))
                break

        # --- estacionamento / marcha-atrás: o que está no corredor do carro ---
        meia = p["largura_carro"] / 2.0 + 0.3
        corredor = [o for o in objs if abs(o["lat"]) < meia]
        livre = min((o["d"] for o in corredor), default=None)
        peoes = [o for o in corredor if o["grupo"] == "peoes" and o["d"] < p["dist_peao"]]
        if peoes:
            alertas.insert(0, ("perigo", f"PEAO ATRAS A {min(o['d'] for o in peoes):.1f} m"))

        tipos = {self.lado_carro("img_esq"): (esq or {}).get("tipo"),
                 self.lado_carro("img_dir"): (dir_ or {}).get("tipo")}
        return dict(seguidor=seg, estado_dist=estado, tipos_linha=tipos, dist_min=dist_min,
                    mudanca=mudanca, alertas=alertas, livre_atras=livre,
                    larg_faixa=self._larg or p["largura_faixa"])
