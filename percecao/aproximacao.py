"""
aproximacao.py — tempo-para-colisão (TTC) dos veículos que seguem atrás.

Porte de analisarAproximacao.m. Distintivo do Tema 2 (câmara traseira): estima
quão depressa cada veículo rastreado se aproxima, pelo CRESCIMENTO aparente da
sua caixa (looming). Calibração-independente: usa só a altura da caixa (px).

TTC ≈ s / (ds/dt), estimado por regressão linear numa janela; só reporta quando
o objeto está mesmo a crescer (a aproximar-se). Caso contrário devolve inf.
"""
import numpy as np


class Aproximacao:
    def __init__(self, janela=15, min_amostras=5, cres_min=0.3, limiar_risco=4.0):
        self.janela = janela
        self.min_amostras = min_amostras
        self.cres_min = cres_min
        self.limiar_risco = limiar_risco
        self._hist = {}   # id -> (lista_t, lista_s)

    def atualizar(self, veic, t):
        """veic: dict com 'bb' e 'id'. Devolve lista de TTC (s) alinhada às caixas."""
        bbs = veic.get("bb", [])
        ids = veic.get("id", [])
        ttc = [float("inf")] * len(bbs)
        for k, bb in enumerate(bbs):
            oid = ids[k] if k < len(ids) else -1
            s = bb[3]                      # altura da caixa (px) = tamanho aparente
            ts, ss = self._hist.get(oid, ([], []))
            ts.append(t); ss.append(s)
            if len(ts) > self.janela:
                ts, ss = ts[-self.janela:], ss[-self.janela:]
            self._hist[oid] = (ts, ss)
            if len(ts) >= self.min_amostras:
                tt = np.array(ts) - ts[0]
                if tt[-1] > 0:
                    p = np.polyfit(tt, np.array(ss), 1)   # s ≈ p[0]*t + p[1]
                    dsdt, s_now = p[0], np.polyval(p, tt[-1])
                    if dsdt > self.cres_min and s_now > 0:
                        ttc[k] = float(s_now / dsdt)
        return ttc

    def ttc_minimo(self, ttc):
        finitos = [v for v in ttc if np.isfinite(v)]
        return min(finitos) if finitos else float("inf")
