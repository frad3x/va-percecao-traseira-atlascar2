"""
aproximacao.py — tempo-para-colisão (TTC) dos veículos que seguem atrás.

Distintivo do Tema 2 (câmara traseira): estima
quão depressa cada veículo rastreado se aproxima, pelo CRESCIMENTO aparente da
sua caixa (looming). Calibração-independente: usa só a altura da caixa (px).

TTC ≈ s / (ds/dt), estimado por regressão linear numa janela; só reporta quando
o objeto está mesmo a crescer (a aproximar-se). Caso contrário devolve inf.

Contra o ruído (carros parados, caixa a "respirar", câmara a tremer): exige um
crescimento relativo mínimo (cres_rel_min por segundo) e que a aproximação se
confirme em 'confirmar' atualizações seguidas.
"""
import numpy as np


class Aproximacao:
    def __init__(self, janela=15, min_amostras=5, cres_min=0.3, limiar_risco=1.0,
                 cres_rel_min=0.05, confirmar=3):
        self.janela = janela
        self.min_amostras = min_amostras
        self.cres_min = cres_min
        self.cres_rel_min = cres_rel_min   # 5%/s: TTC até ~20 s; abaixo disto é ruído
        self.confirmar = confirmar
        self.limiar_risco = limiar_risco
        self._hist = {}   # id -> (lista_t, lista_s)
        self._seguidos = {}   # id -> nº de atualizações seguidas a aproximar-se

    def atualizar(self, veic, t, grupo="veic", forma=None):
        """veic: dict com 'bb' e 'id'. Devolve lista de TTC (s) alinhada às caixas.
        forma = (H, W) da imagem: uma caixa CORTADA pela borda (carro a entrar ou a sair da
        imagem) cresce sem se aproximar — não conta para o TTC."""
        bbs = veic.get("bb", [])
        ids = veic.get("id", [])
        ttc = [float("inf")] * len(bbs)
        for k, bb in enumerate(bbs):
            oid = ids[k] if k < len(ids) else -1
            if oid < 0:                    # sem rastreio não há histórico fiável
                continue
            oid = (grupo, oid)
            if forma is not None:
                H, W = forma[:2]
                x, y, w, h = bb
                m = 0.02 * W
                if x <= m or x + w >= W - m or y + h >= H - 3:
                    self._hist.pop(oid, None)
                    self._seguidos.pop(oid, None)
                    continue
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
                    if dsdt > self.cres_min and s_now > 0 and dsdt / s_now >= self.cres_rel_min:
                        n = self._seguidos[oid] = self._seguidos.get(oid, 0) + 1
                        if n >= self.confirmar:
                            ttc[k] = float(s_now / dsdt)
                    else:
                        self._seguidos[oid] = 0
        # esquece objetos que já não se veem há muito (memória não cresce sem fim)
        velhos = [o for o, (ts, _) in self._hist.items() if t - ts[-1] > 5.0]
        for o in velhos:
            del self._hist[o]
            self._seguidos.pop(o, None)
        return ttc

    def ttc_minimo(self, ttc):
        finitos = [v for v in ttc if np.isfinite(v)]
        return min(finitos) if finitos else float("inf")
