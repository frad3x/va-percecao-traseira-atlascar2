"""
estabilizar.py — caixas estáveis no tempo (por objeto seguido).

O YOLO + ByteTrack dá, em cada imagem, caixas que tremem uns píxeis, objetos que
aparecem numa só imagem (deteções fracas) e objetos que falham uma imagem e voltam.
Desenhadas assim, as caixas "saltam" e piscam. Aqui, por objeto (ID do ByteTrack, ou
associação por sobreposição para os sinais, que não têm ID):

  - suavização do centro e do tamanho À VOLTA de uma previsão a velocidade constante
    (com o carro em marcha os objetos deslizam na imagem; uma média simples ficava
    para trás deles); o tamanho mais suave, porque é dele que vem a distância;
  - confirmação: um objeto novo só aparece depois de visto em `confirmar` imagens
    (ou logo, se a confiança for alta);
  - memória curta: um objeto que falha até `memoria` imagens continua o movimento que
    trazia (marcado 'previsto'), em vez de desaparecer e voltar;
  - IDs de exibição pequenos (1, 2, 3...) em vez dos do ByteTrack (que chegam aos milhares).

Um salto grande (outro objeto com o mesmo ID) reinicia a suavização.

É só para o DESENHO: os dados (distância, TTC, mensagens ROS, métricas) usam as caixas
medidas. Medido na volta real contra um modelo de referência, suavizar as caixas dos
dados baixava o F1 dos veículos de 0,84 para 0,78-0,80 (com o carro em marcha, os
objetos deslizam na imagem e a caixa suavizada fica ligeiramente desfasada).
"""
import numpy as np

GRUPOS = ("veic", "duas_rodas", "peoes", "sinais")


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-6)


class _Faixa:
    __slots__ = ("bb", "vel", "vistos", "falta", "dados", "vis_id", "conf")

    def __init__(self, bb, dados, conf):
        self.bb = np.array(bb, float)
        self.vel = np.zeros(4)                       # variação por imagem de (cx, cy, w, h)
        self.vistos, self.falta = 1, 0
        self.dados, self.vis_id, self.conf = dados, None, conf


class Estabilizador:
    def __init__(self, alpha_pos=0.55, alpha_tam=0.35, confirmar=2, conf_imediata=0.6,
                 memoria=3, confirmar_sinais=3, iou_sinais=0.3):
        self.a_pos, self.a_tam = alpha_pos, alpha_tam
        self.confirmar, self.conf_imediata = confirmar, conf_imediata
        self.memoria = memoria
        self.confirmar_sinais, self.iou_sinais = confirmar_sinais, iou_sinais
        self._faixas = {g: {} for g in GRUPOS}
        self._prox = {g: 0 for g in GRUPOS}        # chaves internas p/ objetos sem ID
        self._vis = 0

    def _suaviza(self, f, bb):
        n = np.array(bb, float)
        o = f.bb
        c_o = o[:2] + o[2:] / 2
        c_n = n[:2] + n[2:] / 2
        # salto grande (o ID passou para outro objeto): recomeça
        if np.abs(c_n - c_o).max() > 0.6 * max(o[3], n[3]) or not (0.6 < n[3] / max(o[3], 1) < 1.6):
            f.bb, f.vel = n, np.zeros(4)
            return
        # previsão a velocidade constante (o carro anda: os objetos deslizam na imagem) e
        # suavização À VOLTA da previsão — sem o atraso de uma média simples
        prev = np.concatenate([c_o, o[2:]]) + f.vel
        med = np.concatenate([c_n, n[2:]])
        a = np.array([self.a_pos, self.a_pos, self.a_tam, self.a_tam])
        novo = a * med + (1 - a) * prev
        f.vel = 0.5 * (novo - np.concatenate([c_o, o[2:]])) + 0.5 * f.vel
        f.bb = np.concatenate([novo[:2] - novo[2:] / 2, novo[2:]])

    @staticmethod
    def _preve(f):
        """Imagem sem medida: a caixa continua o movimento que trazia."""
        c = f.bb[:2] + f.bb[2:] / 2 + f.vel[:2]
        wh = np.maximum(f.bb[2:] + f.vel[2:], 2.0)
        f.bb = np.concatenate([c - wh / 2, wh])

    def atualizar(self, res):
        """Devolve {grupo: caixas suavizadas, só objetos confirmados, + previstos} sem
        mexer em res. 'i_bruto' = índice da mesma caixa em res (None se prevista)."""
        saida = {}
        for g in GRUPOS:
            gr = res.get(g)
            if gr is None:
                continue
            fx = self._faixas[g]
            chaves = list(gr.get("id") or [-1] * len(gr["bb"]))
            if g == "sinais" or all(c is None or c < 0 for c in chaves):
                chaves = self._associa(g, gr["bb"])
            vistas = {}
            n = len(gr["bb"])
            for i in range(n):
                k = chaves[i]
                dados = {c: v[i] for c, v in gr.items() if isinstance(v, list) and len(v) == n}
                sc = float((gr.get("sc") or [1.0] * n)[i])
                if k in fx:
                    f = fx[k]
                    self._suaviza(f, gr["bb"][i])
                    f.vistos += 1
                    f.falta = 0
                    f.dados, f.conf = dados, sc
                else:
                    fx[k] = _Faixa(gr["bb"][i], dados, sc)
                vistas[k] = i
            for k in list(fx):
                if k not in vistas:
                    fx[k].falta += 1
                    self._preve(fx[k])
                    if fx[k].falta > self.memoria:
                        del fx[k]
            # reconstrói o grupo
            novo = {c: [] for c in gr.keys() if isinstance(gr[c], list)}
            novo.setdefault("bb", [])
            novo["previsto"], novo["id_vis"], novo["i_bruto"] = [], [], []
            conf_min = self.confirmar_sinais if g == "sinais" else self.confirmar
            for k, f in fx.items():
                if f.vistos < conf_min and f.conf < self.conf_imediata:
                    continue
                if f.falta > 0 and f.vistos < conf_min + 1:
                    continue                         # mal confirmado e já perdido: some
                if f.vis_id is None:
                    self._vis += 1
                    f.vis_id = self._vis
                for c in novo:
                    if c in ("previsto", "id_vis", "i_bruto"):
                        continue
                    if c == "bb":
                        novo[c].append([float(v) for v in f.bb])
                    else:
                        novo[c].append(f.dados.get(c))
                novo["previsto"].append(f.falta > 0)
                novo["id_vis"].append(f.vis_id)
                novo["i_bruto"].append(vistas.get(k))
            saida[g] = novo
        return saida

    def _associa(self, g, bbs):
        """Objetos sem ID: associa à faixa com maior sobreposição (ou cria uma chave)."""
        fx = self._faixas[g]
        livres = set(fx)
        out = []
        for bb in bbs:
            melhor, mk = self.iou_sinais, None
            for k in livres:
                v = _iou(bb, fx[k].bb)
                if v > melhor:
                    melhor, mk = v, k
            if mk is None:
                self._prox[g] += 1
                mk = ("n", self._prox[g])
            else:
                livres.discard(mk)
            out.append(mk)
        return out
