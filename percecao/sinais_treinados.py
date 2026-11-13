"""
sinais_treinados.py — reconhecimento de sinais com modelos TREINADOS (2 fases).

  1) DETETOR (YOLO treinado no GTSDB): encontra os sinais na imagem e dá a
     categoria (proibição / perigo / obrigação / outros);
  2) CLASSIFICADOR (YOLO-cls treinado no GTSRB, 43 tipos): para cada sinal
     encontrado, diz QUAL é (limite 50, cedência, STOP, rotunda, obras...).

Os dois datasets seguem a convenção europeia de Viena (a mesma dos sinais
portugueses). Os modelos geram-se com treino/treinar_sinais.py.

ReconhecedorSinais(det_pt, cls_pt).detetar(frame) -> {'bb':[...], 'sc':[...], 'tipo':[...]}
(mesmo formato do detetor clássico, para o resto do pipeline não mudar).
"""
from pathlib import Path

# 43 classes do GTSRB/GTSDB (índice = ClassId oficial), em português
NOMES_PT = [
    "limite 20", "limite 30", "limite 50", "limite 60", "limite 70",
    "limite 80", "fim limite 80", "limite 100", "limite 120",
    "proibido ultrapassar", "proibido ultrapassar (pesados)",
    "cruzamento c/ prioridade", "estrada com prioridade", "cedência de passagem",
    "STOP", "trânsito proibido", "proibido pesados", "sentido proibido",
    "perigo", "curva à esquerda", "curva à direita", "curva dupla",
    "piso irregular", "piso escorregadio", "estreitamento à direita", "obras",
    "semáforos", "passadeira", "crianças", "ciclistas", "gelo/neve",
    "animais selvagens", "fim de proibições", "virar à direita",
    "virar à esquerda", "seguir em frente", "frente ou direita",
    "frente ou esquerda", "manter a direita", "manter a esquerda", "rotunda",
    "fim proibição ultrapassar", "fim proib. ultrapassar (pesados)",
]

# categorias do benchmark GTSDB
CATEGORIAS_PT = ["proibição", "perigo", "obrigação", "outros"]
_PROIBICAO = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
_PERIGO = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
_OBRIGACAO = {33, 34, 35, 36, 37, 38, 39, 40}
CATEGORIA = {c: (0 if c in _PROIBICAO else 1 if c in _PERIGO else 2 if c in _OBRIGACAO else 3)
             for c in range(43)}


class ReconhecedorSinais:
    def __init__(self, det_pt, cls_pt=None, conf=0.30, imgsz=1280,
                 conf_cls=0.50, margem=0.15):
        from ultralytics import YOLO
        self.det = YOLO(det_pt)
        self.cls = YOLO(cls_pt) if cls_pt and Path(cls_pt).is_file() else None
        self.conf, self.imgsz = conf, imgsz
        self.conf_cls, self.margem = conf_cls, margem

    def detetar(self, frame):
        out = {"bb": [], "sc": [], "tipo": []}
        r = self.det.predict(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        h, w = frame.shape[:2]
        caixas, recortes, idx_rec = [], [], []
        for i, b in enumerate(r.boxes):
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            caixas.append((x1, y1, x2, y2, int(b.cls[0]), float(b.conf[0])))
            if self.cls is not None:
                mx, my = (x2 - x1) * self.margem, (y2 - y1) * self.margem
                rec = frame[max(0, int(y1 - my)):min(h, int(y2 + my)),
                            max(0, int(x1 - mx)):min(w, int(x2 + mx))]
                if rec.size:
                    recortes.append(rec); idx_rec.append(i)

        tipos = [self._nome_det(c) for *_, c, _ in caixas]
        if recortes:   # 2.ª fase: classificar todos os recortes de uma vez
            for i, rr in zip(idx_rec, self.cls.predict(recortes, imgsz=64, verbose=False)):
                if float(rr.probs.top1conf) >= self.conf_cls:
                    tipos[i] = self._nome_cls(int(rr.probs.top1))

        for (x1, y1, x2, y2, _, s), t in zip(caixas, tipos):
            out["bb"].append([x1, y1, x2 - x1, y2 - y1])
            out["sc"].append(s)
            out["tipo"].append(t)
        return out

    def _nome_det(self, c):
        nome = self.det.names.get(c, str(c)) if isinstance(self.det.names, dict) else self.det.names[c]
        return f"sinal ({nome})"

    def _nome_cls(self, i):
        nome = self.cls.names.get(i, str(i)) if isinstance(self.cls.names, dict) else self.cls.names[i]
        try:                                   # pastas "14_stop" -> índice 14
            return NOMES_PT[int(str(nome).split("_")[0])]
        except (ValueError, IndexError):
            return str(nome)
