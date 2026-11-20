"""
sinais_treinados.py — reconhecimento de sinais com modelos TREINADOS (2 fases).

  1) DETETOR (YOLO treinado no GTSDB): encontra os sinais na imagem e dá a
     categoria (proibição / perigo / obrigação / outros);
  2) CLASSIFICADOR (YOLO-cls treinado no GTSRB, 43 tipos): para cada sinal
     encontrado, diz QUAL é (limite 50, cedência, STOP, rotunda, obras...).

Os dois datasets seguem a convenção europeia de Viena (a mesma dos sinais
portugueses). Os modelos geram-se com treino/treinar_sinais.py.

3) (opcional) CLASSIFICADOR PORTUGUÊS (YOLO-cls treinado com imagens sintéticas dos
   159 sinais A/B/C/D do Regulamento de Sinalização do Trânsito; treino/sinais_pt.py).
   Os dois classificadores combinam-se assim:
     - nos tipos que o GTSRB conhece (41 com equivalente PT) manda o GTSRB — treinado
       com fotografias reais, acerta ~97%; o nome passa ao código PT ("B2 Stop");
     - nos sinais portugueses que o GTSRB NÃO conhece (lomba, estacionamento
       proibido, proibido virar...), manda o classificador português, se confiante.
   Medido em 5760 fotografias reais (GTSRB, 41 tipos comuns): só GTSRB 96,7%; só o
   português 62%; a combinação (conf_pt = 0,80) 93,9% — e passa a reconhecer os
   ~115 sinais portugueses que o GTSRB não tem (83% em imagens sintéticas novas).

ReconhecedorSinais(det_pt, cls_pt, pt_pt).detetar(frame) -> {'bb':[...], 'sc':[...], 'tipo':[...]}
(mesmo formato do detetor clássico, para o resto do pipeline não mudar).
"""
import json
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

# GTSRB (índice) -> código PT equivalente (Regulamento de Sinalização do Trânsito)
GTSRB_PARA_PT = {
    0: "C13-20", 1: "C13-30", 2: "C13-50", 3: "C13-60", 4: "C13-70", 5: "C13-80",
    7: "C13-100", 8: "C13-120", 9: "C14a", 10: "C14b", 11: "B8", 12: "B3", 13: "B1", 14: "B2",
    15: "C2", 16: "C3b", 17: "C1", 18: "A29", 19: "A1b", 20: "A1a", 21: "A1d", 22: "A2c",
    23: "A5", 25: "A23", 26: "A22", 27: "A16a", 28: "A14", 29: "A17", 30: "A11", 31: "A19b",
    32: "C20a", 33: "D1e", 34: "D1d", 35: "D1c", 36: "D2b", 37: "D2a", 38: "D3a", 39: "D3b",
    40: "D4", 41: "C20c", 42: "C20d",
}
COBERTOS_GTSRB = set(GTSRB_PARA_PT.values())

# categorias do benchmark GTSDB
CATEGORIAS_PT = ["proibição", "perigo", "obrigação", "outros"]
_PROIBICAO = {0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16}
_PERIGO = {11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
_OBRIGACAO = {33, 34, 35, 36, 37, 38, 39, 40}
CATEGORIA = {c: (0 if c in _PROIBICAO else 1 if c in _PERIGO else 2 if c in _OBRIGACAO else 3)
             for c in range(43)}


class ReconhecedorSinais:
    def __init__(self, det_pt, cls_pt=None, pt_pt=None, conf=0.30, imgsz=1280,
                 conf_cls=0.50, conf_pt=0.80, margem=0.15):
        from ultralytics import YOLO
        self.det = YOLO(det_pt)
        self.cls = YOLO(cls_pt) if cls_pt and Path(cls_pt).is_file() else None
        self.pt = YOLO(pt_pt) if pt_pt and Path(pt_pt).is_file() else None
        self.nomes_pt = {}
        if self.pt is not None:
            fich = Path(pt_pt).with_name("sinais_pt_nomes.json")
            if fich.is_file():
                self.nomes_pt = json.loads(fich.read_text(encoding="utf-8"))
        self.conf_pt = conf_pt
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
            if self.cls is not None or self.pt is not None:
                mx, my = (x2 - x1) * self.margem, (y2 - y1) * self.margem
                rec = frame[max(0, int(y1 - my)):min(h, int(y2 + my)),
                            max(0, int(x1 - mx)):min(w, int(x2 + mx))]
                if rec.size:
                    recortes.append(rec); idx_rec.append(i)

        tipos = [self._nome_det(c) for *_, c, _ in caixas]
        for i, t in zip(idx_rec, self.classificar(recortes)):
            if t:
                tipos[i] = t

        for (x1, y1, x2, y2, _, s), t in zip(caixas, tipos):
            out["bb"].append([x1, y1, x2 - x1, y2 - y1])
            out["sc"].append(s)
            out["tipo"].append(t)
        return out

    def classificar(self, recortes):
        """Nome de cada recorte (ou None): GTSRB nos tipos que conhece, português nos
        restantes (ver o topo do ficheiro)."""
        if not recortes:
            return []
        out = [None] * len(recortes)
        g = self.cls.predict(recortes, imgsz=64, verbose=False) if self.cls is not None else []
        p = self.pt.predict(recortes, imgsz=64, verbose=False) if self.pt is not None else []
        for k in range(len(recortes)):
            if p:
                cod = self._codigo_pt(int(p[k].probs.top1))
                if cod not in COBERTOS_GTSRB and float(p[k].probs.top1conf) >= self.conf_pt:
                    out[k] = self._nome_pt(cod)          # sinal que o GTSRB não conhece
                    continue
            if g and float(g[k].probs.top1conf) >= self.conf_cls:
                gi = self._indice_gtsrb(int(g[k].probs.top1))
                if self.pt is not None and gi in GTSRB_PARA_PT:
                    out[k] = self._nome_pt(GTSRB_PARA_PT[gi])
                else:
                    out[k] = NOMES_PT[gi] if 0 <= gi < len(NOMES_PT) else None
        return out

    def _nome_det(self, c):
        nome = self.det.names.get(c, str(c)) if isinstance(self.det.names, dict) else self.det.names[c]
        return f"sinal ({nome})"

    def _codigo_pt(self, i):
        """pasta '012_C13-50__limite...' -> 'C13-50'."""
        pasta = str(self.pt.names[i])
        return pasta.split("_", 1)[1].split("__")[0] if "__" in pasta else pasta

    def _nome_pt(self, cod):
        nome = self.nomes_pt.get(cod)
        return f"{cod} {nome}" if nome else cod

    def _indice_gtsrb(self, i):
        nome = self.cls.names.get(i, str(i)) if isinstance(self.cls.names, dict) else self.cls.names[i]
        try:                                   # pastas "14_stop" -> índice 14
            return int(str(nome).split("_")[0])
        except ValueError:
            return -1
