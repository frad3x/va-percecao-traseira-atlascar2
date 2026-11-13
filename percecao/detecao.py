"""
detecao.py — deteção de agentes com YOLO (Ultralytics) + RASTREIO (IDs).

Substitui, num só passo, o detetor + o rastreador que no MATLAB eram separados:
o modo .track() do Ultralytics (ByteTrack) deteta E segue os objetos, dando um
ID estável a cada um — essencial para o TTC e para a trajetória.

detetar(frame) devolve um dict por grupo (os 4 do plano de trabalho):
  {'veic':       {'bb':[[x,y,w,h],...], 'sc':[...], 'id':[...], 'tipo':[...]},
   'duas_rodas': {...},   # bicicletas e motas (o condutor é fundido na caixa)
   'peoes':      {...},   # só pessoas A PÉ
   'sinais':     {...}}
bb = [x, y, w, h] com (x,y) no canto superior esquerdo (píxeis).
"""
from ultralytics import YOLO

# COCO (nomes do YOLO) -> grupos do projeto
GRUPO = {
    "car": "veic", "truck": "veic", "bus": "veic",
    "bicycle": "duas_rodas", "motorcycle": "duas_rodas",
    "person": "peoes",
    "stop sign": "sinais",
}
GRUPOS = ("veic", "duas_rodas", "peoes", "sinais")

# nome PT de cada classe COCO (para as etiquetas)
NOME_PT = {"car": "carro", "truck": "camiao", "bus": "autocarro",
           "bicycle": "bicicleta", "motorcycle": "mota", "person": "peao",
           "stop sign": "STOP"}


def _vazio():
    return {"bb": [], "sc": [], "id": [], "tipo": []}


def _sobreposicao(a, b):
    """Fração da caixa a (x1,y1,x2,y2) que está dentro de b."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return ix * iy / area


class Detetor:
    def __init__(self, modelo="yolo11n.pt", conf=0.25, imgsz=640,
                 usar_track=True, tracker="bytetrack.yaml"):
        self.model = YOLO(modelo)
        self.conf = conf
        self.imgsz = imgsz
        self.usar_track = usar_track
        self.tracker = tracker
        self.names = self.model.names

    def detetar(self, frame):
        if self.usar_track:
            r = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz,
                                 tracker=self.tracker, verbose=False)[0]
        else:
            r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                   verbose=False)[0]

        grupos = {g: _vazio() for g in GRUPOS}
        if r.boxes is None or len(r.boxes) == 0:
            return grupos

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            ids = [-1] * len(cls)

        dets = []
        for i in range(len(cls)):
            nome = self.names[int(cls[i])]
            g = GRUPO.get(nome)
            if g is not None:
                dets.append([g, nome, [float(v) for v in xyxy[i]], float(conf[i]), int(ids[i])])

        # Condutor de bicicleta/mota: o YOLO vê "person" + "bicycle". Não é um peão:
        # a pessoa é absorvida na caixa da duas-rodas (fica uma só caixa: o ciclista).
        rodas = [d for d in dets if d[0] == "duas_rodas"]
        manter = []
        for d in dets:
            if d[0] == "peoes":
                p = d[2]
                pe = [p[0], p[1] + 0.5 * (p[3] - p[1]), p[2], p[3]]   # metade de baixo
                dono = next((r for r in rodas if _sobreposicao(pe, r[2]) > 0.35), None)
                if dono is not None:
                    b = dono[2]
                    dono[2] = [min(b[0], p[0]), min(b[1], p[1]), max(b[2], p[2]), max(b[3], p[3])]
                    continue
            manter.append(d)

        # O mesmo veículo detetado 2x (ex. "truck" + "car", ou só a caixa de carga):
        # uma caixa quase toda DENTRO de outra do mesmo grupo e de altura parecida
        # é duplicada -> fica a de maior confiança.
        manter.sort(key=lambda d: -d[3])
        unicos = []
        for d in manter:
            a = d[2]
            dup = any(u[0] == d[0] != "sinais" and
                      (_sobreposicao(a, u[2]) > 0.85 or _sobreposicao(u[2], a) > 0.85) and
                      min(a[3] - a[1], u[2][3] - u[2][1]) > 0.7 * max(a[3] - a[1], u[2][3] - u[2][1])
                      for u in unicos)
            if not dup:
                unicos.append(d)

        for g, nome, (x1, y1, x2, y2), sc, oid in unicos:
            grupos[g]["bb"].append([x1, y1, x2 - x1, y2 - y1])
            grupos[g]["sc"].append(sc)
            grupos[g]["id"].append(oid)
            grupos[g]["tipo"].append(nome)
        return grupos
