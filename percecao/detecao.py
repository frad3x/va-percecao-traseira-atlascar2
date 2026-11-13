"""
detecao.py — deteção de agentes com YOLO (Ultralytics) + RASTREIO (IDs).

Substitui, num só passo, o detetor + o rastreador que no MATLAB eram separados:
o modo .track() do Ultralytics (ByteTrack) deteta E segue os objetos, dando um
ID estável a cada um — essencial para o TTC e para a trajetória.

detetar(frame) devolve um dict por grupo:
  {'veic':  {'bb':[[x,y,w,h],...], 'sc':[...], 'id':[...], 'tipo':[...]},
   'peoes': {...}, 'sinais': {...}}
bb = [x, y, w, h] com (x,y) no canto superior esquerdo (píxeis).
"""
from ultralytics import YOLO

# COCO (nomes do YOLO) -> grupos do projeto
GRUPO = {
    "car": "veic", "truck": "veic", "bus": "veic",
    "person": "peoes", "bicycle": "peoes", "motorcycle": "peoes",
    "stop sign": "sinais",
}


def _vazio():
    return {"bb": [], "sc": [], "id": [], "tipo": []}


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

        grupos = {"veic": _vazio(), "peoes": _vazio(), "sinais": _vazio()}
        if r.boxes is None or len(r.boxes) == 0:
            return grupos

        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        if r.boxes.id is not None:
            ids = r.boxes.id.cpu().numpy().astype(int)
        else:
            ids = [-1] * len(cls)

        for i in range(len(cls)):
            nome = self.names[int(cls[i])]
            g = GRUPO.get(nome)
            if g is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
            grupos[g]["bb"].append([x1, y1, x2 - x1, y2 - y1])
            grupos[g]["sc"].append(float(conf[i]))
            grupos[g]["id"].append(int(ids[i]))
            grupos[g]["tipo"].append(nome)
        return grupos
