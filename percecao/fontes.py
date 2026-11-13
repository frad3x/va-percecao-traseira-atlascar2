"""
fontes.py — abstrai a origem das imagens (vídeo / pasta de imagens / rosbag MCAP).

Dá uma classe Fonte com:
  .n, .fps, .largura, .altura, .tipo
  .iterar()  -> gera (idx, frame_BGR) por ordem (para o processamento)
  .ler(k)    -> devolve a k-ésima frame (acesso aleatório, para o player)

Para um rosbag MCAP, as frames são exportadas UMA vez para um .avi (em
dados/frames_bag.avi) e a partir daí é tratado como vídeo — assim o player pode
recuar/saltar e não se depende de nenhuma toolbox de ROS.
"""
import glob
import os

import cv2
import numpy as np


# ------------------------- rosbag MCAP -------------------------
def encontrar_mcap(fonte):
    if os.path.isfile(fonte) and fonte.lower().endswith(".mcap"):
        return fonte
    if os.path.isdir(fonte):
        achados = sorted(glob.glob(os.path.join(fonte, "*.mcap")))
        if achados:
            return achados[0]
    return None


def _topico_imagem(mcap_path):
    from mcap.reader import make_reader
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
        schemas = {sid: s.name for sid, s in summary.schemas.items()}
        cands = []
        for c in summary.channels.values():
            nome = schemas.get(c.schema_id, "")
            if nome.endswith("CompressedImage"):
                cands.append((0, c.topic))
            elif nome.endswith("msg/Image"):
                cands.append((1, c.topic))
    assert cands, "Nenhum tópico de imagem no bag."
    cands.sort()
    return cands[0][1]


def _frames_do_mcap(mcap_path, topico):
    from mcap_ros2.reader import read_ros2_messages
    for m in read_ros2_messages(mcap_path, topics=[topico]):
        msg = m.ros_msg
        tipo = m.schema.name
        if tipo.endswith("CompressedImage"):
            buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        else:
            h, w = int(msg.height), int(msg.width)
            data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            enc = getattr(msg, "encoding", "rgb8")
            if enc in ("rgb8", "bgr8"):
                img = data.reshape(h, w, 3)
                if enc == "rgb8":
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif enc == "mono8":
                img = cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
            else:
                img = data.reshape(h, w, -1)[:, :, :3]
        try:
            t = m.log_time / 1e9
        except Exception:
            t = None
        if img is not None:
            yield img, t


def exportar_mcap_para_avi(mcap_path, avi_saida, fps=0.0):
    """Descodifica o bag e grava as frames num .avi MJPG. Devolve (caminho, fps, n)."""
    os.makedirs(os.path.dirname(os.path.abspath(avi_saida)), exist_ok=True)
    topico = _topico_imagem(mcap_path)
    print(f"[fontes] rosbag: {mcap_path}  topico={topico}")
    vw = None
    t0 = None
    prim = None
    n = 0
    fps_real = fps
    for img, t in _frames_do_mcap(mcap_path, topico):
        if vw is None and prim is None and fps <= 0:
            t0, prim = t, img
        elif vw is None:
            if fps > 0:
                fps_real = fps
                vw = _abrir_writer(avi_saida, img, fps_real)
                vw.write(img)
            else:
                dt = (t - t0) if (t is not None and t0 is not None) else None
                fps_real = (1.0 / dt) if (dt and dt > 0) else 15.0
                fps_real = float(min(max(fps_real, 1.0), 60.0))
                vw = _abrir_writer(avi_saida, prim, fps_real)
                vw.write(prim)
                vw.write(img)
        else:
            vw.write(img)
        n += 1
        if n % 200 == 0:
            print(f"[fontes] {n} frames exportadas...", flush=True)
    if vw is None and prim is not None:
        fps_real = 15.0
        vw = _abrir_writer(avi_saida, prim, fps_real)
        vw.write(prim)
    if vw is not None:
        vw.release()
    print(f"[fontes] vídeo exportado: {avi_saida} ({n} frames, {fps_real:.1f} fps)")
    return avi_saida, fps_real, n


def _abrir_writer(caminho, img, fps):
    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(caminho, fourcc, float(fps), (int(w), int(h)))
    if not vw.isOpened():
        raise RuntimeError(f"Não consegui abrir o VideoWriter em {caminho}")
    return vw


# ------------------------- classe Fonte -------------------------
EXT_VIDEO = (".avi", ".mp4", ".mov", ".mkv", ".ogg", ".m4v", ".wmv", ".mpg", ".mpeg")


class Fonte:
    def __init__(self, caminho, topico=None, avi_cache=None):
        self.caminho = caminho
        self.tipo = None
        self._cap = None
        self._ficheiros = None

        ext = os.path.splitext(str(caminho))[1].lower()
        mcap = encontrar_mcap(caminho) if os.path.isdir(caminho) or ext == ".mcap" else None

        if mcap:                                   # rosbag -> exporta para .avi
            if avi_cache is None:
                avi_cache = os.path.join("dados", "frames_bag.avi")
            avi, fps, _ = exportar_mcap_para_avi(mcap, avi_cache)
            self.caminho = avi
            self._abrir_video(self.caminho)
            if fps > 0:
                self.fps = fps
        elif ext in EXT_VIDEO:                      # vídeo
            self._abrir_video(caminho)
        elif os.path.isdir(caminho):               # pasta de imagens
            self._abrir_imagens(caminho)
        else:
            raise ValueError(f"Fonte não suportada: {caminho}")

    # ---- vídeo ----
    def _abrir_video(self, caminho):
        self.tipo = "video"
        self._cap = cv2.VideoCapture(caminho)
        if not self._cap.isOpened():
            raise RuntimeError(f"Não consegui abrir o vídeo: {caminho}")
        self._pos = 0                      # próxima frame que o cap vai devolver
        self.n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 15.0
        self.largura = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.altura = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- pasta de imagens ----
    def _abrir_imagens(self, pasta):
        self.tipo = "imagens"
        fich = []
        for padrao in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            fich += glob.glob(os.path.join(pasta, padrao))
        self._ficheiros = sorted(fich)
        if not self._ficheiros:
            raise RuntimeError(f"Sem imagens em {pasta}")
        im0 = cv2.imread(self._ficheiros[0])
        self.n = len(self._ficheiros)
        self.fps = 15.0
        self.altura, self.largura = im0.shape[:2]

    # ---- API ----
    def ler(self, k):
        """Frame k (0-based), acesso aleatório. Devolve BGR ou None.
        Em reprodução sequencial (k == posição atual) NÃO salta no ficheiro,
        o que torna o play fluido; só procura quando há um salto."""
        k = max(0, min(int(k), self.n - 1))
        if self.tipo == "video":
            if k != self._pos:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, k)
                self._pos = k
            ok, frame = self._cap.read()
            if ok:
                self._pos += 1
                return frame
            if k > 0:                              # última frame pode falhar -> tenta a anterior
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, k - 1)
                ok, frame = self._cap.read()
                self._pos = k if ok else self._pos
                return frame if ok else None
            return None
        return cv2.imread(self._ficheiros[k])

    def iterar(self):
        """Gera (idx, frame) por ordem, para o processamento."""
        if self.tipo == "video":
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._pos = 0
            idx = 0
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                self._pos += 1
                yield idx, frame
                idx += 1
        else:
            for idx, f in enumerate(self._ficheiros):
                img = cv2.imread(f)
                if img is not None:
                    yield idx, img

    def fechar(self):
        if self._cap is not None:
            self._cap.release()
