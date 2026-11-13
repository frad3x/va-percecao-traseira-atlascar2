#!/usr/bin/env python3
"""
Nó 'percecao' — a perceção da câmara traseira, AO VIVO, em ROS 2.

SUBSCREVE
  <image_topic>  (sensor_msgs/CompressedImage ou sensor_msgs/Image)
                 por omissão /atlascar2/camera_tras/image_raw/compressed (o driver usb_cam)
  <velocidade_topic> (std_msgs/Float32, m/s, OPCIONAL) velocidade do carro
                 -> distância de segurança pela regra dos 2 segundos

PUBLICA (no namespace do nó, por omissão /atlascar2/camera_tras/percecao/...)
  agentes              va_percecao_msgs/Agentes        veículos, duas rodas, peões e sinais
                                                       com ID, distância, faixa, TTC
  deteccoes            vision_msgs/Detection2DArray    o mesmo em tipos normalizados
  pavimento            va_percecao_msgs/Pavimento      linhas da faixa (+tipo), passadeiras,
                                                       linhas de paragem
  seguranca            va_percecao_msgs/EstadoSeguranca distância ao veículo de trás, mudança
                                                       de faixa, espaço livre, avisos
  marcadores           visualization_msgs/MarkerArray  agentes em 3D (rviz)
  imagem_anotada/compressed  sensor_msgs/CompressedImage  imagem com o overlay

O processamento é o MESMO da aplicação off-line (pacote Python 'percecao',
classe Pipeline). Corre numa thread à parte e trata sempre a imagem MAIS RECENTE:
se o modelo for mais lento que a câmara, as imagens intermédias são saltadas
(nunca se acumula atraso).
"""
import json
import os
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32
from va_percecao_msgs.msg import Agentes, EstadoSeguranca, Pavimento
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray


def _importar_percecao(pasta):
    """O pacote Python 'percecao' vem do projeto (instalado com pip, ou na pasta dada)."""
    try:
        import percecao  # noqa: F401
    except ImportError:
        if pasta and os.path.isdir(os.path.join(pasta, "percecao")):
            sys.path.insert(0, pasta)
        else:
            raise


class NoPercecao(Node):
    def __init__(self):
        super().__init__("percecao")
        P = self.declare_parameter
        P("image_topic", "/atlascar2/camera_tras/image_raw/compressed")
        P("comprimida", True)                 # False -> sensor_msgs/Image (raw)
        P("velocidade_topic", "/atlascar2/velocidade")
        P("pasta_projeto", os.environ.get("VA_PROJETO", "/ws/projeto"))
        P("modelo", "yolo11m.pt")             # sem GPU: yolo11n.pt + imgsz 640
        P("imgsz", 1280)
        P("conf", 0.25)
        P("dispositivo", "")                  # "" automático | "cpu" | "0"
        P("sinais_det", "modelos/sinais_det.pt")
        P("sinais_cls", "modelos/sinais_cls.pt")
        P("linhas", True)
        P("linhas_params", "{}")              # JSON, ex. '{"altura": 1.1}'
        P("seguranca_params", "{}")           # JSON, ex. '{"dist_seguranca": 15}'
        P("limiar_risco", 4.0)                # TTC (s)
        P("camara_traseira", True)            # False em vídeos de câmara frontal
        P("altura_camara", 1.1)               # m (para os marcadores 3D e escala)
        P("publicar_imagem", True)
        P("jpeg_qualidade", 70)
        P("estacionamento", False)            # guias de estacionamento no overlay
        g = lambda n: self.get_parameter(n).value

        pasta = g("pasta_projeto")
        _importar_percecao(pasta)
        from percecao import desenho
        from percecao.pipeline import Config, Pipeline
        from va_percecao import conversao
        self.desenho, self.conv = desenho, conversao

        def no_projeto(p):
            return p if (not p or os.path.isabs(p) or os.path.exists(p)) else os.path.join(pasta, p)

        linhas_params = json.loads(g("linhas_params"))
        linhas_params.setdefault("altura", float(g("altura_camara")))
        seg_params = json.loads(g("seguranca_params"))
        seg_params["traseira"] = bool(g("camara_traseira"))
        self.limiar = float(g("limiar_risco"))
        self.altura = float(g("altura_camara"))
        self.pub_img_on = bool(g("publicar_imagem"))
        self.jpeg_q = int(g("jpeg_qualidade"))
        self.estacionamento = bool(g("estacionamento"))

        self.get_logger().info(f"a carregar os modelos ({g('modelo')}, imgsz {g('imgsz')})...")
        self.pipe = Pipeline(Config(
            modelo=g("modelo"), imgsz=int(g("imgsz")), conf=float(g("conf")),
            dispositivo=g("dispositivo") or None,
            linhas=bool(g("linhas")), linhas_params=linhas_params,
            sinais_det=no_projeto(g("sinais_det")), sinais_cls=no_projeto(g("sinais_cls")),
            limiar_risco=self.limiar, seguranca_params=seg_params),
            log=lambda m: self.get_logger().info(m))
        self.lado_carro = (self.pipe.seguranca.lado_carro if self.pipe.seguranca else
                           (lambda l: {"img_esq": "direita", "img_dir": "esquerda"}[l]))

        # ---- publicadores ----
        q = 10
        self.pub_agentes = self.create_publisher(Agentes, "percecao/agentes", q)
        self.pub_det = self.create_publisher(Detection2DArray, "percecao/deteccoes", q)
        self.pub_pav = self.create_publisher(Pavimento, "percecao/pavimento", q)
        self.pub_seg = self.create_publisher(EstadoSeguranca, "percecao/seguranca", q)
        self.pub_mk = self.create_publisher(MarkerArray, "percecao/marcadores", q)
        self.pub_img = self.create_publisher(CompressedImage,
                                             "percecao/imagem_anotada/compressed", q)

        # ---- subscrições ----
        tipo = CompressedImage if g("comprimida") else Image
        self.create_subscription(tipo, g("image_topic"), self._on_imagem, qos_profile_sensor_data)
        if g("velocidade_topic"):
            self.create_subscription(Float32, g("velocidade_topic"), self._on_velocidade, 10)

        # ---- thread de processamento (sempre a imagem mais recente) ----
        self._ultima = None
        self._cv = threading.Condition()
        self._parar = False
        self._n_rec = self._n_proc = 0
        self._t_stat = time.time()
        self._thread = threading.Thread(target=self._ciclo, daemon=True)
        self._thread.start()
        self.get_logger().info(f"perceção pronta — a ouvir {g('image_topic')}")

    # ------------------------------------------------------------------
    def _on_imagem(self, msg):
        with self._cv:
            self._ultima = msg            # substitui a anterior se ainda não foi tratada
            self._n_rec += 1
            self._cv.notify()

    def _on_velocidade(self, msg):
        self.pipe.velocidade(msg.data)

    def _descodificar(self, msg):
        import cv2
        if isinstance(msg, CompressedImage):
            return cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        buf = np.frombuffer(bytes(msg.data), np.uint8)
        if msg.encoding in ("bgr8", "rgb8"):
            img = buf.reshape(msg.height, msg.step // 3, 3)[:, :msg.width]
            return img[:, :, ::-1].copy() if msg.encoding == "rgb8" else img.copy()
        if msg.encoding == "mono8":
            return cv2.cvtColor(buf.reshape(msg.height, msg.step)[:, :msg.width], cv2.COLOR_GRAY2BGR)
        if msg.encoding in ("yuv422", "yuv422_yuy2", "yuyv"):
            yuv = buf.reshape(msg.height, msg.width, 2)
            return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUYV)
        self.get_logger().warn(f"encoding não suportado: {msg.encoding}", throttle_duration_sec=5)
        return None

    def _ciclo(self):
        while rclpy.ok() and not self._parar:
            with self._cv:
                while self._ultima is None and not self._parar:
                    self._cv.wait(timeout=0.5)
                msg, self._ultima = self._ultima, None
            if msg is None:
                continue
            try:
                self._processar(msg)
            except Exception as e:                       # nunca deixa o nó morrer
                self.get_logger().error(f"erro a processar a imagem: {e!r}",
                                        throttle_duration_sec=2)

    def _processar(self, msg):
        frame = self._descodificar(msg)
        if frame is None:
            return
        st = msg.header.stamp
        t = st.sec + st.nanosec * 1e-9 or self.get_clock().now().nanoseconds * 1e-9
        res = self.pipe.processar(frame, t)
        h = msg.header
        if not h.frame_id:
            h.frame_id = "camera_tras_link"

        C = self.conv
        ag = C.agentes(res, h, self.lado_carro)
        self.pub_agentes.publish(ag)
        self.pub_det.publish(C.deteccoes(ag))
        self.pub_pav.publish(C.pavimento(res, h, self.lado_carro))
        self.pub_seg.publish(C.seguranca(res, h))
        self.pub_mk.publish(C.marcadores(res, h, self.altura))
        if self.pub_img_on and self.pub_img.get_subscription_count() > 0:
            import cv2
            anot = self.desenho.desenhar(frame, res, ttc_risco=self.limiar,
                                         estacionamento=self.estacionamento)
            ok, enc = cv2.imencode(".jpg", anot, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
            if ok:
                self.pub_img.publish(CompressedImage(header=h, format="jpeg", data=enc.tobytes()))

        self._n_proc += 1
        agora = time.time()
        if agora - self._t_stat >= 10.0:
            dt = agora - self._t_stat
            self.get_logger().info(
                f"{self._n_proc / dt:.1f} fps processadas de {self._n_rec / dt:.1f} recebidas | "
                f"{len(ag.agentes)} agentes | seguranca: {res.get('seg', {}).get('estado_dist')}")
            self._n_rec = self._n_proc = 0
            self._t_stat = agora

    def destroy_node(self):
        self._parar = True
        with self._cv:
            self._cv.notify_all()
        self._thread.join(timeout=2.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    no = NoPercecao()
    try:
        rclpy.spin(no)
    except KeyboardInterrupt:
        pass
    finally:
        no.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
