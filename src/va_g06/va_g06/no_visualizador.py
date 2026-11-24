#!/usr/bin/env python3
"""
Nó 'visualizador' — a REPRESENTAÇÃO VISUAL da perceção, separada do processamento.

Não deteta nada: só usa o que o nó 'percecao' publica (e a imagem da câmara, para
desenhar por cima). Assim a perceção pode ser usada por outros sem o desenho, e o
desenho pode correr noutra máquina.

SUBSCREVE
  <image_topic>          sensor_msgs/CompressedImage (ou Image)   a câmara
  percecao/agentes       va_percecao_msgs/Agentes
  percecao/pavimento     va_percecao_msgs/Pavimento
  percecao/seguranca     va_percecao_msgs/EstadoSeguranca

PUBLICA (no namespace do grupo, por omissão /g06/...)
  visualizacao/imagem_anotada/compressed  sensor_msgs/CompressedImage
        a imagem com o painel, as caixas, as linhas, a faixa do carro e a vista de cima
        (o mesmo desenho da aplicação off-line, percecao/desenho.py)
  visualizacao/marcadores                 visualization_msgs/MarkerArray
        os agentes no chão em 3D (camera_tras_link), para o rviz/Foxglove

As quatro mensagens de uma imagem têm o MESMO carimbo de tempo (a perceção copia o da
imagem): o nó junta-as por esse carimbo e desenha quando tem as quatro.
"""
import os
import sys
from collections import OrderedDict

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image
from va_percecao_msgs.msg import Agentes, EstadoSeguranca, Pavimento
from visualization_msgs.msg import MarkerArray


def _importar_percecao(pasta):
    try:
        import percecao  # noqa: F401
    except ImportError:
        if pasta and os.path.isdir(os.path.join(pasta, "percecao")):
            sys.path.insert(0, pasta)
        else:
            raise


class NoVisualizador(Node):
    def __init__(self):
        super().__init__("visualizador")
        P = self.declare_parameter
        P("image_topic", "/atlascar2/camera_tras/image_raw/compressed")
        P("comprimida", True)
        P("pasta_projeto", os.environ.get("VA_PROJETO", "/ws/projeto"))
        P("altura_camara", 1.75)              # m (marcadores 3D e geometria das linhas)
        P("frame_corpo", "camera_tras_link")  # referencial dos marcadores 3D
        P("limiar_risco", 1.0)                # TTC (s): caixa vermelha se <= isto
        P("camara_traseira", True)
        P("estacionamento", False)            # guias de estacionamento no desenho
        P("jpeg_qualidade", 80)
        P("linhas_params", "{}")              # o mesmo JSON da perceção (horizonte, ...)
        g = lambda n: self.get_parameter(n).value
        _importar_percecao(g("pasta_projeto"))
        import json
        from percecao import desenho
        from percecao.estabilizar import Estabilizador
        from va_g06 import conversao, reconstrucao
        self.desenho, self.conv, self.rec = desenho, conversao, reconstrucao
        self.memoria_sinais = desenho.MemoriaSinais()
        self.estab = Estabilizador()          # caixas suaves no tempo (só no desenho)
        self.altura = float(g("altura_camara"))
        self.frame_corpo = g("frame_corpo")
        self.limiar = float(g("limiar_risco"))
        self.traseira = bool(g("camara_traseira"))
        self.estacionamento = bool(g("estacionamento"))
        self.jpeg_q = int(g("jpeg_qualidade"))
        self.linhas_params = json.loads(g("linhas_params"))
        self.linhas_params.setdefault("altura", self.altura)

        self.pub_img = self.create_publisher(CompressedImage,
                                             "visualizacao/imagem_anotada/compressed", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "visualizacao/marcadores", 10)

        self._pend = OrderedDict()            # carimbo -> {img, ag, pav, seg}
        tipo = CompressedImage if g("comprimida") else Image
        self.create_subscription(tipo, g("image_topic"), lambda m: self._junta(m, "img"),
                                 qos_profile_sensor_data)
        self.create_subscription(Agentes, "percecao/agentes", self._on_agentes, 10)
        self.create_subscription(Pavimento, "percecao/pavimento", lambda m: self._junta(m, "pav"), 10)
        self.create_subscription(EstadoSeguranca, "percecao/seguranca",
                                 lambda m: self._junta(m, "seg"), 10)
        self.get_logger().info("visualizador pronto — a juntar a câmara e a perceção")

    # ------------------------------------------------------------------
    def _on_agentes(self, msg):
        # os marcadores 3D só precisam dos agentes
        self.pub_mk.publish(self.conv.marcadores(msg, self.altura, self.frame_corpo, self.limiar))
        self._junta(msg, "ag")

    def _junta(self, msg, tipo):
        st = msg.header.stamp
        k = (st.sec, st.nanosec)
        d = self._pend.setdefault(k, {})
        d[tipo] = msg
        while len(self._pend) > 60:           # imagens que a perceção saltou: esquecem-se
            self._pend.popitem(last=False)
        if len(d) == 4:
            del self._pend[k]
            for velho in [c for c in self._pend if c < k]:   # mais antigos já não chegam
                del self._pend[velho]
            if self.pub_img.get_subscription_count() > 0:
                try:
                    self._desenhar(d)
                except Exception as e:        # nunca deixa o nó morrer
                    self.get_logger().error(f"erro a desenhar: {e!r}", throttle_duration_sec=2)

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
            return cv2.cvtColor(buf.reshape(msg.height, msg.width, 2), cv2.COLOR_YUV2BGR_YUYV)
        return None

    def _desenhar(self, d):
        import cv2
        frame = self._descodificar(d["img"])
        if frame is None:
            return
        res = self.rec.resultados(d["ag"], d["pav"], d["seg"], frame.shape,
                                  self.linhas_params, self.traseira)
        res["estab"] = self.estab.atualizar(res)   # números de exibição pequenos incluídos
        st = d["img"].header.stamp
        res["sinais_vistos"] = self.memoria_sinais.atualizar(frame, res["sinais"],
                                                             st.sec + st.nanosec * 1e-9)
        anot = self.desenho.desenhar(frame, res, ttc_risco=self.limiar,
                                     estacionamento=self.estacionamento)
        ok, enc = cv2.imencode(".jpg", anot, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        if ok:
            self.pub_img.publish(CompressedImage(header=d["img"].header, format="jpeg",
                                                 data=enc.tobytes()))


def main(args=None):
    rclpy.init(args=args)
    no = NoVisualizador()
    try:
        rclpy.spin(no)
    except KeyboardInterrupt:
        pass
    finally:
        no.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
