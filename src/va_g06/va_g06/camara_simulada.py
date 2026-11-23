"""Câmara traseira SIMULADA: publica imagens como o driver real (usb_cam), sem carro.

  - com `video` (ficheiro em /data): reproduz o vídeo em ciclo;
    `metade: esquerda` usa só a metade esquerda de cada imagem (vídeos gravados
    lado a lado, p. ex. "Orbbec RGB | Orbbec Depth");
  - sem vídeo: gera uma ESTRADA SINTÉTICA (asfalto, linha contínua à esquerda,
    tracejada à direita, em movimento) — chega para testar linhas, grelha e planeador.

Publica (no namespace do nó, normalmente /atlascar2/camera_tras):
  image_raw/compressed   sensor_msgs/CompressedImage   frame_id camera_tras_optical_frame
"""
import os

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


def estrada_sintetica(k, w=1280, h=720):
    """Imagem k de uma estrada vista por uma câmara traseira (sem dependências)."""
    img = np.zeros((h, w, 3), np.uint8)
    hz = int(0.365 * h)                                         # horizonte da câmara real
    img[:hz] = (200, 170, 140)                                  # céu
    img[hz:] = (70, 70, 70)                                     # asfalto
    fuga = (w // 2, hz)

    def linha(x_base, cor, tracejada):
        n = 40
        for i in range(n):
            if tracejada and ((i + k // 2) % 4) >= 2:
                continue
            t0, t1 = i / n, (i + 1) / n
            p0 = (int(fuga[0] + (x_base - fuga[0]) * t0 ** 1.0), int(hz + (h - hz) * t0))
            p1 = (int(fuga[0] + (x_base - fuga[0]) * t1 ** 1.0), int(hz + (h - hz) * t1))
            cv2.line(img, p0, p1, cor, max(1, int(14 * t1)), cv2.LINE_AA)

    linha(int(0.12 * w), (235, 235, 235), False)               # contínua (esq. da imagem)
    linha(int(0.88 * w), (235, 235, 235), True)                # tracejada (dir. da imagem)
    ruido = np.random.default_rng(k).integers(0, 12, img.shape, dtype=np.uint8)
    return cv2.add(img, ruido)


class CamaraSimulada(Node):

    def __init__(self):
        super().__init__('camara_simulada')
        self.declare_parameter('video', '')
        self.declare_parameter('metade', '')            # '' | esquerda | direita
        self.declare_parameter('fps', 15.0)
        self.declare_parameter('frame_id', 'camera_tras_optical_frame')
        self.declare_parameter('jpeg_qualidade', 80)
        self.pub = self.create_publisher(CompressedImage, 'image_raw/compressed', 10)
        video = self.get_parameter('video').value
        self.cap = None
        if video:
            if os.path.isfile(video):
                self.cap = cv2.VideoCapture(video)
                self.get_logger().info(f'a reproduzir {video} em ciclo')
            else:
                self.get_logger().warn(f'vídeo {video} não existe — estrada sintética')
        else:
            self.get_logger().info('sem vídeo — a publicar uma estrada sintética')
        self.k = 0
        self.create_timer(1.0 / float(self.get_parameter('fps').value), self.publicar)

    def _imagem(self):
        if self.cap is None:
            return estrada_sintetica(self.k)
        ok, img = self.cap.read()
        if not ok:                                       # fim do vídeo: volta ao início
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
            if not ok:
                return estrada_sintetica(self.k)
        metade = self.get_parameter('metade').value
        if metade in ('esquerda', 'direita'):
            m = img.shape[1] // 2
            img = img[:, :m] if metade == 'esquerda' else img[:, m:]
        return img

    def publicar(self):
        img = self._imagem()
        self.k += 1
        q = int(self.get_parameter('jpeg_qualidade').value)
        ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return
        msg = CompressedImage(format='jpeg', data=enc.tobytes())
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter('frame_id').value
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = CamaraSimulada()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
