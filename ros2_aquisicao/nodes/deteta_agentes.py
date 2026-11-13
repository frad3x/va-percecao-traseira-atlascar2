#!/usr/bin/env python3
"""Nó de processamento (ROS 2): deteção simples de peões na câmara traseira.

É o 2.º serviço do compose (driver + processamento), como o enunciado pede.
Subscreve a imagem comprimida, deteta peões com o detetor HOG do OpenCV
(clássico, leve, sem redes pesadas) e publica:
  /atlascar2/camera_tras/num_agentes                       (std_msgs/Int32)
  /atlascar2/camera_tras/agentes/image_raw/compressed      (imagem anotada)

Para VER o resultado: rviz2/Foxglove no tópico da imagem anotada, ou
  ros2 run image_view image_view --ros-args \
     -r image:=/atlascar2/camera_tras/agentes/image_raw/compressed \
     -p image_transport:=compressed

Nota: o processamento "a sério" (YOLO, linhas, sinais) é feito off-line em
MATLAB. Este nó é a versão ao-vivo mínima para a arquitetura da Parte 1.
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Int32


class DetetaAgentes(Node):
    def __init__(self):
        super().__init__('deteta_agentes')
        self.declare_parameter('every_n', 5)    # processa 1 em cada N frames (HOG é lento)
        self.declare_parameter('width', 640)    # redimensiona para acelerar
        self.declare_parameter('image_topic', '/atlascar2/camera_tras/image_raw/compressed')
        self.every_n = int(self.get_parameter('every_n').value)
        self.width = int(self.get_parameter('width').value)
        topic = self.get_parameter('image_topic').value

        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.i = 0

        self.sub = self.create_subscription(CompressedImage, topic, self.on_image, 10)
        self.pub_n = self.create_publisher(Int32, '/atlascar2/camera_tras/num_agentes', 10)
        self.pub_img = self.create_publisher(
            CompressedImage, '/atlascar2/camera_tras/agentes/image_raw/compressed', 10)
        self.get_logger().info(f'deteta_agentes a correr (HOG) - a ouvir {topic}')

    def on_image(self, msg):
        self.i += 1
        if self.i % self.every_n != 0:
            return
        img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return

        h0, w0 = img.shape[:2]
        scale = self.width / float(w0)
        small = cv2.resize(img, (self.width, max(1, int(h0 * scale))))
        rects, _ = self.hog.detectMultiScale(small, winStride=(8, 8))

        for (x, y, w, h) in rects:
            x, y, w, h = (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        self.pub_n.publish(Int32(data=int(len(rects))))

        out = CompressedImage()
        out.header = msg.header
        out.format = 'jpeg'
        ok, enc = cv2.imencode('.jpg', img)
        if ok:
            out.data = enc.tobytes()
            self.pub_img.publish(out)


def main():
    rclpy.init()
    node = DetetaAgentes()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
