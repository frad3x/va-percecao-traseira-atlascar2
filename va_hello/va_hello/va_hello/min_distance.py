"""Processamento: distância ao obstáculo mais próximo a partir de um LaserScan."""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

class MinDistance(Node):
    def __init__(self):
        super().__init__('min_distance')
        self.declare_parameter('alarm_distance', 2.0)
        self.alarm = self.get_parameter('alarm_distance').value
        self.pub = self.create_publisher(Float32, '/obstaculo/distancia_minima', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 10)

    def on_scan(self, scan):
        valid = [r for r in scan.ranges if scan.range_min < r < scan.range_max]
        if not valid:
            return
            
        d = min(valid)
        self.pub.publish(Float32(data=d))
        
        if d < self.alarm:
            self.get_logger().warn(f'Obstaculo a {d:.2f} m')

def main():
    rclpy.init()
    node = MinDistance()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
