"""LiDAR 2D simulado: uma parede a 8 m e um obstáculo que se aproxima e afasta."""
import math
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

class FakeScan(Node):
    def __init__(self):
        super().__init__('fake_scan')
        self.declare_parameter('frame_id', 'lidar2d_frente')
        self.declare_parameter('rate_hz', 10.0)
        self.frame_id = self.get_parameter('frame_id').value
        rate = self.get_parameter('rate_hz').value
        
        self.pub = self.create_publisher(LaserScan, '/scan', 10)
        self.t0 = self.get_clock().now()
        self.create_timer(1.0 / rate, self.publish_scan)
        
        # Posição do sensor no carro (base_link -> sensor): 3,5 m à frente, 0,4 m de altura.
        # No trabalho real isto vem do fragmento de URDF do grupo.
        self.tf_static = StaticTransformBroadcaster(self)
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = 'base_link'
        tf.child_frame_id = self.frame_id
        tf.transform.translation.x = 3.5
        tf.transform.translation.z = 0.4
        tf.transform.rotation.w = 1.0
        self.tf_static.sendTransform(tf)

    def publish_scan(self):
        now = self.get_clock().now()
        t = (now - self.t0).nanoseconds * 1e-9
        obstacle = 5.0 + 3.5 * math.sin(0.3 * t) # entre 1,5 m e 8,5 m
        
        scan = LaserScan()
        scan.header.stamp = now.to_msg()
        scan.header.frame_id = self.frame_id
        scan.angle_min = -math.pi / 2
        scan.angle_max = math.pi / 2
        n = 181
        scan.angle_increment = (scan.angle_max - scan.angle_min) / (n - 1)
        scan.range_min = 0.1
        scan.range_max = 30.0
        
        for i in range(n):
            angle = scan.angle_min + i * scan.angle_increment
            r = min(8.0 / max(math.cos(angle), 1e-3), 29.9) # parede a 8 m
            if abs(angle) < math.radians(10) and obstacle < r:
                r = obstacle
            scan.ranges.append(r)
            
        self.pub.publish(scan)

def main():
    rclpy.init()
    node = FakeScan()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
