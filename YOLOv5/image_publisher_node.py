import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher_node')

        self.publisher_ = self.create_publisher(Image, '/image', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0)  # Gunakan 0 untuk webcam, atau path ke video

        if not self.cap.isOpened():
            self.get_logger().error('Kamera tidak bisa diakses.')
            rclpy.shutdown()
            return

        self.get_logger().info('ImagePublisher berhasil dimulai.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Gagal membaca frame dari kamera.')
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().debug('Frame diterbitkan.')

    def destroy_node(self):
        self.cap.release()
        self.get_logger().info('Kamera ditutup.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImagePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('ImagePublisher dihentikan oleh pengguna.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

