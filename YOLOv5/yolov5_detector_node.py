import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import sys
import torch
import numpy as np
import time
import ament_index_python.packages

# Path ke folder YOLOv5
current_dir = os.path.dirname(os.path.abspath(__file__))
yolov5_path = os.path.join(current_dir, 'yolov5')

# Import YOLOv5 modules
sys.path.insert(0, yolov5_path)

try:
    from models.common import DetectMultiBackend
    from utils.general import non_max_suppression, scale_boxes
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
except ImportError:
    print("GAGAL mengimpor modul YOLOv5! Periksa path dan struktur folder.")
    sys.exit(1)

class YoloV5CameraNode(Node):
    def __init__(self):
        super().__init__('yolov5_camera_node')

        self.bridge = CvBridge()

        # OpenCV camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Gagal membuka kamera.')
            rclpy.shutdown()
            return

        # Cari lokasi weights file
        self.get_logger().info('Mencari file model weights...')
        weights_filename = 'bestv5.pt'
        
        # Opsi 1: Cari di src (untuk development)
        weights_path_src = os.path.join('/home/xiaoran/xiaoran_ws/src/yolov5_detector/yolov5_detector/weights', weights_filename)
        if os.path.exists(weights_path_src):
            weights_path = weights_path_src
            self.get_logger().info(f'Menggunakan weights dari src directory: {weights_path}')
        else:
            # Opsi 2: Cari di share directory (cara standard ROS2)
            try:
                package_share_dir = ament_index_python.packages.get_package_share_directory('yolov5_detector')
                weights_path_share = os.path.join(package_share_dir, 'weights', weights_filename)
                if os.path.exists(weights_path_share):
                    weights_path = weights_path_share
                    self.get_logger().info(f'Menggunakan weights dari share directory: {weights_path}')
                else:
                    self.get_logger().error(f'File weights tidak ditemukan di manapun!')
                    self.get_logger().error(f'  - Tidak di: {weights_path_src}')
                    self.get_logger().error(f'  - Tidak di: {weights_path_share}')
                    rclpy.shutdown()
                    return
            except Exception as e:
                self.get_logger().error(f'Error mencari weights: {e}')
                rclpy.shutdown()
                return

        # Device dan model YOLOv5
        try:
            # Konfigurasi untuk kecepatan
            self.device = select_device('')
            self.model = DetectMultiBackend(weights_path, device=self.device)
            self.model.eval()
            
            # Preload dan warmup model
            self.get_logger().info('Melakukan warmup model...')
            self.model.warmup(imgsz=(1, 3, 640, 640))
            
            self.names = self.model.names
            
            # Cetak informasi tentang class names
            self.get_logger().info(f'Model memiliki {len(self.names)} kelas')
            self.get_logger().info(f'Class names: {self.names}')
            
            self.get_logger().info('Model YOLOv5 siap.')
        except Exception as e:
            self.get_logger().error(f'Gagal inisialisasi model: {e}')
            rclpy.shutdown()
            return
            
        self.prev_time = time.time()
        self.fps = 0.0

        # Timer: 30Hz untuk kecepatan lebih tinggi
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().warn('Gagal membaca frame.')
                return

            # Preprocess image dengan ukuran tetap
            im = letterbox(frame, new_shape=640)[0]
            im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB, to CHW
            im = np.ascontiguousarray(im)

            # Convert ke tensor
            im_tensor = torch.from_numpy(im).to(self.device)
            im_tensor = im_tensor.float() / 255.0
            if im_tensor.ndimension() == 3:
                im_tensor = im_tensor.unsqueeze(0)

            # Perform inference
            pred = self.model(im_tensor, augment=False, visualize=False)
            pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

            # Gambar hasil deteksi
            for det in pred:
                if len(det):
                    det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in det:
                        # Pastikan class index ada dalam dictionary names
                        class_idx = int(cls)
                        if class_idx in self.names:
                            label = f'{self.names[class_idx]} {conf:.2f}'
                        else:
                            # Jika class index tidak valid, gunakan index saja
                            self.get_logger().warn(f'Class index {class_idx} tidak ditemukan di model')
                            label = f'Class-{class_idx} {conf:.2f}'
                        
                        frame = self.draw_box(frame, xyxy, label)

            # Hitung dan tampilkan FPS
            curr_time = time.time()
            self.fps = 1.0 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            cv2.putText(frame, f'FPS: {self.fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            cv2.imshow('YOLOv5 Detection', frame)
            cv2.waitKey(1)
            
        except KeyError as e:
            self.get_logger().error(f'KeyError dalam proses deteksi: {e}')
            
        except Exception as e:
            self.get_logger().error(f'Error dalam callback: {e}')

    def draw_box(self, img, xyxy, label, color=(0, 255, 0), thickness=2):
        p1 = (int(xyxy[0]), int(xyxy[1]))
        p2 = (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(img, p1, p2, color, thickness)
        cv2.putText(img, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info('Kamera dan jendela ditutup.')
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloV5CameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Dihentikan oleh pengguna.')
    except Exception as e:
        node.get_logger().error(f'Error tidak terduga: {e}')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from cv_bridge import CvBridge
# import cv2
# import os
# import sys
# import torch
# from pathlib import Path
# import numpy as np
# import time
# import ament_index_python.packages

# # Path ke folder YOLOv5 (misal Anda clone di dalam package)
# current_dir = os.path.dirname(os.path.abspath(__file__))
# yolov5_path = os.path.join(current_dir, 'yolov5')

# # print(f"Mencoba menambahkan path: {yolov5_path}")
# # print(f"Path ini ada: {os.path.exists(yolov5_path)}")
# # print(f"Path ini memiliki models: {os.path.exists(os.path.join(yolov5_path, 'models'))}")

# sys.path.insert(0, yolov5_path)
# #sys.path.append('/home/xiaoran/xiaoran_ws/src/yolov5_detector/yolov5_detector/yolov5')

# from models.common import DetectMultiBackend
# from utils.general import non_max_suppression, scale_boxes
# from utils.torch_utils import select_device
# from utils.augmentations import letterbox

# class YoloV5CameraNode(Node):
#     def __init__(self):
#         super().__init__('yolov5_camera_node')

#         self.bridge = CvBridge()

#         # OpenCV camera
#         self.cap = cv2.VideoCapture(0)
#         if not self.cap.isOpened():
#             self.get_logger().error('Gagal membuka kamera.')
#             rclpy.shutdown()
#             return

#         # Device dan model YOLOv5
#         self.device = select_device('')
#         weights_path = os.path.join('/home/xiaoran/xiaoran_ws/src/yolov5_detector/yolov5_detector/weights', 'best (4).pt')  # Ganti model jika perlu
#         self.model = DetectMultiBackend(weights_path, device=self.device)
#         self.model.eval()

#         self.names = self.model.names
#         self.get_logger().info('Model YOLOv5 siap.')
#         self.prev_time = time.time()
#         self.fps = 0.0

#         # Timer: 10Hz
#         self.timer = self.create_timer(0.1, self.timer_callback)

#     def timer_callback(self):
#         ret, frame = self.cap.read()
#         if not ret:
#             self.get_logger().warn('Gagal membaca frame.')
#             return

#         im = letterbox(frame, new_shape=640)[0]
#         im = im.transpose((2, 0, 1))[::-1]  # BGR to RGB, to CHW
#         im = np.ascontiguousarray(im)

#         im_tensor = torch.from_numpy(im).to(self.device)
#         im_tensor = im_tensor.float() / 255.0  # 0 - 1
#         if im_tensor.ndimension() == 3:
#             im_tensor = im_tensor.unsqueeze(0)

#         pred = self.model(im_tensor, augment=False, visualize=False)
#         pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

#         for det in pred:
#             if len(det):
#                 det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], frame.shape).round()
#                 for *xyxy, conf, cls in det:
#                     label = f'{self.names[int(cls)]} {conf:.2f}'
#                     frame = self.draw_box(frame, xyxy, label)

#         # Hitung dan tampilkan FPS
#         curr_time = time.time()
#         self.fps = 1.0 / (curr_time - self.prev_time)
#         self.prev_time = curr_time

#         cv2.putText(frame, f'FPS: {self.fps:.2f}', (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        
#         cv2.imshow('YOLOv5 Detection', frame)
#         cv2.waitKey(1)


#     def draw_box(self, img, xyxy, label, color=(0, 255, 0), thickness=2):
#         p1 = (int(xyxy[0]), int(xyxy[1]))
#         p2 = (int(xyxy[2]), int(xyxy[3]))
#         cv2.rectangle(img, p1, p2, color, thickness)
#         cv2.putText(img, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
#         return img

#     def destroy_node(self):
#         self.cap.release()
#         cv2.destroyAllWindows()
#         self.get_logger().info('Kamera dan jendela ditutup.')
#         super().destroy_node()

# def main(args=None):
#     rclpy.init(args=args)
#     node = YoloV5CameraNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info('Dihentikan oleh pengguna.')
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()
