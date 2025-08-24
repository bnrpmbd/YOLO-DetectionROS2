# #!/usr/bin/env python3
# # YOLOv12 ROS2 Node
# # Dioptimalkan untuk dijalankan dalam satu terminal

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image, CompressedImage
# from cv_bridge import CvBridge
# from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import time
# import os
# import threading
# from pathlib import Path
# import torch
# from geometry_msgs.msg import PoseWithCovariance, Pose2D
# from std_msgs.msg import Header

# class YOLOv12ROS2Node(Node):
#     def __init__(self):
#         super().__init__('yolov12_detector')
        
#         # Declare parameter dengan default value
#         self.declare_parameter('weights', 'bestv12.pt')
#         self.declare_parameter('imgsz', 640)
#         self.declare_parameter('conf_thres', 0.25)
#         self.declare_parameter('input_topic', '/camera/image_raw')
#         self.declare_parameter('output_topic', '/yolov12/detections')
#         self.declare_parameter('visualize', True)
#         self.declare_parameter('compressed_input', False)
#         self.declare_parameter('terminal_vis', False)
#         self.declare_parameter('display', False)
        
#         # Get parameter
#         self.weights = self.get_parameter('weights').value
#         self.imgsz = self.get_parameter('imgsz').value
#         self.conf_thres = self.get_parameter('conf_thres').value
#         self.input_topic = self.get_parameter('input_topic').value
#         self.output_topic = self.get_parameter('output_topic').value
#         self.visualize = self.get_parameter('visualize').value
#         self.compressed_input = self.get_parameter('compressed_input').value
#         self.terminal_vis = self.get_parameter('terminal_vis').value
#         self.display = self.get_parameter('display').value
        
#         # Cek path file weights
#         weights_path = Path(self.weights)
#         if not weights_path.is_absolute():
#             # Coba mencari di folder package
#             package_path = Path(__file__).parent.parent
#             weights_folder = package_path / 'yolov12_detector' / 'weights'
#             weights_path = weights_folder / self.weights
#             self.weights = str(weights_path)
#             self.get_logger().info(f"Path weights relatif terdeteksi, menggunakan: {self.weights}")
        
#         # Inisialisasi model YOLOv12
#         self.get_logger().info(f"Memuat model YOLOv12: {self.weights}")
#         try:
#             self.model = YOLO(self.weights)
#             self.get_logger().info(f"Model berhasil dimuat")
#         except Exception as e:
#             self.get_logger().error(f"Gagal memuat model: {e}")
#             rclpy.shutdown()
            
#         # Inisialisasi OpenCV bridge
#         self.bridge = CvBridge()
        
#         # Inisialisasi publisher
#         self.detection_pub = self.create_publisher(
#             Detection2DArray, 
#             self.output_topic, 
#             10
#         )
        
#         if self.visualize:
#             self.image_pub = self.create_publisher(
#                 Image, 
#                 '/yolov12/annotated_image', 
#                 10
#             )
        
#         # Inisialisasi subscriber
#         if self.compressed_input:
#             self.image_sub = self.create_subscription(
#                 CompressedImage,
#                 self.input_topic,
#                 self.compressed_image_callback,
#                 10
#             )
#         else:
#             self.image_sub = self.create_subscription(
#                 Image,
#                 self.input_topic,
#                 self.image_callback,
#                 10
#             )
            
#         self.get_logger().info(f"YOLOv12 ROS2 node diinisialisasi")
#         self.get_logger().info(f"Berlangganan ke: {self.input_topic}")
#         self.get_logger().info(f"Mempublikasikan deteksi ke: {self.output_topic}")
        
#         # Inisialisasi frame counter dan timing
#         self.frame_count = 0
#         self.start_time = time.time()
#         self.fps = 0.0
        
#         # Simpan deteksi terakhir untuk visualisasi terminal
#         self.latest_detections = []
#         self.detection_time = None
        
#         # Mulai thread visualisasi terminal jika diminta
#         if self.terminal_vis:
#             self.vis_thread = threading.Thread(target=self.terminal_visualization)
#             self.vis_thread.daemon = True
#             self.vis_thread.start()
            
#     def compressed_image_callback(self, msg):
#         try:
#             # Konversi gambar terkompresi ke gambar CV
#             np_arr = np.frombuffer(msg.data, np.uint8)
#             cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             self.process_image(cv_image, msg.header)
#         except Exception as e:
#             self.get_logger().error(f"Error memproses gambar terkompresi: {e}")
            
#     def image_callback(self, msg):
#         try:
#             # Konversi gambar ROS ke gambar CV
#             cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
#             self.process_image(cv_image, msg.header)
#         except Exception as e:
#             self.get_logger().error(f"Error memproses gambar: {e}")
            
#     def process_image(self, cv_image, header):
#         # Update frame counter untuk kalkulasi FPS
#         self.frame_count += 1
#         if self.frame_count % 10 == 0:
#             now = time.time()
#             self.fps = 10 / (now - self.start_time)
#             self.start_time = now
            
#         # Jalankan inferensi
#         start_inference = time.time()
#         results = self.model.predict(
#             source=cv_image, 
#             imgsz=self.imgsz, 
#             conf=self.conf_thres,
#             verbose=False
#         )
#         inference_time = time.time() - start_inference
        
#         # Buat pesan untuk deteksi
#         detection_array_msg = Detection2DArray()
#         detection_array_msg.header = header
        
#         # Proses hasil
#         detections = results[0]
#         boxes = detections.boxes
        
#         # Simpan deteksi untuk visualisasi terminal
#         self.latest_detections = []
        
#         # Proses setiap deteksi
#         for i, box in enumerate(boxes):
#             # Dapatkan data kotak
#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             conf = float(box.conf[0])
#             cls = int(box.cls[0])
#             class_name = detections.names[cls]
            
#             # Simpan untuk visualisasi terminal
#             self.latest_detections.append({
#                 'class': class_name,
#                 'confidence': conf,
#                 'box': (x1, y1, x2, y2)
#             })
            
#             # Buat pesan deteksi
#             det_msg = Detection2D()
#             det_msg.header = header
            
#             # Set titik tengah
#             center_x = (x1 + x2) / 2
#             center_y = (y1 + y2) / 2
#             det_msg.bbox.center.position.x = center_x
#             det_msg.bbox.center.position.y = center_y
            
#             # Set ukuran
#             det_msg.bbox.size_x = x2 - x1
#             det_msg.bbox.size_y = y2 - y1
            
#             # Tambahkan result hypothesis
#             hypothesis = ObjectHypothesisWithPose()
#             hypothesis.id = str(cls)
#             hypothesis.score = conf
#             hypothesis.pose.pose.position.x = center_x
#             hypothesis.pose.pose.position.y = center_y
#             det_msg.results.append(hypothesis)
            
#             # Tambahkan ke array
#             detection_array_msg.detections.append(det_msg)
        
#         # Update detection time
#         self.detection_time = time.time()
        
#         # Publikasikan pesan deteksi
#         self.detection_pub.publish(detection_array_msg)
        
#         # Visualisasikan jika diminta
#         if self.visualize:
#             # Dapatkan frame yang di-anotasi
#             annotated_frame = results[0].plot()
            
#             # Tambahkan info FPS
#             cv2.putText(
#                 annotated_frame, 
#                 f"FPS: {self.fps:.1f}", 
#                 (20, 40), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 1, 
#                 (0, 255, 0), 
#                 2
#             )
            
#             # Tambahkan waktu inferensi
#             cv2.putText(
#                 annotated_frame, 
#                 f"Inference: {inference_time*1000:.1f}ms", 
#                 (20, 80), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 1, 
#                 (0, 255, 0), 
#                 2
#             )
            
#             # Tambahkan jumlah deteksi
#             cv2.putText(
#                 annotated_frame, 
#                 f"Deteksi: {len(self.latest_detections)}", 
#                 (20, 120), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 
#                 1, 
#                 (0, 255, 0), 
#                 2
#             )
            
#             # Konversi kembali ke gambar ROS dan publikasikan
#             ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, "bgr8")
#             ros_image.header = header
#             self.image_pub.publish(ros_image)
            
#             # Tampilkan di jendela lokal jika diminta
#             if self.display:
#                 cv2.imshow('YOLOv12 Deteksi', annotated_frame)
#                 cv2.waitKey(1)
    
#     def terminal_visualization(self):
#         """Thread untuk visualisasi berbasis terminal dari deteksi"""
#         while rclpy.ok():
#             if self.latest_detections and self.detection_time and time.time() - self.detection_time < 1.0:
#                 # Bersihkan terminal
#                 os.system('clear' if os.name == 'posix' else 'cls')
                
#                 # Cetak header
#                 print(f"{'=' * 70}")
#                 print(f"YOLOv12 Deteksi - FPS: {self.fps:.1f}")
#                 print(f"{'=' * 70}")
#                 print(f"{'Kelas':<20} {'Kepercayaan':<10} {'Posisi (x1,y1,x2,y2)':<30}")
#                 print(f"{'-' * 70}")
                
#                 # Cetak deteksi
#                 for det in self.latest_detections:
#                     box = det['box']
#                     print(f"{det['class']:<20} {det['confidence']:.2f}{'':<5} ({int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])})")
                
#                 print(f"{'=' * 70}")
            
#             # Tidur untuk mencegah penggunaan CPU tinggi
#             time.sleep(0.2)

# def main(args=None):
#     rclpy.init(args=args)
    
#     try:
#         # Buat dan jalankan node
#         node = YOLOv12ROS2Node()
        
#         # Cetak petunjuk cara menghentikan node
#         print("\nYOLOv12 ROS2 Node aktif")
#         print("Tekan Ctrl+C untuk berhenti\n")
        
#         # Spin node
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         print("Mematikan YOLOv12 ROS2 node...")
#     except Exception as e:
#         print(f"Error dalam YOLOv12 ROS2 node: {e}")
#     finally:
#         # Tutup jendela OpenCV jika ada
#         cv2.destroyAllWindows()
        
#         # Shutdown ROS2
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
# Node deteksi objek YOLOv12 untuk ROS2

import argparse
import cv2
import time
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

class YOLOv12Detector(Node):
    def __init__(self):
        super().__init__('yolov12_detector_node')
        
        weights_path = os.path.join('/home/xiaoran/xiaoran_ws/src/yolov12_detector/yolov12_detector/weights', 'bestv12.pt')
        # Parameter untuk sumber video (default: 0 untuk webcam)
        self.declare_parameter('source', '0')
        self.declare_parameter('weights', weights_path)
        self.declare_parameter('conf_thres', 0.25)
        self.declare_parameter('imgsz', 480)
        
        # Ambil nilai parameter
        source = self.get_parameter('source').value
        weights = self.get_parameter('weights').value
        conf_thres = self.get_parameter('conf_thres').value
        imgsz = self.get_parameter('imgsz').value
        
        # Load model YOLOv12
        self.get_logger().info(f"Memuat model YOLOv12 dari: {weights}")
        self.model = YOLO(weights)
        
        # Inisialisasi sumber video
        if source.isnumeric():
            source = int(source)
            self.get_logger().info(f"Membuka webcam {source}")
        else:
            self.get_logger().info(f"Membuka video: {source}")
            
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.get_logger().error(f"Gagal membuka sumber video {source}")
            raise RuntimeError(f"Gagal membuka sumber video {source}")
        
        # CV Bridge untuk konversi gambar
        self.bridge = CvBridge()
        
        # Publisher untuk hasil deteksi
        self.detection_pub = self.create_publisher(Image, 'detection_output', 10)
        
        # Inisialisasi timer & time tracking untuk FPS
        self.prev_time = time.time()
        self.timer = self.create_timer(0.033, self.process_frame)  # ~30 FPS
        
        self.get_logger().info("Node detektor YOLOv12 siap!")
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Sumber video telah habis")
            self.timer.cancel()
            return
        
        # Hitung FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.prev_time + 1e-6)
        self.prev_time = curr_time
        
        # Deteksi objek
        results = self.model.predict(
            source=frame,
            conf=self.get_parameter('conf_thres').value,
            imgsz=self.get_parameter('imgsz').value,
            verbose=False
        )
        
        # Gambar bounding box
        annotated_frame = results[0].plot()
        
        # Gambar FPS di frame
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # Konversi ke ROS Image message dan publish
        ros_image = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
        self.detection_pub.publish(ros_image)
        
        # Tampilkan di jendela lokal
        cv2.imshow('YOLOv12 Detection', annotated_frame)
        cv2.waitKey(1)
    
    def destroy_node(self):
        self.cap.release()
        super().destroy_node()
        self.get_logger().info("Node detektor berhenti")

def main(args=None):
    rclpy.init(args=args)
    node = None  # Inisialisasi sebagai None
    
    try:
        node = YOLOv12Detector()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")  # Cetak error yang terjadi
    finally:
        if node is not None:  # Hanya destroy jika node berhasil dibuat
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()