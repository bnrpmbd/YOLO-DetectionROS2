# YOLO-DetectionROS2

YOLO-DetectionROS2 is a ROS2 package that integrates **YOLOv5** and
**YOLOv12** for real-time object detection in robotics applications.\
It provides ROS2 nodes for running detection, publishing annotated
images, and streaming camera/video input for testing and deployment.

------------------------------------------------------------------------

## 🚀 Features

-   YOLOv5 and YOLOv12 integration with ROS2.
-   Real-time object detection from webcam or video source.
-   Publishes annotated images with bounding boxes and FPS overlay.
-   ROS2-compatible topics for detection results.
-   Includes an image publisher node for testing with camera/video
    input.

------------------------------------------------------------------------

## 📦 Requirements

-   ROS2 (Foxy, Humble, or newer)
-   Python 3.8+
-   OpenCV (`cv2`)
-   PyTorch (`torch`)
-   Ultralytics YOLO (`pip install ultralytics`)
-   `cv_bridge` and `sensor_msgs` (ROS2 dependencies)
-   `vision_msgs` (for extended detection output with YOLOv12)

------------------------------------------------------------------------

## 📂 Package Structure

    YOLO-DetectionROS2/
    ├── yolov5_detector_node.py   # ROS2 node for YOLOv5 detection
    ├── yolov12_detector_node.py  # ROS2 node for YOLOv12 detection
    ├── image_publisher_node.py   # Simple ROS2 image publisher (webcam/video)
    └── weights/                  # Place YOLOv5/YOLOv12 model weights here (bestv5.pt, bestv12.pt)

------------------------------------------------------------------------

## ⚙️ Installation

1.  Clone the repository into your ROS2 workspace (`src` folder):

    ``` bash
    cd ~/ros2_ws/src
    git clone https://github.com/your-username/YOLO-DetectionROS2.git
    ```

2.  Install Python dependencies:

    ``` bash
    pip install ultralytics opencv-python torch torchvision
    ```

3.  Build the workspace:

    ``` bash
    cd ~/ros2_ws
    colcon build
    source install/setup.bash
    ```

------------------------------------------------------------------------

## ▶️ Usage

### Run YOLOv5 Detector

``` bash
ros2 run yolov5_detector yolov5_detector_node
```

### Run YOLOv12 Detector

``` bash
ros2 run yolov12_detector yolov12_detector_node
```

### Run Image Publisher (Webcam/Video)

``` bash
ros2 run yolov5_detector image_publisher_node
```

------------------------------------------------------------------------

## 🔧 Parameters

Each detector node supports configurable parameters:

-   **weights**: Path to YOLO model weights (`.pt` file).
-   **imgsz**: Input image size (default: 640 for YOLOv5, 480 for
    YOLOv12).
-   **conf_thres**: Confidence threshold (default: 0.25).
-   **source**: Camera index or video file path (YOLOv12).

------------------------------------------------------------------------

## 📊 Example Topics

-   `/yolov5/detections` -- YOLOv5 detection results.
-   `/yolov12/detections` -- YOLOv12 detection results.
-   `/yolov12/annotated_image` -- Annotated images from YOLOv12.
-   `/image` -- Raw images from `image_publisher_node`.

------------------------------------------------------------------------

## 🤖 Applications

-   Robotics perception and navigation
-   Autonomous vehicles and drones
-   Surveillance and monitoring systems
-   Real-time object recognition for research

------------------------------------------------------------------------

## 🏷 Keywords

ros2, ros2-nodes, ros2-perception, computer-vision, object-detection,
yolov5, yolov12, deep-learning, realtime-detection, opencv, cv-bridge,
ultralytics, robotics

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License. See the
[LICENSE](LICENSE) file for details.
