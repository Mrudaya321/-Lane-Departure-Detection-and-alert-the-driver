# -Lane-Departure-Detection-and-alert-the-driver
 Modern vehicles require reliable systems to enhance road safety and prevent 
unintentional lane departures, which are a major cause of accidents. Traditional 
lane detection methods often struggle with real-world challenges such as poor 
lighting, worn or obscured lane markings, and varying road conditions. The problem 
is to design and implement an effective lane departure detection system using a 
camera-based vision pipeline that can robustly identify lane boundaries and detect 
departures, providing timely warnings to the driver under diverse driving situations

# The method used :
 Classical Computer Vision Methods
 
 Canny Edge Detection
 
 Gaussian Blur
 
 Region Masking
 
 Color Space Adjustments (HSV, RGB Thresholding)
 
 Hough Transform (Hough Line, Hough Shift)
 
 Averaging/Extrapolating Detected Lines
 
 Kalman Filter

# Why Choose Camera Vision Pipelines?
 
 • Simple - Easy to code using OpenCV (grayscale, edges, Hough lines).
 
 • Low-cost - Needs only a camera and a basic PC.
 
 • Lightweight - Runs on laptops or Raspberry Pi in real-time.
 
 • No sensors needed - Unlike data fusion, which requires vehicle sensors (yaw,
 steering, etc.).
 
 • No GPU or big data needed - Unlike deep learning, which needs powerful GPU 
and
 large labeled datasets.
 Easy to debug - Steps are visible and adjustable.

  # Flowchart 

 <img width="790" height="667" alt="image" src="https://github.com/user-attachments/assets/b3b917f5-6f18-460f-a7b7-eb991db5a462" />


  # Requirements 
  
  numpy
  
  opencv-python
  
  matplotlib

 
 # installation:
```pip install numpy opencv-python```

# Run lane detection script

```python lane_detection.py```



# Clone
https://github.com/Mrudaya321/-Lane-Departure-Detection-and-alert-the-driver.git




## gif of result :

![Demo](output_videos/gif.mp4)


# Screenshots 
<img width="1051" height="573" alt="image" src="https://github.com/user-attachments/assets/3811da22-bbf3-4fad-a179-949e4f9c8113" />



<img width="1045" height="576" alt="image" src="https://github.com/user-attachments/assets/b1a39885-84fa-466c-bdf7-5ebd8171e594" />









# FINAL OBJECTIVE :
To develop a real-time lane departure detection system utilizing camera vision techniques 
that:
 
 • Accurately identifies lane boundaries using a sequence of classical computer vision 
operations (e.g., grayscale conversion, Gaussian blurring, ROI masking, edge detection, 
Hough Transform).
 
 • Calculates the vehicle's lateral position relative to the detected lanes to continuously 
monitor safe lane discipline.
 
 • Detects unintentional lane departures by evaluating vehicle position with respect to lane 
centers and triggers warnings when thresholds are exceeded.
 
 • Demonstrates robustness to typical driving conditions, including variable lighting,
 shadows, slight occlusions, and moderately degraded lane markings.





