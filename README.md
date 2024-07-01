# Title:
Driver Drowsiness and Yawning Detection System

# Description:
This project utilizes computer vision techniques to detect driver drowsiness and yawning in real-time. It employs facial landmark detection and analysis to monitor eye closure and mouth opening, triggering alerts when signs of drowsiness or yawning are detected. The system integrates with VLC media player for audio alerts and can open a web browser to suggest nearby resting locations based on the detected state of the driver.

# Technologies Used:
- OpenCV: Handles video frame capture, image processing, and real-time analysis.
- dlib: Utilizes a pre-trained model for accurate facial landmark detection and shape prediction.
- NumPy: Efficiently manages numerical data arrays and computations.
- VLC: Integrates for playing audio alerts upon detection events.
- Python libraries (sys, webbrowser, datetime)
