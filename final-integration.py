import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
import vlc
import sys, webbrowser, datetime

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# The function computes the Euclidean distances between specific pairs of landmarks around the mouth:
    #     - The upper lip landmarks (landmark 2 and landmark 10).
    #     - The lower lip landmarks (landmark 4 and landmark 8).
    # - It then calculates the ratio of the average distance between the upper and lower lip landmarks to the distance
    #   between the left and right corners of the mouth
# The yawning ratio is calculated by comparing the average distance between the upper and lower lip landmarks
# to the distance between the left and right corners of the mouth. A higher ratio indicates a wider mouth opening,
# typically associated with yawning
# Parameters:
    #     mouth (array): An array containing landmarks of the mouth region.
# Returns:
    #     float: The yawning ratio.
def yawn(mouth):
    return ((euclideanDist(mouth[2], mouth[10])+euclideanDist(mouth[4], mouth[8]))/(2*euclideanDist(mouth[0], mouth[6])))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def getFaceDirection(shape, size):
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                shape[33],    # Nose tip
                                shape[8],     # Chin
                                shape[45],    # Left eye left corner
                                shape[36],    # Right eye right corne
                                shape[54],    # Left Mouth corner
                                shape[48]     # Right mouth corner
                            ], dtype="double")
    
    # 3D model points Defines the corresponding 3D model points for the selected 2D image points. These are based on a generic 3D face model
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            
                            ])
    
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                            [[focal_length, 0, center[0]],
                            [0, focal_length, center[1]],
                            [0, 0, 1]], dtype = "double"
                            )
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return(translation_vector[1][0])
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# This function calculates the Euclidean distance between two points a and b
# It's used to measure distances between facial landmarks
def euclideanDist(a, b):  
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#EAR -> Eye Aspect ratio
# This function calculates the Eye Aspect Ratio (EAR). 
# EAR is used to determine the openness of an eye, which helps in detecting blinks or eye closure. 
# The formula uses specific eye landmarks to calculate the ratio:
#    eye[0] to eye[3]: Horizontal distance between the corners of the eye.
#    eye[1] to eye[5] and eye[2] to eye[4]: Vertical distances between the points on the upper and lower eyelid.
def ear(eye):
    return ((euclideanDist(eye[1], eye[5])+euclideanDist(eye[2], eye[4]))/(2*euclideanDist(eye[0], eye[3])))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# his function extracts the regions around the left and right eyes based on the landmarks provided.
    # It then saves these regions as separate image files
def writeEyes(a, b, img):
    y1 = max(a[1][1], a[2][1])
    y2 = min(a[4][1], a[5][1])
    x1 = a[0][0]
    x2 = a[3][0]
    cv2.imwrite('left-eye.jpg', img[y1:y2, x1:x2])
    y1 = max(b[1][1], b[2][1])
    y2 = min(b[4][1], b[5][1])
    x1 = b[0][0]
    x2 = b[3][0]
    cv2.imwrite('right-eye.jpg', img[y1:y2, x1:x2])
# open_avg = train.getAvg()
# close_avg = train.getAvg()
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
alert = vlc.MediaPlayer('focus.mp3')
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# These variables are thresholds used to determine the level of drowsiness based on the number of 
# consecutive frames in which certain conditions are met. They are used to adjust the sensitivity 
# of the drowsiness detection algorithm. Lower values make the algorithm more sensitive, 
# while higher values make it less sensitive.
frame_thresh_1 = 15
frame_thresh_2 = 10
frame_thresh_3 = 5

# This variable represents the threshold for the eye aspect ratio (EAR) below which the algorithm 
# considers the eyes to be closed. The EAR is a measure of how open the eyes are, 
# and a lower value indicates more closed eyes. This threshold helps detect when the 
# driver's eyes are closing, indicating drowsiness
close_thresh = 0.3 #(close_avg+open_avg)/2.0

#This variable keeps track of the number of consecutive frames in which the eyes are detected as 
# closed or drowsy. It is incremented when the conditions for drowsiness are met in consecutive frames
flag = 0

# This variable is used to track if a yawn has been detected. It is set to 1 when a yawn is detected,
#  triggering additional drowsiness detection logic.
yawn_countdown = 0

# These variables are used to trigger actions after a certain number of consecutive frames meet the 
# conditions for drowsiness. map_counter counts the number of consecutive frames, and map_flag controls
#  whether a certain action (like opening a map) should be triggered.
map_counter = 0
map_flag = 1

# print(close_thresh)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

capture = cv2.VideoCapture(0) # Initializes video capture from the default camera (index 0)
avgEAR = 0
detector = dlib.get_frontal_face_detector() #Initializes dlib's frontal face detector
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # Loads the pre-trained shape predictor model to detect 68 facial landmarks
#Indices to extract left and right eye and mouth landmarks from the detected facial landmarks
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

while(True):
    ret, frame = capture.read() # Captures a frame 
    size = frame.shape # Gets the dimensions of the captured frame
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    rects = detector(gray, 0) # Detects faces in the frame
    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0])) # Predicts facial landmarks for the detected face , Converts the detected landmarks to a NumPy array
        leftEye = shape[leStart:leEnd] # Extracts landmarks for left and right eyes
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye) # Creates a convex hull around the eye landmarks
        rightEyeHull = cv2.convexHull(rightEye)
        # print("Mouth Open Ratio", yawn(shape[mStart:mEnd]))
        leftEAR = ear(leftEye) #Get the left eye aspect ratio
        rightEAR = ear(rightEye) #Get the right eye aspect ratio
        avgEAR = (leftEAR+rightEAR)/2.0
        eyeContourColor = (255, 255, 255)

        if(yawn(shape[mStart:mEnd])>0.6):
            cv2.putText(gray, "Yawn Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            yawn_countdown=1

        if(avgEAR<close_thresh):
            flag+=1
            eyeContourColor = (0,255,255)
            print(flag)
            if(yawn_countdown and flag>=frame_thresh_3):
                eyeContourColor = (147, 20, 255)
                cv2.putText(gray, "Drowsy after yawn", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
            elif(flag>=frame_thresh_2 and getFaceDirection(shape, size)<0):
                eyeContourColor = (255, 0, 0)
                cv2.putText(gray, "Drowsy (Body Posture)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
            elif(flag>=frame_thresh_1):
                eyeContourColor = (0, 0, 255)
                cv2.putText(gray, "Drowsy (Eye closed)", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
                alert.play()
                if(map_flag):
                    map_flag = 0
                    map_counter+=1
        elif(avgEAR>close_thresh and flag):
            print("Flag reseted to 0")
            alert.stop()
            yawn_countdown=0
            map_flag=1
            flag=0

        if(map_counter>=5):
            map_flag=1
            map_counter=0
            vlc.MediaPlayer('take_a_break.mp3').play()
            webbrowser.open("https://www.google.com/maps/search/hotels+or+motels+near+me")

        cv2.drawContours(gray, [leftEyeHull], -1, eyeContourColor, 2)
        cv2.drawContours(gray, [rightEyeHull], -1, eyeContourColor, 2)
        writeEyes(leftEye, rightEye, frame)
    if(avgEAR>close_thresh):
        alert.stop()
    cv2.imshow('Driver', gray)
    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()