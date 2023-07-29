import math
import numpy as np
import cv2
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt
import subprocess
global cnt
cnt=2
global tmp
tmp=2
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, pose, display=True):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                  (landmark.z * width)))
    if display:
        plt.figure(figsize=[22,22])
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    else:
        return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def classifyPose(landmarks, output_image, display=False):
    global cnt
    label = 'Unknown Pose'
    color = (0, 0, 255)
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   

    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    if left_elbow_angle >= 155 and left_elbow_angle <= 170 and left_shoulder_angle>=66 and left_shoulder_angle<=98 or right_elbow_angle >= 194 and right_elbow_angle <= 217 and right_shoulder_angle>=72 and right_shoulder_angle<=100 or right_elbow_angle >= 285 and right_elbow_angle <= 310 and right_shoulder_angle>=55 and right_shoulder_angle<=90:
        label="Shoot Pose"
        #cv2.imwrite("/Users/ashwi/yolov5/DetectedPics/"+'ash'+str(cnt)+'.jpg',frame)
        #cv2.imwrite('ash'+str(cnt)+'.jpg', frame)
        cv2.imwrite("ashwin.jpg",frame)
       # return output_image, label
        # call yolo model herew
        #yolo_call()
        cnt=cnt+1
    if label != 'Unknown Pose':        
        color = (0, 255, 0)  

    cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, color, 3)
    return output_image, label
    if display:
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        return output_image, label

def yolo_call():
    import os
    import shutil
    global tmp
    Gun_Detected=False
    detections=[]
    shutil.rmtree("/Users/ashwi/yolov5/runs/detect",ignore_errors=True)
    subprocess.call(["python", "detect.py","--weights","yolov5s.pt","--source","ashwin.jpg","--save-txt"])
    fname="/Users/ashwi/yolov5/runs/detect/exp/labels/ashwin.txt"
    file_exist=os.path.isfile(fname)
    if(file_exist):
        with open(fname, 'r') as f:
            for line in f:
                words=line.split()
                detections.append(words)
            if(len(detections)!=0):
                Gun_Detected=True;
            if(Gun_Detected):
                print("ALERT Weapon Detected")
                alert()
            else:
                print("no Weapon detected")
    else:
        print("no Weapon detected")
    '''
    path="/Users/ashwi/yolov5/DetectedPics/"
    for img in os.listdir(path):
        print(img) #gives file name off all images in the folder
        subprocess.call(["python", "detect.py","--source",img,"--save-txt"])
        #start from exp2    
        fname="/Users/ashwi/yolov5/runs/detect/exp"+str(tmp)+"/labels/ash"+str(tmp)+".txt"
        detections=[]
        tmp=tmp+1
    #fname="/content/yolov5/runs/detect/Empty/labels/test_83_empty.txt"
        with open(fname, 'r') as f:
            for line in f:
                words=line.split()
                detections.append(words)
                if(len(detections)!=0):
                    Gun_Detected=True;
        if(Gun_Detected):
            print("ALERT")
  '''  
def alert():
    from twilio.rest import Client
    account_id="AC303eaf394a59931829af20076dfb01ed"
    auth_token="4d6566cdffd5134fe50ae42de0465c59"
    client = Client(account_id, auth_token)
    call = client.calls.create(
                    twiml='<Response><Say>The Bank is under attack</Say></Response>',   
                    to='+918838740896',
                    from_='+12184167610'
                    )
    print(call.sid)
    
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

camera_video = cv2.VideoCapture(0)
camera_video.set(3,1280)
camera_video.set(4,960)

cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

while camera_video.isOpened():
    ok, frame = camera_video.read()

    if not ok:
        continue
    frame = cv2.flip(frame, 1)

    frame_height, frame_width, label =  frame.shape

    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    

    frame, landmarks = detectPose(frame, pose_video, display=False)
    
    # Check if the landmarks are detected.
    if landmarks:
        
        # Perform the Pose Classification.
        frame, label = classifyPose(landmarks, frame, display=False)
       # yolo_call(cnt)
    # Display the frame.
    cv2.imshow('Pose Classification', frame)
    #yolo_call()
    if(label=="Shoot Pose"):
       #camera_video.release()
       #cv2.waitKey(0)
       #cv2.destroyWindow('Pose Classification')
       #img_data=cv2.imread("ashwin.jpg")
       #cv2.imshow("window",img_data)
       yolo_call()

    k = cv2.waitKey(1) & 0xFF
    
    if(k == 27):
        
        break

# Release the VideoCapture object and close the windows.
#yolo_call()
camera_video.release()
cv2.destroyAllWindows()
#yolo_call()
#alert()  