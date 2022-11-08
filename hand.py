import socket
import sys
import cv2
import mediapipe as mp
import numpy as np
import struct
from scipy.spatial.transform import Rotation as R
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

Wr = 0
M0 = 9
M3 = 12

Wr = 0
firstF = [1,2,3,4]
secondF = [5,6,7,8]
middleF = [9,10,11,12]
ringF = [13,14,15,16]
pinkyF= [17,18,19,20]

fingerList = [firstF,secondF,middleF,ringF,pinkyF]

Queue = []
CONF_THRESHOLD = 0.6
Q_NUM =5

HOST="192.168.0.110" # 192.168.0.11: 기숙사,  192.168.0.40: 연구실
PORT=8888

def isBent(hand_landmarks,finger):
    bottom = hand_landmarks.landmark[finger[0]].y
    top = hand_landmarks.landmark[finger[3]].y
    if ((top - bottom)>0):
        bent = False
    else:
        bent = True
    return bent


def calVector(finger1,finger2):
    fvector = [hand_landmarks.landmark[finger1].x-hand_landmarks.landmark[finger2].x,
                    hand_landmarks.landmark[finger1].y-hand_landmarks.landmark[finger2].y,
                    hand_landmarks.landmark[finger1].z-hand_landmarks.landmark[finger2].z]
    return fvector

def isclose(point1, point2):
    fvetor = [point1.x-point2.x,point1.y-point2.y,point1.z-point2.z]
    x,y,z = fvetor
    norm = (x**2+y**2+z**2)**(1/2)
    return [x/norm, y/norm, z/norm]

def ivector(vec):
        x,y,z = vec
        norm = (x**2+y**2+z**2)**(1/2)
        return [x/norm, y/norm, z/norm]

def calculate_joint_angle(joint1,joint2):
    return np.arccos(np.clip(np.dot(joint1, joint2), -1.0, 1.0))

with (socket.socket(socket.AF_INET,socket.SOCK_STREAM)) as s:
    s.bind((HOST,PORT))
    print('Socket bind complete')
    s.listen()
    print('Socket now listening')
    conn,addr=s.accept()
    with conn:
        print("Accepted a connection request from")
        print(conn, addr)
        # data = conn.send(isGesture.encode())
        # while True:
        #     conn.send("hi".encode())
# For webcam input:
        cap = cv2.VideoCapture(0)
        with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
            isFirst = True
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.multi_hand_world_landmarks:
                    # print(len(results.multi_handedness))
                    Queue.append([results.multi_handedness, results.multi_hand_world_landmarks])
                else:
                    Queue.append(None)
            
                if len(Queue) > Q_NUM:
                    q = Queue.pop(0)
                    if(isFirst == True):
                        pre_q = q
                    if q and pre_q is not None and len(q)!=0:
                        for i, (hand_handedness, hand_landmarks) in enumerate(zip(q[0],q[1])):
                            if len(q[0])==1 and hand_handedness.classification[0].score < CONF_THRESHOLD:
                                if pre_q is not None and len(list(pre_q))==1:
                                    pre_mlh, pre_hand_landmarks = pre_q
                                    if pre_mlh.classification[0].score >= CONF_THRESHOLD:
                                        hand_landmarks = pre_hand_landmarks 
                            if(hand_handedness.classification[0].label == "Right"):
                                lefthandvector = ivector(calVector(fingerList[0][0],fingerList[0][3]))
                                headvector = ivector(calVector(fingerList[1][0],fingerList[1][3]))
                                bodyvector = isclose(hand_landmarks.landmark[Wr], hand_landmarks.landmark[5])
                                righthandvector = ivector(calVector(fingerList[2][0],fingerList[2][3]))
                                if(isFirst == True or pre_q[1][0].landmark[0] == hand_landmarks.landmark[Wr].z):
                                    locationvector = [0.,0.,0.]
                                    isFirst = False
                                else:
                                    locationvector = isclose(pre_q[1][0].landmark[0], hand_landmarks.landmark[Wr])
                                    
                                # print(righthandvector)
                                # print(headvector)
                                # print(lefthandvector)
                                
                                cv2.putText(image, text="righthand "+str(righthandvector), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                cv2.putText(image, text="head "+str(headvector), org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                cv2.putText(image, text="lefthand "+str(lefthandvector), org=(150,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                # if(locationvector[2]>0.1):
                                #     cv2.putText(image, text="Go front"+str(locationvector), org=(200,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                # elif(locationvector[2]<=0.1 and locationvector[2]>=-0.1):
                                #     cv2.putText(image, text="Stay"+str(locationvector), org=(200,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                                # else:
                                #     cv2.putText(image, text="Go back"+str(locationvector[2]), org=(200,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing_styles.get_default_hand_landmarks_style(),
                                mp_drawing_styles.get_default_hand_connections_style())
                            
                            # leftangle = calculate_joint_angle(bodyvector,lefthandvector)
                            # ㅣ_r = R.from_rotvec(leftangle)
                            # ㅣ_r.as_euler('zxy', degrees=True)
                            # rightangle = calculate_joint_angle(bodyvector,righthandvector)

                            send_data = []
                            # send_data.extend(lefthandvector)
                            # send_data.extend(righthandvector)
                            # send_data.extend(headvector) 
                            # righthandvector[1] = -righthandvector[1]
                            # lefthandvector[1] = -lefthandvector[1]

                            send_data.extend(righthandvector)
                            send_data.extend(lefthandvector)
                            send_data.extend(bodyvector)
                            print(send_data[2:5])

                            data = struct.pack('<9f',*send_data)
                            conn.send(data)
                            # conn.send(str(righthandvector[0]).encode()+str(righthandvector[1]).encode()+str(righthandvector[2]).encode())
                            # if not results.multi_hand_world_landmarks:
                            #     continue
                            # for hand_world_landmarks in results.multi_hand_world_landmarks:
                            #     mp_drawing.plot_landmarks(
                            #         hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
                    pre_q = q
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        cap.release()