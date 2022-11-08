import socket
import sys
import cv2
import mediapipe as mp
import numpy as np
import struct
import matplotlib.pyplot as plt
from pandas import DataFrame
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
pre_List =[]
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


def calcurVector(finger1,finger2):
    fvector = [hand_landmarks.landmark[finger1].x-hand_landmarks.landmark[finger2].x,
                    hand_landmarks.landmark[finger1].y-hand_landmarks.landmark[finger2].y,
                    hand_landmarks.landmark[finger1].z-hand_landmarks.landmark[finger2].z]
    return fvector
def calVector(point1, point2):
    # fvetor = [point1.x-point2.x,point1.y-point2.y,point1.z-point2.z]
    # x,y,z = fvetor
    # norm = (x**2+y**2+z**2)**(1/2)
    fvetor = [point1.x-point2.x,point1.y-point2.y,point1.z-point2.z]
    return fvetor

def ivector(vec):
        x,y,z = vec
        norm = (x**2+y**2+z**2)**(1/2)
        return [x/norm, y/norm, z/norm]

def calculate_joint_angle(joint1):
    joint2 = [1.,0.,1.]
    return np.arccos(np.clip(np.dot(joint1, joint2), -1.0, 1.0))

# def isClose(pre_q):
#     p_velocity = calVector(pre_q[1][0].landmark[0],pre_q[1][0].landmark[5])
#     px_v,py_v,pz_v = p_velocity
#     pd_velocity = (px_v**2+py_v**2+pz_v**2)**(1/2)
#     p_horizontal = calVector(pre_q[1][0].landmark[0],pre_q[1][0].landmark[17])
#     px_h,py_h,pz_h = p_horizontal
#     pd_horizontal = (px_h**2+py_h**2+pz_h**2)**(1/2)
#     # pre_palm = pd_velocity*pd_horizontal

#     c_velocity = calcurVector(0,9)
#     cx_v,cy_v,cz_v = c_velocity
#     cd_velocity = (cx_v**2+cy_v**2+cz_v**2)**(1/2)
#     c_horizontal = calcurVector(1,17)
#     cx_h,cy_h,cz_h = c_horizontal
#     cd_horizontal = (cx_h**2+cy_h**2+cz_h**2)**(1/2)
#     # cur_palm = cd_velocity*cd_horizontal

#     palm_v = cd_velocity/pd_velocity
#     palm_h = cd_horizontal/pd_horizontal
    
#     print("palm_v: ", palm_v)
#     print("palm_h: ", palm_h)

#     if(palm_v<1 and palm_h<1): # back
#         return 0
#     elif(palm_v>1 and palm_h>1): #front
#         return 1
#     else: #stay
#         return 2

def isClose(pre_q):
    ratio = []
    for i in range(5):
        p_ = calVector(pre_q[2][0].landmark[0],pre_q[2][0].landmark[fingerList[i][0]])
        c_ = calVector(hand_world_landmarks.landmark[0],hand_world_landmarks.landmark[fingerList[i][0]])
        px_v,py_v,pz_v = p_
        cx_v,cy_v,cz_v = c_
        p_d = (px_v**2+py_v**2+pz_v**2)**(1/2)
        c_d = (cx_v**2+cy_v**2+cz_v**2)**(1/2)
        c_ratio = c_d/p_d
        ratio.append(c_ratio)

    print("ratio: ", ratio)

    if all(1<x for x in ratio): # front
        return 1
    elif all(1>x for x in ratio): # back
        return 0
    else: #stay
        return 2
isFirst = True
cap = cv2.VideoCapture(0)
data = {"W0":[],"W1":[],"W2":[],"T0":[],"T1":[],"T2":[],"I0":[],"I1":[],"I2":[],"M0":[],"M1":[],"M2":[],"R0":[],"R1":[],"R2":[],"P0":[],"P1":[],"P2":[],"G0":[],"G1":[],"G2":[],}
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    framenum = 0
    while cap.isOpened():
        framenum = framenum+1
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
        
        if (results.multi_hand_landmarks and results.multi_hand_world_landmarks):
            # print(len(results.multi_handedness))
            Queue.append([results.multi_handedness, results.multi_hand_landmarks, results.multi_hand_world_landmarks])
        else:
            Queue.append(None)
    
        if len(Queue) > Q_NUM:
            q = Queue.pop(0)
            
            if(isFirst == True):
                pre_List.append(q)
                pre_List.append(q)
            pre_q = pre_List[0]
            move = pre_List[1]
            if(q is None or len(q)==0):
                for i in Queue:
                    isFail = False
                    if(i is not None):
                        isFail = True
                        break
                if(isFail):
                    q = pre_q
            if q and pre_q is not None and len(q)!=0:
                for i, (hand_handedness, hand_landmarks,hand_world_landmarks) in enumerate(zip(q[0],q[1],q[2])):
                    if len(q[0])==1 and hand_handedness.classification[0].score < CONF_THRESHOLD:
                        if pre_q is not None and len(list(pre_q))==1:
                            pre_mlh, pre_hand_landmarks,pre_hand_world_landmarks = pre_q
                            if pre_mlh.classification[0].score >= CONF_THRESHOLD:
                                hand_landmarks = pre_hand_landmarks 
                                hand_world_landmarks = pre_hand_world_landmarks
                    if(hand_handedness.classification[0].label == "Right"):
                        # righthandvector = ivector(calcurVector(fingerList[0][0],fingerList[0][3]))
                        # headvector = ivector(calcurVector(fingerList[1][0],fingerList[1][3]))
                        # lefthandvector = ivector(calcurVector(fingerList[2][0],fingerList[2][3]))
                        righthandvector = ivector(calcurVector(fingerList[0][3],fingerList[0][0]))
                        headvector = ivector(calcurVector(fingerList[1][3],fingerList[1][0]))
                        lefthandvector = ivector(calcurVector(fingerList[2][3],fingerList[2][0]))
                        if(isFirst == True or pre_q[1][0].landmark[0] == hand_landmarks.landmark[Wr].z):
                            locationvector = [0.,0.,0.]
                            isFirst = False
                        else:
                            locationvector = calVector(hand_world_landmarks.landmark[Wr],pre_q[2][0].landmark[0])
                            # print(pre_q[1][0].landmark[0])
                            # print(hand_landmarks.landmark[Wr])
                        # # print(locationvector)
                        # if(hand_world_landmarks.landmark[Wr].x>move[2][0].landmark[0].x):
                        #     cv2.putText(image, text="left"+str(locationvector), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # elif(hand_world_landmarks.landmark[Wr].x<move[2][0].landmark[0].x):
                        #     cv2.putText(image, text="right "+str(locationvector), org=(50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # else:
                        #     cv2.putText(image, text="Nor/l "+str(locationvector), org=(50,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # print(move[1][0].landmark[0])
                        # print(hand_landmarks.landmark[Wr])
                        # # cv2.putText(image, text="righthand "+str(righthandvector), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # # cv2.putText(image, text="head "+str(righthandvector), org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # # cv2.putText(image, text="lefthand "+str(righthandvector), org=(150,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # isclose = isClose(move)
                        # if(move==q):
                        #     cv2.putText(image, text="Stay"+str(locationvector), org=(200,300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # if(isclose==1):
                        #     cv2.putText(image, text="Go front"+str(locationvector), org=(200,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # elif(isclose==2):
                        #     cv2.putText(image, text="Stay"+str(locationvector), org=(200,300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # elif(isclose==0):
                        #     cv2.putText(image, text="Go back"+str(locationvector[2]), org=(200,250), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        
                        # if(locationvector[2]>0.1):
                        #     cv2.putText(image, text="Go front"+str(locationvector), org=(200,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # elif(locationvector[2]<=0.1 and locationvector[2]>=-0.1):
                        #     cv2.putText(image, text="Stay"+str(locationvector), org=(200,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        # else:
                        #     cv2.putText(image, text="Go back"+str(locationvector[2]), org=(200,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        
                        # for i in range(21):
                        #     point = hand_world_landmarks.landmark[i]
                        #     for j in r
                        #     data[].append(point)
                            
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        
                    # data = struct.pack('<3f',*righthandvector)
                    
                    # print(data.decode("utf-8"))

                    # conn.send(str(righthandvector[0]).encode()+str(righthandvector[1]).encode()+str(righthandvector[2]).encode())
                    # if not results.multi_hand_world_landmarks:

            # if(framenum>100):
            #     pre_List[1] = q
            #     framenum =0
            pre_List[0] = q
            pre_List[1] = q

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            df = DataFrame.from_dict(data)
            # if not os.path.exists('data2.csv'):
            #     df.to_csv('data2.csv', index=False, mode='w')
            # else:
            df.to_csv('data_lefthand.csv', index=False, mode='w', header=False)
            break
cap.release()