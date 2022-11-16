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
successframe = 0 # success한 frame 개수가 몇개인지 판단.

preMatrix = np.empty((12,3))
curMatrix = np.empty((12,3))
moveMatrix = np.zeros((12,3))
confidence = np.zeros((2))

def n_vector(list1, list2):
    fvector = list1-list2
    x,y,z = fvector
    norm = (x**2+y**2+z**2)**(1/2)
    return [x/norm, y/norm, z/norm]


def vector_movement(moveMatrix, successframe, preMatrix, curMatrix):

    curmove = abs(curMatrix - preMatrix)
    moveMatrix = moveMatrix + ((curmove-moveMatrix)/successframe)

    return moveMatrix

def calculate_final(moveMatrix,preMatrix,curMatrix,confidence,successframe):

    # single point prediction update 
    posMatrix = np.zeros((12,3))
    for j, pPoint in enumerate(zip(preMatrix,curMatrix)):
        for i in range(3):
            if  pPoint[1][i] >pPoint[0][i] :
                posMatrix[j][i] = 1
            else:
                posMatrix[j][i] = -1

    # position = np.array([ 1 if i >j else -1  for cjoint, pjoint in enumerate(zip(preMatrix,curMatrix)) for i, j in enumerate(zip(cjoint, pjoint))])
    predict = moveMatrix*posMatrix + preMatrix 

    # Compare current vector and predict vector and update
    curMatrix = (predict * confidence[0] + curMatrix * confidence[1]) / (confidence[0]+confidence[1])

    moveMatrix = vector_movement(moveMatrix,successframe,predict,curMatrix) # Single movement vector update

    confidence[0] = (confidence[1]*confidence[0])/(confidence[1] + confidence[0])

    preMatrix = curMatrix

def make_numpy(hand_handedness):
    fingerlist = [1,2,3,4,5,6,7,8,9,10,11,12]
    point = []
    for i in fingerlist:
        fingerPoint = [hand_handedness[i].x,hand_handedness[i].y,hand_handedness[i].z]
        point.append(fingerPoint)
    return np.array(point)


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    framenum = 0
    isFirst = True
    cap = cv2.VideoCapture(0)
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
                        successframe = successframe+1

                        if successframe == 1 :
                            preMatrix = make_numpy(hand_world_landmarks.landmark)
                            continue
                        curMatrix = make_numpy(hand_world_landmarks.landmark)
                        if successframe == 2:
                            moveMatrix  = vector_movement(moveMatrix, successframe-1, preMatrix, curMatrix)
                            confidence[0] = hand_handedness.classification[0].score 
                            preMatrix = curMatrix
                            continue
                        confidence[1] = hand_handedness.classification[0].score 

                        calculate_final(moveMatrix,preMatrix,curMatrix,confidence,successframe-1)
                        preMatrix = curMatrix

                        lefthandvector = n_vector(curMatrix[0],curMatrix[3])
                        headvector = n_vector(curMatrix[4],curMatrix[7])
                        righthandvector = n_vector(curMatrix[8],curMatrix[11])
                        cv2.putText(image, text="righthand "+str(righthandvector), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        cv2.putText(image, text="head "+str(headvector), org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        cv2.putText(image, text="lefthand "+str(lefthandvector), org=(150,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        
                        
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
            df.to_csv('datapoint.csv', index=False, mode='w', header=False)
            break
cap.release()