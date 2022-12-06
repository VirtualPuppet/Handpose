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
needFinger = [1,4,5,8,9,12]

CONF_THRESHOLD = 0.6
Q_NUM =5
pre_List =[]



def calcurVector(finger1,finger2):
    fvector = [hand_world_landmarks.landmark[finger1].x-hand_world_landmarks.landmark[finger2].x,
                    hand_world_landmarks.landmark[finger1].y-hand_world_landmarks.landmark[finger2].y,
                    hand_world_landmarks.landmark[finger1].z-hand_world_landmarks.landmark[finger2].z]
    return fvector


def ivector(vec):
        x,y,z = vec
        norm = (x**2+y**2+z**2)**(1/2)
        return [x/norm, y/norm, z/norm]

def calculate_joint_angle(joint1):
    joint2 = [1.,0.,1.]
    return np.arccos(np.clip(np.dot(joint1, joint2), -1.0, 1.0))


def vector_movement(moveMatrix, successframe, preMatrix, curMatrix):

    curmove = abs(curMatrix - preMatrix)
    moveMatrix = moveMatrix + ((curmove-moveMatrix)/successframe)

    return moveMatrix

def calculate_final(moveMatrix,preMatrix,curMatrix,confidence,successframe):
    
    # movement [mx,my,mz]
    # confidence [preVariance, curVariance]

    # single point prediction update 
    posMatrix = np.empty((0,3),axis = 0)
    for cpoint, pPoint in enumerate(zip(preMatrix,curMatrix)):
        position = np.empty((0,3),axis = 0)
        for i in range(3):
            if  cpoint[i] >pPoint[i] :
                position[i] = 1
            else:
                position[i] = -1
        np.append((posMatrix,position),axis =0)


    # position = np.array([ 1 if i >j else -1  for cjoint, pjoint in enumerate(zip(preMatrix,curMatrix)) for i, j in enumerate(zip(cjoint, pjoint))])
    predict = moveMatrix*posMatrix + preMatrix 

    # Compare current vector and predict vector and update
    curMatrix = (predict * confidence[1] + curMatrix * confidence[0]) / (confidence[0]+confidence[1])


    moveMatrix = vector_movement(moveMatrix,successframe,predict,curMatrix) # Single movement vector update

    confidence[0] = (confidence[1]*confidence[0])/(confidence[1] + confidence[0])



isFirst = True
cap = cv2.VideoCapture(0)
data = {"W0":[],"W1":[],"W2":[],"T0":[],"T1":[],"T2":[],"I0":[],"I1":[],"I2":[],"M0":[],"M1":[],"M2":[],"R0":[],"R1":[],"R2":[],"P0":[],"P1":[],"P2":[],"G0":[],"G1":[],"G2":[],}
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    framenum = 0
    successframe = 0 # success한 frame 개수가 몇개인지 판단.
    moveMatrix = np.zeros((6,3))
    variance = np.empty((0,2),axis = 0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        framenum = framenum+1
        confidence = [0,0]

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if (results.multi_hand_landmarks and results.multi_hand_world_landmarks):
            successframe = successframe+1
            
            # print(len(results.multi_handedness))
            if (len(pre_List)==0):
                if(results.multi_hand_landmarks.classification[0].score < CONF_THRESHOLD):
                    pre_List.append([results.multi_handedness,results.multi_hand_world_landmarks])
                    confidence[0] = results.multi_handedness.classfication[0].score
                else:
                    continue
            

            pre_q = pre_List[0]

            q = results

            if len(q)!=0:

                # (except first)
                if successframe == 1:
                    continue

                for i, (hand_handedness , hand_world_landmarks) in enumerate(zip(q[0],q[2])):
                    pre_mlh, pre_hand_world_landmarks = pre_q
                    if len(q[0])==1 and hand_handedness.classification[0].score < CONF_THRESHOLD:
                        if pre_q is not None and len(list(pre_q))==1:
                            if pre_mlh.classification[0].score >= CONF_THRESHOLD:

                                hand_world_landmarks = pre_hand_world_landmarks

                    if(hand_handedness.classification[0].label == "Right"):
                        curMatrix = np.empty((0,3),axis = 0)
                        preMatrix = np.empty((0,3),axis = 0)

                        # vector initiate
                        for i in needFinger:
                            f = np.array([[hand_world_landmarks.landmark[i].x, hand_world_landmarks.landmark[i].y, hand_world_landmarks.landmark[i].z]])
                            pf = np.array([[pre_hand_world_landmarks.landmark[i].x, pre_hand_world_landmarks.landmark[i].y, pre_hand_world_landmarks.landmark[i].z]])
                            curMatrix = np.append(curMatrix,f,axis = 0)
                            preMatrix = np.append(preMatrix,pf,axis = 0)

                        # move_variance = [[]]
                        if(successframe == 2):
                            # 2번째는 update 불가능해서 movement랑 variance만 update해주고 끝. 
                            # smoothing 불가능.
                            vector_movement(moveMatrix, successframe-1, preMatrix, curMatrix)
                            continue


                        # vector smoothing (except first)
                        calculate_final(moveMatrix,preMatrix,curMatrix,confidence,successframe-1)

                    
                        righthandvector = ivector(calcurVector(fingerList[0][3],fingerList[0][0]))
                        headvector = ivector(calcurVector(fingerList[1][3],fingerList[1][0]))
                        lefthandvector = ivector(calcurVector(fingerList[2][3],fingerList[2][0]))

                        cv2.putText(image, text="righthand "+str(righthandvector), org=(50,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        cv2.putText(image, text="head "+str(righthandvector), org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                        cv2.putText(image, text="lefthand "+str(righthandvector), org=(150,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                    mp_drawing.draw_landmarks(
                        image,
                        hand_world_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())


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