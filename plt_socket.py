import socket
import numpy
import cv2
import mediapipe as mp
import numpy as np
import struct
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas import DataFrame
import csv
from matplotlib import gridspec

from example.openGL import getdata
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

HOST="192.168.0.110" # 192.168.0.11: 기숙사,  192.168.0.40: 연구실
PORT=8888

isFirst = True
data = {"W0":[],"T0":[],"T1":[],"T2":[],"T3":[],"I0":[],"I1":[],"I2":[],"I3":[],"M0":[],"M1":[],"M2":[],"M3":[],"R0":[],"R1":[],"R2":[],"R3":[],"P0":[],"P1":[],"P2":[],"P3":[]}
data_smoothing = {"W0":[],"T0":[],"T1":[],"T2":[],"T3":[],"I0":[],"I1":[],"I2":[],"I3":[],"M0":[],"M1":[],"M2":[],"M3":[],"R0":[],"R1":[],"R2":[],"R3":[],"P0":[],"P1":[],"P2":[],"P3":[]}
fingerName = ["T","I","M","R","P"]
before_hands = []
after_hands = []
# edges = ((0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20))
edge = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]]

def getdata_a():
    with open('data_lefthand_smoothing_after.csv', 'r') as f:
        rdr = csv.reader(f)
        for i,line in enumerate(rdr):
            myline = []
            for j in range(21):
                my_list = [float(e) for e in line[j][1:-1].split(',')]
                myline.extend(my_list)
                if j ==0:
                    data_smoothing["W0"].append(my_list)
                else:
                    name = fingerName[(j-1)//4]+str((j-1)%4) 
                    data_smoothing[name].append(my_list)
            after_hands.append(myline)
            

    # print(data_smoothing)


def getdata_b():
    with open('data_lefthand_smoothing_before.csv', 'r') as f:
        rdr = csv.reader(f)
        for i,line in enumerate(rdr):
            myline = []
            for j in range(21):
                my_list = [float(e) for e in line[j][1:-1].split(',')]
                myline.extend(my_list)
                if j ==0:
                    data["W0"].append(my_list)

                else:
                    name = fingerName[(j-1)//4]+str((j-1)%4) 
                    data[name].append(my_list)
            before_hands.append(myline)
getdata_b()
getdata_a()


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

    for i in range(200):
        send_data = before_hands[i]
        data = struct.pack('<21f',*send_data)
        conn.send(data)
