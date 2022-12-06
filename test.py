
import numpy as np
def vector_movement(moveMatrix, successframe, preMatrix, curMatrix):

    curmove = abs(curMatrix - preMatrix)
    moveMatrix = moveMatrix + ((curmove-moveMatrix)/successframe)

    return moveMatrix

def calculate_final(moveMatrix,preMatrix,curMatrix,confidence,successframe):
    
    # movement [mx,my,mz]
    # confidence [preVariance, curVariance]

    # single point prediction update 
    posMatrix = np.zeros((2,3))
    for j, pPoint in enumerate(zip(preMatrix,curMatrix)):
        for i in range(3):
            if  pPoint[1][i] >pPoint[0][i] :
                posMatrix[j][i] = 1
            else:
                posMatrix[j][i] = -1

    # position = np.array([ 1 if i >j else -1  for cjoint, pjoint in enumerate(zip(preMatrix,curMatrix)) for i, j in enumerate(zip(cjoint, pjoint))])
    predict = moveMatrix*posMatrix + preMatrix 

    print("posMatrix: ", posMatrix)

    print("moveMatrix: ", moveMatrix)
    
    print("curMatrix: ", curMatrix)

    print("preMatrix: ", preMatrix)

    print("predict:", predict)

    # Compare current vector and predict vector and update
    curMatrix = (predict * confidence[0] + curMatrix * confidence[1]) / (confidence[0]+confidence[1])

    print("update curMatrix:", curMatrix)


    moveMatrix = vector_movement(moveMatrix,successframe,predict,curMatrix) # Single movement vector update

    confidence[0] = (confidence[1]*confidence[0])/(confidence[1] + confidence[0])
    print("confidence:", confidence)

    preMatrix = curMatrix

successframe = 0 # success한 frame 개수가 몇개인지 판단.
AllMatrix = [[[-0.024871081,0.044426795, 0.044565517],[-0.03999114, 0.02157658, 0.024622899]],[[-0.024117704, 0.044149563, 0.04507934],[-0.04028196, 0.021810167, 0.02544101]],[[-0.0029218383,0.052468047,0.028384207],[0.013520679,0.031127322,0.020247133]],[[-0.023280317,0.044956926,0.047245786],[-0.04126036,0.023075785, 0.026474465]],[[-0.023300342,0.044606097,0.044665303],[-0.0410351,0.022898175,0.023897132]]]
Allconfidence = [3.,4.,2.,3.,4.]
preMatrix = np.empty((2,3))
curMatrix = np.empty((2,3))
moveMatrix = np.zeros((2,3))
confidence = np.zeros((2))
print(confidence)
for i in range(5):
    successframe = successframe+1
    if i ==0 :
        preMatrix = np.array(AllMatrix[i])
        continue
    curMatrix = np.array(AllMatrix[i])
    if i == 1:
        moveMatrix  = vector_movement(moveMatrix, successframe-1, preMatrix, curMatrix)
        confidence[0] = Allconfidence[i]
        preMatrix = curMatrix
        continue
    confidence[1] = Allconfidence[i]
    calculate_final(moveMatrix,preMatrix,curMatrix,confidence,successframe-1)
    preMatrix = curMatrix