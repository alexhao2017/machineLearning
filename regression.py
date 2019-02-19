from numpy import *


def loadDataSet(filename):
    #the last item of data represent target value
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[]
    labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        #split line to list
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat
def standRegres(xArr,yArr):
    xMat=mat(xArr)
    yMat=mat(yArr).T
    xTx=xMat.T*xMat
    #calculate determint value if det=0 it means matrix cannot be inverse
    if linalg.det(xTx)==0.0:
        print('This matrix is singular,cannot do inverse')
        return

    ws=xTx.I*(xMat.T*yMat)
    return ws

