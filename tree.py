from math import log


def calcShannomEnt(dataset):
    numEntries = len(dataset)
    lableCounts = {}
    for featVec in dataset:
        currentLable = featVec[-1]
        lableCounts[currentLable] = lableCounts.get(currentLable, 0) + 1  # vote process
    shannonEnt = 0
    for key in lableCounts:
        prob = float(lableCounts[key] / numEntries)
        shannonEnt += -prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']
               ]
    labels = ['no surfacing', 'flippers']
    return dataset, labels
