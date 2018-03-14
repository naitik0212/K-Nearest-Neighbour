import csv
import numpy as np
import operator
import matplotlib.pyplot as plt


def loadTrainDataset(filename):
    trainingimages = []
    traininglabels = []
    with open(filename, newline='') as csvfile:
        csvReader = csv.reader(csvfile,delimiter=',')
        for row in csvReader:
            traininglabels.append(row[0])
            temp = row[1:]
            for i in range(len(temp)):
                temp[i] = int(temp[i])
            trainingimages.append(temp)
    return trainingimages[0:6000],traininglabels[0:6000]


def loadTestDataset(filename):
    testingimages = []
    testinglabels = []
    with open(filename, newline='') as csvfile:
        csvReader = csv.reader(csvfile, delimiter=',')
        for row in csvReader:
            testinglabels.append(row[0])
            temp = row[1:]
            for i in range(len(temp)):
                temp[i] = int(temp[i])
            testingimages.append(temp)
    return testingimages[0:1000], testinglabels[0:1000]


def L2distance(a, b):
    temp = (a - b) ** 2
    return np.sum(temp)


def sortNeighbours(result):
    result.sort(key=operator.itemgetter(0))
    return result


def findneighbours(test_image, train_Data):
    result = []
    for j in range(6000):
        train_image = np.array(train_Data[0][j])
        label_train = (train_Data[1][j])
        distance = L2distance(test_image, train_image)
        result.append((distance, label_train))
    return sortNeighbours(result)


def findKnearest(countneighbours):
    return sorted(countneighbours.items(), key=operator.itemgetter(1), reverse=True)


def find_majority(k, neighbours):
    nearestneighbours = []
    countneighbours = {}

    for x in range(k):
        nearestneighbours.append(neighbours[x][1])

    for m in range(len(nearestneighbours)):
        value = nearestneighbours[m]
        if value in countneighbours:
            countneighbours[value] += 1
        else:
            countneighbours[value] = 1

        total = findKnearest(countneighbours)

    return total[0][0]


def predictCorrectlyClassified(train_Data, test_data):
    k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    correct_classification = [0 for i in range(11)]

    neighbours = []
    for i in range(1000):
        print(i)
        test_image = np.array(test_data[0][i])
        label_test = test_data[1][i]
        neighbours = findneighbours(test_image, train_Data)

        for index in range(11):
            label_predicted = find_majority(k[index], neighbours)
            if label_predicted == label_test:
                correct_classification[index] += 1
    print(correct_classification)
    return correct_classification



def graphPlot(graph1,k):
    plt.plot(graph1, label="K-Nearest Neighbours")
    xdatapoints=[i for i in range(0, len(k))]
    plt.ylabel('Error %')
    plt.xlabel('Value of K')
    plt.xticks(xdatapoints,k)
    plt.legend()
    plt.show()



def main():
    error = [0 for i in range(11)]
    k = [1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]


    training_DataSet = loadTrainDataset('mnist_train.csv')
    testing_DataSet = loadTestDataset('mnist_test.csv')

    correct_classification = predictCorrectlyClassified(training_DataSet, testing_DataSet)

    print(correct_classification)

    for j in range(len(correct_classification)):
        error[j] = (1000-correct_classification[j])/len(testing_DataSet[1])

    print(error)
    graphPlot(error, k)


main()