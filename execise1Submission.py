import numpy as np
import time
from numpy.random import normal
from scipy.optimize import check_grad
from pprint import pprint
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def convertToOneHot(index,size=10):
    oneHot = np.zeros(size)
    oneHot[index] =1
    return oneHot

def normalizeRGB(dataset):
    return dataset/255.

def loadAllBatchs():
    X,Y,y = loadBatch("Datasets/data_batch_1")
    for i in range(2,6):
        path = "Datasets/data_batch_{}".format(i)
        Xn,Yn,yn = loadBatch(path)
        X = np.append(X,Xn,axis=0)
        Y = np.append(Y,Yn,axis=0)
        y = np.append(y,yn,axis=0)
    return X,Y,y

def loadBatch(path):
    """
    reads in the data from batch file and returns the image and label data in separate files
    :param path: path of the batch file
    :return: image and label data in separate files
    """
    import cPickle
    fo = open(path, 'rb')
    dict = cPickle.load(fo)
    # fo.close()
    X = np.asarray(normalizeRGB(dict["data"]))
    # X = np.asarray(dict["data"])
    # X -= np.mean(X, axis=0)
    # cov = np.dot(X.T, X) / X.shape[0]  # get the data covariance matrix
    # U, S, V = np.linalg.svd(cov)
    # X = np.dot(X, U)  # decorrelate the data
    y = np.asarray(dict["labels"],dtype="uint8")
    Y = np.asarray([convertToOneHot(index) for index in y],dtype="uint8")
    return (X,Y,y)

def softmax(zvector):
    expVector = np.exp(zvector)
    expSum = np.sum(expVector)
    return expVector/expSum

def evaluateClassifier(X,W,b):
    """
    implement equations 1 and 2 in the instruction
    :param X: training data,n*d
    :param W: weight matrix
    :param b: offset
    :return: probability matrix consists of vectors of prob for each label
    """
    scores = np.dot(X,W)+b#N*K
    # print(scores.shape)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)#N*K
    return probs


def computeCost(X,Y,W,b,lambdaValue):
    """
    compute the cost function given by eq.5
    :param X: training set,n*d
    :param Y: one-hot representation of labels for each training sample,n*k
    :param W: weight matrix
    :param b: offset
    :param lambdaValue: regularization coefficient
    :return: cost
    """
    regularizationTerm = lambdaValue*np.sum(W**2)

    sizeD = X.shape[0]
    P = evaluateClassifier(X,W,b)
    correct = P*Y#length = n list
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(np.sum(correct,axis=1))
    data_loss = np.sum(corect_logprobs)/ sizeD
    #add regularizationTerm
    cost =data_loss+regularizationTerm
    return cost

def computeAccuracy(X,y,W,b):
    """
    compute the accuracy of the network's predictions given by eq.4 on a set of data
    :param X: data matrix
    :param y: correct label vector
    :param W: weight matrix
    :param b: offset
    :return: accurracy for the parameters
    """
    scores = np.dot(X,W)+b
    predictions = np.argmax(scores, axis=1)
    acc = np.mean(predictions == y)
    return acc

def computeGradsNum(X, Y, W, b, lbda, h):
    # Numerical check for gradients
    k = W.shape[1]
    # print ("d:{}".format(k))
    grad_W = np.zeros(W.shape)#d*k
    grad_b = np.zeros(k)#d

    for i in range(k):
        b_try = np.copy(b)
        b_try1 = np.copy(b)
        # print (b_try.shape)
        b_try[i] += h
        b_try1[i] -= h
        c1 = computeCost(X, Y, W, b_try, lbda)
        c2 = computeCost(X, Y, W, b_try1, lbda)
        grad_b[i] = (c1 - c2) / (2*h)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_try = np.copy(W)
            W_try1 = np.copy(W)
            W_try[i, j] = W_try[i, j] + h
            W_try1[i, j] = W_try1[i, j] - h
            c1 = computeCost(X, Y, W_try, b, lbda)
            c2 = computeCost(X, Y, W_try1, b, lbda)
            grad_W[i, j] = (c1 - c2) /(2*h)

    return grad_W, grad_b

def computeGradients(X,Y,P,W,lambdaValue):
    """
    evaluates the gradients of cost with regard to W and b for a mini-batch using eq.10,11
    :param X: mini-batch samples
    :param Y: one-hod ground truth
    :param P: prediction prob matrix
    :param W: weight matrix
    :param lambdaValue: the coefficient of regularization
    :return: (grad_W,grad_b)
    """
    sizeD = X.shape[0]
    #gradient on scores
    dscores = P
    dscores -=Y
    dscores /= sizeD
    #backpropagate to W and b
    dW = np.dot(X.T,dscores)
    # db = np.sum(dscores,axis=0,keepdims=True)
    db = np.sum(dscores, axis=0, keepdims=False)
    #reg grad for W
    dW += lambdaValue*W
    return (dW,db)

def gradCheck(numW,numb,anaW,anab):
    maxW = np.maximum(np.abs(numW),np.abs(anaW))
    maxb = np.maximum(np.abs(numb),np.abs(anab))
    diffW = np.abs(numW-anaW)
    diffb = np.abs(numb-anab)
    errW = diffW/maxW
    errb = diffb/maxb
    atol = 1e-4#absolute tolerance
    pprint(errW)
    pprint(errb)
    print(np.max(errW),np.max(errb),errW.shape,errb.shape)
    return np.allclose(errW,0,atol=atol) and np.allclose(errb,0,atol=atol)

def miniBatchGD(X,Y,y,GDparams,W,b,lambdaValue,validationX,validationY,validationy):
    """
    :param X:all training images
    :param Y,y:labels for the training images
    :param GDparams: an object containing the hyperparameters: n_batch,learning_rate,n_epochs(100,0.01,20)
    :param W:initial weight matrix
    :param b:initial bias
    :param lambdaValue:regularization coefficient
    :return:Wstar,bstar
    """
    n_batch,learning_rate,n_epochs = GDparams
    decayRate = 0.9#every epoch
    trainingSize = X.shape[0]
    batchIter = trainingSize//n_batch
    trainingLoss = []
    validationLoss = []
    for n in range(n_epochs):
        for i in range(batchIter):
            batchStart = i*n_batch
            batchEnd = (i+1)*n_batch
            X_batch = X[batchStart:batchEnd]
            Y_batch = Y[batchStart:batchEnd]
            P = evaluateClassifier(X_batch,W,b)
            grad_W,grad_b = computeGradients(X_batch,Y_batch,P,W,lambdaValue)
            W -= learning_rate*grad_W
            b -= learning_rate*grad_b
        learning_rate = 0.9*learning_rate
        acc = computeAccuracy(X,y,W,b)
        cost = computeCost(X,Y,W,b,lambdaValue)
        vacc = computeAccuracy(validationX,validationy,W,b)
        vcost = computeCost(validationX,validationY,W,b,lambdaValue)
        trainingLoss.append(cost)
        validationLoss.append(vcost)
        # print("b:{}".format(b))
        if n==(n_epochs-1):
            print("Epoch {} training accuracy:{} training cost: {} valiation accuracy: {} validation cost: {}".format(n,acc,cost,vacc,vcost))
    return (W,b,trainingLoss,validationLoss)

def plotLoss(trainingLoss,validationLoss):
    plt.plot(trainingLoss,label="training loss")
    plt.plot(validationLoss,label = "validation loss")
    plt.legend()
    plt.show()

def plotWeightMatrix(W,n):
    #preprocess W
    maxV = np.max(W)
    minV = np.min(W)
    rangeV = maxV-minV
    W = (W-minV)/rangeV
    W = W.T
    for i in range(10):
        squared_image = np.rot90(np.reshape(W[i], (32, 32, 3), order='F'), k=3)
        plt.subplot(1, 10, i + 1)
        plt.imshow(squared_image, interpolation='gaussian')
        plt.axis("off")
    plt.savefig("{}".format(n),bbox_inches = 'tight',pad_inches = 0)

def main():
    np.random.seed(123)
    K,N,d = 10,10000,3072
    W_initial = 0.01 * normal(0, 1, (d, K))
    b_initial = 0.01 * normal(0, 1, (K,))
    # plotWeightMatrix(W_initial,0)


    #training data and validation data in the instruction
    X,Y,y = loadBatch("Datasets/data_batch_1")#10000*3072,10000*10,10000,
    validationX,validationY,validationy = loadBatch("Datasets/data_batch_2")
    # squared_image = np.rot90(np.reshape(X[1000], (32, 32, 3), order='F'), k=3)
    # print (squared_image)
    # plt.imshow(squared_image)
    # plt.axis('off')
    # plt.title('try')
    # plt.show()
    # testX,textY,testy = loadBatch("Datasets/test_batch")

    #Test the analytical gradient
    # X_test = X[0:2]
    # Y_test = Y[0:2]`
    # P_test = evaluateClassifier(X_test,W_initial,b_initial)
    # analyticW,analyticb = computeGradients(X_test,Y_test,P_test,W_initial,0)
    # numericW,numericb = computeGradsNum(X_test,Y_test,W_initial,b_initial,0,0.00001)
    # checkRes = gradCheck(numericW,numericb,analyticW,analyticb)
    # print (checkRes)

    #test the mini-batch gradient
    testParams = [(100,0.1,40,0),(100,0.01,40,0),(100,0.01,40,0.1),(100,0.01,40,1)]
    for i,params in enumerate(testParams):
        GDparams = params[:3]#n_batch,learning_rate,n_epochs
        reg = params[3]
        newW,newb,trainingLoss,validationLoss = miniBatchGD(X,Y,y,GDparams,W_initial,b_initial,reg,validationX,validationY,validationy)
        # print (np.max(newW),np.min(newW))
        plotWeightMatrix(newW,i+1)
        # plotLoss(trainingLoss,validationLoss)

if __name__ =="__main__":
    main()
