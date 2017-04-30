import numpy as np
import time
from numpy.random import normal
from pprint import pprint
import matplotlib.pylab as plt


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
        Xn,Yn,yn = loadBatch(path,training=False)
        X = np.append(X,Xn,axis=0)
        Y = np.append(Y,Yn,axis=0)
        y = np.append(y,yn,axis=0)
    return X,Y,y

def loadBatch(path,training=True):
    """
    reads in the data from batch file and returns the image and label data in separate files
    :param path: path of the batch file
    :return: image and label data in separate files
    """
    import cPickle
    fo = open(path, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    X = np.asarray(normalizeRGB(dict["data"]))
    if training:
        X -= np.mean(X, axis=0)#should not perform this on validation set and test set,since we may just see one sample at a time where mean value is not available
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

def evaluateClassifier(X,W1,b1,W2,b2):
    """
    implement equations 1 and 2 in the instruction
    :param X: training data,N*d
    :param W: weight matrix
    :param b: offset
    :return: probability matrix consists of vectors of prob for each label
    """
    s1 = np.dot(X,W1)+b1#W1: d*h; X: N*d; b1: (h.)-->s1:N*h
    hidden_layer = np.maximum(0,s1)  # ReLU activation
    # print("X: {} W1: {} W2: {}".format(X.shape,W1.shape,W2.shape))
    s2 = np.dot(hidden_layer,W2)+b2#h: N*h; W2: h*K; b2: (K,)-->s2:N*K
    exp_scores = np.exp(s2)#p = softmax(s2)?
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)#N*K
    return probs

def computeCost(X,Y,W1,b1,W2,b2,lambdaValue):
    """
    compute the cost function given by eq.5
    :param X: training set,n*d
    :param Y: one-hot representation of labels for each training sample,n*k
    :param W: weight matrix
    :param b: offset
    :param lambdaValue: regularization coefficient
    :return: cost
    """
    regularizationTerm = lambdaValue*(np.sum(W1**2)+np.sum(W2**2))

    sizeD = X.shape[0]
    P = evaluateClassifier(X,W1,b1,W2,b2)
    correct = P*Y#length = n list
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(np.sum(correct,axis=1))
    data_loss = np.sum(corect_logprobs)*1./ sizeD
    #add regularizationTerm
    cost =data_loss+regularizationTerm
    return cost

def computeAccuracy(X,y,W1,b1,W2,b2):
    """
    compute the accuracy of the network's predictions given by eq.4 on a set of data
    :param X: data matrix
    :param y: correct label vector
    :param W: weight matrix
    :param b: offset
    :return: accurracy for the parameters
    """
    s1 = np.dot(X,W1)+b1
    h = np.maximum(s1,0)
    s2 = np.dot(h,W2)+b2
    p = softmax(s2)
    predictions = np.argmax(p, axis=1)
    acc = np.mean(predictions == y)
    return acc

def computeGradsNum(X, Y, W1, b1, W2, b2, lbda, delta=0.00001):
    # Numerical check for gradients
    k = W2.shape[1]
    h = W1.shape[1]#hidden layer size

    grad_W1 = np.zeros(W1.shape)#d*k
    grad_b1 = np.zeros(W1.shape[1])#d
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(k)

    #calculate gradient for b1
    for i in range(h):
        b1_try = np.copy(b1)
        b1_try1 = np.copy(b1)
        # print (b_try.shape)
        b1_try[i] += delta
        b1_try1[i] -= delta
        c1 = computeCost(X, Y, W1, b1_try,W2, b2, lbda)
        c2 = computeCost(X, Y, W1, b1_try1,W2, b2, lbda)
        grad_b1[i] = (c1 - c2)/(2.*delta)
    # print "grad_b1"
    # pprint(grad_b1)


    # calculate gradient for b2
    for i in range(k):
        b2_try = np.copy(b2)
        b2_try1 = np.copy(b2)
        # print (b_try.shape)
        b2_try[i] += delta
        b2_try1[i] -= delta
        c1 = computeCost(X, Y, W1, b1, W2, b2_try, lbda)
        c2 = computeCost(X, Y, W1, b1, W2, b2_try1, lbda)
        grad_b2[i] = (c1 - c2) / (2. * delta)
    # print "grad_b2"
    # pprint(grad_b2)

    # calculate gradient for W2
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.copy(W2)
            W2_try1 = np.copy(W2)
            W2_try[i, j] += delta
            W2_try1[i, j] -= delta
            c1 = computeCost(X, Y, W1, b1, W2_try, b2, lbda)
            c2 = computeCost(X, Y, W1, b1, W2_try1, b2, lbda)
            # if i == 1:
            #     print "i=1,c1: {} c2: {}".format(c1,c2)
            grad_W2[i, j] = (c1 - c2) / (2. * delta)
    # print "grad_W2"
    # pprint(grad_W2)

    # calculate gradient for W1
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.copy(W1)
            W1_try1 = np.copy(W1)
            W1_try[i, j] += delta
            W1_try1[i, j] -= delta
            c1 = computeCost(X, Y, W1_try, b1, W2, b2, lbda)
            c2 = computeCost(X, Y, W1_try1, b1, W2, b2, lbda)
            grad_W1[i, j] = (c1 - c2) / (2. * delta)
    # print "grad_W1"
    # pprint(grad_W1)

    return grad_W1, grad_b1,grad_W2,grad_b2

def computeGradients(X,Y,W1,b1,W2,b2,lambdaValue):
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

    #forward pass
    s1 = np.dot(X, W1) + b1  # N*h
    h = np.maximum(0, s1)  # ReLU activation, N*h
    s2 = np.dot(h, W2) + b2#N*K
    exp_scores = np.exp(s2)  # p = softmax(s2)?
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # N*K

    #gradient on scores
    dscores = probs#N*K
    dscores -=Y#N*K
    dscores /= sizeD
    #backpropagate to W2 and b2 first
    dW2 = np.dot(h.T,dscores)#h*K
    db2 = np.sum(dscores,axis=0,keepdims=False)#K
    #backpropagate to hidden layer
    dh = np.dot(dscores,W2.T)
    dh[h<=0] = 0#ds1,h: N*h,
    #backpropagate to W1 and b1
    dW1 = np.dot(X.T,dh)
    db1 = np.sum(dh,axis=0,keepdims=False)
    #reg grad for W1 and W2
    dW1 += lambdaValue*W1
    dW2 += lambdaValue*W2
    # print "b1"
    # pprint(db1)
    # print "W1"
    # pprint(dW1)
    # print "b2"
    # pprint(db2)
    # print "W2"
    # pprint(dW2)
    return (dW1,db1,dW2,db2)

def gradCheck(numW1,numb1,numW2,numb2,anaW1,anab1,anaW2,anab2):
    maxW1 = np.maximum(np.abs(numW1),np.abs(anaW1))
    maxb1 = np.maximum(np.abs(numb1),np.abs(anab1))
    maxW2 = np.maximum(np.abs(numW2), np.abs(anaW2))
    maxb2 = np.maximum(np.abs(numb2), np.abs(anab2))
    diffW1 = np.abs(numW1-anaW1)
    diffb1 = np.abs(numb1-anab1)
    diffW2 = np.abs(numW2 - anaW2)
    diffb2 = np.abs(numb2 - anab2)
    errW1 = diffW1/maxW1
    errb1 = diffb1/maxb1
    errW2 = diffW2 / maxW2
    errb2 = diffb2 / maxb2
    atol = 1e-4#absolute tolerance
    print "diffW1"
    pprint(diffW1)
    print "diffb1"
    pprint(diffb1)
    print "diffW2"
    pprint(diffW2)
    print "diffb2"
    pprint(diffb2)
    print(np.max(errW1),np.max(errb1),np.max(errW2),np.max(errb2))
    return np.allclose(errW1,0,atol=atol) and np.allclose(errb1,0,atol=atol) and np.allclose(errW2,0,atol=atol) and np.allclose(errb2,0,atol=atol)

def miniBatchGD(X,Y,y,GDparams,W1,b1,W2,b2,lambdaValue,validationX,validationY,validationy):
    """
    :param X:all training images
    :param Y,y:labels for the training images
    :param GDparams: an object containing the hyperparameters: n_batch,learning_rate,n_epochs(100,0.01,20)
    :param W:initial weight matrix
    :param b:initial bias
    :param lambdaValue:regularization coefficient
    :return:Wstar,bstar
    """
    n_batch,learning_rate,n_epochs,rho = GDparams
    decayRate = 0.9#every ten epochs
    trainingSize = X.shape[0]
    batchIter = trainingSize//n_batch
    trainingLoss = []
    validationLoss = []
    vw1 = 0
    vb1 = 0
    vw2 = 0
    vb2 = 0
    # acc = computeAccuracy(X, y, W1, b1, W2, b2)
    # cost = computeCost(X, Y, W1, b1, W2, b2, lambdaValue)
    # vacc = computeAccuracy(validationX, validationy, W1, b1, W2, b2)
    # vcost = computeCost(validationX, validationY, W1, b1, W2, b2, lambdaValue)
    # print(
    # "Epoch {} training accuracy:{} training cost: {} valiation accuracy: {} validation cost: {}".format(0, acc, cost,
    #                                                                                                     vacc, vcost))
    for n in range(n_epochs):
        for i in range(batchIter):
            batchStart = i*n_batch
            batchEnd = (i+1)*n_batch
            X_batch = X[batchStart:batchEnd]
            Y_batch = Y[batchStart:batchEnd]
            # P = evaluateClassifier(X_batch,W,b)
            grad_W1,grad_b1,grad_W2,grad_b2 = computeGradients(X_batch,Y_batch,W1,b1,W2,b2,lambdaValue)
            #momentum update
            vw1 = rho*vw1 + learning_rate*grad_W1
            W1 -= vw1
            vb1 = rho*vb1 + learning_rate*grad_b1
            b1 -= vb1
            vw2 = rho*vw2 + learning_rate*grad_W2
            W2 -= vw2
            vb2 = rho*vb2 + learning_rate*grad_b2
            b2 -= vb2

            #vanilla update
            # W1 -=learning_rate*grad_W1
            # b1 -=learning_rate*grad_b1
            # W2 -=learning_rate*grad_W2
            # b2 -=learning_rate*grad_b2
        if n%10==0:
            learning_rate = decayRate*learning_rate
        acc = computeAccuracy(X,y,W1,b1,W2,b2)
        cost = computeCost(X,Y,W1,b1,W2,b2,lambdaValue)
        vacc = computeAccuracy(validationX,validationy,W1,b1,W2,b2)
        vcost = computeCost(validationX,validationY,W1,b1,W2,b2,lambdaValue)
        trainingLoss.append(cost)
        validationLoss.append(vcost)
        # print("b:{}".format(b))
        # print("Epoch {} training accuracy:{} training cost: {} valiation accuracy: {} validation cost: {}".format(n, acc,cost, vacc,vcost))
        if n==(n_epochs-1):
            print("Epoch {} training accuracy:{} training cost: {} valiation accuracy: {} validation cost: {}".format(n,acc,cost,vacc,vcost))
    return (W1,b1,W2,b2,trainingLoss,validationLoss)

def plotLoss(trainingLoss,validationLoss):
    n_epochs = len(trainingLoss)
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

def paramsInit(K,h,d,sd):
    """
    :param K: number of classes
    :param h: size of the hidden layer
    :param d: size of the input image
    :param sd: standard diviation of the initialization
    :return: Weight matrices and biad vectors
    """
    # np.random.seed(123)
    W1_initial = normal(0, sd, (d, h))
    b1_initial = normal(0, sd, (h,))
    W2_initial = normal(0,sd,(h,K))
    b2_initial = normal(0,sd,(K,))
    return W1_initial,b1_initial,W2_initial,b2_initial

def main():
    W1_init, b1_init,W2_init,b2_init = paramsInit(K=10,h=50,d=3072,sd=0.001)
    # print (W2_init.shape)

    #exercise 1: Read in the data and initializa the parameters of the network
    X_train,Y_train,y_train = loadBatch("Datasets/data_batch_1")
    # X_validation,Y_validation,y_validation = loadBatch("Datasets/data_batch_2",training=False)
    # X_test,Y_test,y_test = loadBatch("Datasets/test_batch",training=False)

    #grad check for grad, NB: kinks in ReLu will give analytical results exact 0 but not for numerical results
    # checkSize =2
    # Xcheck = X_train[:checkSize]
    # Ycheck = Y_train[:checkSize]
    # # ycheck = y_train[:checkSize]
    # anaW1, anab1, anaW2, anab2 = computeGradients(Xcheck, Ycheck, W1_init, b1_init, W2_init, b2_init, 0)
    # numW1,numb1,numW2,numb2 = computeGradsNum(Xcheck,Ycheck,W1_init,b1_init,W2_init,b2_init,0)
    #
    # checkRes = gradCheck(numW1,numb1,numW2,numb2,anaW1,anab1,anaW2,anab2)
    # print (checkRes)



    # emin =-1.1#on log scale
    # emax =-0.7#on log scale
    # lmin =-6
    # lmax = -2
    # for i in range(100):
    #     W1_init, b1_init, W2_init, b2_init = paramsInit(K=10, h=10, d=3072, sd=0.001)
    #     #random search in reasonable eta and lambda
    #     e = emin + (emax-emin)*np.random.rand(1)
    #     eta = 10**e
    #     l = lmin + (lmax - lmin) * np.random.rand(1)
    #     lam = 10 ** l
    #     print ("eta: {} lam: {}".format(eta,lam))
    #     GDparams = (100,eta,10,0.9)#n_batch,learning_rate,n_epochs,rho
    #     W1, b1, W2, b2, trainingLoss, validationLoss = miniBatchGD(X_train,Y_train,y_train,GDparams,W1_init,b1_init,W2_init,b2_init,lam,X_validation,Y_validation,y_validation)
        # plotLoss(trainingLoss,validationLoss)


    # using as much data as possible
    X_train, Y_train, y_train = loadAllBatchs()
    X_train =X_train[1000:]
    Y_train =Y_train[1000:]
    y_train =y_train[1000:]
    X_validation, Y_validation, y_validation = loadBatch("Datasets/data_batch_1", training=False)
    X_validation = X_validation[:1000]
    Y_validation = Y_validation[:1000]
    y_validation = y_validation[:1000]
    X_test, Y_test, y_test = loadBatch("Datasets/test_batch", training=False)
    GDparams = (100, 0.01, 30, 0.9)  # n_batch,learning_rate,n_epochs,rho
    W1, b1, W2, b2, trainingLoss, validationLoss = miniBatchGD(X_train, Y_train, y_train, GDparams, W1_init, b1_init,W2_init, b2_init, 1e-6, X_validation, Y_validation,y_validation)
    plotLoss(trainingLoss, validationLoss)
    acc = computeAccuracy(X_test, y_test, W1, b1, W2, b2)
    cost = computeCost(X_test, Y_test, W1, b1, W2, b2, 1e-6)
    print("test accuracy: {} test cost: {}".format(acc,cost))


if __name__ =="__main__":
    main()
