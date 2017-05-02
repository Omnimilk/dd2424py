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

def loadAllBatchs(zeroCentering=True):
    X,Y,y,batchMean = loadBatch("Datasets/data_batch_1",zeroCentering=zeroCentering)
    for i in range(2,6):
        path = "Datasets/data_batch_{}".format(i)
        Xn,Yn,yn,batchMean = loadBatch(path,zeroCentering=zeroCentering)
        # Xn, Yn, yn = loadBatch(path, training=False)
        X = np.append(X,Xn,axis=0)
        Y = np.append(Y,Yn,axis=0)
        y = np.append(y,yn,axis=0)
    return X,Y,y

def loadBatch(path,zeroCentering=True):
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
    batchMean = None
    if zeroCentering:
        batchMean = np.mean(X, axis=0)
        X -= batchMean#should not perform this on validation set and test set,since we may just see one sample at a time where mean value is not available
    y = np.asarray(dict["labels"],dtype="uint8")
    Y = np.asarray([convertToOneHot(index) for index in y],dtype="uint8")
    return (X,Y,y,batchMean)

def softmax(zvector):
    expVector = np.exp(zvector)
    expSum = np.sum(expVector)
    return expVector/expSum

def evaluateClassifier(X,Ws,bs):
    """
    implement equations 1 and 2 in the instruction
    :param X: training data,N*d
    :param W: weight matrix
    :param b: offset
    :return: probability matrix consists of vectors of prob for each label
    """
    netSize = len(Ws)
    assert netSize>1 ,"net size must be greater than 1"
    scores = []
    hidden_layers = []
    epsilon = 1e-8
    #initializa exponential moving average term to

    #normalize input data
    # mean = np.mean(X, axis=0, keepdims=True)
    # X = X - mean
    # variance = np.mean(X ** 2, axis=0)
    # sqrtVarT = 1. / np.sqrt(variance + epsilon)
    # X = X * sqrtVarT
    for i in range(netSize-1):
        s = np.dot(X,Ws[i])+bs[i]

        #batch normalization in forward pass
        # mean = np.mean(s,axis=0,keepdims=True)
        # s = s-mean
        # variance = np.mean(s**2,axis=0)
        # sqrtVarT = 1./np.sqrt(variance+epsilon)
        # s = s*sqrtVarT#TODO: add gamma and beta for BN as learnable parameters
        #TODO: Exponential moving average for batch means and variance afeter each epoch,eq.27 and eq.28
        #TODO: BN for backward pass
        #TODO: storing cache in forward pass or refactoring it completely

        hidden_layer = np.maximum(0, s)  # ReLU activation
        X = hidden_layer
        hidden_layers.append(hidden_layer)
        scores.append(s)
    s = np.dot(hidden_layers[-1], Ws[-1]) + bs[-1]
    scores.append(s)
    exp_scores = np.exp(scores[-1])#no ReLu in last layer
    # print (exp_scores.shape)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)#N*K
    return scores,hidden_layers,probs

def computeCost(X,Y,Ws,bs,lambdaValue):
    """
    compute the cost function given by eq.5
    :param X: training set,n*d
    :param Y: one-hot representation of labels for each training sample,n*k
    :param W: weight matrix
    :param b: offset
    :param lambdaValue: regularization coefficient
    :return: cost
    """
    regularizationTerm = lambdaValue*sum([np.sum(W**2) for W in Ws])

    sizeD = X.shape[0]
    scores, hidden_layers,P = evaluateClassifier(X,Ws,bs)
    correct = P*Y#length = n list
    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(np.sum(correct,axis=1))
    data_loss = np.sum(correct_logprobs)*1./ sizeD
    #add regularizationTerm
    cost =data_loss+regularizationTerm
    return cost

def computeAccuracy(X,y,Ws,bs):
    """
    compute the accuracy of the network's predictions given by eq.4 on a set of data
    :param X: data matrix
    :param y: correct label vector
    :param W: weight matrix
    :param b: offset
    :return: accurracy for the parameters
    """
    s,h,p = evaluateClassifier(X,Ws,bs)
    predictions = np.argmax(p, axis=1)
    acc = np.mean(predictions == y)
    return acc

def computeGradsNum(X, Y, Ws, bs, lbda, delta=0.00001):
    #compute numerical gradient
    netSize = len(Ws)

    grad_Ws = []
    grad_bs = []
    for i in range(netSize):
        grad_Wi = np.zeros(Ws[i].shape)
        grad_bi = np.zeros(bs[i].shape)
        #calculate gradient for bi
        for j in range(bs[i].shape[0]):
            b_try = np.copy(bs)
            b_try1 = np.copy(bs)
            # print (b_try.shape)
            b_try[i][j] += delta
            b_try1[i][j] -= delta
            c1 = computeCost(X, Y, Ws,b_try, lbda)
            c2 = computeCost(X, Y, Ws,b_try1, lbda)
            # print "c1: {} c2: {}".format(c1,c2)
            grad_bi[j] = (c1 - c2)/(2.*delta)
        #add to results
        grad_bs.append(grad_bi)

        # calculate gradient for Wi
        for m in range(Ws[i].shape[0]):
            for n in range(Ws[i].shape[1]):
                Ws_try = np.copy(Ws)
                Ws_try1 = np.copy(Ws)
                Ws_try[i][m, n] += delta
                Ws_try1[i][m, n] -= delta
                c1 = computeCost(X, Y, Ws_try,bs, lbda)
                c2 = computeCost(X, Y, Ws_try1,bs, lbda)
                grad_Wi[m, n] = (c1 - c2) / (2. * delta)
        #add to results
        grad_Ws.append(grad_Wi)

    return grad_Ws, grad_bs

def computeGradients(X,Y,Ws,bs,lambdaValue):
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
    netSize = len(Ws)
    assert netSize>=2, "network size must be greater than 2"

    #forward pass
    scores,hiddenlayers,P = evaluateClassifier(X,Ws,bs)

    #backward pass
    dWs = []
    dbs = []

    #gradient on final scores
    dscores = P#N*K
    dscores -=Y#N*K
    dscores /= sizeD

    #from last score to
    for i in range(netSize-1):
        dWi = np.dot(hiddenlayers[netSize-i-2].T,dscores)#netSize must be greater than 2
        dbi = np.sum(dscores,axis=0,keepdims=False)
        dh = np.dot(dscores,Ws[netSize-i-1].T)
        dh[hiddenlayers[netSize-i-2]<=0] = 0
        dscores = dh
        dWs.append(dWi)
        dbs.append(dbi)

    #backpropagate to W1 and b1
    dW1 = np.dot(X.T,dscores)
    db1 = np.sum(dscores,axis=0,keepdims=False)
    dWs.append(dW1)
    dbs.append(db1)

    # reverse dWs and dbs
    dWs.reverse()
    dbs.reverse()

    #reg grad for Ws
    for i in range(netSize):
        dWs[i] += lambdaValue*Ws[i]
    return dWs,dbs

def gradCheck(numWs,numbs,anaWs,anabs):
    netSize = len(numWs)
    atol = 1e-4  # absolute tolerance
    res = True
    for i in range(netSize):
        maxWi = np.maximum(np.abs(numWs[i]),np.abs(anaWs[i]))
        maxbi = np.maximum(np.abs(numbs[i]),np.abs(anabs[i]))
        diffWi = np.abs(numWs[i]-anaWs[i])
        diffbi = np.abs(numbs[i]-anabs[i])
        errWi = diffWi/maxWi
        errbi = diffbi/maxbi
        print "diffW{}".format(i+1)
        pprint(diffWi)
        print "diffb{}".format(i+1)
        pprint(diffbi)
        print(np.max(errWi),np.max(errbi))
        res = res and np.allclose(errWi,0,atol=atol) and np.allclose(errbi,0,atol=atol)

    return  res

def miniBatchGD(X,Y,y,validationX,validationY,validationy,GDparams,Ws,bs,lambdaValue):
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
    netSize = len(Ws)
    decayRate = 0.9#every ten epochs
    trainingSize = X.shape[0]
    batchIter = trainingSize//n_batch
    trainingLoss = []
    validationLoss = []
    #init momentum terms
    vWs = [0]*netSize
    vbs = [0]*netSize

    #train; update
    for n in range(n_epochs):
        for i in range(batchIter):
            batchStart = i*n_batch
            batchEnd = (i+1)*n_batch
            X_batch = X[batchStart:batchEnd]
            Y_batch = Y[batchStart:batchEnd]
            gWs,gbs = computeGradients(X_batch,Y_batch,Ws,bs,lambdaValue)
            #momentum update
            for j in range(netSize):
                vWs[j] = rho*vWs[j] + learning_rate*gWs[j]
                Ws[j] -=vWs[j]
                vbs[j] = rho*vbs[j] + learning_rate*gbs[j]
                bs[j] -=vbs[j]

            #vanilla update
            # for j in range(netSize):
            #     Ws[j] -=gWs[j]
            #     bs[j] -=gbs[j]
        if n%10==0:
            learning_rate = decayRate*learning_rate
        acc = computeAccuracy(X,y,Ws,bs)
        cost = computeCost(X,Y,Ws,bs,lambdaValue)
        vacc = computeAccuracy(validationX,validationy,Ws,bs)
        vcost = computeCost(validationX,validationY,Ws,bs,lambdaValue)
        trainingLoss.append(cost)
        validationLoss.append(vcost)
        # print("b:{}".format(b))
        # print("Epoch {} training accuracy:{} training cost: {} valiation accuracy: {} validation cost: {}".format(n, acc,cost, vacc,vcost))
        if n==(n_epochs-1):
            print("Epoch {} training accuracy:{} training cost: {} valiation accuracy: {} validation cost: {}".format(n+1,acc,cost,vacc,vcost))
    return (Ws,bs,trainingLoss,validationLoss)

def plotLoss(trainingLoss,validationLoss):
    # n_epochs = len(trainingLoss)
    plt.plot(trainingLoss,label="training loss")
    plt.plot(validationLoss,label = "validation loss")
    # plt.ylim((1.4,2.4))
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

def paramsInit(d,h_sizes,K,sd):
    """
    :param K: number of classes
    :param h: size of the hidden layer
    :param d: size of the input image
    :param sd: standard diviation of the initialization
    :return: Weight matrices and biad vectors
    """
    # np.random.seed(123)
    Ws = []#weight matrices
    bs = []#bias terms
    if not h_sizes:#no hidden layer, input layer is directly connected to out put layer
        Ws.append(normal(0, sd, (d, K)))
        bs.append(normal(0, sd, (K,)))
    else:
        for i,h_size in enumerate(h_sizes):
            if i==0:
                Ws.append(normal(0,sd,(d,h_size)))
                bs.append(normal(0,sd,(h_size,)))
            else:
                Ws.append(normal(0, sd, (h_sizes[i-1], h_size)))
                bs.append(normal(0, sd, (h_size,)))
            if i==(len(h_sizes)-1):
                Ws.append(normal(0, sd, (h_size, K)))
                bs.append(normal(0, sd, (K,)))
    return Ws,bs

def main():
    #Initializa the parameters of the network
    Ws,bs = paramsInit(d=3072,h_sizes=[100],K=10,sd=0.001)
    #Read in the data
    # X_train,Y_train,y_train,trainBatchMean = loadBatch("Datasets/data_batch_1",zeroCentering=False)
    # X_validation,Y_validation,y_validation,batchMeanV = loadBatch("Datasets/data_batch_2",zeroCentering=False)
    # if trainBatchMean is not None:
    #     print ("batch Mean shape: {}".format(trainBatchMean.shape))
    #     X_validation -= trainBatchMean
    # X_test,Y_test,y_test = loadBatch("Datasets/test_batch",training=False)


    # emin =-1.1#on log scale
    # emax =-0.7#on log scale
    # lmin =-6
    # lmax = -2
    # for i in range(100):
    #     Ws, bs = paramsInit(d=3072, h_sizes=[50], K=10, sd=0.001)
    #     #random search in reasonable eta and lambda
    #     e = emin + (emax-emin)*np.random.rand(1)
    #     eta = 10**e
    #     l = lmin + (lmax - lmin) * np.random.rand(1)
    #     lam = 10 ** l
    #     print ("eta: {} lam: {}".format(eta,lam))
    #     GDparams = (100,eta,10,0.9)#n_batch,learning_rate,n_epochs,rho
    #     Ws_after,bs_after, trainingLoss, validationLoss = miniBatchGD(X_train,Y_train,y_train,X_validation,Y_validation,y_validation,GDparams,Ws,bs,lam)
    #     # plotLoss(trainingLoss,validationLoss)


    # using as much data as possible
    X_train, Y_train, y_train = loadAllBatchs(zeroCentering=False)
    X_train =X_train[1000:]
    Y_train =Y_train[1000:]
    y_train =y_train[1000:]
    X_validation, Y_validation, y_validation,_ = loadBatch("Datasets/data_batch_1", zeroCentering=False)
    X_validation = X_validation[:1000]
    Y_validation = Y_validation[:1000]
    y_validation = y_validation[:1000]
    X_test, Y_test, y_test,_ = loadBatch("Datasets/test_batch", zeroCentering=False)
    GDparams = (100, 0.01, 10, 0.9)  # n_batch,learning_rate,n_epochs,rho
    Ws, bs, trainingLoss, validationLoss = miniBatchGD(X_train, Y_train, y_train,X_validation, Y_validation,y_validation, GDparams, Ws,bs, 1e-6)
    plotLoss(trainingLoss, validationLoss)
    acc = computeAccuracy(X_test,y_test,Ws,bs)
    print "test acc: {}".format(acc)



if __name__ =="__main__":
    main()



