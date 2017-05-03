import numpy as np
import time
from numpy.random import normal
from pprint import pprint
import matplotlib.pylab as plt


def convertToOneHot(index,size=10):
    oneHot = np.zeros(size)
    oneHot[index] =1
    return oneHot

def dataAugmentation(data):
    pass

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

def BN_forward(x,epsilon):
    N, D = x.shape
    #batch normalization
    mean = np.mean(x, axis=0)#D,
    x_centered = x - mean#N*D
    sq = x_centered ** 2#N*D
    var = np.mean(sq, axis=0)#D,
    sqrtvar = np.sqrt(var + epsilon)#D,
    ivar = 1. / sqrtvar#D,
    xhat = x_centered * ivar#N*D
    # print "mean:{} x_centered:{} sq:{} var:{} sqrtvar:{} ivar:{} xhat:{}".format(mean.shape,x_centered.shape,sq.shape,var.shape,sqrtvar.shape,ivar.shape,xhat.shape)
    # store intermediate
    cache = (xhat,mean, x_centered,ivar, sqrtvar, var, epsilon)
    return xhat, cache

def BN_backward(dout,cache):
    # unfold the variables stored in cache
    xhat,mean,x_centered,ivar, sqrtvar, var, epsilon = cache

    N, D = dout.shape
    #BP along the computational graph
    divar = np.sum(dout * x_centered, axis=0)#D,
    dx_centered = dout * ivar#N*D
    dsqrtvar = -1. / (sqrtvar ** 2) * divar
    dvar = 0.5 * 1. / np.sqrt(var + epsilon) * dsqrtvar
    dsq = 1. / N * np.ones((N, D)) * dvar
    dx_centered  += 2 * x_centered * dsq
    dmean = -1 * np.sum(dx_centered, axis=0)
    dx = 1. / N * np.ones((N, D)) * dmean
    dx += dx_centered
    return dx

def evaluateClassifier(X,Ws,bs,epsilon = 1e-8):
    """
    implement equations 1 and 2 in the instruction
    :param X: training data,N*d
    :param W: weight matrix
    :param b: offset
    :return: probability matrix consists of vectors of prob for each label
    """
    netSize = len(Ws)
    assert netSize>1 ,"net size must be greater than 1"
    hidden_layers = []
    caches = []
    #do BN for input data
    X,cache = BN_forward(X,epsilon)
    caches.append(cache)
    for i in range(netSize-1):
        s = np.dot(X,Ws[i])+bs[i]
        #batch normalization in forward pass
        shat,cache = BN_forward(s,epsilon)
        caches.append(cache)
        hidden_layer = np.maximum(0, shat)  # ReLU activation
        # hidden_layer = np.maximum(0, s)  # ReLU activation
        X = hidden_layer
        hidden_layers.append(hidden_layer)
    s = np.dot(hidden_layers[-1], Ws[-1]) + bs[-1]
    exp_scores = np.exp(s)#no ReLu in last layer
    # print (exp_scores.shape)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)#N*K
    return hidden_layers,caches,probs

def testClassifer(X,Ws,bs,EMA_mus,EMA_ivars):
    netSize = len(Ws)
    hidden_layers = []
    assert netSize > 1, "net size must be greater than 1"

    X_centered = X- EMA_mus[0]
    Xhat = X_centered*EMA_ivars[0]
    for i in range(netSize - 1):
        s = np.dot(Xhat, Ws[i]) + bs[i]
        # batch normalization in forward pass
        shat = (s-EMA_mus[i+1])*EMA_ivars[i+1]
        hidden_layer = np.maximum(0, shat)  # ReLU activation
        Xhat = hidden_layer
        hidden_layers.append(hidden_layer)
    s = np.dot(hidden_layers[-1], Ws[-1]) + bs[-1]
    exp_scores = np.exp(s)  # no ReLu in last layer
    # print (exp_scores.shape)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # N*K
    return probs

def computeCost(X,Y,Ws,bs,lambdaValue,EMA_mus=None,EMA_ivars=None):
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
    if EMA_mus is None:
        _,_,P = evaluateClassifier(X,Ws,bs)
    else:
        P = testClassifer(X,Ws,bs,EMA_mus,EMA_ivars)
    correct = P*Y#length = n list
    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(np.sum(correct,axis=1))
    data_loss = np.sum(correct_logprobs)*1./ sizeD
    #add regularizationTerm
    cost =data_loss+regularizationTerm
    return cost

def computeAccuracy(X,y,Ws,bs,EMA_mus=None,EMA_ivars=None):
    """
    compute the accuracy of the network's predictions given by eq.4 on a set of data
    :param X: data matrix
    :param y: correct label vector
    :param W: weight matrix
    :param b: offset
    :return: accurracy for the parameters
    """
    if EMA_ivars is None:
        h,_,p = evaluateClassifier(X,Ws,bs)
    else:
        # h, _, p_ = evaluateClassifier(X, Ws, bs)
        p = testClassifer(X,Ws,bs,EMA_mus,EMA_ivars)#Issue: all near zero
        # diffp = p-p_
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

    #keep a record of mus and vars for each layer
    mus = []
    ivars = []

    #forward pass
    hiddenlayers,caches,P = evaluateClassifier(X,Ws,bs)
    # extract mus and vars in each layer
    for cache in caches:
        xhat, mean, x_centered, ivar, sqrtvar, var, epsilon = cache
        mus.append(mean)
        ivars.append(ivar)
    mus = np.asarray(mus)
    ivars = np.asarray(ivars)

    # print("caches: {}, scores: {}, hiddenlayers:{}".format(len(caches),len(scores),len(hiddenlayers)))

    #backward pass
    dWs = []
    dbs = []

    #gradient on final scores
    dscores = P#N*K
    dscores -=Y#N*K
    dscores /= sizeD

    #from last score to
    for i in range(netSize-1):
        #from s to W,b and h
        dWi = np.dot(hiddenlayers[netSize-i-2].T,dscores)#netSize must be greater than 2
        dbi = np.sum(dscores,axis=0,keepdims=False)
        dh = np.dot(dscores,Ws[netSize-i-1].T)
        #from h to shat
        dh[hiddenlayers[netSize-i-2]<=0] = 0
        dshat = dh
        # # #back propogate from shat to s
        dscores = BN_backward(dshat,caches[netSize-i-1])
        # dscores = dh
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
    return dWs,dbs,mus,ivars

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

def miniBatchGD(X,Y,y,validationX,validationY,validationy,GDparams,Ws,bs,lambdaValue,alpha=0.99):
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

    # exponential moving average mu and var for each layer
    EMA_mus = None
    EMA_ivars = None

    #train; update
    for n in range(n_epochs):
        for i in range(batchIter):
            batchStart = i*n_batch
            batchEnd = (i+1)*n_batch
            X_batch = X[batchStart:batchEnd]
            Y_batch = Y[batchStart:batchEnd]
            gWs,gbs,mus,ivars = computeGradients(X_batch,Y_batch,Ws,bs,lambdaValue)
            # print ("mus shape: {} ivars shape: {}".format(mus.shape,ivars.shape))
            if EMA_mus is not None:#if EMA terms has been initialized, then apply 27 and 28
                EMA_mus = alpha*EMA_mus+(1-alpha)*mus
                EMA_ivars = alpha*EMA_ivars +(1-alpha)*ivars
            else:#initialize EMA with values from first batch
                EMA_mus = mus
                EMA_ivars = ivars


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
        if (n+1)%10==0:
            learning_rate = decayRate*learning_rate
        #without batchnormalization
        # acc = computeAccuracy(X, y, Ws, bs)
        # cost = computeCost(X, Y, Ws, bs, lambdaValue)
        # vacc = computeAccuracy(validationX, validationy, Ws, bs)
        # vcost = computeCost(validationX, validationY, Ws, bs, lambdaValue)

        #with batch normalization
        acc = computeAccuracy(X,y,Ws,bs,EMA_mus,EMA_ivars)
        cost = computeCost(X,Y,Ws,bs,lambdaValue,EMA_mus,EMA_ivars)
        vacc = computeAccuracy(validationX,validationy,Ws,bs,EMA_mus,EMA_ivars)
        vcost = computeCost(validationX,validationY,Ws,bs,lambdaValue,EMA_mus,EMA_ivars)
        trainingLoss.append(cost)
        validationLoss.append(vcost)
        # print("b:{}".format(b))
        # print("Epoch {} training accuracy:{} training cost: {} valiation accuracy: {} validation cost: {}".format(n, acc,cost, vacc,vcost))
        if n==(n_epochs-1):
            print("Epoch {} training accuracy:{} training cost: {} valiation accuracy: {} validation cost: {}".format(n+1,acc,cost,vacc,vcost))
    return (Ws,bs,trainingLoss,validationLoss,EMA_mus,EMA_ivars)

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
    """
    change the archetecture by simply change hidden layer size in paramsInit()
    """
    #Initializa the parameters of the network
    Ws,bs = paramsInit(d=3072,h_sizes=[50],K=10,sd=0.05)
    #Read in the data
    # X_train,Y_train,y_train,trainBatchMean = loadBatch("Datasets/data_batch_1",zeroCentering=False)
    # X_validation,Y_validation,y_validation,_ = loadBatch("Datasets/data_batch_2",zeroCentering=False)
    # if trainBatchMean is not None:
    #     print ("batch Mean shape: {}".format(trainBatchMean.shape))
    #     X_validation -= trainBatchMean
    # X_test,Y_test,y_test,_ = loadBatch("Datasets/test_batch",zeroCentering=False)


    # emin =-2#on log scale
    # emax =-1#on log scale
    # lmin =-4
    # lmax = -2
    # for i in range(300):
    #     Ws, bs = paramsInit(d=3072, h_sizes=[50,30], K=10, sd=0.001)
    #     #random search in reasonable eta and lambda
    #     e = emin + (emax-emin)*np.random.rand(1)
    #     eta = 10**e
    #     l = lmin + (lmax - lmin) * np.random.rand(1)
    #     lam = 10 ** l
    #     print ("i: {} eta: {} lam: {}".format(i,eta,lam))
    #     GDparams = (100,eta,6,0.9)#n_batch,learning_rate,n_epochs,rho
    #     Ws_after,bs_after, trainingLoss, validationLoss,_,_ = miniBatchGD(X_train,Y_train,y_train,X_validation,Y_validation,y_validation,GDparams,Ws,bs,lam)
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
    GDparams = (100, 0.0307, 10, 0.9)  # n_batch,learning_rate,n_epochs,rho
    Ws, bs, trainingLoss, validationLoss,EMA_mus,EMA_ivars = miniBatchGD(X_train, Y_train, y_train,X_validation, Y_validation,y_validation, GDparams, Ws,bs, 0.00139)
    plotLoss(trainingLoss, validationLoss)
    acc = computeAccuracy(X_test,y_test,Ws,bs,EMA_mus,EMA_ivars)
    # # print ("EMA shape:{} {} ivars, {}{}".format(EMA_mus[0].shape,EMA_mus.shape,EMA_ivars[0].shape,EMA_ivars[1].shape))
    print "test acc: {}".format(acc)



if __name__ =="__main__":
    main()



