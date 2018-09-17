import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
from scipy.stats import multivariate_normal
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example

    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 

    a= []  
    b= []
    for n in np.unique(y[:,0]):
        n = int(n)
        mask = y[:,0] == n
        a.append(list(np.array(np.mean(X[mask], axis = 0))))
    means = np.array(a)
    means = means.transpose()
    covmat = np.cov(X.transpose())
   
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD

    a= []  
    b= []

    for n in np.unique(y[:,0]):
        n = int(n)
        mask = y[:,0] == n
        a.append(list(np.array(np.mean(X[mask], axis = 0))))
        b.append(list(np.array(np.cov(X[mask].transpose()))))
        
    a_list = a
    means = np.array(a)
    means = means.transpose()
    covmats = np.array(b)

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example

    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD

    mw = means.transpose()
    pdf = []
    pdf2 = []
    ypred = []
    yp =[]
    acc = 0
    unique_cols = means.shape[1]
    #unique_cols = len(np.unique(ytest))

    for i in range(unique_cols):
        pdf.append(np.array(multivariate_normal.pdf(Xtest, mw[i], covmat)))
    p = np.array(pdf)
    p = p.transpose()

    for i in range(0,Xtest.shape[0]):
        ypred.append(np.array(np.where(p[i] == np.max(p[i]))[0]))

    for i in range(0,Xtest.shape[0]):
        ypred[i] = ypred[i] + 1
        if ypred[i] == ytest[i]:
            acc = acc + 1
            
    ypred = np.array(ypred)
    #print(ypred.shape)
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example

    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    mw = means.transpose()
    pdf = []
    pdf2 = []
    ypred = []
    yp =[]
    acc = 0

    # unique_cols = len(np.unique(ytest))
    unique_cols = means.shape[1]
    for i in range(unique_cols):
        pdf.append(np.array(multivariate_normal.pdf(Xtest, mw[i], covmats[i])))
    p = np.array(pdf)
    p = p.transpose()

    for i in range(0,Xtest.shape[0]):
        ypred.append(np.array(np.where(p[i] == np.max(p[i]))[0]))

    for i in range(0,Xtest.shape[0]):
        ypred[i] = ypred[i] + 1
        if ypred[i] == ytest[i]:
            acc = acc + 1
            
    ypred = np.array(ypred)
    #print(ypred)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1
    
    # Output: 
    # w = d x 1
    # IMPLEMENT THIS METHOD         
    # w = inv(XXt) * Xt * y
    Xt = np.transpose(X)
    Xs = np.dot(Xt, X)
    Xsi = np.linalg.inv(Xs)
    l = np.dot(Xsi, (Xt))
    w = np.dot(l, y)

    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)

    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD        
    # w = inv(lambd*I + Xt*X)*Xt*y
    xt = X.transpose()
    h = lambd*np.identity(np.shape(X)[1])
    kk = h + np.dot(xt, X)
    z = np.linalg.inv(kk)
    f = np.dot(z, xt)
    w = np.dot(f, y)
    #print(w)                                           
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1

    # Output:
    # mse
        
    # IMPLEMENT THIS METHOD
    wtx = np.dot(Xtest,w)    #get Wt*x
    u = (ytest - wtx)  #get (y-wt*x)
    ut = u.transpose()   # get (y-wt*x) transpose
    j1 = np.dot(ut, u)   # multiply the two
    k1 = 1/np.shape(Xtest)[0]    # for 1 over N
    mse = (k1)*j1
    
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD

    #print(w.shape)
    w = w.reshape(np.shape(w)[0],1)
    Xw = np.dot(X,w)
    z = y-Xw
    zt = z.transpose()
    wt = w.transpose()
    xt = X.transpose()
    zs = np.dot(zt,z)
    ws = np.dot(wt,w)
    error = (0.5 * zs) + (0.5 * lambd * ws)
    v = np.dot(xt, y-Xw)            # first term in differentitation
    error_grad = (-1*v) +  lambd * w #after differentiting the error
    error = error.flatten()
    error_grad = error_grad.flatten()

    #print("error =" , np.shape(error))
    #print("error_grad = ", np.shape(error_grad))                                         
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - intege1r (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 

    # IMPLEMENT THIS METHOD
    
    r = x.shape[0]
    Xd = np.ones((r, p+1))
    for i in range(p):
        Xd[:,i+1] = np.power(x,i+1)

    #print(Xd)
    #print(Xd.shape)    
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest[:,0])
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))


# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
#lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
opt_val = np.argmin(mses3)
lambda_opt = lambdas[opt_val]
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()