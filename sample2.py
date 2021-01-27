import math
import numpy as np
from img2feat import *

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

(Itrain, Ytrain), (Itest, Ytest) = antbee.load()
cnn = CNN('alexnet')

###############################
Xtrain = cnn( Itrain )
Xtest = cnn( Itest )

lr = LinearRegression()
lr.fit( Xtrain, Ytrain )

Ypred = lr.predict( Xtest )
rmse = math.sqrt( mean_squared_error( Ytest, Ypred ) )
acc = accuracy_score( Ytest, (Ypred>0.5).astype(np.int) )
print( 'Linear: ', rmse, acc )

###############################
tc = TenCrop( scales=[256] )
Itrain_aug = tc( Itrain )
Ytrain_aug = np.kron( Ytrain, np.ones( (tc.nb_aug,), dtype=np.int) )
Xtrain_aug = cnn( Itrain_aug )

lr = LinearRegression()
lr.fit( Xtrain_aug, Ytrain_aug )

Itest_aug = tc( Itest)
Xtest_aug = cnn( Itest_aug )
Ypred = lr.predict( Xtest_aug )
Ypred = np.reshape( Ypred, (Ypred.size//tc.nb_aug, tc.nb_aug), order='C')
Ypred = np.mean( Ypred, axis=1 )

rmse = math.sqrt( mean_squared_error( Ytest, Ypred ) )
acc = accuracy_score( Ytest, (Ypred>0.5).astype(np.int) )
print( 'Ensemble: ', rmse, acc )

###############################
lrs=[]
for i in range(tc.nb_aug):
    X = Xtrain_aug[i::tc.nb_aug,:]
    Y = Ytrain_aug[i::tc.nb_aug]
    lr = LinearRegression()
    lr.fit( X, Y )
    lrs.append( lr )

Ypred = np.zeros( (Xtest.shape[0], tc.nb_aug), dtype=np.float32 )
for i in range(tc.nb_aug):
    X = Xtest_aug[i::tc.nb_aug,:]
    P = lrs[i].predict( X )
    Ypred[:,i] = P

Ypred = np.mean( Ypred, axis=1 )

rmse = math.sqrt( mean_squared_error( Ytest, Ypred ) )
acc = accuracy_score( Ytest, (Ypred>0.5).astype(np.int) )
print( 'Bagging Ensemble: ', rmse, acc )
