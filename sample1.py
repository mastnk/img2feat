import math
import numpy as np
from img2feat import *

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error, accuracy_score


(Xtrain, Ytrain), (Xtest, Ytest) = antbee.load_squared_npy('alexnet')
lr = LinearRegression()
lr.fit( Xtrain, Ytrain )

Ypred = lr.predict( Xtest )
rmse = math.sqrt( mean_squared_error( Ytest, Ypred ) )
acc = accuracy_score( Ytest, (Ypred>0.5).astype(np.int) )

print( 'Linear: ', rmse, acc )

ratios = [0.8, 0.6, 0.4, 0.2]
for ratio in ratios:
    bg_linear = BaggingRegressor( LinearRegression(), max_features = ratio )
    bg_linear.fit( Xtrain, Ytrain )

    Ypred = bg_linear.predict( Xtest )
    rmse = math.sqrt( mean_squared_error( Ytest, Ypred ) )
    acc = accuracy_score( Ytest, (Ypred>0.5).astype(np.int) )

    print( ratio, rmse, acc )
