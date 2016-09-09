# @author: Yunus Emrah Bulut
# This is a solution for the Bike Rental Demand competition of Kaggle. 
# I train linear regression, Ridge regression and Gradient Boosting regression 
# and choose the best model.

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
import math
 

if __name__ == '__main__': #This is for the sake of parallelization
    
    #take the original data
    data_path = "C:/.....wherever your files are located....."
    
    data = pd.read_csv(data_path+"hour.csv", header=0, delimiter=",")
    
    #load the data on a data frame
    df = pd.DataFrame(data)
    
    #create a variable called isWorking
    work_day = df['workingday'].as_matrix()
    holiday = df['holiday'].as_matrix()                       
    df['isWorking'] = np.where(np.logical_and(work_day == 1, holiday == 0), 1, 0)
    
    #prepare some lists of features for the visualization and feature engineering
    #bar is a list of categorical variables
    bar = ["season","yr","mnth","holiday","weekday","workingday","weathersit","isWorking"]
    #scat is a list of continuous variables
    scat = ["temp","atemp","hum","windspeed"]
    
    #bar plots and scatter plots
    i=0
    fig, axes= plt.subplots(nrows=4,ncols=2)
    for col in bar:
        means = df.groupby(col).cnt.mean()
        colInd = i%2
        rowInd = math.floor(i/2)
        i=i+1
        ax = axes[rowInd,colInd]
        means.plot(kind = 'bar', x = 0, y = 1, ax = ax, alpha = 0.5,color='r',fontsize=12,figsize=(10,15))
            
    fig.savefig('bar_cnt.png')
    
    i=0
    fig, axes= plt.subplots(nrows=2,ncols=2)
    for col in scat:
        means = df[[col,"cnt"]]
        colInd = i%2
        rowInd = math.floor(i/2)
        i=i+1
        ax = axes[rowInd,colInd]
        means.plot(kind = 'scatter', x = 0, y = 1, ax = ax, alpha = 0.5,color='r', fontsize=12,figsize=(10,15))
               
    fig.savefig('scatter_cnt.png')
    
    i=0
    fig, axes= plt.subplots(nrows=4,ncols=2)
    for col in bar:
        means = df.groupby(col).casual.mean()
        colInd = i%2
        rowInd = math.floor(i/2)
        i=i+1
        ax = axes[rowInd,colInd]
        means.plot(kind = 'bar', x = 0, y = 1, ax = ax, alpha = 0.5,color='r',fontsize=12,figsize=(10,15))
            
    fig.savefig('bar_casual.png')
    
    i=0
    fig, axes= plt.subplots(nrows=2,ncols=2)
    for col in scat:
        means = df[[col,"casual"]]
        colInd = i%2
        rowInd = math.floor(i/2)
        i=i+1
        ax = axes[rowInd,colInd]
        means.plot(kind = 'scatter', x = 0, y = 1, ax = ax, alpha = 0.5,color='r', fontsize=12,figsize=(10,15))
               
    fig.savefig('scatter_casual.png')
    
    
    i=0
    fig, axes= plt.subplots(nrows=4,ncols=2)
    for col in bar:
        means = df.groupby(col).registered.mean()
        colInd = i%2
        rowInd = math.floor(i/2)
        i=i+1
        ax = axes[rowInd,colInd]
        means.plot(kind = 'bar', x = 0, y = 1, ax = ax, alpha = 0.5,color='r',fontsize=12,figsize=(10,15))
            
    fig.savefig('bar_reg.png')
    
    i=0
    fig, axes= plt.subplots(nrows=2,ncols=2)
    for col in scat:
        means = df[[col,"registered"]]
        colInd = i%2
        rowInd = math.floor(i/2)
        i=i+1
        ax = axes[rowInd,colInd]
        means.plot(kind = 'scatter', x = 0, y = 1, ax = ax, alpha = 0.5,color='r', fontsize=12,figsize=(10,15))
               
    fig.savefig('scatter_reg.png')
    
    #split the original data randomly to training and test set.    
    train, test = train_test_split(data, test_size=0.20, random_state=98745)
    
    #use the features below for prediction
    features1 = ['season', 'holiday', 'workingday', 'weathersit',
            'atemp', 'hum', 'windspeed', 'yr',
            'mnth', 'weekday', 'hr']
    
       
    #fit a linear regression model with features1
    regr = linear_model.LinearRegression()
    regr.fit(train[features1], train["cnt"])
    resultRegr1 = regr.predict(test[features1])
    
    #print some metrics
    print("Linear regression on count - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultRegr1 - test["cnt"]) ** 2)))
    
    
    #fit a linear regression model with features1 on casual
    regr.fit(train[features1], train["casual"])
    resultRegr3 = regr.predict(test[features1])
    
    #print some metrics
    print("Linear regression on casual - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultRegr3 - test["casual"]) ** 2)))
    
    
    #fit a linear regression model with features1 on registered
    regr.fit(train[features1], train["registered"])
    resultRegr4 = regr.predict(test[features1])
    
    #print some metrics
    print("Linear regression on registered - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultRegr4 - test["registered"]) ** 2)))
    
    print("Linear regression on casual and registered combined - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultRegr3 + resultRegr4 - test["cnt"]) ** 2)))
    
    
    #This includes several values for alpha to tune
    param_grid = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 50, 100, 1000]}
    
    #fit a Ridge regression model with features1
    rig = Ridge()
    rigBest = GridSearchCV(rig, param_grid, n_jobs=-1).fit(train[features1], train['cnt'])
    
    # print the best hyperparameters
    print("Best alpha of the ridge regression on count: " + str(rigBest.best_params_)) 
    resultRidge = rigBest.predict(test[features1])
        
    #print some metrics
    print("Ridge ression on count - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultRidge - test["cnt"]) ** 2)))
    
    
    #fit a Ridge regression model with features1 on casual
    rigBest = GridSearchCV(rig, param_grid, n_jobs=-1).fit(train[features1], train['casual'])
    
    # print the best hyperparameters
    print("Best alpha of the ridge regression on casual: " + str(rigBest.best_params_)) 
    resultRidgeCas = rigBest.predict(test[features1])
        
    #print some metrics
    print("Ridge ression on casual - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultRidgeCas - test["casual"]) ** 2)))
    
    
    #fit a linear regression model with features1 on registered
    rigBest = GridSearchCV(rig, param_grid, n_jobs=-1).fit(train[features1], train['registered'])
    
    # print the best hyperparameters
    print("Best alpha of the ridge regression on registered: " + str(rigBest.best_params_)) 
    resultRidgeReg = rigBest.predict(test[features1])
        
    #print some metrics
    print("Ridge ression on registered - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultRidgeReg - test["registered"]) ** 2)))
    
    print("Ridge ression on casual and registered combined - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultRidgeCas + resultRidgeReg - test["cnt"]) ** 2)))
    
    
    # This includes some values for the hyperparameters of GBR for tuning
    param_grid = {'n_estimators': [100, 200,300],'learning_rate': [0.1, 0.05, 0.01], 'max_depth': [10, 20],'min_samples_leaf': [3, 7, 11]}
     
    #fit a GBR model on cnt
    error = 1000000
    est = ensemble.GradientBoostingRegressor(warm_start=True)
    gbr1 = GridSearchCV(est, param_grid, n_jobs=-1).fit(train[features1], train['cnt'])
    
    # print the best hyperparameters
    print(gbr1.best_params_) 
    resultGbr1 = gbr1.predict(test[features1])
    
    #print some metrics
    print("GBR (cnt) - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultGbr1 - test["cnt"]) ** 2)))
    
    #replicate the best model for visualization        
    est = ensemble.GradientBoostingRegressor(n_estimators=gbr1.best_params_['n_estimators'],
                                                 learning_rate=gbr1.best_params_['learning_rate'],
                                                 max_depth=gbr1.best_params_['max_depth'],
                                                 min_samples_leaf=gbr1.best_params_['min_samples_leaf'])
    est.fit(train[features1], train['cnt'])
    
    # compute test set deviance
    test_score = np.zeros((max(param_grid['n_estimators']),), dtype=np.float64)
    
    for i, y_pred in enumerate(est.staged_predict(test[features1])):
        test_score[i] = est.loss_(test["cnt"], y_pred)
    
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(max(param_grid['n_estimators'])) + 1, est.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(max(param_grid['n_estimators'])) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    
    # Plot feature importance
    feature_importance = est.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos,[features1[i] for i in sorted_idx], fontsize=8)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    fig.savefig("importance.png")


    #fit a GBR model on casual
    est = ensemble.GradientBoostingRegressor(warm_start=True)
    gbrCasual = GridSearchCV(est, param_grid, n_jobs=-1).fit(train[features1], train['casual'])
    
    # print the best hyperparameters
    print(gbrCasual.best_params_) 
            
    resultGbrCasual = gbrCasual.predict(test[features1])
    print("GBR (Casual) - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultGbrCasual - test["casual"]) ** 2)))


    #fit a GBR model on registered
    est = ensemble.GradientBoostingRegressor(warm_start=True)
    gbrRegistered = GridSearchCV(est, param_grid, n_jobs=-1).fit(train[features1], train['registered'])
   
    # print the best hyperparameters
    print(gbrRegistered.best_params_) 
            
    resultGbrRegistered = gbrRegistered.predict(test[features1])
    print("GBR (Registered) - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultGbrRegistered - test["registered"]) ** 2)))
    
    print("GBR (Casual + Registered) - Root Mean Squared Error: %.2f"
          % np.sqrt(np.mean((resultGbrCasual + resultGbrRegistered - test["cnt"]) ** 2)))




