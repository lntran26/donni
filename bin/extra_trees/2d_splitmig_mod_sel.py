import numpy as np
import dadi
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# import training data
list_train_dict = pickle.load(open('data/2d-splitmig/train-data','rb'))
train_dict = list_train_dict[0]

train_features = [train_dict[params].data.flatten() for params in train_dict]
train_label = [params for params in train_dict]

# import ML algorithms to be optimized
rfr = RandomForestRegressor()
parameters = {
    "n_estimators":[5,10,50,100,250],
    "max_depth":[2,4,8,16,32,None]  
}

# vrfr = ExtraTreesRegressor()
# parameters = {
#     "n_estimators":[5,10,50,100,250],
#     "max_depth":[2,4,8,16,32,None]  
# }

cv = GridSearchCV(rfr, parameters, n_jobs=-2)
# cv = GridSearchCV(vrfr, parameters, n_jobs=-2)

cv.fit(train_features,train_label)
def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

display(cv)