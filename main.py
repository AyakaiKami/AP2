from pandas import *
from utils import data_parser

from algs import linear_regression, regression_tree
from sklearn import linear_model, tree

import numpy as np
import matplotlib.pyplot as plt

def drawCasement(predictions_and_values:map):
    size=len(predictions_and_values)+1
    fig , ax = plt.subplots()
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect('equal')

    ax.set_xticks(range(0, size))
    ax.set_xlabel("True Values")

    ax.set_yticks(range(0, size))
    ax.set_ylabel("Predictions")

    for key,value in predictions_and_values.items():
        x=value[0]
        y=value[1]
        print(f"{key}:(x:{x} y:{y})")

        circle = plt.Circle((x, y), 0.5, color='blue', alpha=0.5)
        ax.text(x, y, key, color='black', fontsize=10, ha='center', va='center')

        ax.add_patch(circle)


    plt.show()


file_path="Data/tourism_dataset.csv"

GET_INPUT = False

LINEAR_REGRESSION = 0
REGRESSION_TREE = 1
ALGORITHM_USED = LINEAR_REGRESSION


whole_data = data_parser.load_data(file_path)

country = "Brazil" if GET_INPUT == False else input("Give a country for which to estimate the revenue: ")

whole_data = whole_data[whole_data[:,1]==country]
if whole_data.size == 0:
    print('Country does not exist in DB!')
    exit(1)

# we don't need the country and the location
whole_data = whole_data[:,2:]

if ALGORITHM_USED in {LINEAR_REGRESSION}:
    data_parser.convert_to_numeric(whole_data)
    td, vd = data_parser.split_train_validation(whole_data,0.9)

    # Training data
    a_1, t_1 = data_parser.split_attrib_target(td,3)
    a_1 = np.array(a_1,dtype=np.float64)
    t_1 = np.array(t_1,dtype=np.float64)

    reg = linear_model.LinearRegression()
    #reg2 = linear_regression.LinearRegression()
    #reg.fit(a_1,t_1)
    #linear_regression.train(a_1,t_1,20)
    #reg2.train(a_1,t_1,10000)




    # Testing data
    a_2, t_2 = data_parser.split_attrib_target(vd,3)
    a_2 = np.array(a_2,dtype=np.float64)
    t_2 = np.array(t_2,dtype=np.float64)

    predictions = reg.predict(a_2)
    #predictions = reg2.predict(a_2)
    real        = t_2

    categoryPredictions = {}
    categoryReals       = {}

    for vals, p, r in zip(a_2,predictions,real):
        type = vals[0]
        if categoryPredictions.get(type) == None:
            categoryPredictions[type] = p
            categoryReals[type]       = r
        else:
            categoryPredictions[type] += p
            categoryReals[type]       += r


    final_ranking = []
    for key in categoryPredictions.keys():
        final_ranking.append([key,categoryPredictions[key],categoryReals[key]])

    ranking_pred = sorted(final_ranking,key=lambda x:x[1],reverse=True)
    ranking_real = sorted(final_ranking,key=lambda x:x[2],reverse=True)
    for x, y in zip(ranking_pred,ranking_real):
        print(x,y)        
    
elif ALGORITHM_USED == REGRESSION_TREE:
    data_parser.convert_to_numeric(whole_data)
    td, vd = data_parser.split_train_validation(whole_data,0.9)

    # Training data
    a_1, t_1 = data_parser.split_attrib_target(td,3)
    a_1 = np.array(a_1,dtype=np.float64)
    t_1 = np.array(t_1,dtype=np.float64)

    #rt = tree.DecisionTreeRegressor()
    #rt = rt.fit(a_1,t_1)
    rt = regression_tree.ID3((a_1,t_1),10)

    # Testing data
    a_2, t_2 = data_parser.split_attrib_target(vd,3)
    a_2 = np.array(a_2,dtype=np.float64)
    t_2 = np.array(t_2,dtype=np.float64)

    #predictions = rt.predict(a_2)
    predictions = rt.predict(a_2)
    #predictions = reg2.predict(a_2)
    real        = t_2

    categoryPredictions = {}
    categoryReals       = {}

    categoryPredictions = {}
    categoryReals       = {}

    for vals, p, r in zip(a_2,predictions,real):
        type = vals[0]
        if categoryPredictions.get(type) == None:
            categoryPredictions[type] = p
            categoryReals[type]       = r
        else:
            categoryPredictions[type] += p
            categoryReals[type]       += r


    final_ranking = []
    for key in categoryPredictions.keys():
        final_ranking.append([key,categoryPredictions[key],categoryReals[key]])

    ranking_pred = sorted(final_ranking,key=lambda x:x[1],reverse=True)
    ranking_real = sorted(final_ranking,key=lambda x:x[2],reverse=True)

    predictions_and_values={}
    nr_instance=0
    for x, y in zip(ranking_pred,ranking_real):
        #print(x[0],y[0])
        nr_instance+=1
        predictions_and_values['instance'+str(nr_instance)]=(x[0],y[0])            
        print(predictions_and_values['instance'+str(nr_instance)])

    drawCasement(predictions_and_values)

    pass
    


    
