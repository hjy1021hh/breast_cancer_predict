#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


# In[2]:


plt.rcParams['font.sans-serif'] = ['Simhei']  #显示中文
plt.rcParams['axes.unicode_minus'] = False 


# In[3]:


class logsitic_regression:
    def __init__(self, xtrain,ytrain,xtest,ytest):
        self.train_x = xtrain
        self.train_y = ytrain
        self.test_x = xtest
        self.test_y = ytest
    def best_parameters(self,penaltys,Cs):
        tuned_parameters=dict(penalty = penaltys,C = Cs)
        lr_penalty=LogisticRegression(solver='liblinear',random_state = 0)
        grid = GridSearchCV(lr_penalty,tuned_parameters,cv=10,scoring='accuracy',n_jobs=4)
        grid.fit(self.train_x,self.train_y)
        print('best_score:',grid.best_score_)
        print('best_params:',grid.best_params_)
    def fitting(self,pena,C):
        grid = LogisticRegression(penalty=pena,C = C,solver='liblinear',random_state = 0)
        grid.fit(self.train_x,self.train_y)
        self.coef = grid.coef_
        self.score = grid.score(self.test_x,self.test_y)
        print("the coefficient of variables:\n",grid.coef_)
        print("the accuracy score on test:",grid.score(self.test_x,self.test_y))


# In[4]:


from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import  DictVectorizer
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt


# In[5]:


class decision_tree:
    def __init__(self, xtrain,ytrain,xtest,ytest,feature_name):
        self.train_x = xtrain
        self.train_y = ytrain
        self.test_x = xtest
        self.test_y = ytest
        self.feature_name = feature_name
    def fitting(self,max_depth=None,feature_importance_plotting=0):
        dec = DecisionTreeClassifier(max_depth=max_depth,random_state=50)
        dec.fit(self.train_x,self.train_y)
        self.score = dec.score(self.test_x, self.test_y)
        print("the accuracy score on test:",dec.score(self.test_x, self.test_y))
        #生成决策树图片
        dot_data = export_graphviz(dec, out_file=None, feature_names=self.feature_name,class_names=['0','1'],filled=True,rounded=True)
        graph = graphviz.Source(dot_data)
        graph.render() 
        self.feature_importance=[*zip(self.feature_name,dec.feature_importances_)]

        #特征重要性绘图
        if feature_importance_plotting:
            feature_names = list(self.feature_name)
            feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': dec.feature_importances_})
            feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
            fig, ax = plt.subplots(figsize=(16, 6))
            ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
            ax.invert_yaxis()
            ax.set_xlabel('特征重要性', fontsize=12)
            for i, v in enumerate(feature_importances_df['importance']):
                ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=8)
            plt.show()
            plt.savefig('决策树特征重要性.png')

    #调参,限制树的最大深度，超过设定深度的树枝全部剪掉
    def max_depth_finding(self,n):
        test = []
        for i in range(n):
            dec = DecisionTreeClassifier(max_depth=i+1,random_state=50)
            dec.fit(self.train_x,self.train_y)
            score = dec.score(self.test_x, self.test_y)
            test.append(score) 
        print("the best depth of the tree:",test.index(max(test))+1)
        print("the best score on test:",max(test))
        plt.plot(range(1,n+1),test,color="coral",label="max_depth")
        plt.legend()
        plt.xlabel("max_depth")
        plt.ylabel("score")
        plt.savefig( 'max_depth.png')
        plt.show()
        
    def score_with_depth_plotting(self):
        tr = []
        te = []
        for i in range(10):
            dec = DecisionTreeClassifier(random_state=50,max_depth=i+1)
            dec.fit(self.train_x,self.train_y)
            score_tr = dec.score(self.train_x,self.train_y)
            score_te = dec.score(self.test_x, self.test_y)
            tr.append(score_tr)
            te.append(score_te)
        plt.plot(range(1,11),tr,color="red",label="train")
        plt.plot(range(1,11),te,color="blue",label="test")
        plt.xticks(range(1,11))
        plt.legend()
        plt.xlabel("max_depth")
        plt.ylabel("score")
        plt.savefig('决策树_过拟合.png')
        plt.show()


# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
class randomforest:
    def __init__(self, xtrain,ytrain,xtest,ytest,feature_name,X,Y):
        self.train_x = xtrain
        self.train_y = ytrain
        self.test_x = xtest
        self.test_y = ytest
        self.feature_name = feature_name
        self.X = X
        self.Y = Y
    def best_n_estimators(self,n):
        superpa = []
        for i in range(n):
            rfc = RandomForestClassifier(n_estimators=i+1,n_jobs=-1,random_state=0)
            rfc_s = cross_val_score(rfc,self.X,self.Y,cv=10).mean()
            superpa.append(rfc_s)
        print("the best n_estimators:",superpa.index(max(superpa))+1)
        print("the best score of cross_validation:",max(superpa))
        plt.figure(figsize=[20,5])
        plt.plot(range(1,n+1),superpa)
        plt.xlabel("n_estimators")
        plt.ylabel("score")
        plt.savefig('随机森林.png')
        plt.show()
    def fitting(self,n_estimators,feature_importance_plotting=1):
        rfc = RandomForestClassifier(n_estimators=n_estimators,random_state=0) 
        rfc = rfc.fit(self.train_x,self.train_y)
        self.score = rfc.score(self.test_x, self.test_y)
        self.feature_importance=[*zip(self.feature_name,rfc.feature_importances_)]
        print("the best score on test:",rfc.score(self.test_x, self.test_y))

        #特征重要性绘图
        if feature_importance_plotting:
            feature_names = list(self.feature_name)
            feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': rfc.feature_importances_})
            feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)
            colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
            ax.invert_yaxis()
            ax.set_xlabel('特征重要性', fontsize=12)
            for i, v in enumerate(feature_importances_df['importance']):
                ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=8)
            plt.show()
            plt.savefig("随机森林特征重要性.png")


# In[7]:


try:   
    get_ipython().system('jupyter nbconvert --to python model_fitting.ipynb')
except:
    pass


# In[ ]:




