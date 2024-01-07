#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import joypy
from matplotlib import cm

plt.rcParams['font.sans-serif'] = ['Simhei']  #显示中文
plt.rcParams['axes.unicode_minus'] = False 


# In[3]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from plotnine import *


# In[4]:


class dataplotshow:
    def __init__(self, data):
        self.data = data
    def histplotting(self):
        try:
            self.data.hist(figsize=(12, 10), bins=31)
            plt.tight_layout()
            plt.savefig( '变量分布直方图.png')
        except AttributeError as e:
            print("Error:",e)
    def corrplotting(self):
        try:
            plt.figure(figsize=(12,12))
            sns.heatmap(self.data.corr(), center=0,
                    square=True, linewidths=.5,annot=True, cbar_kws={"shrink": .5},annot_kws={'size': 10}, fmt='.1f')
            plt.savefig( '相关系数图1.png')
            plt.tight_layout()
        except AttributeError as e:
            print("Error:",e)
    def vifplotting(self):
        try:
            data1 = add_constant(self.data)
            vif = pd.DataFrame()
            vif["VIF Factor"] = [variance_inflation_factor(data1.values, i) for i in range(data1.shape[1])]
            vif["features"] = data1.columns
            vif = vif.sort_values(by='VIF Factor', ascending=True)
            vif['features'] = pd.Categorical(vif['features'], categories=vif['features'], ordered=True)
            p=(
                ggplot(vif, aes('VIF Factor', 'features'))
                + geom_segment(aes(x=0, xend='VIF Factor', y='features', yend='features'))
                + geom_point(shape='o', size=2, color='k', fill='#FC4E07')
                + theme(text=element_text(family="SimHei"))
                + scale_x_log10()
                + labs(x='VIF')
            )
            p.save(filename='vif.png',dpi=1000)
            print(p)
        except:
            print("Something wrong happened in vifplotting")
    def boxplotting(self,index):
        try:
            plt.figure(figsize=(12, 10))
            sns.boxplot(data=self.data, orient="v")
            plt.xticks(range(len(index)),index, rotation=90)
            plt.xlabel("Features")
            plt.ylabel("Value")
            plt.title("Boxplot of Features")
            plt.savefig( '箱型图.png')
            plt.show()
        except AttributeError as e:
            print("Error:",e)
    


# In[5]:


def corrplotting_sub4(data1,data2,data3,data4):
    try:
        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot(221)
        sns.heatmap(data1.corr(),square=True,cbar=0,annot=True,linewidths=.1,fmt='.1f',center=0)
        ax2 = plt.subplot(222)
        sns.heatmap(data2.corr(),square=True,cbar=0,annot=True,linewidths=.1,fmt='.1f',center=0)
        ax3 = plt.subplot(223)
        sns.heatmap(data3.corr(),square=True,cbar=0,annot=True,linewidths=.1,fmt='.1f',center=0)
        ax4 = plt.subplot(224)
        sns.heatmap(data4.corr(),square=True,cbar=0,annot=True,linewidths=.1,fmt='.1f',center=0)
        plt.savefig( '相关系数图2.png')
        plt.show()
    except (AttributeError,TypeError) as e:
        print("Error:",e)


# In[6]:


try:   
    get_ipython().system('jupyter nbconvert --to python plotting.ipynb')
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
except:
    pass


# In[ ]:




