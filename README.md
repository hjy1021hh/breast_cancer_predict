# breast_cancer_predict
predict based on breast cancer cells' features

### 项目简介：

项目根据源于威斯康星州乳腺癌数据集进行数据分析，包括探索性数据分析和模型建立两部分。探索性数据分析中包括数据清洗、变量分布信息、变量相关性，处理异常值等，这部分中的绘图通过调用‘plotting.ipynb'模块实现，该模块主要目的是实现实现dataframe的分布直方图、变量的相关系数热力图、方差膨胀因子VIF的棒棒糖图，以及箱线图的绘制。模型建立通过调用’model.fitting.ipynb'实现，该模块包含三个类，可以实现逻辑回归，决策树，随机森林三种模型的拟合，可以进行调参，输出相关信息，以及绘图等功能。

### 开发环境：

python：3.11.5

jupyter notebook:  7.0.6

使用的包的版本：

pandas: 2.1.1

numpy: 1.24.3

seaborn: 0.12.2

matplotlib: 3.8.0

sklearn: 1.3.2

graphviz: 0.20.1

plotnine: 0.12.3

### 复现方法：

将breast_cancer_analysis_main.ipynb,model_fitting.ipynb,plotting.ipynb,model_fitting.py,plotting.py放到文件下同一目录下，运行breast_cancer_analysis_main.ipynb



