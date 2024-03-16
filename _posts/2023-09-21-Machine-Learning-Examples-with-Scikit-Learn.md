---
title:  "Machine Learning Examples with Scikit-Learn"
date:   2023-09-21 12:00:00 -500
categories: [Blog]
tags: [machine learning, Scikit-Learn, data science]
layout: post
comments: true
---


**Note:** All code is available on [Github](https://github.com/gallileugenesis/Machine-Learning-Examples-with-Scikit-Learn)

This Jupyter Notebook provides basic examples of supervised and unsupervised machine learning algorithms using scikit-learn.

## Supervised Learning:

### K-Nearest Neighbors (KNN)
- Example: KNN

In this example, we demonstrate how to use the K-Nearest Neighbors (KNN) algorithm for classification.

### Decision Tree
- Example: DecisionTree

This example showcases the Decision Tree algorithm for classification.

### Support Vector Machine (SVM)
- Example: SVM

Here, we use the Support Vector Machine (SVM) algorithm for classification.

## Unsupervised Learning:

### K-Means Clustering
- Example: KMeans_Clustering.ipynb

In this example, we demonstrate the K-Means clustering algorithm for unsupervised learning.

## How to Use the Notebooks:

1. Open the respective Jupyter Notebook file for the algorithm you want to explore.
2. Follow the code and comments to understand the algorithm and its usage.
3. Run the code cells to execute the examples.

## Requirements:

- Python
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- mlxtend
- pydotplus (for decision tree visualization)

You can install the required packages using pip:

```shell
pip install scikit-learn pandas numpy matplotlib mlxtend pydotplus

**Example of Supervised algorithm: KNN**


```python
# Importing Libraries
import pylab as pl
from sklearn import neighbors, datasets
import pandas as pd
import numpy as np
```


```python
# Creating a routine to use the Iris dataset
iris = datasets.load_iris()
```


```python
# Converting the Iris dataset into a DataFrame
df_iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                       columns=iris['feature_names'] + ['target'])


df_iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Transforming the data into arrays
X = df_iris.iloc[:, :-1].values  # Input data
y = df_iris.iloc[:, 4].values  # Outputs or target
```


```python
# Splitting the data into training and testing sets
from sklearn.model_selection import train_test_split  # Function for dataset split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # Split 20% for testing
```


```python
# Performing data normalization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  # Object for data normalization
scaler.fit(X_train)  # Perform data normalization

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
```


```python
# Training the model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)  # Using 5 neighbors for classification
classifier.fit(X_train, y_train)  # Apply classification
```




<style>#sk-container-id-7 {color: black;background-color: white;}#sk-container-id-7 pre{padding: 0;}#sk-container-id-7 div.sk-toggleable {background-color: white;}#sk-container-id-7 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-7 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-7 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-7 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-7 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-7 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-7 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-7 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-7 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-7 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-7 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-7 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-7 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-7 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-7 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-7 div.sk-item {position: relative;z-index: 1;}#sk-container-id-7 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-7 div.sk-item::before, #sk-container-id-7 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-7 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-7 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-7 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-7 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-7 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-7 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-7 div.sk-label-container {text-align: center;}#sk-container-id-7 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-7 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-7" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>KNeighborsClassifier()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" checked><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">KNeighborsClassifier</label><div class="sk-toggleable__content"><pre>KNeighborsClassifier()</pre></div></div></div></div></div>




```python
# Making predictions
y_pred = classifier.predict(X_test)
```


```python
# Building the confusion matrix to compare the created model
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

    [[12  0  0]
     [ 0 10  0]
     [ 0  0  8]]
                  precision    recall  f1-score   support
    
             0.0       1.00      1.00      1.00        12
             1.0       1.00      1.00      1.00        10
             2.0       1.00      1.00      1.00         8
    
        accuracy                           1.00        30
       macro avg       1.00      1.00      1.00        30
    weighted avg       1.00      1.00      1.00        30
    
    


```python
# Plotting the confusion matrix
from mlxtend.plotting import plot_confusion_matrix

confusion_matrix_plot = confusion_matrix(y_test, y_pred)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_plot)
plt.show()
```
    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-09-21-Machine-Learning-Examples-with-Scikit-Learn/output_11_0.png?raw=true)
    


**Example of Supervised Algorithm: Decision Tree**


```python
from sklearn.tree import DecisionTreeClassifier  # Importing the Decision Tree classifier
from sklearn import metrics  # Importing metrics for evaluation

# Creating the classification object
clf = DecisionTreeClassifier()

# Training the classifier
clf = clf.fit(X_train, y_train)
```


```python
# Making classification predictions
y_pred = clf.predict(X_test)
```


```python
# Evaluating the model

# Plotting the confusion matrix
confusion_matrix_plot = confusion_matrix(y_test, y_pred)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_plot)
plt.show()
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-09-21-Machine-Learning-Examples-with-Scikit-Learn/output_15_0.png?raw=true)
    



```python
# Decision Tree visualization
# Importing the necessary libraries for Decision Tree visualization
from sklearn.tree import export_graphviz
from io import StringIO
from IPython.display import Image
!pip install pydotplus
import pydotplus
```

    Requirement already satisfied: pydotplus in /opt/conda/lib/python3.10/site-packages (2.0.2)
    Requirement already satisfied: pyparsing>=2.0.1 in /opt/conda/lib/python3.10/site-packages (from pydotplus) (3.0.9)
    


```python
# Building the Decision Tree for the Iris dataset
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=iris.feature_names, class_names=['0', '1', '2'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('iris.png')
Image(graph.create_png())
```




    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-09-21-Machine-Learning-Examples-with-Scikit-Learn/output_17_0.png?raw=true)
    



**Example of Supervised Algorithm: Support Vector Machine (SVM)**


```python
# Library required for SVM construction
from sklearn.svm import SVC

# Creating the SVM object
clf = SVC(gamma='auto')  # Choosing the linear kernel

# Performing classification via SVM
clf.fit(X_train, y_train)
```




<style>#sk-container-id-8 {color: black;background-color: white;}#sk-container-id-8 pre{padding: 0;}#sk-container-id-8 div.sk-toggleable {background-color: white;}#sk-container-id-8 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-8 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-8 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-8 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-8 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-8 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-8 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-8 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-8 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-8 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-8 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-8 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-8 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-8 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-8 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-8 div.sk-item {position: relative;z-index: 1;}#sk-container-id-8 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-8 div.sk-item::before, #sk-container-id-8 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-8 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-8 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-8 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-8 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-8 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-8 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-8 div.sk-label-container {text-align: center;}#sk-container-id-8 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-8 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-8" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>SVC(gamma=&#x27;auto&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" checked><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC(gamma=&#x27;auto&#x27;)</pre></div></div></div></div></div>




```python
# Making classification predictions
y_pred = clf.predict(X_test)
```


```python
# Evaluating the model
from mlxtend.plotting import plot_confusion_matrix

# Plotting the confusion matrix
confusion_matrix_plot = confusion_matrix(y_test, y_pred)

fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix_plot)
plt.show()

```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-09-21-Machine-Learning-Examples-with-Scikit-Learn/output_21_0.png?raw=true)
    


**Example of Unsupervised Algorithm: k-Means**


```python
# Importing the required libraries
from pandas import DataFrame
import matplotlib.pyplot as plt

# Creating random data
data = {'x': [25, 34, 22, 27, 33, 33, 31, 22, 35, 34, 67, 54, 57, 43, 50, 57, 59, 52, 65, 47, 49, 48, 35, 33, 44, 45, 38, 43, 51, 46],
        'y': [79, 51, 53, 78, 59, 74, 73, 57, 69, 75, 51, 32, 40, 47, 53, 36, 35, 58, 59, 50, 25, 20, 14, 12, 20, 5, 29, 27, 8, 7]
       }

# Creating the DataFrame
df = DataFrame(data, columns=['x', 'y'])
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>79</td>
    </tr>
    <tr>
      <th>1</th>
      <td>34</td>
      <td>51</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>53</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27</td>
      <td>78</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.plot()
```




    <Axes: >




    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-09-21-Machine-Learning-Examples-with-Scikit-Learn/output_24_1.png?raw=true)
    



```python
# Adding libraries to build the algorithm
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)  # Creating the k-means algorithm object to find 2 clusters
kmeans.fit(df)  # Applying the algorithm
centroids = kmeans.cluster_centers_  # Finding the centroid coordinates
print(centroids)
```

    [[38.75       61.625     ]
     [47.07142857 22.14285714]]
    

    /opt/conda/lib/python3.10/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
      warnings.warn(
    


```python
# Plotting the output graph
plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel("X")
plt.ylabel("Y")
```




    Text(0, 0.5, 'Y')




    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-09-21-Machine-Learning-Examples-with-Scikit-Learn/output_26_1.png?raw=true)
    

