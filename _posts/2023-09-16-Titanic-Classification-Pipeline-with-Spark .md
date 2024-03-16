---
title:  "Pipeline de classficação do Titanic com Spark"
date:   2023-09-16 12:00:00 -500
categories: [Blog, Notebooks, Kaggle]
tags: [machine learning, classificação, spark, big data, ciência de dados, kaggle]
layout: post
comments: true
---



**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/Titanic-Classification-Pipeline-with-Spark)

Este tutorial demonstra como construir um pipeline de classificação completo usando Apache Spark e o conjunto de dados Titanic. O pipeline inclui pré-processamento de dados, engenharia de recursos, treinamento de modelo e avaliação


```python
! pip install pyspark
```

    Requirement already satisfied: pyspark in /opt/conda/lib/python3.10/site-packages (3.4.1)
    Requirement already satisfied: py4j==0.10.9.7 in /opt/conda/lib/python3.10/site-packages (from pyspark) (0.10.9.7)
    

# Iniciar Sessão Spark


```python
from pyspark.sql import SparkSession

spark = SparkSession \
        .builder \
        .appName("Titanic-ML") \
        .getOrCreate()

spark.version
```




    '3.4.1'



# Carregar o conjunto de dados

Carregamos o conjunto de dados no formato dataframe do spark. A função printSchema() mostra as colunas com seus respectivos tipos de dados.


```python
train_df = spark.read.csv('data/train.csv', header='True', inferSchema='True')

train_df.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
    
    

# Checar valores ausentes

A maioria dos modelos de aprendizado de máquina não aceitam dados ausentes ou nulos. O código a seguir mostra a quantidade de valores nulos em cada coluna. Temos 177 na coluna 'Age', 687 em 'Cabin' e 2 em 'Embarked'.


```python
from pyspark.sql.functions import col

# Iterate through the columns and count the null values
null_counts = []
for column in train_df.columns:
    null_count = train_df.filter(col(column).isNull()).count()
    null_counts.append((column, null_count))

# Display the columns and their respective null counts
for column, count in null_counts:
    print(f"Column '{column}': {count} null values")
```

    Column 'PassengerId': 0 null values
    Column 'Survived': 0 null values
    Column 'Pclass': 0 null values
    Column 'Name': 0 null values
    Column 'Sex': 0 null values
    Column 'Age': 177 null values
    Column 'SibSp': 0 null values
    Column 'Parch': 0 null values
    Column 'Ticket': 0 null values
    Column 'Fare': 0 null values
    Column 'Cabin': 687 null values
    Column 'Embarked': 2 null values
    

# Função StringIndexer

A função StringIndexer no Apache Spark é usada para converter colunas de strings (categorias) em números inteiros.

No código abaixo a coluna "SexIndexer" contém o mapeamento das categoriasda coluna "Sex" ("male" e "female") para os número inteiros (0, 1).


```python
# demostrando o sex_indexer
from pyspark.ml.feature import StringIndexer

sex_indexer = StringIndexer(inputCol='Sex', outputCol='SexIndexer')
sex_indexer.fit(train_df).transform(train_df).select("PassengerId", "Sex","SexIndexer").show(10)
```

    +-----------+------+----------+
    |PassengerId|   Sex|SexIndexer|
    +-----------+------+----------+
    |          1|  male|       0.0|
    |          2|female|       1.0|
    |          3|female|       1.0|
    |          4|female|       1.0|
    |          5|  male|       0.0|
    |          6|  male|       0.0|
    |          7|  male|       0.0|
    |          8|  male|       0.0|
    |          9|female|       1.0|
    |         10|female|       1.0|
    +-----------+------+----------+
    only showing top 10 rows
    
    

# Função OneHotEncoder

A função OneHotEncoder é usada para converter colunas categóricas em representações binárias (vetores de 0s e 1s) para que possam ser usadas como entradas em algoritmos de aprendizado de máquina que exigem entradas numéricas.

O OneHotEncoder funciona criando um conjunto de n-1 colunas binárias, para n categorias diferentes, atribuindo 1 para a categoria correspondente e 0 para as outras.  

No código abaixo a coluna "SexVector" apresenta 1 para a categoria "male" e fazio para "female".


```python
# demostrando o sex_encoder
from pyspark.ml.feature import OneHotEncoder
sex_encoder = OneHotEncoder(inputCol='SexIndexer', outputCol='SexVector')

sex_indexer_model = sex_indexer.fit(train_df).transform(train_df)

# https://stackoverflow.com/questions/42295001/how-to-interpret-results-of-spark-onehotencoder
# primeiro valor: tamanho do vetor
# segundo valor: índices dos valores que não são zero
# terceiro valor: valores que não são zero
sex_encoder.fit(sex_indexer_model).transform(sex_indexer_model).select("PassengerId", "Sex","SexVector").show(10)
```

    +-----------+------+-------------+
    |PassengerId|   Sex|    SexVector|
    +-----------+------+-------------+
    |          1|  male|(1,[0],[1.0])|
    |          2|female|    (1,[],[])|
    |          3|female|    (1,[],[])|
    |          4|female|    (1,[],[])|
    |          5|  male|(1,[0],[1.0])|
    |          6|  male|(1,[0],[1.0])|
    |          7|  male|(1,[0],[1.0])|
    |          8|  male|(1,[0],[1.0])|
    |          9|female|    (1,[],[])|
    |         10|female|    (1,[],[])|
    +-----------+------+-------------+
    only showing top 10 rows
    
    

# Função Imputer  

Tratar dados faltantes é uma das etapas mais importantes do fluxo de desemvolvimento de modelos de aprendizado. Existem diversas estratégias para lidar com dados ausentes, tanto em dados categóricos, quanto dados numéricos. 

Como vimos, a coluna 'Age' apresenta vários dados ausentes. Como trata-se de uma coluna numérica, uma estratégia possível pode ser imputar os valores faltando com a média de todas as idades.


```python
# Média das idades 
mean_age = train_df.agg({'Age': 'mean'}).collect()[0][0]
mean_age
```




    29.69911764705882



Podemos fazer esse tratamento diretamente no dataframe, como abaixo.


```python
train_df2 = train_df.fillna(mean_age, subset=['Age'])
```

E agora não temos mais dados ausentes na coluna 'Age'


```python
train_df2.filter(col('Age').isNull()).count()
```




    0



No entanto a melhor forma de fazer isso, é com a função Imputer, que faz examente a mesma coisa, mas pode ser incorporada no pipeline, automatizado esse processo.


```python
from pyspark.ml.feature import Imputer  

# Definir as colunas de entrada e saída
input_cols = ["Age"]
output_cols = ["Age"]

# Instanciar a função Imputer com a estratégia média
imputer = Imputer(
    inputCols=input_cols,
    outputCols=output_cols,
    strategy="mean"   
)
```

# Função VectorAssembler

VectorAssembler é uma função que é usada para combinar várias colunas de um DataFrame em uma única coluna de vetor. Isso é especialmente útil ao trabalhar com algoritmos de aprendizado de máquina que exigem que as entradas sejam representadas como um único vetor, como é o caso de muitos modelos do Spark MLlib.

Abaixo é criado o objeto "assembler" a partir da combinação das colunas 'Age' e 'SexVector' em uma única coluna chamada "features".


```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=['Age','SexVector'], outputCol='features')
```

# Instanciar um modelo de Árvore de decisão

O código abaixo mostra como instanciar o modelo DecisionTreeClassifier do Spark MLlib para criar um modelo de árvore de decisão para classificação. Este modelo será treinado para prever a coluna "Survived" (rótulos) com base nas características (features) fornecidas na coluna "features".


```python
from pyspark.ml.classification import DecisionTreeClassifier

classifier = DecisionTreeClassifier(labelCol='Survived', featuresCol='features')
```

# Criar um Pipeline

Um pipeline é uma sequência de estágios (transformações e modelos) usados para processar dados e executar tarefas em fluxo contínuo.

O Pipeline abaixo tem 5 estágios: os estágios de StringIndexer (sex_indexer) e  OneHotEncoder (sex_encoder) para a coluna Sex; o estágio imputer, para lidar com dados ausentes; o "assembler", que é o estágio usado para criar uma única coluna de características (features) a partir de várias colunas no DataFrame; e "classifier" que é o estágio que contém o modelo, no caso DecisionTreeClassifier.


```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[sex_indexer, sex_encoder, imputer, assembler, classifier])

```

# Separar os dados em treinamento e teste

A função randomSplit é usada para dividir um DataFrame em múltiplos DataFrames menores com base em proporções especificadas. Cada DataFrame resultante conterá uma parte dos dados originais e a divisão é feita aleatoriamente.

Nesse caso, vamos dividir o DataFrame em dois subconjuntos, um para treinamento e outro para teste, com as proporções [0.7, 0.3]. Ou seja, dividiremos o DataFrame original em dois DataFrames, um com aproximadamente 70%  dos dados (treinamento) e outro com aproximadamente 30% dos dados (teste).


```python
train_data, test_data = train_df.randomSplit([0.7, 0.3])
```

# Treinar o modelo

Para treinar o modelo basta chamar o método fit do Pipeline e usar os dados de treinamentos.


```python
%time predict_survived_model = pipeline.fit(train_data)
```

    CPU times: user 45.2 ms, sys: 13 ms, total: 58.2 ms
    Wall time: 1.46 s
    

# Fazer predições


```python
predictions = predict_survived_model.transform(test_data)
predictions.select('passengerId', 'age', 'sex', 'rawPrediction', 'prediction', 'Survived').show(50)
```

    +-----------+------------------+------+-------------+----------+--------+
    |passengerId|               age|   sex|rawPrediction|prediction|Survived|
    +-----------+------------------+------+-------------+----------+--------+
    |         10|              14.0|female|  [42.0,96.0]|       1.0|       1|
    |         11|               4.0|female|    [2.0,6.0]|       1.0|       1|
    |         26|              38.0|female|   [8.0,36.0]|       1.0|       1|
    |         27|30.211492842535787|  male| [301.0,65.0]|       0.0|       0|
    |         30|30.211492842535787|  male| [301.0,65.0]|       0.0|       0|
    |         31|              40.0|  male| [301.0,65.0]|       0.0|       0|
    |         32|30.211492842535787|female|  [42.0,96.0]|       1.0|       1|
    |         41|              40.0|female|   [8.0,36.0]|       1.0|       0|
    |         45|              19.0|female|  [42.0,96.0]|       1.0|       1|
    |         48|30.211492842535787|female|  [42.0,96.0]|       1.0|       1|
    |         52|              21.0|  male| [301.0,65.0]|       0.0|       0|
    |         58|              28.5|  male| [301.0,65.0]|       0.0|       0|
    |         64|               4.0|  male|    [5.0,8.0]|       1.0|       0|
    |         70|              26.0|  male| [301.0,65.0]|       0.0|       0|
    |         73|              21.0|  male| [301.0,65.0]|       0.0|       0|
    |         78|30.211492842535787|  male| [301.0,65.0]|       0.0|       0|
    |         80|              30.0|female|  [42.0,96.0]|       1.0|       1|
    |         82|              29.0|  male| [301.0,65.0]|       0.0|       1|
    |         85|              17.0|female|  [42.0,96.0]|       1.0|       1|
    |         90|              24.0|  male| [301.0,65.0]|       0.0|       0|
    |         92|              20.0|  male| [301.0,65.0]|       0.0|       0|
    |         95|              59.0|  male| [301.0,65.0]|       0.0|       0|
    |         96|30.211492842535787|  male| [301.0,65.0]|       0.0|       0|
    |         98|              23.0|  male| [301.0,65.0]|       0.0|       1|
    |        100|              34.0|  male| [301.0,65.0]|       0.0|       0|
    |        101|              28.0|female|  [42.0,96.0]|       1.0|       0|
    |        102|30.211492842535787|  male| [301.0,65.0]|       0.0|       0|
    |        104|              33.0|  male| [301.0,65.0]|       0.0|       0|
    |        108|30.211492842535787|  male| [301.0,65.0]|       0.0|       1|
    |        109|              38.0|  male| [301.0,65.0]|       0.0|       0|
    |        118|              29.0|  male| [301.0,65.0]|       0.0|       0|
    |        127|30.211492842535787|  male| [301.0,65.0]|       0.0|       0|
    |        128|              24.0|  male| [301.0,65.0]|       0.0|       1|
    |        129|30.211492842535787|female|  [42.0,96.0]|       1.0|       1|
    |        132|              20.0|  male| [301.0,65.0]|       0.0|       0|
    |        136|              23.0|  male| [301.0,65.0]|       0.0|       0|
    |        139|              16.0|  male| [301.0,65.0]|       0.0|       0|
    |        143|              24.0|female|  [42.0,96.0]|       1.0|       1|
    |        144|              19.0|  male| [301.0,65.0]|       0.0|       0|
    |        146|              19.0|  male| [301.0,65.0]|       0.0|       0|
    |        149|              36.5|  male| [301.0,65.0]|       0.0|       0|
    |        150|              42.0|  male| [301.0,65.0]|       0.0|       0|
    |        152|              22.0|female|  [42.0,96.0]|       1.0|       1|
    |        156|              51.0|  male| [301.0,65.0]|       0.0|       0|
    |        158|              30.0|  male| [301.0,65.0]|       0.0|       0|
    |        164|              17.0|  male| [301.0,65.0]|       0.0|       0|
    |        165|               1.0|  male|    [5.0,8.0]|       1.0|       0|
    |        168|              45.0|female|   [8.0,36.0]|       1.0|       0|
    |        169|30.211492842535787|  male| [301.0,65.0]|       0.0|       0|
    |        171|              61.0|  male| [301.0,65.0]|       0.0|       0|
    +-----------+------------------+------+-------------+----------+--------+
    only showing top 50 rows
    
    

# Avaliar o modelo


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import SparkSession, Row

# Define evaluation metrics
evaluator_acc = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='accuracy')
evaluator_precision = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='weightedPrecision')
evaluator_recall = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='weightedRecall')
evaluator_f1 = MulticlassClassificationEvaluator(labelCol='Survived', predictionCol='prediction', metricName='f1')
evaluator_auc = BinaryClassificationEvaluator(labelCol='Survived', rawPredictionCol='rawPrediction', metricName='areaUnderROC')

# Calculate the evaluation metrics
accuracy = evaluator_acc.evaluate(predictions)
precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)
auc = evaluator_auc.evaluate(predictions)

# Create a list of rows with the metrics
metrics_data = [
    Row(Metric="Accuracy", Value=round(accuracy,4)),
    Row(Metric="Precision", Value=round(precision,4)),
    Row(Metric="Recall", Value=round(recall,4)),
    Row(Metric="F1 Score", Value=round(f1,4)),
    Row(Metric="AUC", Value=round(auc,4)),
]

# Create a DataFrame from the list of rows
metrics_df = spark.createDataFrame(metrics_data)

# Show the DataFrame
metrics_df.show()
```

    +---------+------+
    |   Metric| Value|
    +---------+------+
    | Accuracy|0.8165|
    |Precision|0.8145|
    |   Recall|0.8165|
    | F1 Score| 0.815|
    |      AUC|0.4972|
    +---------+------+
    
    

# Explorar o Pipeline

Podemos recuperar os estágios do pipeline chamando o método stages e indicando o indice do estágio de interesse. Por exemplo, chamando o último estágio, recuperamos o modelo. 


```python
decision_tree_model = predict_survived_model.stages[-1]
decision_tree_model
```




    DecisionTreeClassificationModel: uid=DecisionTreeClassifier_f81d13a4a73a, depth=5, numNodes=19, numClasses=2, numFeatures=2



No segundo estágio recuperamos o OneHotEncoderModel e assim por diante.


```python
predict_survived_model.stages[1]
```




    OneHotEncoderModel: uid=OneHotEncoder_114cef040893, dropLast=true, handleInvalid=error



Podemos recuperar também a representação de string do modelo de árvore de decisão.


```python
# Obter a string de depuração do modelo de árvore de decisão
dot_data = decision_tree_model.toDebugString
dot_data
```




    'DecisionTreeClassificationModel: uid=DecisionTreeClassifier_f81d13a4a73a, depth=5, numNodes=19, numClasses=2, numFeatures=2\n  If (feature 1 in {1.0})\n   If (feature 0 <= 10.5)\n    If (feature 0 <= 4.5)\n     Predict: 1.0\n    Else (feature 0 > 4.5)\n     Predict: 0.0\n   Else (feature 0 > 10.5)\n    Predict: 0.0\n  Else (feature 1 not in {1.0})\n   If (feature 0 <= 30.75)\n    If (feature 0 <= 30.355746421267895)\n     If (feature 0 <= 10.5)\n      If (feature 0 <= 4.5)\n       Predict: 1.0\n      Else (feature 0 > 4.5)\n       Predict: 0.0\n     Else (feature 0 > 10.5)\n      Predict: 1.0\n    Else (feature 0 > 30.355746421267895)\n     Predict: 0.0\n   Else (feature 0 > 30.75)\n    If (feature 0 <= 36.5)\n     Predict: 1.0\n    Else (feature 0 > 36.5)\n     If (feature 0 <= 37.5)\n      Predict: 0.0\n     Else (feature 0 > 37.5)\n      Predict: 1.0\n'



Recuperando o assembler podemos acessar as features usadas na construção dos modelos. 


```python
assembler = predict_survived_model.stages[-2]
assembler.getInputCols()
```




    ['Age', 'SexVector']



# Feature importances

Podemos usar o atributo featureImportances para obter as importâncias dos recursos do modelo. As importâncias dos recursos representam a importância relativa de cada recurso no processo de tomada de decisão da árvore. Quanto maior a importância, mais influente será o recurso na realização de previsões.

Nesse caso, o artibuto 0 teve uma importância de 14%, enquanto o atributo 1 teve 86% de importância nas decisões do modelo.


```python
decision_tree_model.featureImportances
```




    SparseVector(2, {0: 0.1423, 1: 0.8577})



Podemos visualizar os nomes dos atribuitos e suas importâncias. Nesse caso, o atributo 0 é a idade e o 1 o sexo, esse último que exerceu o maior peso nas decisões.


```python
list(zip(assembler.getInputCols(), decision_tree_model.featureImportances))
```




    [('Age', 0.1423290472612345), ('SexVector', 0.8576709527387656)]




```python
import matplotlib.pyplot as plt


feature_names = assembler.getInputCols()
feature_importances = decision_tree_model.featureImportances

plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_importances)), feature_importances, align="center")
plt.yticks(range(len(feature_importances)), feature_names)
plt.xlabel("Feature Importances")
plt.title("Feature Importances of Decision Tree Model")
plt.gca().invert_yaxis()  
plt.show()
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-09-16-Titanic-Classification-Pipeline-with-Spark/output_48_0.png?raw=true)
    


# Salvar o modelo

Podemos salvar o pipeline criado, como a seguir.


```python
pipeline.write().overwrite().save("decisionTreeModel")
```
