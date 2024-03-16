---
title:  "Spark SQL na prática"
date:   2023-09-30 12:00:00 -500
categories: [Blog]
tags: [machine learning, spark, sql, data science]
layout: post
comments: true
---


**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/spark-sql-hands-on)

Nesse notebook utilizamos a biblioteca PySpark para realizar análises via Spark SQL e criar um modelo de árvore de decisão para prever a chance de derrame (stroke) com base em um conjunto de dados.


```python
! pip install pyspark
```

    Collecting pyspark
      Downloading pyspark-3.5.0.tar.gz (316.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m316.9/316.9 MB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25h  Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: py4j==0.10.9.7 in /opt/conda/lib/python3.10/site-packages (from pyspark) (0.10.9.7)
    Building wheels for collected packages: pyspark
      Building wheel for pyspark (setup.py) ... [?25ldone
    [?25h  Created wheel for pyspark: filename=pyspark-3.5.0-py2.py3-none-any.whl size=317425350 sha256=9313dc5382022bf0880e5549b239d90e668ac4c3917f203be60ca0cc78816c68
      Stored in directory: /root/.cache/pip/wheels/41/4e/10/c2cf2467f71c678cfc8a6b9ac9241e5e44a01940da8fbb17fc
    Successfully built pyspark
    Installing collected packages: pyspark
    Successfully installed pyspark-3.5.0
    

# Iniciar Sessão Spark


```python
from pyspark.sql import SparkSession

spark = SparkSession \
        .builder \
        .appName("Titanic-ML") \
        .getOrCreate()

spark.version
```

    Setting default log level to "WARN".
    To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
    23/09/30 00:11:40 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
    




    '3.5.0'



# Carregar o conjunto de dados



```python
df = spark.read.csv('/kaggle/input/stroke-data', header='True', inferSchema='True')

df.printSchema()
```

    [Stage 1:>                                                          (0 + 1) / 1]

    root
     |-- 0: integer (nullable = true)
     |-- gender: string (nullable = true)
     |-- age: double (nullable = true)
     |-- hypertension: integer (nullable = true)
     |-- heart_disease: integer (nullable = true)
     |-- ever_married: string (nullable = true)
     |-- work_type: string (nullable = true)
     |-- Residence_type: string (nullable = true)
     |-- avg_glucose_level: double (nullable = true)
     |-- bmi: double (nullable = true)
     |-- smoking_status: string (nullable = true)
     |-- stroke: integer (nullable = true)
    
    

                                                                                    

### Quantos registros existem no arquivo?


```python
num_records = df.count()
print(f"O Dataset possui {num_records} registros.")
```

    O Dataset possui 67135 registros.
    

### Quantas colunas existem no arquivo? 


```python
num_columns = len(df.columns)

print(f"O DataFrame possui {num_columns} colunas.")
```

    O DataFrame possui 12 colunas.
    

### Quantas são numéricas? 


```python
from pyspark.sql.types import NumericType

num_numeric_cols = 0

for col in df.columns:
    data_type = df.schema[col].dataType
    if isinstance(data_type, NumericType):
        num_numeric_cols += 1
        
print(f"O DataFrame possui {num_numeric_cols} colunas numéricas.")
```

    O DataFrame possui 7 colunas numéricas.
    

### Quantos pacientes sofreram e não sofreram derrame (stroke), respectivamente?


```python
df.createOrReplaceTempView('stroke_table')

stroke_count = spark.sql("SELECT stroke, COUNT(*) as count FROM stroke_table GROUP BY stroke")
stroke_count.show()
```

    [Stage 5:>                                                          (0 + 1) / 1]

    +------+-----+
    |stroke|count|
    +------+-----+
    |     1|40287|
    |     0|26848|
    +------+-----+
    
    

                                                                                    

### Quantos pacientes tiveram derrame por tipo de trabalho (work_type)?

Quantos pacientes sofreram derrame e trabalhavam respectivamente, no setor privado, de forma independente, no governo e quantas são crianças?


```python
stroke_by_work_type = spark.sql("SELECT work_type, COUNT(*) as count FROM stroke_table WHERE stroke = 1 GROUP BY work_type")
stroke_by_work_type.show()
```

    +-------------+-----+
    |    work_type|count|
    +-------------+-----+
    | Never_worked|   85|
    |Self-employed|10807|
    |      Private|23711|
    |     children|  520|
    |     Govt_job| 5164|
    +-------------+-----+
    
    

                                                                                    

### Qual a proporção, por gênero, de participantes do estudo. 

A maioria dos participantes é?


```python
gender = spark.sql("SELECT gender, COUNT(*) as count FROM stroke_table GROUP BY gender")
gender.show()
```

    +------+-----+
    |gender|count|
    +------+-----+
    |Female|39530|
    | Other|   11|
    |  Male|27594|
    +------+-----+
    
    

### Quem tem mais probabilidade de sofrer derrame: hipertensos ou não-hipertensos?
 


```python
total = spark.sql("SELECT hypertension, COUNT(*) as total FROM stroke_table GROUP BY hypertension")
total.show()
```

    +------------+-----+
    |hypertension|total|
    +------------+-----+
    |           1|11017|
    |           0|56118|
    +------------+-----+
    
    


```python
total_stroke = spark.sql("SELECT hypertension, COUNT(*) as total_stroke FROM stroke_table WHERE stroke = 1 GROUP BY hypertension")
total_stroke.show()
```

    +------------+------------+
    |hypertension|total_stroke|
    +------------+------------+
    |           1|        8817|
    |           0|       31470|
    +------------+------------+
    
    


```python
result = total.join(total_stroke, 'hypertension', 'left_outer')

result = result.withColumn("Probs_stroke", result["total_stroke"] / result["total"])

result.show()
```

    +------------+-----+------------+------------------+
    |hypertension|total|total_stroke|      Probs_stroke|
    +------------+-----+------------+------------------+
    |           1|11017|        8817|0.8003086139602432|
    |           0|56118|       31470|0.5607826365871913|
    +------------+-----+------------+------------------+
    
    

### Qual o número de pessoas que sofreram derrame por idade?





```python
stroke_by_age = spark.sql("SELECT age, COUNT(*) as count FROM stroke_table WHERE stroke = 1 GROUP BY age")
stroke_by_age.show()
```

    +----+-----+
    | age|count|
    +----+-----+
    |70.0|  881|
    |67.0|  801|
    |69.0|  677|
    |49.0|  315|
    |29.0|  306|
    |64.0|  376|
    |75.0|  809|
    |47.0|  472|
    |42.0|  318|
    |44.0|  292|
    |62.0|  550|
    |35.0|  281|
    |18.0|  218|
    |80.0| 1858|
    |39.0|  295|
    |37.0|  260|
    |34.0|  289|
    |25.0|  226|
    |36.0|  293|
    |41.0|  324|
    +----+-----+
    only showing top 20 rows
    
    

###  Com qual idade o maior número de pessoas do conjunto de dados sofreu derrame?


```python
# Ordenar os resultados em ordem decrescente pela contagem
stroke_by_age_ordered = stroke_by_age.orderBy("count", ascending=False)

# Selecionar a primeira linha, que terá a idade com a maior contagem
greater_age = stroke_by_age_ordered.first()
greater_age
```




    Row(age=79.0, count=2916)



###  Quantas pessoas sofreram derrames após os 50 anos?


```python
stroke_age_greater_50 = spark.sql("SELECT age FROM stroke_table WHERE stroke = 1 AND age>50").count()
stroke_age_greater_50 
```




    28938



#### Qual o nível médio de glicose para pessoas que, respectivamente, sofreram e não sofreram derrame?


```python
avg_glucose_level = spark.sql("SELECT stroke, AVG(avg_glucose_level) as avg_glucose FROM stroke_table GROUP BY stroke")
avg_glucose_level.show()
```

    +------+------------------+
    |stroke|       avg_glucose|
    +------+------------------+
    |     1|119.95307046938272|
    |     0|103.60273130214506|
    +------+------------------+
    
    

### Qual é o BMI (IMC = índice de massa corpórea) médio de quem sofreu e não sofreu derrame?


```python
avg_imc = spark.sql("SELECT stroke, AVG(bmi) as avg_bmi FROM stroke_table GROUP BY stroke")
avg_imc.show()
```

    +------+------------------+
    |stroke|           avg_bmi|
    +------+------------------+
    |     1|29.942490629729495|
    |     0|27.989678933253657|
    +------+------------------+
    
    

# Modelo de árvore de decisão para prevê a chance de derrame (stroke) 



```python
train_data, test_data = df.randomSplit([0.7, 0.3])
```


```python
from pyspark.ml.feature import VectorAssembler

# usar as variáveis contínuas/categóricas binárias: 
# idade, BMI, hipertensão, doença do coração, nível médio de glicose.  
numerical_cols = ['age', 'bmi', 'hypertension', 'heart_disease', 'avg_glucose_level']

assembler = VectorAssembler(inputCols=numerical_cols, outputCol='features')
```


```python
from pyspark.ml.classification import DecisionTreeClassifier

classifier = DecisionTreeClassifier(labelCol='stroke', featuresCol='features')
```


```python
from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[ assembler, classifier])
```


```python
%time predict_pipeline = pipeline.fit(train_data)
```

                                                                                    

    CPU times: user 23.3 ms, sys: 9.06 ms, total: 32.3 ms
    Wall time: 5.53 s
    


```python
predictions = predict_pipeline.transform(test_data)
predictions.select('0', 'rawPrediction', 'prediction', 'stroke').show(50)
```

    +---+---------------+----------+------+
    |  0|  rawPrediction|prediction|stroke|
    +---+---------------+----------+------+
    |  3|[5079.0,5346.0]|       1.0|     0|
    |  5|[1196.0,7837.0]|       1.0|     1|
    |  9|[5079.0,5346.0]|       1.0|     1|
    | 11|[3886.0,3336.0]|       0.0|     1|
    | 15|[3886.0,3336.0]|       0.0|     1|
    | 16|[5079.0,5346.0]|       1.0|     1|
    | 17|[3886.0,3336.0]|       0.0|     1|
    | 23|[1196.0,7837.0]|       1.0|     1|
    | 25|[3886.0,3336.0]|       0.0|     0|
    | 26| [563.0,1070.0]|       1.0|     1|
    | 34|[5079.0,5346.0]|       1.0|     1|
    | 38|[1196.0,7837.0]|       1.0|     1|
    | 43| [544.0,1466.0]|       1.0|     1|
    | 47|[2701.0,4991.0]|       1.0|     1|
    | 52|  [870.0,351.0]|       0.0|     0|
    | 55|[3886.0,3336.0]|       0.0|     0|
    | 59|[2701.0,4991.0]|       1.0|     0|
    | 66|[5079.0,5346.0]|       1.0|     1|
    | 74| [499.0,2223.0]|       1.0|     1|
    | 75| [499.0,2223.0]|       1.0|     1|
    | 81|[3886.0,3336.0]|       0.0|     0|
    | 89|[2701.0,4991.0]|       1.0|     1|
    | 90|   [2994.0,0.0]|       0.0|     0|
    | 91|[5079.0,5346.0]|       1.0|     1|
    | 93|[5079.0,5346.0]|       1.0|     1|
    |101|[1196.0,7837.0]|       1.0|     1|
    |103|[2701.0,4991.0]|       1.0|     1|
    |104| [269.0,1302.0]|       1.0|     0|
    |105|[1196.0,7837.0]|       1.0|     1|
    |110|[1196.0,7837.0]|       1.0|     1|
    |114|[3886.0,3336.0]|       0.0|     0|
    |115|[5079.0,5346.0]|       1.0|     1|
    |119|[3886.0,3336.0]|       0.0|     1|
    |127| [499.0,2223.0]|       1.0|     1|
    |131|[5079.0,5346.0]|       1.0|     0|
    |132|[2701.0,4991.0]|       1.0|     0|
    |135|[5079.0,5346.0]|       1.0|     0|
    |136|[3886.0,3336.0]|       0.0|     0|
    |137|  [870.0,351.0]|       0.0|     0|
    |139|[2701.0,4991.0]|       1.0|     0|
    |145| [499.0,2223.0]|       1.0|     0|
    |147|[5079.0,5346.0]|       1.0|     0|
    |151| [544.0,1466.0]|       1.0|     1|
    |153|[1196.0,7837.0]|       1.0|     1|
    |154|[2701.0,4991.0]|       1.0|     1|
    |164|[1196.0,7837.0]|       1.0|     1|
    |165|[5079.0,5346.0]|       1.0|     0|
    |167|[1196.0,7837.0]|       1.0|     1|
    |168|  [870.0,351.0]|       0.0|     1|
    |169|[5079.0,5346.0]|       1.0|     1|
    +---+---------------+----------+------+
    only showing top 50 rows
    
    


```python
df.show(5)
```

    +---+------+----+------------+-------------+------------+-------------+--------------+-----------------+-----+---------------+------+
    |  0|gender| age|hypertension|heart_disease|ever_married|    work_type|Residence_type|avg_glucose_level|  bmi| smoking_status|stroke|
    +---+------+----+------------+-------------+------------+-------------+--------------+-----------------+-----+---------------+------+
    |  1|Female|18.0|           0|            0|          No|      Private|         Urban|            94.19|12.12|         smokes|     1|
    |  2|  Male|58.0|           1|            0|         Yes|      Private|         Rural|           154.24| 33.7|   never_smoked|     0|
    |  3|Female|36.0|           0|            0|         Yes|     Govt_job|         Urban|            72.63| 24.7|         smokes|     0|
    |  4|Female|62.0|           0|            0|         Yes|Self-employed|         Rural|            85.52| 31.2|formerly smoked|     0|
    |  5|Female|82.0|           0|            0|         Yes|      Private|         Rural|            59.32| 33.2|         smokes|     1|
    +---+------+----+------------+-------------+------------+-------------+--------------+-----------------+-----+---------------+------+
    only showing top 5 rows
    
    

### Métricas do modelo


```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.sql import SparkSession, Row

def evaluator(predictions):
    
    # Define as métricas de avaliação
    evaluator_acc = MulticlassClassificationEvaluator(labelCol='stroke', predictionCol='prediction', metricName='accuracy')
    evaluator_precision = MulticlassClassificationEvaluator(labelCol='stroke', predictionCol='prediction', metricName='weightedPrecision')
    evaluator_recall = MulticlassClassificationEvaluator(labelCol='stroke', predictionCol='prediction', metricName='weightedRecall')
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol='stroke', predictionCol='prediction', metricName='f1')
    evaluator_auc = BinaryClassificationEvaluator(labelCol='stroke', rawPredictionCol='rawPrediction', metricName='areaUnderROC')

    # Calcula as métricas
    accuracy = evaluator_acc.evaluate(predictions)
    precision = evaluator_precision.evaluate(predictions)
    recall = evaluator_recall.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    auc = evaluator_auc.evaluate(predictions)

    metrics_data = [
        Row(Metric="Accuracy", Value=round(accuracy,4)),
        Row(Metric="Precision", Value=round(precision,4)),
        Row(Metric="Recall", Value=round(recall,4)),
        Row(Metric="F1 Score", Value=round(f1,4)),
        Row(Metric="AUC", Value=round(auc,4)),
    ]

    # Create a DataFrame from the list of rows
    metrics_df = spark.createDataFrame(metrics_data)
    
    return metrics_df
```


```python
metrics_df = evaluator(predictions)
# Mostra o DataFrame
metrics_df.show()
```

                                                                                    

    +---------+------+
    |   Metric| Value|
    +---------+------+
    | Accuracy|0.6884|
    |Precision|0.6854|
    |   Recall|0.6884|
    | F1 Score| 0.669|
    |      AUC| 0.638|
    +---------+------+
    
    

### Adicionar ao modelo as variáveis categóricas: gênero e status de fumante


```python
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder

# Define as colunas a serem tratadas 
categorical_cols = ["gender", "smoking_status"]
# Cria os StringIndexers para as colunas categóricas
string_indexers = [StringIndexer(inputCol=col, outputCol=col + '_index') for col in categorical_cols]
# Cria o OneHotEncoder para as colunas indexadas
one_hot_encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=indexer.getOutputCol() + '_OHE') for indexer in string_indexers]

```


```python
# Crie uma lista de todas as colunas codificadas
encoded_cols = [encoder.getOutputCol() for encoder in one_hot_encoders]
all_cols = numerical_cols + encoded_cols
all_cols
```




    ['age',
     'bmi',
     'hypertension',
     'heart_disease',
     'avg_glucose_level',
     'gender_index_OHE',
     'smoking_status_index_OHE']




```python
assembler = VectorAssembler(inputCols=all_cols, outputCol='features')
```


```python

# Lista de estágios do pipeline
stages = string_indexers + one_hot_encoders + [assembler, classifier]
# Criar um objeto Pipeline
pipeline = Pipeline(stages=stages)
```


```python
%time predict_pipeline = pipeline.fit(train_data)
```

                                                                                    

    CPU times: user 57.2 ms, sys: 10.5 ms, total: 67.7 ms
    Wall time: 5.32 s
    


```python
predictions = predict_pipeline.transform(test_data)
predictions.select('0', 'rawPrediction', 'prediction', 'stroke').show(50)
```

    +---+----------------+----------+------+
    |  0|   rawPrediction|prediction|stroke|
    +---+----------------+----------+------+
    |  3|[3963.0,12146.0]|       1.0|     0|
    |  5|  [168.0,5102.0]|       1.0|     1|
    |  9|[3963.0,12146.0]|       1.0|     1|
    | 11|[3963.0,12146.0]|       1.0|     1|
    | 15|[3963.0,12146.0]|       1.0|     1|
    | 16|[3963.0,12146.0]|       1.0|     1|
    | 17|[3963.0,12146.0]|       1.0|     1|
    | 23|  [168.0,5102.0]|       1.0|     1|
    | 25|  [9195.0,102.0]|       0.0|     0|
    | 26|   [111.0,178.0]|       1.0|     1|
    | 34|[3963.0,12146.0]|       1.0|     1|
    | 38|  [770.0,3316.0]|       1.0|     1|
    | 43| [1426.0,3195.0]|       1.0|     1|
    | 47| [1426.0,3195.0]|       1.0|     1|
    | 52|  [9195.0,102.0]|       0.0|     0|
    | 55|[3963.0,12146.0]|       1.0|     0|
    | 59|[3963.0,12146.0]|       1.0|     0|
    | 66|  [2463.0,576.0]|       0.0|     1|
    | 74|  [369.0,3343.0]|       1.0|     1|
    | 75|  [770.0,3316.0]|       1.0|     1|
    | 81|  [9195.0,102.0]|       0.0|     0|
    | 89| [1426.0,3195.0]|       1.0|     1|
    | 90|  [9195.0,102.0]|       0.0|     0|
    | 91|[3963.0,12146.0]|       1.0|     1|
    | 93|[3963.0,12146.0]|       1.0|     1|
    |101|  [770.0,3316.0]|       1.0|     1|
    |103| [1426.0,3195.0]|       1.0|     1|
    |104| [1426.0,3195.0]|       1.0|     0|
    |105|  [168.0,5102.0]|       1.0|     1|
    |110|  [168.0,5102.0]|       1.0|     1|
    |114|  [2463.0,576.0]|       0.0|     0|
    |115|[3963.0,12146.0]|       1.0|     1|
    |119|[3963.0,12146.0]|       1.0|     1|
    |127|  [369.0,3343.0]|       1.0|     1|
    |131|  [2463.0,576.0]|       0.0|     0|
    |132|  [9195.0,102.0]|       0.0|     0|
    |135|  [9195.0,102.0]|       0.0|     0|
    |136|  [9195.0,102.0]|       0.0|     0|
    |137|  [9195.0,102.0]|       0.0|     0|
    |139|  [9195.0,102.0]|       0.0|     0|
    |145|  [770.0,3316.0]|       1.0|     0|
    |147|[3963.0,12146.0]|       1.0|     0|
    |151| [1426.0,3195.0]|       1.0|     1|
    |153|  [770.0,3316.0]|       1.0|     1|
    |154|[3963.0,12146.0]|       1.0|     1|
    |164|  [168.0,5102.0]|       1.0|     1|
    |165|  [9195.0,102.0]|       0.0|     0|
    |167|  [168.0,5102.0]|       1.0|     1|
    |168|[3963.0,12146.0]|       1.0|     1|
    |169|[3963.0,12146.0]|       1.0|     1|
    +---+----------------+----------+------+
    only showing top 50 rows
    
    


```python
# Mostrar as métricas
metrics_df = evaluator(predictions)
metrics_df.show()
```

                                                                                    

    +---------+------+
    |   Metric| Value|
    +---------+------+
    | Accuracy|0.8401|
    |Precision|0.8557|
    |   Recall|0.8401|
    | F1 Score|0.8328|
    |      AUC|0.8043|
    +---------+------+
    
    

### Qual dessas variáveis é mais importante no modelo de árvore de decisão?


```python
model = predict_pipeline.stages[-1]

list(zip(assembler.getInputCols(), model.featureImportances))
```




    [('age', 0.1684319037330913),
     ('bmi', 0.0015314049882113054),
     ('hypertension', 0.005462801295789414),
     ('heart_disease', 0.0),
     ('avg_glucose_level', 0.007867663668840353),
     ('gender_index_OHE', 0.00022678244704680607),
     ('smoking_status_index_OHE', 0.0)]



### Qual a profundidade da árvore de decisão? 


```python
model.depth
```




    5



### Quantos nodos a árvore de decisão possui?


```python
model.numNodes
```




    23


