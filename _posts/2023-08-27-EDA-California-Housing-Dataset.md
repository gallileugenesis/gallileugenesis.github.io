---
title:  "EDA - California Housing Dataset"
date:   2023-08-27 12:00:00 -500
categories: [Blog, Notebooks, Kaggle]
tags: [EDA, competições kaggle, data science, análise de dados, python]
layout: post
comments: true
---


**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/California-Housing-Dataset)

###### Objetivos:
O objetivo da Análise Exploratória de Dados (EDA) é entender a estrutura e os relacionamentos dentro do conjunto de dados e identificar inconsistências, "sujeiras", padrões, outliers e outros recursos de interesse. É uma etapa inicial no processo de análise de dados e é uma etapa crucial antes de aplicar qualquer modelo estatístico ou de aprendizado de máquina aos dados.

Alguns objetivos específicos da EDA incluem:

- Familiarizar-se com os dados e sua estrutura
- Identificando dados ausentes ou incorretos
- Efetuar eventuais correções nos dados
- Detecção de outliers e anomalias
- Entendendo a distribuição de cada variável
- Identificando relações e padrões entre variáveis
- Gerando hipóteses para análise ou modelagem posterior

Em geral, o objetivo da EDA é usar métodos visuais e estatísticos para obter insights sobre os dados e identificar áreas para investigação posterior. É um processo iterativo que permite ao analista refinar e melhorar sua compreensão dos dados à medida que novos insights são obtidos.


## Parte I

O objetivo dessa etapa é realizar uma exploração mais visual, a fim de familiarizar-se com os dados e sua estrutura e identificar dados ausentes, duplicados ou incorretos e, por fim, realizar eventuais correções, se for o caso.


```python
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.2f}'.format)

import warnings
warnings.filterwarnings("ignore")

bold = ['\033[1m', '\033[0m']
```

### Banco de dados

O banco de dados de treinamento possui 37137 amostras (entradas/linhas), 8 colunas (features/recursos), além das colunas id e target.


```python
train = pd.read_csv("Data/train.csv")
print(train.shape)
train.head()
```

    (37137, 10)
    




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
      <th>id</th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseVal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2.39</td>
      <td>15.00</td>
      <td>3.83</td>
      <td>1.11</td>
      <td>1280.00</td>
      <td>2.49</td>
      <td>34.60</td>
      <td>-120.12</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3.72</td>
      <td>17.00</td>
      <td>6.01</td>
      <td>1.05</td>
      <td>1504.00</td>
      <td>3.81</td>
      <td>38.69</td>
      <td>-121.22</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>4.78</td>
      <td>27.00</td>
      <td>6.54</td>
      <td>1.10</td>
      <td>1061.00</td>
      <td>2.46</td>
      <td>34.71</td>
      <td>-120.45</td>
      <td>1.58</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2.41</td>
      <td>16.00</td>
      <td>3.35</td>
      <td>0.97</td>
      <td>1255.00</td>
      <td>2.09</td>
      <td>32.66</td>
      <td>-117.09</td>
      <td>1.34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>3.75</td>
      <td>52.00</td>
      <td>4.28</td>
      <td>1.07</td>
      <td>1793.00</td>
      <td>1.60</td>
      <td>37.80</td>
      <td>-122.41</td>
      <td>4.50</td>
    </tr>
  </tbody>
</table>
</div>



O banco de dados de teste possui 24759 amostras (entradas/linhas), 8 colunas (features/recursos), além da coluna id.


```python
test = pd.read_csv("Data/test.csv")
print(test.shape)
test.head()
```

    (24759, 9)
    




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
      <th>id</th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37137</td>
      <td>1.71</td>
      <td>35.00</td>
      <td>4.97</td>
      <td>1.10</td>
      <td>1318.00</td>
      <td>2.84</td>
      <td>39.75</td>
      <td>-121.85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37138</td>
      <td>1.39</td>
      <td>22.00</td>
      <td>4.19</td>
      <td>1.10</td>
      <td>2296.00</td>
      <td>3.18</td>
      <td>33.95</td>
      <td>-118.29</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37139</td>
      <td>7.72</td>
      <td>21.00</td>
      <td>7.13</td>
      <td>0.96</td>
      <td>1535.00</td>
      <td>2.89</td>
      <td>33.61</td>
      <td>-117.81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37140</td>
      <td>4.68</td>
      <td>49.00</td>
      <td>4.77</td>
      <td>1.05</td>
      <td>707.00</td>
      <td>1.74</td>
      <td>34.17</td>
      <td>-118.34</td>
    </tr>
    <tr>
      <th>4</th>
      <td>37141</td>
      <td>3.13</td>
      <td>25.00</td>
      <td>3.77</td>
      <td>1.08</td>
      <td>4716.00</td>
      <td>2.00</td>
      <td>34.17</td>
      <td>-118.29</td>
    </tr>
  </tbody>
</table>
</div>



#### Informação de dados ausentes:

Abaixo estão três métodos para analisar a existência de dados ausentes (você só precisa usar um, se quiser). Nenhum dado ausente foi encontrado (tão bom que parece mentira).


```python
# vamos obter um resumo conciso do DataFrame.
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 37137 entries, 0 to 37136
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   id           37137 non-null  int64  
     1   MedInc       37137 non-null  float64
     2   HouseAge     37137 non-null  float64
     3   AveRooms     37137 non-null  float64
     4   AveBedrms    37137 non-null  float64
     5   Population   37137 non-null  float64
     6   AveOccup     37137 non-null  float64
     7   Latitude     37137 non-null  float64
     8   Longitude    37137 non-null  float64
     9   MedHouseVal  37137 non-null  float64
    dtypes: float64(9), int64(1)
    memory usage: 2.8 MB
    


```python
# vamos obter um resumo conciso do DataFrame.
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 24759 entries, 0 to 24758
    Data columns (total 9 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   id          24759 non-null  int64  
     1   MedInc      24759 non-null  float64
     2   HouseAge    24759 non-null  float64
     3   AveRooms    24759 non-null  float64
     4   AveBedrms   24759 non-null  float64
     5   Population  24759 non-null  float64
     6   AveOccup    24759 non-null  float64
     7   Latitude    24759 non-null  float64
     8   Longitude   24759 non-null  float64
    dtypes: float64(8), int64(1)
    memory usage: 1.7 MB
    

Percebemos que a quantidade de valores não nulos é igual ao total de entradas do dataframe, o que indica que não há dados ausentes. Notamos também que todas as entradas são numéricas do tipo float64, com exceção do id que é do tipo int64, ocupando 2.8 e 1.4 MB de memória (treinamento e teste, respectivamente). 


```python
# O método a seguir mostra de forma mais direta a quantidade de dados ausente em cada coluna
train.isnull().sum()
```




    id             0
    MedInc         0
    HouseAge       0
    AveRooms       0
    AveBedrms      0
    Population     0
    AveOccup       0
    Latitude       0
    Longitude      0
    MedHouseVal    0
    dtype: int64




```python
test.isnull().sum()
```




    id            0
    MedInc        0
    HouseAge      0
    AveRooms      0
    AveBedrms     0
    Population    0
    AveOccup      0
    Latitude      0
    Longitude     0
    dtype: int64



#### Avaliar dados duplicados

Não temos dados duplicados.


```python
# Linhas inteiramente duplicadas
train_duplicated = train[train.duplicated(keep = False)]
print(train_duplicated.shape)
train_duplicated.head()
```

    (0, 10)
    




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
      <th>id</th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseVal</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
test_duplicated = test[test.duplicated(keep = False)]
print(test_duplicated.shape)
test_duplicated.head()
```

    (0, 9)
    




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
      <th>id</th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



#### Descrição estatística básica


```python
train.describe()
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
      <th>id</th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>MedHouseVal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>37137.00</td>
      <td>37137.00</td>
      <td>37137.00</td>
      <td>37137.00</td>
      <td>37137.00</td>
      <td>37137.00</td>
      <td>37137.00</td>
      <td>37137.00</td>
      <td>37137.00</td>
      <td>37137.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>18568.00</td>
      <td>3.85</td>
      <td>26.06</td>
      <td>5.16</td>
      <td>1.06</td>
      <td>1660.78</td>
      <td>2.83</td>
      <td>35.57</td>
      <td>-119.55</td>
      <td>2.08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10720.67</td>
      <td>1.80</td>
      <td>12.16</td>
      <td>1.21</td>
      <td>0.10</td>
      <td>1302.47</td>
      <td>2.70</td>
      <td>2.08</td>
      <td>1.97</td>
      <td>1.16</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>0.50</td>
      <td>2.00</td>
      <td>0.85</td>
      <td>0.50</td>
      <td>3.00</td>
      <td>0.95</td>
      <td>32.55</td>
      <td>-124.35</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9284.00</td>
      <td>2.60</td>
      <td>17.00</td>
      <td>4.36</td>
      <td>1.02</td>
      <td>952.00</td>
      <td>2.39</td>
      <td>33.93</td>
      <td>-121.80</td>
      <td>1.21</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>18568.00</td>
      <td>3.52</td>
      <td>25.00</td>
      <td>5.07</td>
      <td>1.05</td>
      <td>1383.00</td>
      <td>2.74</td>
      <td>34.19</td>
      <td>-118.45</td>
      <td>1.81</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>27852.00</td>
      <td>4.70</td>
      <td>35.00</td>
      <td>5.86</td>
      <td>1.09</td>
      <td>1856.00</td>
      <td>3.13</td>
      <td>37.70</td>
      <td>-118.02</td>
      <td>2.66</td>
    </tr>
    <tr>
      <th>max</th>
      <td>37136.00</td>
      <td>15.00</td>
      <td>52.00</td>
      <td>28.84</td>
      <td>5.87</td>
      <td>35682.00</td>
      <td>502.99</td>
      <td>41.95</td>
      <td>-114.55</td>
      <td>5.00</td>
    </tr>
  </tbody>
</table>
</div>



#### Comentários:
**MedInc (Renda Média):**
- A média da renda média é aproximadamente 3.85, com um desvio padrão de 1.80. Isso sugere que a maioria das áreas tem renda média próxima a esse valor médio.
- renda mínima é 0.50, enquanto a renda máxima é 15.00. Essa variação indica que existem áreas com renda consideravelmente baixa e alta.
- O percentil 25 (primeiro quartil) é 2.60, o percentil 50 (mediana) é 3.52 e o percentil 75 (terceiro quartil) é 4.70. Isso sugere que a maioria das áreas tem renda média entre 2.60 e 4.70.
 

**HouseAge (Idade das Casas):**
- A idade média das casas é cerca de 26.06 anos, com um desvio padrão de 12.16 anos. Isso sugere uma variabilidade nas idades das casas entre diferentes áreas.
- A idade mínima das casas é 2 anos, enquanto a idade máxima é 52 anos.
- O percentil 25 (primeiro quartil) é 17 anos, o percentil 50 (mediana) é 25 anos e o percentil 75 (terceiro quartil) é 35 anos. Isso sugere que a maioria das casas nas áreas tem idades entre 17 e 35 anos.

**AveRooms (Média de Quartos por Casa):**
- A média de quartos por casa é aproximadamente 5.16, com um desvio padrão de 1.21. Isso sugere uma relativa consistência na média de quartos entre diferentes áreas.
- O número mínimo médio de quartos por casa é 0.85, enquanto o número máximo é 28.84, que pode ser um valor atípico.
- O percentil 25 (primeiro quartil) é 4.36, o percentil 50 (mediana) é 5.07 e o percentil 75 (terceiro quartil) é 5.86. Isso indica que a maioria das casas nas áreas tem entre 4.36 e 5.86 quartos em média.

**AveBedrms (Média de Quartos de Dormir por Casa):**
- A média de quartos de dormir por casa é aproximadamente 1.06, com um desvio padrão de 0.10. Isso sugere uma consistência razoável nessa média entre diferentes áreas.
- O número mínimo médio de quartos de dormir por casa é 0.50, enquanto o número máximo é 5.87.
- O percentil 25 (primeiro quartil) é 1.02, o percentil 50 (mediana) é 1.05 e o percentil 75 (terceiro quartil) é 1.09. Isso indica que a maioria das casas nas áreas tem entre 1.02 e 1.09 quartos de dormir em média.

**Population (População):**
- A população média é cerca de 1660.78, com um desvio padrão de 1302.47. Isso sugere uma variação considerável nas populações entre diferentes áreas.
- A população mínima é 3, enquanto a população máxima é 35682.
- O percentil 25 (primeiro quartil) é 952, o percentil 50 (mediana) é 1383 e o percentil 75 (terceiro quartil) é 1856. Isso sugere que a maioria das áreas tem populações abaixo de 1856.

**AveOccup (Média de Ocupantes por Casa):**
- A média de ocupantes por casa é cerca de 2.83, com um desvio padrão de 2.70. Isso sugere uma variação razoável na média de ocupantes entre diferentes áreas.
- O número mínimo médio de ocupantes por casa é 0.95, enquanto o número máximo é 502.99, que parece ser um valor atípico.
- O percentil 25 (primeiro quartil) é 2.39, o percentil 50 (mediana) é 2.74 e o percentil 75 (terceiro quartil) é 3.13. Isso indica que a maioria das casas nas áreas tem entre 2.39 e 3.13 ocupantes em média.

**Latitude e Longitude:**
- As coordenadas de latitude têm média em torno de 35.57 e desvio padrão de 2.08. As coordenadas de longitude têm média próxima de -119.55 e desvio padrão de 1.97. Isso sugere alguma variação nas localizações das áreas.
- Os valores mínimos e máximos de latitude e longitude refletem a diversidade das áreas cobertas pelo conjunto de dados.

**MedHouseVal (Valor Médio das Casas):**
- A média do valor médio das casas é cerca de 2.08, com um desvio padrão de 1.16. Isso sugere que os valores do valor médio das casas estão em torno dessa média, com alguma variação.
- O valor mínimo do valor médio das casas é 0.15, enquanto o valor máximo é 5.00.
- O percentil 25 (primeiro quartil) é 1.21, o percentil 50 (mediana) é 1.81 e o percentil 75 (terceiro quartil) é 2.66. Isso sugere que a maioria das áreas tem valores do valor médio das casas entre 1.21 e 2.66.

## Parte II

Vamos, nessa etapa, fazer uma exploração mais aprofundada dos dados, buscando identificar padrões e insight, a partir do interrelacionamento das variáveis e suas distribuições. 



```python
def plot_histogram(df_train, df_test, numerical_col):
    """
    Plota um histograma com ou sem estimativa de densidade de kernel para variáveis numéricas dos dados de treinamento e teste
    
    Parâmetros:
    df_train (pd.DataFrame): O DataFrame contendo os dados de treinamento.
    df_test (pd.DataFrame): O DataFrame contendo os dados de teste.
    numerical_col (str): O nome da coluna numérica a ser plotada.
    
    Retorna:
    None (exibe o gráfico)
    """
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df_train, x=numerical_col, kde=True, color="skyblue", alpha = 1.0,  label='Treinamento')
    if numerical_col in df_test.columns.tolist():
        sns.histplot(data=df_test, x=numerical_col, kde=True, color="orange", alpha = 1.0, label='Teste')
    plt.legend()
    plt.title(f'Histograma de {numerical_col}')
    plt.xlabel(numerical_col)
    plt.ylabel('Frequência')
    plt.show()


def plot_boxplot(df_train, df_test, numerical_col):
    """
     Traça boxplots lado a lado para uma coluna numérica nos conjuntos de dados de treinamento e teste.
    
     Parâmetros:
     df_train (pd.DataFrame): O DataFrame que contém os dados de treinamento.
     df_test (pd.DataFrame): O DataFrame que contém os dados de teste.
     numerical_col (str): O nome da coluna numérica a ser plotada.
    
     Returns:
     None (exibe o gráfico)
     """

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=df_train, x=numerical_col)
    plt.title(f'Boxplot of {numerical_col} - Training Data')
    
    if numerical_col in df_test.columns.tolist():
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df_test, x=numerical_col)
        plt.title(f'Boxplot of {numerical_col} - Test Data')
    
    plt.tight_layout()
    plt.show()
    
def plot_violin(df_train, df_test, numerical_col):
    """
    Plots side-by-side violin plots for a numerical column in the training and test datasets.
    
    Parameters:
    df_train (pd.DataFrame): The DataFrame containing the training data.
    df_test (pd.DataFrame): The DataFrame containing the test data.
    numerical_col (str): The name of the numerical column to plot.
    
    Returns:
    None (displays the plot)
    """
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    sns.violinplot(data=df_train, x=numerical_col)
    plt.title(f'Violin Plot of {numerical_col} - Training Data')
    
    if numerical_col in df_test.columns.tolist():
        plt.subplot(1, 2, 2)
        sns.violinplot(data=df_test, x=numerical_col)
        plt.title(f'Violin Plot of {numerical_col} - Test Data')
    
    plt.tight_layout()
    plt.show()

    
def plot_pairplot(df_train, df_test):
    plt.figure(figsize=(16, 4))
    sns.pairplot(data=df_train)
    plt.title(f'Training Data')
    
    sns.pairplot(data=df_test)
    plt.title(f'Test Data')
    plt.show()

def plot_heatmap(df):
    plt.figure(figsize=(8, 4))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()
```

#### Comentários:
- Os gráficos abaixo mostram que a distribuição dos dados de treinamento e teste são muito semelhantes.
- É possível notar também o impacto de valores extremos nas distribuições.
- A localiazação das casas se concentram em torno da latitude 34 e 38 e longitude -122 e -118. A localização parece ser uma ótima candidata a gerar novos insumos para os modelos.
- Pode-se nota que a variável MedHouseVal tem um pico de dados em 5, saindo completamente da tendência que se seguia.


```python
numerical_features = train.drop(['id'], axis=1).columns.tolist()
for col in numerical_features:
    plot_histogram(train, test, col)
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_0.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_1.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_2.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_3.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_4.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_5.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_6.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_7.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_23_8.png?raw=true)



```python
for col in numerical_features:
    plot_boxplot(train, test, col)
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_0.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_1.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_2.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_3.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_4.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_5.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_6.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_7.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_24_8.png?raw=true)
    



```python
for col in numerical_features:
    plot_violin(train, test, col)
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_0.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_1.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_2.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_3.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_4.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_5.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_6.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_7.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_25_8.png?raw=true)
    


#### Comentários:
- Algumas variáveis apresentam boa correlação positiva com a variável resposta (MedHouseVal), como e MedInc e AveRooms



```python
plot_pairplot(train, test)
```


    <Figure size 1600x400 with 0 Axes>



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_27_1.png?raw=true)
    



    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_27_2.png?raw=true)
    



```python
# Correção para os dados de trainamento
plot_heatmap(train[numerical_features])
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_28_0.png?raw=true)
    



```python
# Correção para os dados de teste
plot_heatmap(test[numerical_features[:-1]])
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_29_0.png?raw=true)
    


Analisando a distribuição espacial das casas, confirmamos o que os gráficos antiriores anteciparam. Pode-se observar a formação de alguns clusters populacionais em torno de (34,-118) e (38,-122). Um comportamento semelhante pode ser observado quando analisamos a distribuição espacial das idades das casas. 


```python
plt.figure(figsize=(14, 14))
plt.title('Target distrution', size=22, y=0.97, fontname='Calibri', 
          fontweight='bold', color='#444444')
a = sns.scatterplot(data=train, x='Latitude', y='Longitude', hue='MedHouseVal', 
                    palette=sns.color_palette('YlOrBr', as_cmap=True), s=25, alpha=0.5)
plt.xticks(size=11)
plt.yticks(size=11)
plt.xlabel('Latitude', labelpad=10, fontsize=15)
plt.ylabel('Longitude', labelpad=10, fontsize=15)

for j in ['right', 'top']:
    a.spines[j].set_visible(False)
    
plt.show()
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_31_0.png?raw=true)
    



```python
plt.figure(figsize=(14, 14))
plt.title('Target distrution', size=22, y=0.97, fontname='Calibri', 
          fontweight='bold', color='#444444')
a = sns.scatterplot(data=train, x='Latitude', y='Longitude', hue='HouseAge', 
                    palette=sns.color_palette('YlOrBr', as_cmap=True), s=25, alpha=0.5)
plt.xticks(size=11)
plt.yticks(size=11)
plt.xlabel('Latitude', labelpad=10, fontsize=15)
plt.ylabel('Longitude', labelpad=10, fontsize=15)

for j in ['right', 'top']:
    a.spines[j].set_visible(False)
    
plt.show()
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-27-EDA-California-Housing-Dataset/output_32_0.png?raw=true)
    

**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/California-Housing-Dataset)