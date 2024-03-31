---
title:  "Tudo o que você precisa saber sobre o algoritmo Naive Bayes"
date:   2024-03-12 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, modelos, GNB]
layout: post
comments: true
---

<!-- Linking MathJax (put this in the header or somewhere at the beginning of your document) -->
<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-12-GaussianNB/header_image.jpeg?raw=true)


## Introdução

Em Machine Learning, um problema de classificação consiste em prever alguma classe ou rótulo, com base em um conjunto de dados.

Naive Bayes (NB) é um dos mais antigos algoritmos de classificação utilizados para esse fim, tendo sido desenvolvido na década de 50 do século passado. Ele tem algumas características que o tornam bastante popular: é fácil de entender, fácil de implementar e apresenta ótimos resultados, mesmo para problemas relativamente complexos.

A base matemática de sua aplicação é, como você deve supor, o teorema de Bayes, que foi proposto no século 18 pelo reverendo inglês [Thomas Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes) (1701–1761). Ele propôs esse teorema com o humilde objetivo de provar a existência de Deus (não se sabe se ele conseguiu).

No vasto universo do aprendizado de máquina, os algoritmos NB se destacam por sua simplicidade, eficiência e eficácia, especialmente em tarefas de classificação. Entre as variantes do Naive Bayes, o Modelo Gaussiano Naive Bayes (GNB) ocupa uma posição de destaque, graças à sua capacidade de trabalhar diretamente com dados contínuos, assumindo que os valores de cada característica são distribuídos segundo uma distribuição Gaussiana ou [normal](https://en.wikipedia.org/wiki/Normal_distribution). 

Este artigo explora em detalhes o funcionamento do NB, sua fundamentação matemática, características, vantagens e desvantagens. 

## Fundamentos Matemáticos do NB

<p>
A ideia básica de um algorítimo de classificação é que ele consiga, com base em conjunto de dados de treinamento \((𝑋,y)\) usado para ajustar o modelo, aprender e atribuir corretamente uma classe para novos valores de entrada. Em outras palavras, um algoritmo de classificação cria uma função matemática \(𝑦=𝑓(𝑥)\) que, ajustada pelos dados de treinamento, mapeia um certo conjunto de dados de entrada \(X = [x_1, x_2,...,x_m]\) para um outro conjunto de dados \(y = [y_1,y_2,...,y_K]\), composto por \(K\) classes distintas.

Podemos tratar esse problema de classificação probabilisticamente, avaliando a probabilidade condicional da ocorrência de uma classe \(𝑦_k\), dado o conjunto de dados \(X\). 
</p>

<p>
Matematicamente, isso pode ser escrito da seguinte forma:
\[
P(y_k|X) = P(y_k|x_1, x_2,...,x_m)
\]

Lemos "A probabilidade de ocorrência da classe \(𝑦_k\), dado o conjunto de dados \(𝑋\)".
</p>

<p> 
Tudo o que temos que fazer é calcular essa probabilidade para todas as classes em \(y\), e a classe com maior probabilidade é escolhida. 
</p>

<p> 
So easy, né? Bem...não tão depressa.
</p>

<p> 
A primeira pergunta a ser feita é: como calcular essas probabilidades? Como obter o resultado dessa equação para todas as classes em \(y\)? É aqui que entra o reverendo Bayes e seu teorema quase divino.
</p>

<p> 
Aplicando o teorema de Bayes à equação acima, obtemos:
\[
P(y_k|X) = \frac{P(X|y_k)P(y_k)}{P(X)}
\]
</p>

<p> 
ou, de uma forma mais extensa:
\[
P(y_k|X) = \frac{P(x_1, x_2,...,x_m|y_k)P(y_k)}{P(x_1, x_2,...,x_m)}
\]
</p>

<p>
Nesse caso, a probabilidade de interesse, \(𝑃(𝑦_k|𝑋)\), é chamada probabilidade a Posteriori, e \(𝑃(𝑦_k)\) probabilidade a Priori. Já \(𝑃(X|𝑦_k)\) é a probabilidade de ocorrência dos dados de \(𝑋\), se a classe \(𝑦_k\) for verdadeira. Este termo é, por vezes, chamado Verossimilhança (Likelihood). E, por fim, \(𝑃(X)\) é a probabilidade dos dados de \(X\), independentemente da classe em questão, também chamado de Evidência.
</p>

<p>
Em termos gerais, \(𝑃(𝑦_k)\) pode ser calculada via a frequência relativa de cada classe no próprio conjunto de dados de treinamento. No entanto, o cálculo da probabilidade conjunta \(P(x_1, x_2,...,x_m|y_k)\) não é trivial, pois todas as variáveis possuem interdependência e, portanto, precisamos estimar as distribuições de todas as combinações possíveis. Isso requer uma quantidade de dados muito grande o que, por sua vez, aumenta o esforço computacional necessário para se efetuar o cálculo do teorema de Bayes diretamente.
</p>

### Simplificação do Teorema de Bayes

<p>
Bem, como já deu pra notar, a vida é séria e a guerra é dura. Precisamos simplificar as coisas para tornar esse cálculo viável. Uma simplificação extrema é simplesmente considerar que todos os componentes de \(𝑋\) são independentes. Por exemplo, suponha que seu banco de dados (\(X\)) possua medidas diárias de temperatura, velocidade do vento e umidade relativa do ar, e seu interesse é prever se irá ou não chover em um determinado dia. Para aplicar o teorema de Bayes nesse caso, teríamos que supor que essas variáveis são absolutamente independentes uma das outras.
</p>

<p>
Nesse momento você salta da sua confortável cadeira e branda revoltado "mas considerar a independência total entre todas as variáveis de \(𝑋\) é uma suposição bastante forte, tola pra ser mais preciso!". Sim, de fato é verdade, essa consideração é bastante ingênua, já que na prática não se pode esperar algo tão perfeito assim, principalmente em problemas complexos; e é justamente dai que vem o nome do método. No entanto, é surpreendente como o danado funciona bem quando a problemas reais.
</p>

<p>
Com essa consideração, temos então o cálculo de uma probabilidade condicional com variáveis independentes, cujo numerador é composto pelas probabilidades condicionais de cada elemento de \(𝑋\) dado um elemento de \(y_k\), assim:

\[
P(y_k|X) = \frac{P(x_1|y_k)P(x_2|y_k)...P(x_m|y_k)P(y_k)}{P(x_1)P(x_2)...P(x_m)}
\]

ou, de uma forma mais chique:
\[
P(y_k|X) = \dfrac{P(y_k) \prod_{i=1}^{m}P(x_i|y_k)}{P(x_1)P(x_2)...P(x_m)}
\]

O denominador da expressão acima é constante para o cálculo das probabilidades condicionais de todas as \(K\) classes em \(y\). Logo, por uma questão de economia computacional, pode-se omitir essa parcela dos cálculos.
</p>

<p>
Nesse caso, dizemos que \(𝑃(𝑦_k|𝑋)\) é proporcional a \(𝑃(x_1|𝑦_k)𝑃(x_2|𝑦_k)…𝑃(x_m|𝑦_k)P(y_k)\). 

Matematicamente, escrevemos:

\[
P(y_k|X) \propto P(x_1|y_k)P(x_2|y_k)...P(x_m|y_k)P(y_k)
\]

ou, como antes:
\[
P(y_k|X) \propto P(y_k)\prod_{i=1}^{m}P(x_i|y_k)
\]

Pronto, com isso nós temos nosso classificador probabilístico Bayesiano hiper master plus+.
</p>

### Máxima Probabilidade a Posteriori

<p>
Como já comentamos, essa probabilidade condicional (probabilidade a posteriori) é realizada para todas as \(K\) classes do nosso conjunto de dados e a classe com maior probabilidade é então selecionada. Tecnicamente, esse procedimento é chamado de máxima probabilidade a posteriori, ou MAP, na sigla em inglês.
</p>

<p>
Sendo assim, a classe predita pelo modelo será dada por:
\[
\widehat{y} = \arg\max_{y_k} P(y_k|X) = \arg\max_{y_k} P(y_k)\prod_{i=1}^{m}P(x_i|y_k)
\]

Em muitas ocasiões, principalmente quando temos uma grande quantidade de dados, é conveniente expressarmos as probabilidades da equação anterior na forma de logaritmo.

\[
\widehat{y} = \arg\max_{y_k} [ln(P(y_k)\prod_{i=1}^{m}P(x_i|y_k))]
\]

Como o logaritmo do produto é igual à soma dos logaritmos, ficamos com:

\[
\widehat{y} = \arg\max_{y_k} [ln (P(y_k)) + \sum_{i=1}^{m}ln(P(x_i|y_k))]
\]

Reparem que trocamos o produto de probabilidades por somas de probabilidades, o que é computacionalmente mais eficiente.
</p>

<p>
Então, em resumo, tudo o que precisamos calcular para treinar o modelo são as probabilidade envolvidas na equação acima. Não há coeficientes que precisem ser ajustados via alguma algoritmo de otimização, como é comum em outros algoritmos de Machine Learning. Consequência? Temos um algoritmo de aprendizagem rápida e fácil implementação.
</p>

## Tipos de algoritmos NB

<p>
Para aplicar o algoritmo NB em problemas reais, precisamos estimar uma distribuição de probabilidades para os recursos de \(X\). Essa estimativa pode ser guiada pelo tipo de dado que compõe o recurso. Se as variável são categóricas binárias, usa-se uma distribuição de bernoulli para representa-las. Caso haja multiclasses, usa-se uma distribuição multinomial. E, por fim, se estivermos lidando com dados contínuos, usa-se uma distribuição Gaussiana.
</p>
<p>
E disto surgem os três tipos de classificadores Naive Bayes:
<ul>
    <li>Bernoulli naive Bayes (BNB).</li>
    <li>Multinomial naive Bayes (MNB).</li>
    <li>Gaussian naive Bayes (GNB).</li>
</ul>

É importante frisar que, caso \(X\) possua recursos com diferentes tipos de dados, pode-se atribuir diferentes distribuições de probabilidades para cada um deles. 
</p>

<p>
Outro ponto importante é que, apesar de serem as distribuições mais comumente usadas em um classificador Naive Bayes, se os dados de entrada são melhores descritos por outra distribuição de probabilidade, deve-se usá-la. Ou ainda, se não temos certeza sobre a distribuição de probabilidade que melhor descreve esses dados, pode-se usar um estimador de distribuição de probabilidades, também chamado, KDE (Kernel density estimation).
</p>

## Gaussian Naive Bayes

<p>
Como discutido anteriormente, a depender da natureza dos recursos de \(X\), pode-se assumir que estes sigam determinada distribuição de probabilidade. Quando se trata de dados contínuos, na maioria das vezes, assume-se que estes seguem uma distribuição de probabilidade Gaussiana. Nesse caso, as probabilidades da  Verossimilhança (Likelihood) pode ser obtida pela equação abaixo:

\[
P(x_i|y_k) = \dfrac{1}{\sqrt(2 \pi \sigma_k^2)} e^{-\dfrac{(x_i - \mu_k)^2}{2 \sigma_k^2}}
\]

E, desse modo, os únicos parâmetros que precisamos estimar, a partir dos dados de treinamento, são a média \(\mu_k\) e o desvio padrão \(\sigma_k\) de cada classe \(y_k\).
</p>

<p>
Simples, né?
</p>

### Vantagens do GNB

<ul>
    <li>Eficiente com grandes dimensões de dados.</li>
    <li>Boa performance com um pequeno conjunto de dados.</li>
    <li>Trata bem dados contínuos.</li>
</ul>

### Desvantagens do GNB

<ul>
    <li>Suposição de independência entre os atributos.</li>
    <li>Sensibilidade a dados não representativos.</li>
</ul>

## Aplicação prática GNB

Para ilustrar a aplicação do Modelo Gaussiano Naive Bayes, vamos implementá-lo "from scratch" e também utilizar a biblioteca scikit-learn para classificar o conjunto de dados [Iris](https://archive.ics.uci.edu/dataset/53/iris).

### Implementação *"from Scratch"*
Primeiro, definiremos uma classe GaussianNBFromScratch que calculará as médias e variâncias para cada classe e característica, e usará essas estatísticas para fazer previsões baseadas na formulação matemática apresentada anteriormente.


```python
import numpy as np

class GaussianNBFromScratch:
    """
    Implementação simplificada do Gaussian Naive Bayes para classificação.

    Atributos:
        classes (np.array): Array único das classes no conjunto de dados.
        parameters (dict): Dicionário contendo parâmetros (média, variância e priori) 
                            para cada classe.
    """
    
    def fit(self, X, y):
        """
        Treina o modelo Gaussian Naive Bayes com os dados fornecidos.

        Parâmetros:
            X (np.array): Conjunto de dados de características, onde cada linha 
                          representa uma amostra e cada coluna uma característica.
            y (np.array): Vetor de rótulos de classe para cada amostra no conjunto 
                          de dados X.
        """
        
        self.classes = np.unique(y)  # Identifica e armazena as classes únicas no vetor de rótulos
        self.parameters = {}  # Inicializa o dicionário para armazenar os parâmetros para cada classe
        
        # Calcula e armazena os parâmetros para cada classe
        for c in self.classes:
            X_c = X[y == c]  # Seleciona as amostras pertencentes à classe c
            self.parameters[c] = {
                'mean': X_c.mean(axis=0),  # Calcula a média para cada característica
                'var': X_c.var(axis=0),  # Calcula a variância para cada característica
                'prior': X_c.shape[0] / X.shape[0]  # Calcula a probabilidade a priori da classe
            }
    
    def predict(self, X):
        """
        Realiza a classificação das amostras fornecidas.

        Parâmetros:
            X (np.array): Conjunto de dados de características a serem classificadas.

        Retorna:
            np.array: Um vetor de rótulos de classe previstos para cada amostra.
        """
        y_pred = [self._predict(x) for x in X]  # Usa a função auxiliar _predict para cada amostra
        return np.array(y_pred)  # Retorna as previsões como um array numpy
    
    def _predict(self, x):
        """
        Auxiliar que calcula a classe mais provável para uma única amostra.

        Parâmetros:
            x (np.array): Uma amostra única a ser classificada.

        Retorna:
            int: A classe prevista para a amostra.
        """
        posteriors = []  # Lista para armazenar as probabilidades posteriores para cada classe
        
        # Calcula a probabilidade posterior para cada classe
        for c, params in self.parameters.items():
            prior = np.log(params['prior'])  # Log da probabilidade a priori da classe
            conditional = np.sum(np.log(self._pdf(x, params['mean'], params['var'])))  # Log da probabilidade condicional
            posterior = prior + conditional  # Log da probabilidade posterior
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]  # Retorna a classe com a maior probabilidade posterior
    
    def _pdf(self, x, mean, var):
        """
        Calcula a função densidade de probabilidade (PDF) gaussiana.

        Parâmetros:
            x (float): O valor da característica a ser avaliada.
            mean (float): A média da característica para a classe.
            var (float): A variância da característica para a classe.

        Retorna:
            float: O valor da PDF gaussiana para x.
        """
        return (1. / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))

```

### Banco de dados Iris


A base de dados [Iris](https://archive.ics.uci.edu/dataset/53/iris) é um dos conjuntos de dados mais icônicos e amplamente utilizados no campo do aprendizado de máquina e estatística. Introduzida pelo estatístico britânico Ronald Fisher em 1936, ela contém 150 amostras de três espécies diferentes de flores de íris (Iris setosa, Iris virginica e Iris versicolor), cada uma com 50 amostras. Para cada amostra, são medidas e registradas quatro características: o comprimento e a largura das sépalas e das pétalas.


```python
from sklearn.datasets import load_iris
import pandas as pd

# Carregar os dados
data = load_iris()

# Criar um dataframe pandas
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target

# Exibir as primeiras linhas do DataFrame para verificação
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
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import seaborn as sns
import matplotlib.pyplot as plt

# Configurando o tema do Seaborn
sns.set_theme(style="whitegrid")

# Criando um pairplot com o dataset Iris
sns.pairplot(df, hue="target", diag_kind="kde", markers=["o", "s", "D"], palette="bright")

plt.show()
```


    
![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-12-GaussianNB/output_5_0.png?raw=true)    


### Dividir os dados em treino e teste


```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)  # Isso remove a coluna 'target', deixando apenas as características
y = df['target']  # Isso seleciona apenas a coluna 'target' como o alvo

# Divide dados em treinamento e teste, como um proporção de 70 e 30%, respectivamente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

### Modelo *"from scratch"*


```python
# Criar instância do modelo 
model_from_scratch = GaussianNBFromScratch()
# Treinar o modelo
model_from_scratch.fit(X_train, y_train)
```

### Modelo Scikit-learn 


```python
from sklearn.naive_bayes import GaussianNB

# Inicializar e treinar o modelo
model_sklearn = GaussianNB()
model_sklearn.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GaussianNB()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;GaussianNB<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.naive_bayes.GaussianNB.html">?<span>Documentation for GaussianNB</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>GaussianNB()</pre></div> </div></div></div></div>



### Avaliar os modelos 


```python
from sklearn.metrics import accuracy_score

# Fazer previsões e avaliar os modelos
y_pred_from_scratch = model_sklearn.predict(X_test)
accuracy_from_scratch = accuracy_score(y_test, y_pred_from_scratch)

y_pred_sklearn = model_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"Acurácia do nosso modelo from scratch: {accuracy_from_scratch:.2f}")
print(f"Acurácia do modelo sklearn: {accuracy_sklearn:.2f}")
```

    Acurácia do nosso modelo from scratch: 0.98
    Acurácia do modelo sklearn: 0.98
    
**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/gaussian-naive-bayes)