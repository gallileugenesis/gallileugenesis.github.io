---
title:  "Escalonamento de dados: normalização vs padronização"
date:   2025-01-27 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, pré-processamento, normalização, padronização]
layout: post
comments: true
---

<!-- Linking MathJax (put this in the header or somewhere at the beginning of your document) -->
<script src="https://polyfill.io/v3/polyfill.min.js?*Features*=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2025-01-27-standardization-vs-normalization/header_image.jpeg?raw=true)


*Features* com maior escala e variância tendem a dominar o aprendizado do modelo, enquanto as de menor escala e variância têm menor influência.

Modelos baseados em cálculos de distâncias, como o K-Means, KNN e SVM, tendem a ter seus resultados distorcidos pelas dimensões com maior escala e variância. Modelos lineares, como a regressão logística ou regressão linear, podem levar mais tempo para convergir ou apresentar coeficientes inadequados devido à escalas desiguais entre as variáveis.

Normalização e padronização são técnicas de pré-processamento para escalonamento de dados. Ou seja, colocam os dados em uma mesma escala, mas de formas diferentes.

**Normalização:** Reduz os valores para um intervalo fixo, geralmente [0, 1]. 
- Método **MinMaxScaler** no Scikit-learn.

<p>
Propriedades:
</p>
<ol>
    <li>Após a normalização, os valores estarão no intervalo [0,1]:
        <ul>
            <li>0 corresponde ao valor mínimo (\(\min(x)\)).</li>
            <li>1 corresponde ao valor máximo (\(\max(x)\)).</li>
        </ul>
    </li>
    <li>Preserva a distribuição relativa dos dados, mas comprime outliers para dentro do intervalo definido.</li>
</ol>

<p>
\[  
x_{\text{normalizado}} = \frac{x - \min(x)}{\max(x) - \min(x)}
\]
</p>

**Padronização:** Transforma os dados para uma distribuição com média zero e desvio padrão igual a 1. 
- Método **StandardScaler** no Scikit-learn.

Propriedades:
1. Após a padronização, a distribuição dos dados terá média igual a 0 e desvio padrão igual a 1.
2. Mantém a forma da distribuição original, mas ajusta os valores em termos de desvios padrão em relação à média.
3. Não é sensível a outliers, mas os outliers permanecem, podendo ter valores padronizados altos ou baixos.

<p>
\[
z = \frac{x - \mu}{\sigma}
\]

Onde:
- $x$ é o valor original,
- $\mu$ é a média da distribuição,
- $\sigma$ é o desvio padrão.
</p>

Abaixo temos um exemplo simples de como fazer a normalização e padronização usando a biblioteca sklearn.


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler 

# Gerando uma distribuição de dados aleatórios (100 amostras, 2 *Features*)
np.random.seed(42)
feature1 = np.random.randint(1, 10, size=(100, ))
feature2 = np.random.randint(100, 1000, size=(100, ))

data = np.array([feature1, feature2]).T 
```


```python
# Aplicando MinMaxScaler
max_min_scaler = MinMaxScaler()
normalized_data = max_min_scaler.fit_transform(data)

# Aplicando StandardScaler
standard_scaler = StandardScaler()
standardized_data = standard_scaler.fit_transform(data)
```


```python
# Plotando os dados originais, normalizados e padronizados lado a lado
plt.figure(figsize=(18, 6))

# Gráfico dos dados originais
plt.subplot(1, 3, 1)
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Original Data', s=50)
plt.title("Dados Originais", fontsize=18)
plt.xlabel("Feature 1", fontsize=14)
plt.ylabel("Feature 2", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.5)
plt.xlim(-1000,1000)

# Gráfico dos dados normalizados
plt.subplot(1, 3, 2)
plt.scatter(normalized_data[:, 0], normalized_data[:, 1], color='green', label='Normalized Data', s=50)
plt.title("Dados Normalizados", fontsize=18)
plt.xlabel("Feature 1 (Normalized)", fontsize=14)
plt.ylabel("Feature 2 (Normalized)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.5)

# Gráfico dos dados padronizados
plt.subplot(1, 3, 3)
plt.scatter(standardized_data[:, 0], standardized_data[:, 1], color='red', label='Standardized Data', s=50)
plt.title("Dados Padronizados", fontsize=18)
plt.xlabel("Feature 1 (Standardized)", fontsize=14)
plt.ylabel("Feature 2 (Standardized)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.5)

# Ajusta o layout para evitar sobreposição e salva o gráfico
plt.tight_layout()
plt.savefig("MinMaxScaler_StandardScaler.png")
plt.show()
```


![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2025-01-27-standardization-vs-normalization/output_8_0.png?raw=true)
  

A primeira figura da esquerda mostra a visualização dos dados com os eixos variando na mesma magnitude da feature de maior escala (feature 2). Claramente a visualização fica prejudicada pela diferença de escala entre as *Features*. Essa diferença também seria prejudicial para alguns modelos de machine learning, como já discutido.

No gráfico do meio temos os dados normalizados, agora ambas as *Features* variam entre 0 e 1. Isso retira o efeito da escala. Do mesmo modo, a escala já não tem impacto no gráfico da direita, onde os dados foram padronizando, passando a ter média zero e desvio padrão 1. 