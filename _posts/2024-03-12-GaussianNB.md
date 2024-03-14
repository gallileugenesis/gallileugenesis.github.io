---
title:  "Gaussiano Naive Bayes"
date:   2024-03-12 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, model, GNB]
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-12-GaussianNB/header_image.png?raw=true)

## Introdução

Em Machine Learning, um problema de classificação consiste em prever alguma classe ou rótulo, com base em um conjunto de dados.

Naive Bayes (NB) é um dos mais antigos algoritmos de classificação utilizados para esse fim, tendo sido desenvolvido na década de 50 do século passado. Ele tem algumas características que o tornam bastante popular: é fácil de entender, fácil de implementar e apresenta ótimos resultados, mesmo para problemas relativamente complexos.

A base matemática de sua aplicação é, como você deve supor, o teorema de Bayes, que foi proposto no século 18 pelo reverendo inglês [Thomas Bayes](https://en.wikipedia.org/wiki/Thomas_Bayes) (1701–1761). Ele propôs esse teorema com o humilde objetivo de provar a existência de Deus (não se sabe se ele conseguiu).

No vasto universo do aprendizado de máquina, os algoritmos Naive Bayes se destacam por sua simplicidade, eficiência e eficácia, especialmente em tarefas de classificação. Entre as variantes do Naive Bayes, o Modelo Gaussiano Naive Bayes (GNB) ocupa uma posição de destaque, graças à sua capacidade de trabalhar diretamente com dados contínuos, assumindo que os valores de cada característica são distribuídos segundo uma distribuição Gaussiana ou [normal](https://en.wikipedia.org/wiki/Normal_distribution). 

Este artigo explora em detalhes o funcionamento do GNB, sua fundamentação matemática, características, vantagens e desvantagens. No final, faremos uma aplicação prática desse algoritmo.

## Fundamentos Matemáticos do GNB

A ideia básica de um algorítimo de classificação é que ele consiga, com base em conjunto de dados de treinamento ***(𝑋,y)*** usado para ajustar o modelo, aprender e atribuir corretamente uma classe para novos valores de entrada (valores inda não observados pelo modelo). Em outras palavras, um algoritmo de classificação cria uma função matemática ***(𝑦=𝑓(𝑥))*** que, ajustada pelos dados de treinamento, mapeia um certo conjunto de dados ***X*** (dados de entrada) para um outro conjunto de dados ***y*** (classes).

Podemos tratar esse problema de classificação probabilisticamente, avaliando a probabilidade condicional da ocorrência de uma classe ***𝑦k***, dado o conjunto de dados ***𝑋***. Matematicamente, isso pode ser escrito da seguinte forma:

<!-- Inline Equation Example -->
<p>To display an inline equation, like \(E=mc^2\), you can write it directly into your paragraph.
</p>


```
$P(y_k|X) = P(y_k|X_1, X_2,...,X_m)$
```


O GNB baseia-se no Teorema de Bayes para prever a classe de uma observação. Assume-se que os valores dos atributos seguem uma distribuição gaussiana. A probabilidade de uma característica, dado que pertence a uma classe específica, é modelada pela distribuição Gaussiana:

```
P(x_i|y) = (1 / sqrt(2 * pi * sigma_y^2)) * exp(-((x_i - mu_y)^2 / (2 * sigma_y^2)))
```

Aqui, `mu_y` é a média dos valores de um atributo para a classe `y`, e `sigma_y^2` é a variância.

## Implementação e Comparação Prática

Implementamos o GNB 'from scratch' e comparamos seu desempenho com a versão do scikit-learn usando o dataset Iris. Ambas as implementações alcançaram uma acurácia de 100% na classificação do conjunto de teste, destacando a eficácia do GNB mesmo em sua forma mais simples.

## Vantagens do GNB

- **Eficiente com grandes dimensões de dados**
- **Boa performance com um pequeno conjunto de dados**
- **Trata bem dados contínuos**

## Desvantagens do GNB

- **Suposição de independência entre os atributos**
- **Sensibilidade a dados não representativos**

O GNB se destaca pela sua capacidade de fornecer resultados rápidos e precisos, mesmo com suas suposições simplificadoras. Sua aplicação vai desde a classificação de textos até o reconhecimento de padrões em dados biológicos, provando ser uma ferramenta valiosa no arsenal de qualquer cientista de dados.
