---
title:  "Gaussiano Naive Bayes (GNB)"
date:   2024-03-12 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, GNB]
---


No vasto universo do aprendizado de máquina, os algoritmos Naive Bayes se destacam por sua simplicidade, eficiência e eficácia, especialmente em tarefas de classificação. Entre as variantes do Naive Bayes, o Modelo Gaussiano Naive Bayes (GNB) ocupa uma posição de destaque, graças à sua capacidade de trabalhar diretamente com dados contínuos, assumindo que os valores de cada característica são distribuídos segundo uma distribuição Gaussiana (normal). Este artigo explora em detalhes o funcionamento do GNB, sua concepção matemática, características, vantagens e desvantagens, culminando em uma aplicação prática utilizando tanto a implementação "from scratch" quanto a disponível na biblioteca scikit-learn.

No mundo do aprendizado de máquina, os modelos Naive Bayes são conhecidos por sua simplicidade e eficiência, sendo o Modelo Gaussiano Naive Bayes (GNB) uma variante popular quando se trata de dados contínuos. Neste artigo, vamos mergulhar na mecânica, matemática, vantagens e desvantagens do GNB, culminando com uma aplicação prática.

## Fundamentos Matemáticos do GNB

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
