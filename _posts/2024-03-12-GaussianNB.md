---
title:  "Tudo o que você precisa saber sobre o algoritmo Naive Bayes"
date:   2024-03-12 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, models, GNB]
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

Este artigo explora em detalhes o funcionamento do GNB, sua fundamentação matemática, características, vantagens e desvantagens. No final, faremos uma aplicação prática com a construção do zero em comparação com o modelo da biblioteca scikit-learn.

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
Nesse caso, a probabilidade de interesse, \(𝑃(𝑦_k|𝑋)\), é chamada probabilidade a Posteriori, e \(𝑃(𝑦_k)\) probabilidade a Priori. Já \(𝑃(X|𝑦_k)\) é a probabilidade de ocorrência dos dados de \(𝑋\), se a classe \(𝑦_k\) for verdadeira. Este termo é, por vezes, chamado Verossimilhança (Likelihood). E, por fim, \(𝑃(X)\) é a probabilidade dos dados de \(X\), independentemente da classe em questão, também chamado de *Evidência*.
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
\widehat{y} = \arg\max_{y_k, k \in 1,2,...,K} P(y_k|X) = \arg\max_{y_k} P(y_k)\prod_{i=1}^{m}P(x_i|y_k)
\]

Em muitas ocasiões, principalmente quando temos uma grande quantidade de dados, é conveniente expressarmos as probabilidades da equação anterior na forma de logaritmo.

\[
\widehat{y} = \arg\max_{y_k} [ln(P(y_k)\prod_{i=1}^{m}P(x_i|y_k))]
\]

Como o logaritmo do produto é igual à soma dos logaritmos, ficamos com:

\[
\widehat{y} = \arg\max_{y_k} [ln P(y_k) + \prod_{i=1}^{m}lnP(x_i|y_k)]
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

Como discutido anteriormente, a depender da natureza dos recursos de \(X\), pode-se assumir que estes sigam determinada distribuição de probabilidade. Quando se trata de dados contínuos, na maioria das vezes, assume-se que estes seguem uma distribuição de probabilidade Gaussiana. Nesse caso, as probabilidades na parcela do somatório na Equação 1, pode ser obtida pela Equação 2 abaixo:

\[
P(x_i|y) = (1 / sqrt(2 * pi * sigma_y^2)) * exp(-((x_i - mu_y)^2 / (2 * sigma_y^2)))
\]

Aqui, `mu_y` é a média dos valores de um atributo para a classe `y`, e `sigma_y^2` é a variância.


## Vantagens do GNB

- **Eficiente com grandes dimensões de dados**
- **Boa performance com um pequeno conjunto de dados**
- **Trata bem dados contínuos**

## Desvantagens do GNB

- **Suposição de independência entre os atributos**
- **Sensibilidade a dados não representativos**

