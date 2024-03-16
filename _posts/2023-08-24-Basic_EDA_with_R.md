---
layout: post
comments: true
title:  "Análise exploratória de dados com R"
date:   2023-08-24 12:00:00 -500
categories: [Blog, Notebooks]
tags: [EDA, blog, ciência de dados, análise de dados, R]
---


A Análise Exploratória de Dados (EDA) é uma etapa fundamental no processo de compreensão e extração de insights de conjuntos de dados. Esse notebook realiza uma EDA básica aplicada a dados fictícios da venda de café em um supermercado. Este é um exemplo destinado a usuários iniciantes e tem fins puramente educativos. 

O objetivo é que, ao final desta experiência, você tenha uma compreensão básica e sólida de como iniciar uma análise de dados. Vamos explorar estatísticas descritivas, gráficos informativos e até mesmo criar algumas variáveis adicionais para ganhar insights mais profundos.

O código começa por oferecer uma visão geral dos dados, fornecendo estatísticas descritivas das variáveis presentes no conjunto de dados. Em seguida, prossegue para calcular os desvios padrão das medidas de vendas e preços, proporcionando uma compreensão da dispersão dos valores em torno das médias.

A partir daí, o código mergulha na visualização dos dados. Histogramas são criados para ilustrar a distribuição dos preços do café, oferecendo uma representação gráfica da frequência com que os diferentes intervalos de preço ocorrem. Além disso, gráficos de dispersão são utilizados para investigar a relação entre as vendas e os preços do café, incorporando elementos visuais como cores para destacar os dias de promoção.

O código também introduz a criação de variáveis derivadas, como aquela que indica se as vendas em determinado dia estão acima ou abaixo da média histórica. Essa variável adicional é usada para construir gráficos de barras e tabelas de contagem, facilitando a análise da distribuição das vendas em relação a essa média.

Por fim, o código explora a representação de dados através de boxplots, permitindo identificar padrões de dispersão, mediana e possíveis outliers nas vendas e preços do café. Além disso, realiza comparações entre vendas com e sem promoção, proporcionando uma visão comparativa das distribuições desses dados.

No conjunto, esse código de Análise Exploratória de Dados oferece uma jornada informativa e visualmente rica, demonstrando como a combinação de estatísticas e gráficos pode fornecer insights valiosos para tomadas de decisões informadas.

**Nota:** todo o código está disponível no [Github](https://github.com/gallileugenesis/basic-EDA-with-R/tree/main)

```R
# Cria o data frame contendo o histórico de vendas do cafe
dados <- data.frame(Vendas_Cafe = c(18, 20, 23, 23, 23, 23, 24, 25, 26, 26, 26, 26, 27, 28, 28,
                                    29, 29, 30, 30, 31, 31, 33, 34, 35, 38, 39, 41, 44, 44, 46),
                    Preco_Cafe = c(4.77, 4.67, 4.75, 4.74, 4.63, 4.56, 4.59, 4.75, 4.75, 4.49,
                                   4.41, 4.32, 4.68, 4.66, 4.42, 4.71, 4.66, 4.46, 4.36, 4.47, 4.43,
                                   4.4, 4.61, 4.09, 3.73, 3.89, 4.35, 3.84, 3.81, 3.79),
                    Promocao = c("Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Nao", "Sim",
                                 "Nao", "Sim", "Nao", "Nao", "Sim", "Sim", "Nao", "Sim", "Sim",
                                 "Sim", "Nao", "Nao", "Sim", "Sim", "Sim", "Nao", "Sim", "Sim",
                                 "Sim", "Sim", "Sim") )
```


```R
# Visualizar as 5 primeiras linhas
head(dados, 5)
```


<table>
<thead><tr><th scope=col>Vendas_Cafe</th><th scope=col>Preco_Cafe</th><th scope=col>Promocao</th></tr></thead>
<tbody>
	<tr><td>18  </td><td>4.77</td><td>Nao </td></tr>
	<tr><td>20  </td><td>4.67</td><td>Nao </td></tr>
	<tr><td>23  </td><td>4.75</td><td>Nao </td></tr>
	<tr><td>23  </td><td>4.74</td><td>Nao </td></tr>
	<tr><td>23  </td><td>4.63</td><td>Nao </td></tr>
</tbody>
</table>




```R
# Obtendo a quantidade de linhas
quantidade_linhas <- nrow(dados)

# Obtendo a quantidade de colunas
quantidade_colunas <- ncol(dados)

# Imprimindo os resultados
cat("Quantidade de linhas:", quantidade_linhas, "\n")
cat("Quantidade de colunas:", quantidade_colunas, "\n")

```

    Quantidade de linhas: 30 
    Quantidade de colunas: 3 
    

A seguir são mostradas algumas estatísticas descritivas do conjunto de dados. 

**Vendas_Cafe:** 
- A venda mínima de café registrada foi 18 unidades, enquanto a venda máxima foi de 46 unidades.
- A mediana (valor do meio quando os dados estão ordenados) das vendas é 28.50, indicando que metade das observações estão abaixo desse valor e metade acima.
- A média das vendas é 30.00, sugerindo que, em média, as vendas estão ligeiramente acima da mediana.
- O desvio padrão das vendas é aproximadamente 7.31, indicando a dispersão dos valores em relação à média.
- O primeiro quartil (25% dos dados) está em 25.25 e o terceiro quartil (75% dos dados) está em 33.75, o que indica a distribuição das vendas no conjunto de dados.

**Preco_Cafe:**
- O menor preço de café registrado foi 3.730 e o preço máximo foi 4.770.
- A mediana dos preços é 4.480, enquanto a média é 4.426. Isso sugere que a distribuição dos preços é relativamente simétrica, com a média próxima da mediana.
- O desvio padrão dos preços é aproximadamente 0.322 (30 centavos)
- O primeiro quartil está em 4.353 e o terceiro quartil em 4.668, indicando a distribuição dos preços.

**Promocao:**
- Há um total de 15 registros em que "Promocao" é "Nao" e 15 registros em que é "Sim". Isso sugere um equilíbrio entre as observações de promoção e não promoção.


```R
#visualiza a media (mean) e outras estatisticas descritivas das variaveis
summary(dados)
```


      Vendas_Cafe      Preco_Cafe    Promocao
     Min.   :18.00   Min.   :3.730   Nao:15  
     1st Qu.:25.25   1st Qu.:4.353   Sim:15  
     Median :28.50   Median :4.480           
     Mean   :30.00   Mean   :4.426           
     3rd Qu.:33.75   3rd Qu.:4.668           
     Max.   :46.00   Max.   :4.770           



```R
#Visualiza desvio padrao (standard deviation) das variaveis
sd(dados$Vendas_Cafe)
sd(dados$Preco_Cafe)
sd(dados$Preco_Leite)
```


7.31083277486696



0.322056788024478



0.255808162676553


O gráfico de histograma abaixo sugere que a distribuição dos dados do preço do café não é simétrica em relação à média, sendo observada uma assimetria negativa, com a maior parte das observações estando entre 4.4 e 4.8


```R
#Visualiza atraves de um histograma a distribuicao da variavel Preco_Cafe

# Define o tamanho das figuras para o Jupyter Notebook
options(repr.plot.width = 6, repr.plot.height = 6)

# Cria o histograma
hist(dados$Preco_Cafe)

```


<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_8_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
     



```R
# Customizando o histograma
options(repr.plot.width = 6, repr.plot.height = 6)
hist(dados$Preco_Cafe,
     col = 'blue',
     main = 'Distribuicao dos Preços Praticados para o Café')
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_9_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    


Podemos observar mais de uma variável na mesma figura, isso facilita a análise. Abaixo temos a distribuição das vendas e dos preços. Às vezes parecem ter uma distribuição mais simétrica, com uma leve assimetria positiva


```R
#Visualiza o histograma das tres variaveis numericas na mesma pagina
#Configura layout para posicionar os graficos em duas linhas e duas colunas
options(repr.plot.width = 8, repr.plot.height = 4)
par(mfrow=c(1,2)) 

hist(dados$Vendas_Cafe,
     col = 'blue',
     main = 'Distribuicao das Vendas do Café')
hist(dados$Preco_Cafe,
     col = 'blue',
     main = 'Distribuicao dos Preços do Café')
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_11_0.png?raw=true" alt="Python Logo" width="900" height="600">
</div> 
    



```R
#limpa os graficos e volta o layout para configuracao normal
dev.off() 
```


<strong>null device:</strong> 1


O gráfico de dispersão a seguir indica que a relação entre o preço e as vendas têm boa correlação negativa, ou seja, na medida em que os preços aumentam, as vendas tendem a diminuir.


```R
#Visualiza relacao entre as vendas do café o preço do café
options(repr.plot.width = 6, repr.plot.height = 6)
plot(y = dados$Vendas_Cafe,
     x = dados$Preco_Cafe)
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_14_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    



```R
#Customiza o grafico
options(repr.plot.width = 6, repr.plot.height = 6)
plot(y = dados$Vendas_Cafe,
     x = dados$Preco_Cafe,
     pch = 16,
     col = 'blue',
     xlab = 'Preço',
     ylab = 'Quantidade Vendidade',
     main = 'Relação entre o Preço e as Vendas do Café')

#este comando adiciona linhas de grade ao grafico
grid() 
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_15_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    


Adicionando as informações sobre a promoção, vemos que as vendas tendem a diminuir mesmo com a promoção, se os preços não estiverem de fato abaixo da média. 


```R
#Colore os pontos em que havia promoção naquele dia
options(repr.plot.width = 6, repr.plot.height = 6)
plot(y = dados$Vendas_Cafe,
     x = dados$Preco_Cafe,
     col = factor(dados$Promocao),
     pch = 16,
     xlab = 'Preço',
     ylab = 'Quantidade Vendidade',
     main = 'Relação entre o Preço e as Vendas do Café')


#adiciona legenda
legend(x=4.4,y=45,
       c("Promoção","Sem_Promoção"),
       col=c("red","black"),
       pch=c(16,16))
grid()
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_17_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    


O gráfico de barras a seguir mostra que a quantidade de unidades vendidas foi maior quando os preços estavam abaixo da média.


```R
#Cria uma nova variavel informando se naquele dia vendeu acima ou abaixo da media historica
media <- mean(dados$Vendas_Cafe) #armazena a media em uma variavel
variavel <- ifelse(dados$Vendas_Cafe > media,
                   'Acima_da_media',
                   'Abaixo_da_media')

variavel <- factor(variavel) #converte nova variavel para factor
```


```R
options(repr.plot.width = 6, repr.plot.height = 6)
plot(variavel) #grafico com a qtde abaixo e acima da media
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_20_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    



```R
table(variavel) #visualiza a qtde abaixo e acima da media
```


    variavel
    Abaixo_da_media  Acima_da_media 
                 19              11 


O gráfico de box-plot abaixo a distribuição dos dados de venda, não indicando a presença de nenhum outlier.


```R
#Gera boxplot das vendas
options(repr.plot.width = 6, repr.plot.height = 6)
boxplot(dados$Vendas_Cafe)
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_23_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    


Já os dados de preço mostram alguns valores discrepantes no limite inferior, possivelmente de promoções.


```R
# Gera boxplot do preco
options(repr.plot.width = 6, repr.plot.height = 6)
boxplot(dados$Preco_Cafe)
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_25_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    


Se analisarmos os box-plots das vendas para dias com e sem promoção, notamos que a mediana das vendas dos dias com promoção é maior do que as dos dias sem. No entanto, há maior variação nesses casos. Então, para concluirmos com maior segurança essa associação, é interessante a realização de testes de hipóteses para confirmar se essas diferenças são estatisticamente significativas.


```R
#Gera boxplot comparativo das vendas quando houve promocao e de quando nao houve
options(repr.plot.width = 6, repr.plot.height = 6)
boxplot(dados$Vendas_Cafe~dados$Promocao)
```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_27_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    



```R
#Customizando o boxplot
options(repr.plot.width = 6, repr.plot.height = 6)
boxplot(dados$Vendas_Cafe~dados$Promocao,
        col = 'gray',
        pch = 16,
        xlab = 'Promoção',
        ylab = 'Vendas',
        main = 'Vendas com promoção vs Vendas sem promoção')

```


    
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/basic-EDA-with-R/blob/main/Figures/output_28_0.png?raw=true" alt="Python Logo" width="600" height="600">
</div> 
    

**Nota:** todo o código está disponível no [Github](https://github.com/gallileugenesis/basic-EDA-with-R/tree/main)