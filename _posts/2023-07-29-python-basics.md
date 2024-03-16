---
title:  "Curso básico de python"
date:   2023-07-29 12:00:00 -500
categories: [Blog, Notebooks]
tags: [python, programação, curso]
layout: post
comments: true
---



<div style="text-align:center;">
  <img src="https://logos-world.net/wp-content/uploads/2021/10/Python-Logo.png"  alt="Python Logo" width="800" height="400">
</div>

# Curso básico de Python
Prof. Gallileu Genesis

### Tópicos:

#### 1 - Introdução ao Python:
- O que é Python
- Aplicações
- Ferramentas de desenvolvimento
- Instalação do Python e ambiente de desenvolvimento
- Executando o primeiro programa em Python

#### 2 - Tipos de dados e variáveis:
- Tipos de dados em Python (números, strings, booleanos)
- Declaração de variáveis
- Operações com variáveis
- Exercícios

#### 3 - Estruturas de controle de fluxo:
- Estruturas Condicionais (if/else)
- Estruturas de Repetição (for, while)
- Instruções de controle de fluxo (break, continue)
- Exercícios

#### 4 - Estruturas de dados:
- Listas
- Acesso e manipulação de elementos em Listas
- Tuplas
- Acesso e manipulação de elementos em Tuplas
- Dicionários
- Acesso e manipulação de elementos em dicionários
- Exercícios

## 1 - Introdução ao Python

### 1.1 - O que é Python:

Python é uma linguagem de programação de alto nível, interpretada, orientada a objetos e multiplataforma.

- **Linguagem de Programação de Alto Nível:** Significa que a linguagem foi projetada com um alto nível de abstração, tornando-a mais próxima da linguagem humana e permitindo que os desenvolvedores expressem suas ideias de maneira mais clara e concisa. Linguagens de alto nível geralmente têm uma sintaxe mais amigável e oferecem construções de alto nível para lidar com tarefas complexas.

- **Interpretada**: Uma linguagem interpretada é aquela em que o código-fonte é executado diretamente por um interpretador, que é um programa que lê o código linha por linha e o executa em tempo real. Isso é diferente de uma linguagem compilada, onde o código-fonte é transformado em código de máquina (ou bytecode) antes da execução.

- **Orientada a Objetos**: Refere-se a um paradigma de programação onde o código é organizado em objetos, que são entidades que encapsulam dados e comportamentos relacionados. A programação orientada a objetos (POO) permite que os desenvolvedores criem abstrações mais significativas, facilitando a reutilização de código e a manutenção do software.

- **Multiplataforma**: Significa que a linguagem pode ser executada em diferentes plataformas de hardware ou sistemas operacionais sem a necessidade de grandes modificações. Isso é alcançado através do uso de ambientes de execução (como máquinas virtuais ou interpretadores) que permitem que o código seja executado em ambientes diferentes, desde que o ambiente de execução esteja disponível para a plataforma específica.

### 1.2 - Aplicações:

Python é uma linguagem de programação versátil e amplamente utilizada em várias aplicações:

- **Inteligência Artificial e Aprendizado de Máquina:** Python é uma das principais linguagens usadas em projetos de IA e ML, com bibliotecas como *TensorFlow*, *Keras* e *PyTorch* permitindo a criação de modelos complexos de aprendizado de máquina e redes neurais.

- **Análise e ciência de dados:** Python é uma escolha comum para análise e manipulação de dados devido às bibliotecas poderosas, como *Pandas* e *NumPy*, além de ferramentas para visualização de dados, como *Matplotlib* e *Seaborn*.

- **Internet das Coisas (IoT):** Python é uma escolha popular para projetos de IoT devido à sua facilidade de integração com dispositivos e sensores.

- **Desenvolvimento de aplicações científicas:** Python é amplamente usado em aplicações científicas para resolver problemas complexos, simulação e modelagem matemática.

- **Desenvolvimento web**: Python é muito popular para criar aplicativos web, devido a frameworks como *Django* e *Flask*, que permitem o desenvolvimento rápido e eficiente de aplicações web robustas e escaláveis.

- **Automação e scripts**: Python é frequentemente usado para escrever *scripts* e automatizar tarefas repetitivas no sistema operacional, tornando-o uma excelente escolha para administração de sistemas e tarefas de rotina.

- **Desenvolvimento de jogos:** Python pode ser usado para o desenvolvimento de jogos com a ajuda de bibliotecas como *Pygame*, que fornecem funcionalidades de gráficos e interação com o usuário.

- **Desenvolvimento de aplicativos de desktop:** Python pode ser usado para desenvolver aplicativos de desktop com interfaces gráficas usando ferramentas como *Tkinter*, *PyQT ou WXPython*.

### 1.3 - Ferramentas de desenvolvimento

Temos várias opções quando se trata de escolher onde desenvolver nossos códigos Python. As opções variam de *ambientes de desenvolvimento integrado (IDEs)* a *editores de código e ambientes interativos*. Cada opção possui suas próprias vantagens e recursos, e a escolha dependerá das preferências pessoais do desenvolvedor e das necessidades específicas do projeto.

IDEs:
- **PyCharm:** desenvolvido pela JetBrains, é uma das IDEs mais poderosas e completas para o desenvolvimento Python. Disponível em uma edição Community gratuita e em uma edição Professional paga, o PyCharm oferece recursos avançados, como edição de código inteligente, depuração integrada, suporte a frameworks populares, gerenciamento de projetos e muito mais.

- **Visual Studio Code (VS Code):** desenvolvido pela Microsoft, é um editor de código leve, altamente customizável e amplamente utilizado pela comunidade Python. Com uma grande variedade de extensões disponíveis, o VS Code pode ser adaptado para atender às necessidades específicas do desenvolvedor, incluindo suporte para Python, depuração e integração com sistemas de controle de versão.

- **Spyder:** é uma IDE projetada especificamente para análise de dados e ciência de dados em Python. Ele oferece recursos como edição de código, console interativo e visualização de variáveis, tornando-o ideal para tarefas de análise e manipulação de dados.

- **IDLE:** é um ambiente de desenvolvimento padrão incluído na distribuição oficial do Python. Embora seja uma opção mais básica em comparação com as IDEs mencionadas acima, o IDLE oferece recursos de edição de código e um console interativo para testar código rapidamente.

Ambientes interativos:
- **Jupyter Notebook:** é uma aplicação web interativa que permite criar documentos interativos chamados notebooks. Amplamente utilizado na área de ciência de dados e análise de dados, o Jupyter Notebook permite combinar código Python, texto formatado e visualizações em um único documento.

- **Google Colab (Google Colaboratory):** é um serviço gratuito baseado em nuvem que permite criar e executar notebooks do Jupyter diretamente no navegador. É uma opção popular para desenvolver códigos Python, especialmente para projetos de aprendizado de máquina e ciência de dados. O Colab oferece acesso gratuito a recursos de computação na nuvem e a diversas bibliotecas populares de Python pré-instaladas, tornando-o uma escolha conveniente para experimentar algoritmos de aprendizado de máquina e realizar análise de dados de maneira colaborativa.

### 1.4 - Instalação do Python e ambiente de desenvolvimento:

Existem várias opções para instalar o Python em seu computador, mas uma das mais populares é a distribuição Anaconda, que inclui o Python e outras bibliotecas populares para ciência de dados e aprendizado de máquina. O ambiente de desenvolvimento mais comumente usado para Python é o IDLE (Integrated Development and Learning Environment), que é uma interface gráfica de usuário para escrever e executar código Python.

##### Instalando o Python pela página oficial:

    1 - Acesse o site oficial do Python em https://www.python.org/downloads/

    2 - Clique no botão "Download" na página inicial.
    Selecione a versão mais recente do Python e o sistema operacional correspondente. Por exemplo, para Windows, clique em "Windows" e, em seguida, na versão mais recente do Python.
    3 - Baixe o instalador do Python e salve-o em seu computador.
    4 - Execute o arquivo baixado e siga as instruções do instalador. Certifique-se de marcar a opção "Add Python to PATH" durante a instalação para que o Python possa ser facilmente acessado a partir do prompt de comando ou de qualquer outro ambiente de desenvolvimento.
    5 - Depois de concluir a instalação, abra o prompt de comando e digite python --version para verificar se o Python foi instalado corretamente. Você deve ver a versão do Python que acabou de instalar.
    
##### Instalando o Python com Anaconda
    1 - O Anaconda é uma distribuição de Python popular para ciência de dados e aprendizado de máquina, que inclui muitas bibliotecas e ferramentas populares. Aqui está como instalar o Python com Anaconda:

    2 - Baixe o instalador do Anaconda em https://www.anaconda.com/products/individual
    3 - Selecione a versão mais recente do Anaconda para o seu sistema operacional e baixe o instalador.
    4 - Execute o arquivo baixado e siga as instruções do instalador.
    5 - Durante a instalação, certifique-se de marcar as opções "Add Anaconda to my PATH environment variable" e "Register Anaconda as my default Python" para garantir que o Anaconda seja facilmente acessado e configurado como seu ambiente de desenvolvimento padrão.
    
Depois de concluir a instalação, abra o Anaconda Navigator (ou use o prompt de comando) para acessar o ambiente de desenvolvimento do Python e começar a programar.
Com o Python e o ambiente de desenvolvimento instalados corretamente, você está pronto para começar a programar em Python!

### 1.5 - Executando o primeiro programa em Python:

O código Python mais simples é uma declaração "print" que imprime uma mensagem na tela. Aqui está um exemplo:


```python
print("Olá, mundo!")
```

    Olá, mundo!
    

**Nota:** Linhas iniciadas com # são comentários, que são ignorados pelo Python e servem para explicar o código.


```python
# Esse texto está comentado e será ignorado
print(f"Olá, mundo!")
```

    Olá, mundo!
    

## 2 - Tipos de dados e variáveis:

Tecnicamente, podemos definir uma variável como sendo um identificador de um valor dentro da memória do computador. Se você preferir, pode imaginar uma variável como uma "caixa de memória" na qual é guardado algum valor. E que são esses valores? São os dados, informações que são armazenadas e manipuladas em um programa.

### 2.1 - Tipos de dados em Python:

- **Numéricos:** Em Python, existem dois tipos principais de dados numéricos, inteiros (*int*) e de ponto flutuante (*float*). Inteiros são números inteiros positivos ou negativos, enquanto números de ponto flutuante são números reais com um ponto decimal.

Exemplos:


```python
a = 2 # inteiro (int)
b = 3.14 # ponto flutuante (float)

# mostra as variáveis a e b e seus respectivos tipos.
print(f'Inteiro: {a}, {type(a)}, Ponto flutuante: {b}, {type(b)}')
```

    Inteiro: 2, <class 'int'>, Ponto flutuante: 3.14, <class 'float'>
    


```python
type(a)
```




    int



- **Strings:** Strings são sequências de caracteres. Em Python, eles são definidos entre aspas simples ou duplas.

Exemplos:


```python
nome = 'João' #entre aspas simples
mensagem = "Olá, mundo!" #entre aspas duplas

print(nome)
print(mensagem)
```

    João
    Olá, mundo!
    


```python
print(f"Variável: {nome}, tipo: {type(nome)}")
```

    Variável: João, tipo: <class 'str'>
    

- **Booleanos:** Booleanos são valores lógicos que podem ser verdadeiros (True) ou falsos (False).

Exemplos:


```python
verdadeiro = True
falso = False
```


```python
print(f"Variável: {verdadeiro}, tipo: {type(verdadeiro)}")
```

    Variável: True, tipo: <class 'bool'>
    

### 2.2 - Declaração de variáveis:

Em Python, as variáveis são declaradas atribuindo um valor a um nome. O nome da variável pode ser qualquer combinação de letras, números e sublinhados, **desde que comece com uma letra**. Uma vez declarada, a variável pode ser chamada no programa sempre que necessitarmos acessar seu valor.

Exemplos:


```python
idade = 25
nome = "Maria"
```

**Nota:** Não podemos declarar uma variável cujo nome comece com um número. Isso gera um **erro de sintaxe**.


```python
25anos = 25
print(25anos)
```


      Cell In[10], line 1
        25anos = 25
          ^
    SyntaxError: invalid syntax
    



```python
idade_25 = 25
print(idade_25)
```

    25
    

**Nota:** Algumas palavras são reservados pelo python e não pode ser usadas como nomes de variáveis, é o caso abaixo. 


```python
if = 20
```


      Cell In[12], line 1
        if = 20
           ^
    SyntaxError: invalid syntax
    


### 2.3 - Operações com variáveis:

As variáveis em Python podem ser usadas em várias operações matemáticas e lógicas.

#### 2.3.1 - Operações matemáticas:


```python
# Definição das variáveis a e b
a = 10
b = 8
```


```python
# Operações
c = a + b #adição
d = a - b #subtração
e = a * b #multiplicação
f = a / b #divisão
g = a % b #módulo (resto da divisão)
h = a ** b #exponenciação
```


```python
# Nos comando abaixo função print utiliza(chama) as variáveis definidas anteriormente para escrever seus valores na tela
print(f'a + b = {c}')
print(f'a - b = {d}')
print(f'a * b = {e}')
print(f'a / b = {f}')
print(f'a % b = {g}')
print(f'a ^ b = {h}')
```

    a + b = 18
    a - b = 2
    a * b = 80
    a / b = 1.25
    a % b = 2
    a ^ b = 100000000
    

**Nota:** Em python, o operador igual (=) não possui o mesmo significado e comportamento que na matemática. Na matemática, ele é um operador bidirecional: A = B seria a mesma coisa que B = A. No Python, ele é o que chamamos de um operador de **ATRIBUIÇÃO:** A variável da esquerda apenas armazena o valor da direita. Em outras palavras, o valor da direita é atribuido a variável à esquerda.

#### 2.3.2 - Operações lógicas e relacionais:

Operadores lógicos e relacionais são usados para realizar diferentes tipos de comparações entre valores.

Os operadores relacionais são usados para comparar valores e retornar um valor booleano (true ou false) com base na relação entre eles.

Em Python existem 6 operadores relacionais:

- **Maior que: >**
- **Maior ou igual: >=**
- **Menor que: <**
- **Menor ou igual: <=**
- **Igual: ==**
- **Diferente: !=**

**Nota:** Observe que o operador que compara se 2 valores são iguais é **==**, e não **=**. Isso ocorre porque o operador = é o nosso operador de **atribuição:** ele diz que a variável à sua esquerda deve receber o valor da expressão à direita. O operador == irá testar se o valor à sua esquerda é igual ao valor à sua direita e irá responder **True** ou **False**, assim como ocorre com todos os outros operadores de comparação.

Os operadores lógicos são usados para avaliar condições booleanas, ou seja, expressões que podem ser verdadeiras ou falsas. Eles são usados para combinar ou inverter condições, e incluem os operadores:

- **AND:** retorna verdadeiro (true) se **TODAS** as condições forem verdadeiras, caso contrário, retorna falso (false).
- **OR:** retorna verdadeiro (true) se **AO MENOS UMA** das condições for verdadeira, caso contrário, retorna falso (false).
- **NOT:** **INVERTE** o resultado booleano de uma certa condição. Retorna true se a condição avaliada for falsa e vice-versa.


```python
a = 10
b = 8

resultado1 = a > b # verdadeiro se a for maior que b
resultado2 = a == b # verdadeiro se a for igual a b
resultado3 = a < b # verdadeiro se a for menor que b
resultado4 = not(resultado3) # inverte o resultado da condição a < b

print(f'a = {a}, b = {b}, a > b = {resultado1}')
print(f'a = {a}, b = {b}, a == b = {resultado2}')
print(f'a = {a}, b = {b}, a < b = {resultado3}')
print(f'a = {a}, b = {b}, ~(a < b) = {resultado4}')
```

    a = 10, b = 8, a > b = True
    a = 10, b = 8, a == b = False
    a = 10, b = 8, a < b = False
    a = 10, b = 8, ~(a < b) = True
    

Podemos obter o resultado lógico da comparação de duas ou mais relações, por exemplo:


```python
operacao1 = 7 > 2 and 6 > 1 # Verdadeira se as duas condições são verdadeiras
operacao2 = 5 == 3 and 6 > 3 # Verdadeira se as duas condições são verdadeiras
operacao3 = 5 == 3 or 6 > 3 # Verdadeira se ao menos uma duas condições é verdadeira
operacao4 = 5 != 3 and 6 > 3 and 1 > 0 # Verdadeira se as três condições são verdadeiras

print('Resultado da operação 1: ', operacao1)
print('Resultado da operação 2: ', operacao2)
print('Resultado da operação 3: ', operacao3)
print('Resultado da operação 4: ', operacao4)
```

    Resultado da operação 1:  True
    Resultado da operação 2:  False
    Resultado da operação 3:  True
    Resultado da operação 4:  True
    

**Nota:** é possível ainda fazer um tipo especial de operação com *strings* utilizando o operador **+**, denominada concatenação. O resultado nada mais é do que a junção de duas ou mais strings em uma única string, como mostra o exemplo abaixo:


```python
primeiro_nome = 'james'
ultimo_nome = 'Bond'
nome_completo = primeiro_nome + ' ' + ultimo_nome # Adicionamos um espaço entre os nomes
print(f'Meu nome é {ultimo_nome}, {nome_completo}.')
```

    Meu nome é Bond, james Bond.
    

### 2.4 - Exercícios:

#### Exercício 1:
    a) Declare uma variável chamada "nome_produto" e atribua a ela uma string com o nome do produto que você deseja comprar.
    b) Declare uma variável chamada "preco_produto" e atribua a ela o valor numérico do preço do produto que você deseja comprar.
    c) Declare uma variável chamada "quantidade" e atribua a ela um número inteiro que represente quantos itens desse produto você pretende comprar.
    d) Declare uma variável chamada "total_compra" e calcule o valor total da compra multiplicando o preço do produto pela quantidade desejada.
    e) Utilize a função print() no padrão que temos usado até aqui para mostrar o nome do produto, o valor, a quantidade comprada e valor total da compra.


```python
# a)
nome_produto = "Celular"

# b)
preco_produto = 1200.50

# c)
quantidade = 3

# d)
total_compra = preco_produto * quantidade

# e)
print("Nome do produto:", nome_produto)
print("Preço do produto:", preco_produto)
print("Quantidade comprada:", quantidade)
print("Valor total da compra:", total_compra)
```

    Nome do produto: Celular
    Preço do produto: 1200.5
    Quantidade comprada: 3
    Valor total da compra: 3601.5
    

#### Exercício 2:
    a) Declare uma variável chamada "saldo" e atribua a ela um valor numérico que represente o saldo da sua conta bancária.
    b) Declare uma variável chamada "gastos" e atribua a ela um valor numérico que represente quanto você gastou em uma compra.
    c) Atualize o valor da variável "saldo" subtraindo o valor da variável "gastos".
    d) Crie uma variável booleana chamada "tem_saldo_positivo" e atribua a ela o valor True se o saldo for maior ou igual a 0, caso contrário, atribua o valor False.


```python
# a)
saldo = 1500.75

# b)
gastos = 350.20

# c)
saldo = saldo - gastos

# d)
tem_saldo_positivo = saldo >= 0

print("Saldo atual:", saldo)
print("Tem saldo positivo?", tem_saldo_positivo)

```

    Saldo atual: 1150.55
    Tem saldo positivo? True
    

#### Exercício 3:
    a) Declare uma variável chamada "frase" e atribua a ela uma string com uma frase de sua escolha.
    b) Crie uma variável chamada "tamanho_frase" e calcule o tamanho da string "frase" usando a função len().
    c) Declare uma variável chamada "caractere" e atribua a ela uma letra qualquer da "frase".
    d) Crie uma variável chamada "quantidade_caractere" e conte quantas vezes o "caractere" escolhido aparece na "frase" usando o método count().


```python
# a)
frase = "A vida é bela e cheia de surpresas."

# b)
tamanho_frase = len(frase)

# c)
caractere = "e"

# d)
quantidade_caractere = frase.count(caractere)

# Mostra o tamanho da frase e a quantidade de vezes que o caractere escolhido aparece na frase
print("Frase:", frase)
print("Tamanho da frase:", tamanho_frase)
print("Caractere escolhido:", caractere)
print("Quantidade do caractere na frase:", quantidade_caractere)
```

    Frase: A vida é bela e cheia de surpresas.
    Tamanho da frase: 35
    Caractere escolhido: e
    Quantidade do caractere na frase: 5
    

## 3 - Estruturas de controle de fluxo:

As estruturas de controle de fluxo são recursos fundamentais em programação, pois permitem controlar a sequência de execução das instruções em um programa. Elas permitem que você tome decisões e repita ações com base em determinadas condições. Essas estruturas permitem que você crie programas mais flexíveis e dinâmicos, adaptando o comportamento do programa de acordo com as condições e requisitos específicos. Combinando essas estruturas de maneira adequada, é possível construir algoritmos complexos e resolver uma ampla variedade de problemas de programação.

Existem três principais estruturas de controle de fluxo:

### 3.1 - Estruturas Condicionais

As instruções condicionais permitem que você tome decisões com base em condições específicas. A estrutura mais comum é o "if-else" (se-senão), que avalia uma condição e executa um bloco de código se a condição for verdadeira e outro bloco se a condição for falsa. Também existe a estrutura "if-elif-else" (se-senão se-senão), que permite avaliar múltiplas condições sequencialmente.

#### 3.1.1 if/else:
A estrutura if/else é usada para testar uma condição e executar diferentes blocos de código, dependendo se a condição é verdadeira ou falsa. No caso do if, queremos que o programa execute um código se uma dada condição for atendida. Já se estamos interessados que o programa escolha entre 2 casos mutuamente exclusivos, utilizamos o if e o else conjuntamente. O else não possui condição para verificar e sempre vem imediatamente após um if e é executado se o if for ignorado.

A sintaxe básica é a seguinte:
# Caso do if isolado
if condicao:
    # codigo a ser executado se a condição for verdadeira# Caso do if e else em conjunto
if condicao:
    # codigo a ser executado se a condição for verdadeira
else:
    # codigo a ser executado se a condição for falsa
Exemplos:


```python
# Caso do if isolado
condicao = True
if condicao:
  # codigo a ser executado se a condição for verdadeira
  print('A condição é verdadeira')
```

    A condição é verdadeira
    


```python
# Caso do if isolado
condicao = False
if condicao:
  # codigo a ser executado se a condição for verdadeira
  print('A condição é verdadeira')
```


```python
# Comentario sobre a identação
condicao = False
if condicao:
  # codigo a ser executado se a condição for verdadeira
  print('A condição é verdadeira')

print('Isso não faz parte do if')
```

    Isso não faz parte do if
    


```python
# Caso do if e else em conjunto
condicao = True
if condicao:
    # codigo a ser executado se a condição for verdadeira
    print('A condição é verdadeira')
else:
    # codigo a ser executado se a condição for falsa
    print('A condição é falsa')
```

    A condição é verdadeira
    


```python
# Caso do if e else em conjunto
condicao = False
if condicao:
    # codigo a ser executado se a condição for verdadeira
    print('A condição é verdadeira')
else:
    # codigo a ser executado se a condição for falsa
    print('A condição é falsa')
```

    A condição é falsa
    

Mais um exemplo: suponha que queremos escrever um código que verifica se um número é positivo ou negativo:


```python
numero = int(input("Digite um número: "))

if numero > 0:
    print("O número é positivo")
else:
    print("O número é negativo ou zero")
```

    Digite um número: 18
    O número é positivo
    

**Nota:** Utilizamos um 'tab' antes de cada linha pertencente ao if. Esse processo é chamado de "indentação", cujo objetivo é melhorar a legibilidade e organização do código.

É possível encadear diversos if's e else's, como mostra o programa abaixo:


```python
tempo = float(input('Digite a temperatura atual: '))
if tempo >= 30:
    resposta = input('Gostaria de se refrescar em uma piscina? ')
    if resposta == 'sim':
        print('Ótimo! A piscina está aberta, aproveite!')
    else:
        print('Tudo bem, aproveite o seu dia!')
else:
    print('A temperatura não está alta o suficiente para aproveitar uma piscina neste momento.')
```

    Digite a temperatura atual: 36
    Gostaria de se refrescar em uma piscina? sim
    Ótimo! A piscina está aberta, aproveite!
    

Por fim, podemos testar diversos casos mutuamente exclusivos utilizando o 'elif'.

O comando elif é a contração de "else if" - ou seja, caso um if não seja executado, você pode propor uma nova condição para ser testada.


```python
exercicios = int(input('Quantos exercícios de Python você já fez?'))

if exercicios > 30:
    print('Já está ficando profissional!')
elif exercicios > 20:
    print('Tá indo bem, bora fazer mais alguns!')
elif exercicios > 10:
    print('Vamos tirar o atraso?')
else:
    print('Xiiii...')
```

    Quantos exercícios de Python você já fez?56
    Já está ficando profissional!
    

A vantagem de usar a estrutura elif é a possibilidade de verificar múltiplas condições de forma mais concisa e eficiente.

Ao usar uma série de instruções if independentes, todas as condições são verificadas, mesmo que uma delas já tenha sido satisfeita. Isso pode resultar em um desperdício de recursos computacionais, especialmente quando há muitas condições a serem verificadas.

Por outro lado, o uso do elif permite que o programa verifique as condições em sequência e execute apenas o bloco de código correspondente à primeira condição verdadeira encontrada. Isso significa que assim que uma condição for satisfeita, as condições subsequentes serão ignoradas.

Essa abordagem economiza tempo de processamento, pois o programa não precisa verificar todas as condições se a primeira já for verdadeira. Além disso, torna o código mais legível e organizado, especialmente quando há uma lógica complexa com várias condições a serem avaliadas.

### 3.2 Estruturas de Repetição

Os loops permitem que você repita a execução de um bloco de código várias vezes. Existem dois tipos principais de loops: "while" (enquanto) e "for". O loop "while" repete o bloco de código enquanto uma determinada condição for verdadeira. O loop "for" é usado para iterar sobre uma sequência de elementos, como uma lista, e executar o bloco de código para cada elemento da sequência.

#### 3.2.1 while:

A estrutura while é usada para repetir um bloco de código enquanto uma condição é verdadeira. A sintaxe básica é a seguinte:
while condicao:
    # codigo a ser repetido enquanto a condicao for verdadeira
Por exemplo, suponha que queremos escrever um código que imprime os números de 1 a 10:


```python
numero = 1

while numero <= 10:
    print(numero)
    numero += 1
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    

#### 3.2.2 for:

A estrutura for é usada para iterar sobre uma sequência, como uma lista ou uma string. A sintaxe básica é a seguinte:
for elemento in sequencia:
    # codigo a ser executado para cada elemento da sequencia
Por exemplo, suponha que queremos escrever um código que imprime os elementos de uma lista:


```python
lista_frutas = ["maçã", "banana", "laranja"]

for fruta in lista_frutas:
    print(fruta)
```

    maçã
    banana
    laranja
    


```python
# Mostre os número de 0 a 10 usando o for
lista_numeros = [0,1,2,3,4,5,6,7,8,9,10]

for numero in lista_numeros:
    print(numero)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    


```python
# Mostrar os números de 1 a 10
for numero in range(1,11):
    print(numero)
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    


```python
for numero in range(0,11):
    print(numero)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    


```python
lista_salarios = [1000, 2000]
for salario in lista_salarios:
  novo_salario = salario*1.2
  print(novo_salario)
```

    1200.0
    2400.0
    

### 3.3 - Instruções de controle de fluxo

Existem também as instruções de controle de fluxo em Python, como o break e o continue, que permitem que você interrompa ou pule partes de um loop. Essas estruturas são úteis em muitas situações, como quando você quer sair de um loop assim que encontrar o resultado desejado.

Dominar as estruturas de controle de fluxo é fundamental para programar em Python e em qualquer outra linguagem de programação. Elas permitem que você crie programas mais sofisticados e eficientes, que podem lidar com uma variedade de situações e condições.

#### 3.3.1 Break
A instrução break é usada para interromper a execução de um loop imediatamente, mesmo que a condição do loop ainda não tenha sido completamente avaliada. Quando o break é encontrado, o programa sai do loop e continua a execução a partir da próxima linha após o loop.

O loop abaixo em tese seria infinito, mas se a condição do if for verificada, o break é executado e conseguimos escapar do loop:


```python
while True:
    resposta = input('Digite OK: ')
    if resposta == 'OK':
        break
```

    Digite OK: Não
    Digite OK: Não
    Digite OK: OK
    

Um outro exemplo, seria:


```python
numeros = [1, 2, 3, 4, 5, 6]

for numero in numeros:
    if numero == 4:
        break
    print(numero)
```

    1
    2
    3
    

#### 3.3.2 Continue

A instrução continue é usada para pular o restante do código dentro do loop e continuar para a próxima iteração. Quando o continue é encontrado, o programa ignora qualquer código restante dentro do bloco do loop e passa para a próxima iteração.



```python
numeros = [1, 2, 3, 4, 5, 6]

for numero in numeros:
    if numero % 2 == 0:
        continue
    print(numero)
```

    1
    3
    5
    

### 3.4 - Exercícios

**Exercício 1:** Faça um programa que peça ao usuário a idade e verifique se a pessoa já pode votar (idade maior ou igual a 16 anos).


```python
# Solicita a idade do usuário
idade = int(input("Digite sua idade: "))

# Verifica se a pessoa pode votar (idade maior ou igual a 16 anos)
if idade >= 16:
    print("Você já pode votar!")
else:
    print("Você ainda não pode votar.")
```

    Digite sua idade: 56
    Você já pode votar!
    

**Exercício 2:** Escreva um programa que solicite ao usuário um número e exiba se esse número é positivo, negativo ou igual a zero.


```python
# Solicita ao usuário um número
numero = int(input("Digite um número: "))

# Verifica se o número é positivo, negativo ou igual a zero
if numero > 0:
    print("O número é positivo.")
elif numero < 0:
    print("O número é negativo.")
else:
    print("O número é igual a zero.")

```

    Digite um número: 6
    O número é positivo.
    

**Exercício 3:** Escreva um programa que solicite ao usuário um número inteiro positivo e faça uma contagem regressiva a partir desse número até zero. O programa deve exibir os números na tela enquanto realiza a contagem.


```python
# Solicita ao usuário um número inteiro positivo
numero_str = input("Digite um número inteiro positivo: ")

# Converte o número para um valor inteiro
numero = int(numero_str)

# Verifica se o número é positivo
if numero <= 0:
    print("O número digitado não é um inteiro positivo.")
else:
    # Contagem regressiva
    print("Contagem regressiva:")
    while numero >= 0:
        print(numero)
        numero -= 1
```

    Digite um número inteiro positivo: 12
    Contagem regressiva:
    12
    11
    10
    9
    8
    7
    6
    5
    4
    3
    2
    1
    0
    

**Exercício 4:** Escreva um programa que exiba os números pares de 0 a 20 usando um loop while.


```python
# Inicializa o contador com o valor 0
numero = 0

# Loop while para exibir os números pares de 0 a 20
print("Números pares de 0 a 20:")
while numero <= 20:
    print(numero)
    numero += 2
```

    Números pares de 0 a 20:
    0
    2
    4
    6
    8
    10
    12
    14
    16
    18
    20
    

**Exercício 5:** Escreva um programa que exiba os números pares de 0 a 20 usando um loop for.


```python
# Loop for para exibir os números pares de 0 a 20
print("Números pares de 0 a 20:")
for numero in range(0, 21, 2):
    print(numero)
```

    Números pares de 0 a 20:
    0
    2
    4
    6
    8
    10
    12
    14
    16
    18
    20
    

**Exercício 6:** Crie um programa que calcule a soma dos números de 1 a 100 (1+2+3+...+100) usando um loop for.


```python
# Inicializa a variável para armazenar a soma
soma = 0
# Loop for para somar os números de 1 a 100
for numero in range(1, 101):
    soma += numero

# Exibe o resultado da soma
print("A soma dos números de 1 a 100 é:", soma)
```

    A soma dos números de 1 a 100 é: 5050
    

**Exercício 7:** Escreva um programa que imprima os números de 1 a 10, exceto o número 5. Use a instrução "continue" para pular o número 5.


```python
# Loop for para imprimir os números de 1 a 10, exceto o número 5
print("Números de 1 a 10, exceto o número 5:")
for numero in range(1, 11):
    if numero == 5:
        continue
    print(numero)
```

    Números de 1 a 10, exceto o número 5:
    1
    2
    3
    4
    6
    7
    8
    9
    10
    

**Exercício 8:** Escreva um programa que exiba os números de 1 a 20, mas interrompa o loop quando encontrar um número divisível por 7. Use a instrução "break" para sair do loop quando essa condição for atendida.


```python
# Loop for para exibir os números de 1 a 20, interrompendo quando encontrar um número divisível por 7
print("Números de 1 a 20 (com interrupção no primeiro número divisível por 7):")
for numero in range(1, 21):
    print(numero)
    if numero % 7 == 0:
        break
```

    Números de 1 a 20 (com interrupção no primeiro número divisível por 7):
    1
    2
    3
    4
    5
    6
    7
    

## 4 - Estruturas de Dados:

As estruturas de dados são fundamentais para organizar e manipular informações em um programa de forma eficiente. Compreender o funcionamento e a utilização dessas estruturas é essencial para o desenvolvimento de aplicações robustas e versáteis.

### 4.1 Listas

Listas são estruturas de dados versáteis que permitem armazenar uma coleção ordenada de elementos em uma única variável.

Características das listas:
- Listas são mutáveis, ou seja, seus elementos podem ser modificados após a criação.
- São definidas utilizando colchetes ([]).
- Os elementos em uma lista podem ser de tipos diferentes.

Exemplo de criação de uma lista:


```python
lista_numeros = [1, 2, 3, 4, 5]
```

#### 4.1.2 Acesso e manipulação de elementos em Listas

Para acessar elementos em uma lista, utilizamos índices numéricos, começando por 0 para o primeiro elemento (lista[0]).

Exemplo de acesso a elementos:


```python
lista_frutas = ['maçã', 'banana', 'laranja', 'morango']

print(lista_frutas[0]) # Acessa o primeiro elemnento
print(lista_frutas[2]) # Acessa o terceiro elemento
print(lista_frutas[3]) # Acessa o quarto e último elemento
print(lista_frutas[-1]) # Acessa o último elemento
```

    maçã
    laranja
    morango
    morango
    

Para alterar elementos em uma lista, basta atribuir um novo valor ao índice desejado.

Exemplo de manipulação de elementos:


```python
# Substitui o segundo elemento
lista_frutas[1] = 'abacaxi'
print(lista_frutas)
```

    ['maçã', 'abacaxi', 'laranja', 'morango']
    


```python
# Remove o primeiro elemento

del lista_frutas[0]
print(lista_frutas)
```

    ['abacaxi', 'laranja', 'morango']
    


```python
# Usando o método pop()
lista_frutas.pop(2)
print(lista_frutas)
```

    ['abacaxi', 'laranja']
    


```python
# Adiciona novos elementos à lista (método append())
lista_frutas.append('Maracujá')
print(lista_frutas)
```

    ['abacaxi', 'laranja', 'Maracujá']
    

Usando loops:


```python
for fruta in lista_frutas:
    print(fruta)
```

    abacaxi
    laranja
    Maracujá
    


```python
for index in range(len(lista_frutas)):
    print(index, lista_frutas[index])
```

    0 abacaxi
    1 laranja
    2 Maracujá
    


```python
index = 0
while index < len(lista_frutas):
    print(index, lista_frutas[index])
    index +=1
```

    0 abacaxi
    1 laranja
    2 Maracujá
    


```python
# Cria uma lista com os números de 0 a 10
lista_numeros = []

for numero in range(11):
    lista_numeros.append(numero)

print(lista_numeros)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    

### 4.2 Tuplas em Python:

Tuplas são estruturas de dados **imutáveis** que permitem armazenar uma coleção ordenada de elementos em uma única variável.

- Tuplas são definidas utilizando parênteses ()
- Os elementos em uma tupla podem ser de tipos diferentes
- Os elementos em uma tupla NÃO podem ser modificados

Exemplo de criação de uma tupla:


```python
tupla_coordenadas = (10, 20)
```

#### 4.2.1 Acesso e Manipulação de elementos em Tuplas:
Para acessar elementos em uma tupla, também utilizamos índices numéricos, começando por 0 para o primeiro elemento (tupla[0]).

Exemplo de acesso a elementos:


```python
ponto = (3, 5)
print(ponto[0])
print(ponto[1])
```

    3
    5
    

Concatenação


```python
tupla1 = (1,2)
tupla2 = (3,4)
concatenacao = tupla1 + tupla2
print(concatenacao)
```

    (1, 2, 3, 4)
    

Usando loops:


```python
for elemento in ponto:
    print(elemento)
```

    3
    5
    


```python
for index in range(len(ponto)):
    print(index, ponto[index])
```

    0 3
    1 5
    


```python
index = 0
while index < len(ponto):
    print(index, ponto[index])
    index +=1
```

    0 3
    1 5
    

**Nota desnecessária:** Diferentemente das listas, as tuplas são imutáveis, o que significa que seus elementos não podem ser alterados após a criação.

### 4.3 Dicionários em Python:
Dicionários são estruturas de dados que permitem armazenar elementos em um formato de chave-valor. Cada elemento (valor) é associado a uma chave exclusiva que permite acessá-lo de forma rápida.

Dicionários são definidos utilizando chaves ({}) com pares chave-valor separados por dois pontos (:).
As chaves devem ser únicas e imutáveis, geralmente strings ou números.

Exemplo de criação de um dicionário:


```python
dicionario_idade_alunos = {
    'João': 18,
    'Maria': 20,
    'Pedro': 19,
    'Ana': 21
}
```


```python
dicionario_idade_alunos = {}
dicionario_idade_alunos['João'] = 18
dicionario_idade_alunos['Maria'] = 20
dicionario_idade_alunos['Pedro'] = 19
dicionario_idade_alunos['Ana'] = 21
```

#### 4.3.1 Acesso e Manipulação de elementos em Dicionários:
Para acessar elementos em um dicionário, utilizamos as chaves como índices.

Exemplo de acesso a elementos:


```python
print(dicionario_idade_alunos['Maria'])
print(dicionario_idade_alunos['Pedro'])
```

    20
    19
    

Para adicionar, atualizar ou remover elementos de um dicionário, basta atribuir um novo valor a uma chave ou utilizar o método del.

Exemplo de manipulação de elementos:


```python
# Adicionar um novo aluno
dicionario_idade_alunos['Mariana'] = 22

print(dicionario_idade_alunos)
```

    {'João': 18, 'Maria': 20, 'Pedro': 19, 'Ana': 21, 'Mariana': 22}
    


```python
# Atualizar a idade de um aluno
dicionario_idade_alunos['Pedro'] = 20

print(dicionario_idade_alunos)
```

    {'João': 18, 'Maria': 20, 'Pedro': 20, 'Ana': 21, 'Mariana': 22}
    


```python
# Remover um aluno
del dicionario_idade_alunos['Ana']

print(dicionario_idade_alunos)
```

    {'João': 18, 'Maria': 20, 'Pedro': 20, 'Mariana': 22}
    

Usando loops:


```python
# Acesso às chaves do dicionário usando um loop for
for nome in dicionario_idade_alunos:
    print(nome)
```

    João
    Maria
    Pedro
    Mariana
    


```python
# Acesso aos valores do dicionário usando um loop for
for idade in dicionario_idade_alunos.values():
    print(idade)
```

    18
    20
    20
    22
    


```python
# Acesso às chaves e valores do dicionário usando um loop for
for nome, idade in dicionario_idade_alunos.items():
    print("Nome:", nome, "- Idade:", idade)
```

    Nome: João - Idade: 18
    Nome: Maria - Idade: 20
    Nome: Pedro - Idade: 20
    Nome: Mariana - Idade: 22
    

### 4.4 - Exercícios:

#### Exercício 1:
    a) Crie uma lista vazia chamada "numeros". Em seguida, adicione os números de 1 a 10 a essa lista utilizando um loop.
    b) Dada a lista "frutas" abaixo, remova o último elemento da lista e adicione "morango" no seu lugar.
    frutas = ["maçã", "banana", "laranja", "abacaxi"]
    c) Dada a lista "notas" com as notas de um estudante, calcule a média aritmética das notas.
    notas = [8.5, 7.2, 6.8, 9.0, 7.5]


```python
# a)
numeros = []
for i in range(1, 11):
    numeros.append(i)

print(numeros)
```

    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    


```python
# b)
frutas = ["maçã", "banana", "laranja", "abacaxi"]
frutas.pop()  # Remove o último elemento da lista
frutas.append("morango")  # Adiciona "morango" no lugar do último elemento removido

print(frutas)
```

    ['maçã', 'banana', 'laranja', 'morango']
    


```python
# c)
notas = [8.5, 7.2, 6.8, 9.0, 7.5]
media = sum(notas) / len(notas)

print("Notas:", notas)
print("Média:", media)

```

    Notas: [8.5, 7.2, 6.8, 9.0, 7.5]
    Média: 7.8
    

#### Exercício 2:
    a) Crie uma lista com os números de 1 a 10. Acesse o primeiro e o último elemento da lista.
    b) Dada a lista "nomes" abaixo, substitua o segundo elemento por "Maria".
    nomes = ["João", "Pedro", "Carlos", "Ana"]


```python
# a)
numeros = list(range(1, 11))

primeiro_elemento = numeros[0]
ultimo_elemento = numeros[-1]

print("Lista de números:", numeros)
print("Primeiro elemento:", primeiro_elemento)
print("Último elemento:", ultimo_elemento)

```

    Lista de números: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    Primeiro elemento: 1
    Último elemento: 10
    


```python
# b)
nomes = ["João", "Pedro", "Carlos", "Ana"]
nomes[1] = "Maria"  # Substitui o segundo elemento (índice 1) por "Maria"

print("Lista de nomes:", nomes)
```

    Lista de nomes: ['João', 'Maria', 'Carlos', 'Ana']
    

#### Exercício 3:
    a) Dada a tupla "coordenadas" abaixo, extraia os valores de latitude e longitude em variáveis separadas.
    coordenadas = (40.7128, -74.0060)
    b) Concatene as tuplas "tupla1" e "tupla2" abaixo em uma única tupla chamada "tupla_concatenada".
    tupla1 = (1, 2, 3)
    tupla2 = (4, 5, 6)


```python
# a)
coordenadas = (40.7128, -74.0060)
latitude, longitude = coordenadas

print("Latitude:", latitude)
print("Longitude:", longitude)

```

    Latitude: 40.7128
    Longitude: -74.006
    


```python
# b)
tupla1 = (1, 2, 3)
tupla2 = (4, 5, 6)
tupla_concatenada = tupla1 + tupla2

print("Tupla 1:", tupla1)
print("Tupla 2:", tupla2)
print("Tupla concatenada:", tupla_concatenada)

```

    Tupla 1: (1, 2, 3)
    Tupla 2: (4, 5, 6)
    Tupla concatenada: (1, 2, 3, 4, 5, 6)
    

#### Exercício 4:
    a) Crie um dicionário vazio chamado "contatos". Adicione três contatos (nome e e-mail) ao dicionário.
    b) Crie um dicionário chamado "pessoa" com as chaves "nome", "idade" e "cidade", preenchendo com seus respectivos valores.
    c) Dado o dicionário "pontuacoes" abaixo, atualize a pontuação de "Maria" para 95.
    pontuacoes = {"João": 85, "Maria": 90, "Carlos": 88}
    d) Imprima todas as chaves do dicionário "estoque_produtos" abaixo.
    estoque_produtos = {"arroz": 10, "feijão": 5, "macarrão": 8, "óleo": 15}


```python
# a)
contatos = {}

# Adicionando contatos ao dicionário
contatos["João"] = "joao@email.com"
contatos["Maria"] = "maria@email.com"
contatos["Pedro"] = "pedro@email.com"

print("Dicionário contatos:", contatos)
```

    Dicionário contatos: {'João': 'joao@email.com', 'Maria': 'maria@email.com', 'Pedro': 'pedro@email.com'}
    


```python
# b)
pessoa = {
    "nome": "João",
    "idade": 30,
    "cidade": "São Paulo"
}

print("Dicionário pessoa:", pessoa)

```

    Dicionário pessoa: {'nome': 'João', 'idade': 30, 'cidade': 'São Paulo'}
    


```python
# c)
pontuacoes = {"João": 85, "Maria": 90, "Carlos": 88}
pontuacoes["Maria"] = 95  # Atualiza a pontuação de "Maria" para 95

print("Dicionário pontuações atualizado:", pontuacoes)

```

    Dicionário pontuações atualizado: {'João': 85, 'Maria': 95, 'Carlos': 88}
    


```python
# d)
estoque_produtos = {"arroz": 10, "feijão": 5, "macarrão": 8, "óleo": 15}
chaves_estoque = estoque_produtos.keys()

print("Chaves do dicionário estoque_produtos:", chaves_estoque)

```

    Chaves do dicionário estoque_produtos: dict_keys(['arroz', 'feijão', 'macarrão', 'óleo'])
    


```python

```
