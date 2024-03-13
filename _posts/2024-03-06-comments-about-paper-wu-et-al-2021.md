---
title:  "Comentários sobre o artigo: Wu et al. 2021 - Recursively Summarizing Books with Human Feedback"
date:   2024-03-06 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, research, paper, NLP, LLM]
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-06-comments-about-paper-wu-et-al-2021/header_image.png?raw=true)


Em inteligência artificial, o problema de alinhamento engloba as preocupações em garantir que os modelos de aprendizado de máquina atuem de acordo com as regras, normas e diretrizes humanas, definidas em um certo escopo, ou de acordo com princípios éticos mais universais.

A forma mais direta de se verificar se os resultados gerados pelos modelos estejam dentro do escopo esperado é passar esses resultados por uma "curadoria" humana.

Evidentemente, isso se torna um desafio quando lidamos com uma solução escalável, onde os resultados do modelo são difíceis ou demorados para serem avaliados por humanos.

No artigo [Recursively Summarizing Books with Human Feedback](https://arxiv.org/abs/2109.10862), os autores propuseram uma abordagem interessante para superar esse desafio, no caso de resumos de livros inteiros: eles combinaram aprendizagem por reforço a partir de feedback humano e decomposição recursiva de tarefas.

Obter o feedback humano para o resumo de um livro inteiro seria demasiado trabalhoso, já que um ser humano precisaria ler o livro inteiro, o que levaria muitas horas. É ai que entra a decomposição recursiva de tarefas: basicamente, o que se faz é dividir uma tarefa difícil em tarefas mais fáceis. Neste caso, o conteúdo do livro inteiro é subdivido em vários trechos mais curtos, sobre o qual o modelo realiza o resumo.  Isso permite que os humanos avaliem os resumos dos modelos mais rapidamente, usando resumos de partes menores do livro, em vez de ler o texto fonte. Os resumos podem ser combinados e resumidos novamente seguindo a mesma lógica até que um resumo final seja obtido.

A figura abaixo mostra o fluxo do procedimento de resumo em forma de árvore, como proposto pelos autores. 

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-06-comments-about-paper-wu-et-al-2021/image.png?raw=true)

Primeiro o conteúdo do livro é decomposto em vários pedaços fixos (chunks) (altura 0). Na sequência, o modelo é treinado usando o algoritmo *behavioral cloning* (BC) e resumos dos chunks feitos por humanos. As ações de resumo,  feito pelo modelo ou por um humano, são representadas na ilustração do fluxo com o símbulo de um lápis.

Posteriormente, os resumos gerados pelo modelo são avaliados por humanos e, a partir desse feedback, o modelo é treinado usando um modelo de recompensa. Na sequência, os resumos são concatenados para passar por mais uma rodada de resumos, que segue a mesma política e se repete até se resumir o livro inteiro. 

Os autores reconhecem que com essa estrutura, alguns resumos intermediários podem não ser bem sucedidos por não ter um contexto adequado. Para lidar com esse problema, foi proposto concatenar resumos anteriores e coloca-los em contexto na mesma profundidade.  

Os resultados mostraram que o modelo resultante gera resumos sensatos de livros inteiros, igualando até mesmo a qualidade de resumos escritos por humanos em alguns casos (∼ 5% dos livros).
