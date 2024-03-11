---
title:  "Comentários sobre o artigo: Wu et al. 2021 - Recursively Summarizing Books with Human Feedback"
date:   2024-03-06 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, research, NLP, LLM]
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-06-comments-about-paper-wu-et-al-2021/header_image.png?raw=true)


Em inteligência artificial, o problema de alinhamento engloba as preocupações em garantir que os modelos de aprendizado de máquina atuem de acordo com as regras, normas e diretrizes humanas, definidas em um certo escopo, ou de acordo com princípios éticos mais universais.

A forma mais direta de se verificar se os resultados gerados pelos modelos estejam dentro do escopo esperado é passar esses resultados por uma "curadoria" humana.

Evidentemente, isso se torna um desafio quando lidamos com uma solução escalável, onde os resultados do modelo são difíceis ou demorados para serem avaliados por humanos.

No artigo [Recursively Summarizing Books with Human Feedback](https://arxiv.org/abs/2109.10862), os autores propuseram uma abordagem interessante para superar esse desafio, no caso de resumos de livros inteiros: eles combinaram aprendizagem por reforço a partir de feedback humano e decomposição recursiva de tarefas.

Obter o feedback humano para o resumo de um livro inteiro seria demasiado trabalhoso, já que um ser humano precisaria ler o livro inteiro, o que levaria muitas horas. É ai que entra a decomposição recursiva de tarefas: basicamente, o que se faz é dividir uma tarefa difícil em tarefas mais fáceis. Neste caso, o conteúdo do livro inteiro é subdivido em vários trechos mais curtos, sobre o qual o modelo realiza o resumo.  Isso permite que os humanos avaliem os resumos dos modelos mais rapidamente, usando resumos de partes menores do livro, em vez de ler o texto fonte. Os resumos podem ser combinados e resumidos novamente seguindo a mesma lógica até que um resumo final seja obtido.

Os resultados mostraram que o modelo resultante gera resumos sensatos de livros inteiros, igualando até mesmo a qualidade de resumos escritos por humanos em alguns casos (∼ 5% dos livros).

A figura abaixo mostram o fluxo do procedimento de resumo, proposto pelos autores. 

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-06-comments-about-paper-wu-et-al-2021/image.png?raw=true)