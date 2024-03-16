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

Os modelos foram construídos de tal forma a serem capazes de resumir de forma abstrata, traçando arcos narrativos e temas maiores, e não somente listar séries de eventos. 

A principal métrica de avaliação dos modelos foram as notas atribuídas por julgadores humanos sobre a qualidade geral do resumo em uma [escala Likert](https://www.scribbr.com/methodology/likert-scale/#:~:text=A%20Likert%20scale%20is%20a,five%20or%20seven%20answer%20statements.) de 1 a 7. Além disso, houve uma avalição mais abstrata, onde os julgadores avaliaram a precisão do resumo, a cobertura do texto fonte, a coerência e a quantidade de abstração.

Os resumos intermediários foram definidos para conter uma taxa de compactação do texto de entrada em um fator de 5 a 10x, com limites superiores de comprimento de 128 a 384 tokens. A avaliação humana foi condicionada a quantidade de palavras dos resumos, para se evitar a preferência por resumos mais longos. 

Foram avaliados dois tamanhos de modelo 175B e 6B, ou seja, modelos com 175 bilhões 6 bilhões de parâmetros, respectivamente. Para cada tamanho de modelo, foram testados dois tipos de treinamento: clonagem comportamental (*behavioral cloning* (BC)) e aprendizagem por reforço (*reinforcement learning* (RL)). Além disso, foram testadas temperaturas T=0,0, 0,3 e 0,6. 

Um fato interessante é que a concordância entre os julgamentos humanos quanto à qualidade relativa dos resumos escritos por modelos foi de quase 80%.

Os resultados mostram que os melhores modelos geraram resumos sensatos de livros inteiros, igualando até mesmo a qualidade de resumos escritos por humanos em alguns casos. Em aproximadamente 5% dos casos, os resumos do melhor modelo 175B receberam uma pontuação de 6 de 7, e mais de 15% receberam 5 de 7. No entanto, na média, os resumos dos modelos ainda são significativamente piores do que os resumos escritos por humanos.

Um outro resultado interessante apresentado no trabalho foi de que as pontuações Likert para os resumos completos dos livros foram significativamente mais baixas do que as pontuações Likert de qualquer uma das tarefas individuais decompostas, ou seja, de resumos intermediários das frações do livro. Isso se deve ao fato de que os erros acumulados em cada profundidade são todos refletidos na pontuação completa do resumo do livro. 

Além da avaliação por julgamento humano, os modelos foram avaliados no conjunto de dados [BookSum](https://paperswithcode.com/dataset/booksum), usando as métricas  [ROUGE](https://en.wikipedia.org/wiki/ROUGE_(metric) e [BERTScore](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=BERTScore:_Evaluating_Text_Generation_with_BERT).

Nessa avaliação os modelos testados superaram praticamente todas as linhas de base na métrica ROUGE em 3-4 pontos. Com relação à metrica BERTScore, o resultado foi ainda melhor, com os modelos superando significativamente todas as linhas de base. 