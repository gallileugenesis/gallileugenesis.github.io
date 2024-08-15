---
title:  "LLMs aprendem quando treinados com dados gerados por outras LLMs?"
date:   2024-08-15 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, research, paper, NLP, LLM]
layout: post
comments: true
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-15-08-comments-about-paper-Shumailov-et-al-2024/header_image.png?raw=true)


Se os atuais grandes modelos de linguagem - Large language models (LLMs) - foram treinados a partir de dados extraídos da web e, se esses dados foram, predominantemente, criados por humanos, e, se a internet está sendo inundada cada vez mais por dados criados por LLMs, o que irá acontecer com as próximas gerações de LLMs quando forem treinados por dados gerados por outros modelos e não mais por humanos? 

Essas foram as ótimas questões levantadas por Ilia Shumailov e seus parceiros no artigo  [“AI models collapse when trained on recursively generated data”](https://www.nature.com/articles/s41586-024-07566-y?s=08), publicado esse ano na Nature.

Ter modelos que aprendam razoavelmente bem os padrões da realidade a partir de dados gerados por outros modelos é, sem dúvidas, um Santo Graal para a área de inteligência artificial. 

Bem, se depender dos resultados desse artigo, esse sonho está cada vez mais difícil, talvez impossível. Os resultados indicam a degradação do desempenho de modelos quando treinados sequencialmente com dados gerados por outros modelos. Nas palavras dos autores: 

“*Descobrimos que o uso indiscriminado de conteúdo gerado por modelos no treinamento causa defeitos irreversíveis nos modelos resultantes, nos quais as caudas da distribuição original desaparecem.*”

Os autores denominaram esse fenômeno de "colapso do modelo" — “*um processo degenerativo pelo qual, ao longo do tempo, os modelos esquecem a verdadeira distribuição de dados subjacente, mesmo na ausência de uma mudança na distribuição ao longo do tempo.*”

Isso significa que “*o valor dos dados coletados sobre interações humanas genuínas com sistemas será cada vez mais valioso na presença de conteúdo gerado por LLM em dados rastreados da Internet.*”