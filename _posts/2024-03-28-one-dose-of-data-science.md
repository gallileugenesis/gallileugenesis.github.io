---
title:  "Uma dose de ciência de dados: Ciclo de vida LLMOps"
date:   2024-03-28 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, LLMOps, LLM]
layout: post
comments: true
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-28-one-dose-of-data-science/LLMOps_lifecycle.jpeg?raw=true)

Existem seis etapas principais principais pelas quais um modelo de linguagem normalmente passa:

- **Preparação e exploração de dados:** Semelhante ao aprendizado de máquina tradicional, você precisa preparar, explorar e limpar seus dados.
- **Pré treino:** O pré-treinamento normalmente é opcional, pois ao criar modelos de linguagem grandes, muitas vezes você usa um modelo que já foi pré-treinado.
- **Ajuste fino do modelo e engenharia de *prompt*:** É aqui que você melhora o desempenho do seu modelo em relação à sua tarefa específica, seja coletando novos dados para treinar (ajuste fino) ou solicitando o modelo com mais precisão (engenharia *prompt*).
- **Avaliação e depuração de modelo:** Depois de ter um modelo, você precisa avaliar sua qualidade. Você pode definir qualidade de diferentes maneiras. Isso inclui abordar e testar diferentes táticas e métricas de avaliação para avaliar a qualidade, bem como estratégias para testar a segurança do modelo.
- **Implantação do modelo:** A implantação costumava ser um dos maiores bloqueadores em projetos de ML, mas agora existem muitas soluções que facilitam a implantação de um modelo em produção. 
- **Monitoramento e manutenção do modelo:** Depois que um modelo é implantado, o trabalho árduo começa. Com um LLM, queremos monitorar mais do que faríamos em um modelo de ML tradicional. Queremos rastrear os usuários, como eles usam o aplicativo, quão consistente é o desempenho do modelo. Queremos coletar feedback direto do usuário, se possível, para nos ajudar a ajustar o modelo posteriormente. E, claro, queremos monitorar preocupações clássicas de MLOps, como desvio de modelo e latência.