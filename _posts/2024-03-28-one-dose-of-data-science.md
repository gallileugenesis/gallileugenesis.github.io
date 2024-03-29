---
title:  "Uma dose de ciência de dados: Ciclo de vida LLMOps"
date:   2024-03-28 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, LLMOps, LLM]
layout: post
comments: true
---

## LLMOps

LLMOps (*Large Language Model Ops*) se refere ao conjunto de práticas, ferramentas e metodologias usadas para gerenciar, implantar, monitorar e escalar grandes modelos de linguagem - LLMs (*Large Language Models*).  

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-28-one-dose-of-data-science/LLMOps_lifecycle.png?raw=true)

Existem seis etapas principais pelas quais um modelo de linguagem normalmente passa:

- **Preparação e exploração de dados:** Semelhante ao aprendizado de máquina tradicional, onde ocorrem as etapas de preparação, exploração e limpeza dos dados.
- **Pré-treino:** O pré-treinamento normalmente é opcional, pois ao criar LLMs, muitas vezes se usa um modelo que já foi pré-treinado.
- **Ajuste fino do modelo e engenharia de *prompt*:** É aqui que se melhora o desempenho do modelo em relação à uma tarefa específica, seja coletando novos dados para treinar (ajuste fino) ou solicitando o modelo com mais precisão (engenharia *prompt*).
- **Avaliação e depuração de modelo:** Depois de ter um modelo, é preciso avaliar sua qualidade. A avaliação do modelo pode ser realizada de diferentes maneiras. Isso inclui abordar e testar diferentes táticas e métricas para avaliar a qualidade, bem como estratégias para testar a segurança do modelo.
- **Implantação do modelo:** Essa etapa costumava ser um dos maiores bloqueadores em projetos de ML, mas atualmente existem muitas soluções que facilitam a implantação de um modelo em produção.
- **Monitoramento e manutenção do modelo:** Depois que um modelo é implantado, o trabalho árduo começa. Em uma solução de LLM, o processo de monitoramento é mais detalhado e profundo do que em um modelo de ML tradicional. Por exemplo, para implementação de LLM em um aplicativo, o monitoramento pode envolver rastrear os usuários, analisar como eles usam o aplicativo, quão consistente é o desempenho do modelo, etc. Além disso, pode-se coletar feedback direto do usuário, se possível, para ajudar a ajustar o modelo posteriormente. E, claro, deve-se monitorar preocupações clássicas de MLOps, como desvio de modelo e latência.

## Ferramentas de LLMOps

Para cada uma dessas etapas existem diversas ferramentas que fornecem funcionalidades diferentes que permitem trabalhar com grandes modelos de linguagem de uma maneira muito mais fluida e flexivel. Essas ferramentas vão desde os grandes fornecedores de modelos de linguagem, como OpenAI e Anthropic,  ferramentas de gerenciamento de experimentos e modelos como o Comet e Mlflow, frameworks, como Hugging Face, Langchain, Griptape e muitos outros, e, por fim, serviços de “infraestrutura, como Databricks, Azure e Snowflake, dentre outros.

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-03-28-one-dose-of-data-science/LLMOps_tools.jpeg?raw=true)


## LLMOps vs. MLOps

Existem muitas sobreposições entre os conceitos presentes nos LLMOps e os que também aparecem nos MLOps (*Machine Learning Operations*), como depuração, manutenção de modelo, rastreamento de experimentos e muito mais. Isso ocorre porque LLMOps é uma subárea do MLOps. Na verdade, a maneira mais útil de conceituar LLMOps é como uma porção muito específica de MLOps, aplicando-se especificamente a grandes modelos de linguagem.

Mas existem diferenças. As considerações que se faz nos LLMOps serão diferentes do restante dos MLOps de várias maneiras:

- **Requisitos de dados:** Grandes modelos de linguagem, especialmente se você mesmo estiver conduzindo um pré-treinamento, exigem grandes quantidades de dados e computação.
- **Experimentação:** É provável que um “experimento” em LLMOps inclua coisas como sua estratégia de *prompt* ou cadeias de inferência, não apenas dados sobre suas execuções de treinamento.
- **Estratégias de avaliação:** É notoriamente difícil avaliar LLMs. Suas tarefas costumam ser muito gerais para testes discretos e, simplesmente, os modelos são muito poderosos para muitos dos *benchmarks* simples que costuma-se usar para modelos de ML tradicionais.
- **Custos e latência:** Relacionado ao primeiro ponto sobre requisitos de dados, os LLMs são caros. Os custos de computação por si só podem ser exorbitantes, especialmente quando se deseja latência rápida. Este não é um problema fácil de resolver e requer muita estratégia em torno da infraestrutura.

# Referências

- [Databricks - LLMOps](https://www.databricks.com/glossary/llmops)
- [Udacity - LLMOps: Building Real-World Applications With Large Language Models](https://learn.udacity.com/paid-courses/cd13455)
- [IBM - What are large language model operations (LLMOps)?](https://www.ibm.com/topics/llmops)