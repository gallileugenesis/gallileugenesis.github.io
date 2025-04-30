---
title:  "10 motivos para usar engenharia de prompt em vez de fine-tuning"
date:   2025-02-04 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, prompt engineering, fine-tuning, LLM]
layout: post
comments: true
---

<!-- ![png](https://github.com/gallileugenesis/gallileugenesis.github.io
/blob/main/post-img/2025-02-04-Prompting-vs-fine-tuning/header_image.png?raw=true) -->

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2025-01-27-standardization-vs-normalization/header_image.jpeg?raw=true)


Projetos baseados em LLMs passaram de exóticos aos queridinhos do mercado. E não por acaso. Eu costumo dizer que os LLMs tem o maior poder disruptivo dentro do universo da inteligência artificial, pois já tem grande parte da infraestrutura construída pelos grandes players do mercado e também porque consegue resolver uma série de problemas do mundo real de forma assustadoramente eficiente e, principalmente, adaptativa e customizada. 

Dado isto, eu tenho ouvido muito o termo "fine-tuning" nas discussões técnicas sobre para qual rumo os projetos devem seguir. Acredito que a maioria das pessoas não tem noção do quão complexo é fazer um fine-tuning. Quero dizer, fazer um fine-tuning corretamente.

Outro dia, numa dessas discussões, me perguntaram "quando devemos fazer fine-tuning?". Minha resposta foi categórica e talvez frustrante para a audiência: quando nada mais funcionar.

A engenharia de prompt é muito mais rápida do que outros métodos de controle de comportamento do modelo, como RAG e o próprio fine-tuning, e pode frequentemente gerar ganhos de desempenho em muito menos tempo.Eu trouxe aqui 10 pontos que ajudam a justificar meu ponto de vista:

1. **Eficiência de recursos:** o fine-tuning requer GPUs de ponta e grande memória, enquanto a engenharia de prompt precisa apenas de entrada de texto, o que a torna muito mais amigável aos recursos.
2. **Custo-benefício:** para serviços de IA baseados em nuvem, o fine-tuning incorre em custos significativos. A engenharia de prompt usa o modelo base, que normalmente é mais barato.
3. **Manutenção de atualizações do modelo:** quando os provedores atualizam modelos, as versões com fine-tuning podem precisar de retreinamento. Os prompts geralmente funcionam em todas as versões sem alterações.
4. **Economia de tempo:** o fine-tuning pode levar horas ou até dias. Em contraste, a engenharia de prompt fornece resultados quase instantâneos, permitindo uma rápida resolução de problemas.
5. **Necessidades mínimas de dados:** o fine-tuning precisa de dados substanciais específicos da tarefa e rotulados, que podem ser escassos ou caros. Já a engenharia de prompt explora o conhecimento pré-treinado do modelo e pode operar com poucos exemplos no próprio prompt (few-shot) ou até sem exemplos explícitos (zero-shot).
6. **Flexibilidade e iteração rápida:** com a engenharia de prompt se pode experimentar várias abordagens rapidamente, ajustar os prompts e ver resultados imediatamente. Essa experimentação rápida é difícil com o fine-tuning.
7. **Adaptação de domínio:** optando pela engenharia de prompt pode-se adaptar facilmente os modelos a novos domínios, fornecendo contexto específico do domínio em prompts, sem retreinamento.
8. **Melhorias de compreensão:** a engenharia de prompt é muito mais eficaz do que o fine-tuning para ajudar os modelos a entender e utilizar melhor o conteúdo externo, como documentos recuperados.
9. **Preserva o conhecimento geral:** o fine-tuning corre o risco de esquecimento catastrófico, onde o modelo perde o conhecimento geral. A engenharia de prompt mantém os amplos recursos do modelo.
10. **Transparência:** os prompts são legíveis por humanos, mostrando exatamente quais informações o modelo recebe. Essa transparência auxilia na compreensão e depuração.