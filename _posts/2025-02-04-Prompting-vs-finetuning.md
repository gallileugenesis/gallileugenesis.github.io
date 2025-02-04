---
title:  "10 motivos para usar engenharia de prompt em vez de finetuning"
date:   2025-02-04 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, prompt engineering, finetuning, LLM]
layout: post
comments: true
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2025-02-04-Prompting-vs-finetuning/header_image.jpeg?raw=true)


A engenharia de prompt é muito mais rápida do que outros métodos de controle de comportamento do modelo, como finetuning, e pode frequentemente gerar ganhos de desempenho em muito menos tempo. Aqui estão 10 motivos para considerar a engenharia de prompt em vez do finetuning:

1. **Eficiência de recursos:** o finetuning requer GPUs de ponta e grande memória, enquanto a engenharia de prompt precisa apenas de entrada de texto, o que a torna muito mais amigável aos recursos.
2. **Custo-benefício:** para serviços de IA baseados em nuvem, o finetuning incorre em custos significativos. A engenharia de prompt usa o modelo base, que normalmente é mais barato.
3. **Manutenção de atualizações do modelo:** quando os provedores atualizam modelos, as versões com finetuning podem precisar de retreinamento. Os prompts geralmente funcionam em todas as versões sem alterações.
4. **Economia de tempo:** o finetuning pode levar horas ou até dias. Em contraste, a engenharia de prompt fornece resultados quase instantâneos, permitindo uma rápida resolução de problemas.
5. **Necessidades mínimas de dados:** o finetuning precisa de dados substanciais específicos da tarefa e rotulados, que podem ser escassos ou caros. Já a engenharia de prompt explora o conhecimento pré-treinado do modelo e pode operar com poucos exemplos no próprio prompt (few-shot) ou até sem exemplos explícitos (zero-shot).
6. **Flexibilidade e iteração rápida:** com a engenharia de prompt se pode experimentar várias abordagens rapidamente, ajustar os prompts e ver resultados imediatamente. Essa experimentação rápida é difícil com o finetuning.
7. **Adaptação de domínio:** optando pela engenharia de prompt pode-se adaptar facilmente os modelos a novos domínios, fornecendo contexto específico do domínio em prompts, sem retreinamento.
8. **Melhorias de compreensão:** a engenharia de prompt é muito mais eficaz do que o finetuning para ajudar os modelos a entender e utilizar melhor o conteúdo externo, como documentos recuperados.
9. **Preserva o conhecimento geral:** o finetuning corre o risco de esquecimento catastrófico, onde o modelo perde o conhecimento geral. A engenharia de prompt mantém os amplos recursos do modelo.
10. **Transparência:** os prompts são legíveis por humanos, mostrando exatamente quais informações o modelo recebe. Essa transparência auxilia na compreensão e depuração.