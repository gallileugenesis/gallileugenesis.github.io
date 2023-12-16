---
title:  "Projeto de Sumarizador de Texto com GPT"
date:   2023-12-15 12:00:00 -500
categories: [Projetos]
tags: [data science, machine learning, NLP, LLM]
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-12-16-text-summarization-with-gpt/header_image.png?raw=true)


**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/text-summarization-with-gpt)


Bem-vindo ao nosso blog técnico onde exploramos as últimas inovações no campo da ciência de dados e aprendizado de máquina. Hoje, vamos compartilhar o projeto de um sumarizador de texto alimentado pelos mais avançados modelos de linguagem natural da OpenAI, o GPT-3.5 e GPT-4.

## Visão Geral do Projeto

O objetivo deste projeto é criar uma ferramenta que possa resumir textos extensos de maneira eficiente e eficaz, utilizando a capacidade de compreensão e geração de texto dos modelos GPT. O projeto é composto por duas partes principais:

- **Backend (`summarizer_model.py`)**: Integração com a API da OpenAI para processamento e sumarização de texto.
- **Frontend (`summarizer_app.py`)**: Uma interface web construída com Streamlit, oferecendo aos usuários uma forma interativa de usar a ferramenta.

### A Interface do Sumarizador

![Interface do Sumarizador](link_para_imagem_da_interface)

A interface do usuário foi desenhada para ser intuitiva e fácil de usar. Os usuários podem escolher entre diferentes modelos de linguagem, ajustar parâmetros de sumarização e inserir textos de várias maneiras.

## Funcionalidades Detalhadas

### Escolha de Modelos

Os usuários têm a opção de escolher entre diferentes modelos de linguagem, como GPT-3.5 e GPT-4, adaptando-se às suas necessidades específicas de sumarização.

![Seleção de Modelos](link_para_imagem_selecao_modelos)

### Personalização de Sumarização

A ferramenta permite que os usuários definam o estilo de sumarização baseado no tipo de pessoa (cientista, estudante, etc.), tornando os resumos mais personalizados.

![Estilos de Sumarização](link_para_imagem_estilos_sumarizacao)

### Ajuste de Hiperparâmetros

Os usuários podem ajustar hiperparâmetros como tokens, temperatura, nucleus sampling e frequência de penalidade para refinar os resultados da sumarização.

![Ajuste de Hiperparâmetros](link_para_imagem_ajuste_hiperparametros)

### Entrada de Texto e Upload de Arquivos

Além de digitar o texto diretamente, os usuários podem carregar arquivos em formatos como .txt, .pdf e .docx para sumarização.

![Upload de Arquivos](link_para_imagem_upload_arquivos)

## Como o Sumarizador Funciona

1. **Entrada de Texto**: O usuário insere o texto ou carrega um arquivo.
2. **Processamento**: O backend envia o texto para o modelo GPT escolhido, processando-o de acordo com os parâmetros definidos.
3. **Geração de Sumário**: O modelo gera um sumário que é então exibido na interface.

## Tecnologias Utilizadas

- **Python**: Linguagem de programação para o desenvolvimento do backend e frontend.
- **OpenAI API**: Para acessar os modelos GPT.
- **Streamlit**: Para construir a interface web interativa.
- **python-dotenv**: Para gerenciar variáveis de ambiente.

## Conclusão

Este projeto ilustra a potência dos modelos de linguagem modernos na tarefa de sumarização de texto. Com sua interface fácil de usar e a capacidade de personalização, ele representa um grande avanço para profissionais que precisam de resumos rápidos e eficientes de grandes volumes de texto.

