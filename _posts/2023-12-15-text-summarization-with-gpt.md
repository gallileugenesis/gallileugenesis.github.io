---
title:  "Projeto de Sumarizador de Texto com GPT"
date:   2023-12-15 12:00:00 -500
categories: [Projetos]
tags: [data science, machine learning, NLP, LLM]
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-12-15-text-summarization-with-gpt/header_image.png?raw=true)


**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/text-summarization-with-gpt)


Bem-vindo ao nosso blog técnico onde exploramos temos como inovação e técnologia nos campos da ciência de dados e aprendizado de máquina. Hoje, vamos compartilhar o projeto de um sumarizador de texto alimentado pelos mais avançados modelos de linguagem natural da OpenAI, o GPT-3.5 e GPT-4.

## Visão Geral do Projeto

O objetivo deste projeto é criar uma ferramenta que possa resumir textos extensos de maneira eficiente e eficaz, utilizando a capacidade de compreensão e geração de texto dos modelos GPT. O projeto é composto por duas partes principais:

- **Backend (`summarizer_model.py`)**: Integração com a API da OpenAI para processamento e sumarização de texto.
- **Frontend (`summarizer_app.py`)**: Uma interface web construída com Streamlit, oferecendo aos usuários uma forma interativa de usar a ferramenta.

### Backend

O backend deste projeto é construído em Python e faz uso da API da OpenAI para acessar os modelos GPT. O script summarizer_model.py é responsável por configurar a conexão com a API e definir a função de sumarização. Veja abaixo o código-chave do backend:

```python

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_summarizer(
    model,
    max_tokens,
    top_p,
    frequency_penalty,
    temperature,
    prompt,
    person_type,
):
    chat = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        messages=[
         {"role": "system", "content": "You are a helpful assistant for text summarization."},
         {"role": "user", "content": f"Summarize this for a {person_type}: {prompt}"},
        ],
    )
    return chat.choices[0].message.content
```

A função *generate_summarizer* utiliza vários parâmetros para gerar resumos usando a API da OpenAI. Cada parâmetro tem um propósito específico para controlar como a geração de texto é realizada:

- **model:** Este parâmetro especifica qual modelo de linguagem da OpenAI será usado para gerar o texto. Modelos diferentes podem incluir versões como "gpt-3.5-turbo" ou "gpt-4", que se referem às versões e capacidades do modelo de linguagem da OpenAI.

- **max_tokens:** Define o número máximo de tokens (palavras ou peças de palavras) que podem ser gerados pelo modelo. Um "token" pode ser uma palavra inteira ou parte de uma palavra, por isso não é exatamente o mesmo que a contagem de palavras. Este limite ajuda a controlar o comprimento do texto de saída.

- **top_p (Nucleus Sampling):** É um método de amostragem que considera apenas as previsões mais prováveis do modelo. O valor de top_p define o limite para essas previsões. Por exemplo, um top_p de 0.1 significa que apenas as previsões que constituem os 10% superiores em termos de probabilidade serão consideradas para a geração do texto.

- **frequency_penalty:** Este parâmetro ajuda a evitar a repetição. Um valor positivo desencoraja a repetição de palavras ou frases já usadas, enquanto um valor negativo pode encorajar a repetição. Um valor de 0 significa que não há penalidade de frequência aplicada.

- **temperature:** Este parâmetro controla o nível de aleatoriedade ou criatividade na resposta do modelo. Uma temperatura mais alta (próxima de 1) resulta em respostas mais variadas e criativas, enquanto uma temperatura mais baixa (próxima de 0) gera respostas mais previsíveis e conservadoras.

- **prompt:** O texto de entrada que serve como ponto de partida para o modelo gerar o resumo. Este é o texto que você deseja resumir.

- **person_type:** Este parâmetro é usado para personalizar a saída do modelo com base no tipo de pessoa para quem o resumo é destinado, como "cientista", "estudante", etc. Isso é útil para ajustar o estilo e o nível de detalhe do resumo de acordo com o público-alvo.

Esses parâmetros permitem uma grande flexibilidade e personalização no processo de geração de texto, tornando a ferramenta adaptável a uma variedade de necessidades e contextos de sumarização.

**Obs:** Note que para utilizar o modelo é necessário a chave de API fornecida pela OpenAI.

### Frontend

O frontend do projeto foi desenvolvido usando Streamlit, uma biblioteca Python que facilita a criação de aplicações web para análise de dados. Nessa interface os usuários podem escolher entre diferentes modelos de linguagem, ajustar parâmetros de sumarização e inserir textos escritos ou em arquivos.

![Interface do Sumarizador](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-12-15-text-summarization-with-gpt/interface.png?raw=true)

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

