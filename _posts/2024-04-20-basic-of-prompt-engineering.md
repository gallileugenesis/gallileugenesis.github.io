---
title:  "Noções básicas de engenharia de prompt"
date:   2024-04-20 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, prompt engineering, GenAI, NPL, LLM]
layout: post
comments: true
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-04-20-basic-of-prompt-engineering/image.png?raw=true)

## Introdução

Quando se está construindo soluções baseadas em LLMs (*Large Language Models*), uma das principais etapas é a engenharia de *prompt* (*prompt engineering*), que consiste na criação cuidadosa de instruções ou de perguntas (*prompts*) para os grandes modelos de linguagem, como GPT (*Generative Pretrained Transformer*) da OpenAI ou Gemini, do Google. O principal objetivo da engenharia de *prompt* é maximizar a eficácia e a precisão dos modelos ao executar suas tarefas, seja responder perguntas ou gerar conteúdo, por exemplo. Esta é uma das formas mais eficazes de melhorar a qualidade dos resultados dos LLMs e é particularmente relevante em modelos que dependem de interações baseadas em texto para entender e executar solicitações dos usuários.

Mas, a engenharia de *prompt* não se resume a busca por melhorar a qualidade das respostas dos modelos. Ela pode ser usada para controlar adequadamente suas saídas, seja quanto ao seu formato/padrão, ou com relação às políticas e regras da organização, reduzir o custo/latência das respostas, etc.

Abaixo está uma lista de melhores práticas para se realizar a tarefa de engenharia de *prompt* de maneira eficaz e desenvolver bons *prompts* para seus modelos: 

1. **Especificidade e clareza:** Evite instruções ambíguas. Seja o mais explícito possível em suas solicitações para garantir resultados precisos e relevantes.
2. **Use delimitadores:** Os delimitadores ajudam a estruturar e esclarecer os *prompts*, especialmente ao lidar com múltiplas seções de informações.
3. **Especifique o formato de saída:** Indique explicitamente o formato desejado da saída, seja texto livre, markdown, HTML ou um formato estruturado, como JSON.
4. **Pense passo a passo:** Instruir o modelo a raciocinar passo a passo sobre um problema pode melhorar o desempenho, especialmente em tarefas que envolvem raciocínio complexo.
5. **Interpretação de papéis (*Role-Playing*):** Implementar *role-playing* em *prompts*, onde o modelo assume um papel específico, como um assistente de pesquisa, pode melhorar a qualidade e a relevância das respostas.

Além disso, diversas técnicas avançadas de engenharia de prompts foram desenvolvidas para aprimorar ainda mais este processo, incluindo:

- **Encadeamento de *prompts*:** Em tarefas complexas o LLM pode ter dificuldade em resolver se for solicitado com um único prompt muito detalhado. Uma alternativa que pode melhorar significativamente o seu desempenho é dividir a tarefa em várias subtarefas mais simples, onde o LLM recebe uma subtarefa e sua resposta é usada como entrada para outro prompt, criando uma cadeia de operações de prompts.
- **Aprendizado de poucos exemplos em contexto (*Few-shot in-context learning*):** Esta técnica envolve fornecer ao LLM alguns exemplos do tipo de resposta desejada dentro do *prompt*, ajudando o modelo a entender melhor o contexto e a produzir resultados semelhantes.
- **Cadeia de Pensamento (*Chain-of-Thought*):** Com essa abordagem, os *prompts* são elaborados para encorajar o LLM a explicar o raciocínio por trás de suas respostas, passo a passo, como se estivesse "pensando alto".
- ***ReAct*:** Uma técnica que foca na reação do LLM a diferentes estímulos ou instruções dentro do *prompt*, ajustando a abordagem com base na resposta obtida.

Abaixo são apresentados exemplos para alguns dos pontos listados, usando o modelo gpt-3.5-turbo da OpenAI.


```python
from openai import OpenAI
import IPython

import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```


```python
# Função para gerar a resposta do modelo
def get_completion(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=300):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content
```

## 1 - Seja específico e claro
Escreva instruções tão claras e específicas quanto possível para obter os comportamentos desejados do LLM:


```python
menu = [
    "Feijoada Completa",
    "Ratatouille",
    "Paella Valenciana",
    "Sushi",
    "Pasta Carbonara",
    "Tom Yum Goong",
    "Churrasco",
    "Moussaka",
    "Biryani",
    "Tacos"
]

system_message = """
Sua tarefa é recomendar um prato para um cliente.

Você é responsável por recomendar um prato do nosso menu {menu}.

Você deve evitar pedir preferências ao usuário e não solicitar informações pessoais.

"""

user_request = """
Por favor, me recomende um prato do seu menu.
"""

message = [
    {
        "role": "system",
        "content": system_message.format(menu=menu)
    },
    {
        "role": "user",
        "content": user_request
    }
]

response = get_completion(message)
print(response)

```

    Recomendo experimentar a "Paella Valenciana". É um prato espanhol delicioso, feito com arroz, frutos do mar, frango, legumes e temperos especiais. Tenho certeza de que você vai adorar!
    

Quanto mais específico for o comportamento desejado do modelo, mais específicas deverão ser as instruções e a lógica. Abaixo está um exemplo em que o cliente fornece informações sobre seus gostos:


```python
menu = [
    "Feijoada Completa",
    "Ratatouille",
    "Paella Valenciana",
    "Sushi",
    "Pasta Carbonara",
    "Tom Yum Goong",
    "Churrasco",
    "Moussaka",
    "Biryani",
    "Tacos"
]


system_message = """
Sua tarefa é recomendar um prato para um cliente.

Você é responsável por recomendar um prato do nosso menu {menu}.

Você deve evitar pedir preferências ao usuário e não solicitar informações pessoais.

"""

user_request = """
Eu adoro comida italiana. Por favor, me recomende um prato do seu menu.
"""

message = [
    {
        "role": "system",
        "content": system_message.format(menu=menu)
    },
    {
        "role": "user",
        "content": user_request
    }
]

response = get_completion(message)
print(response)

```

    Com base em seus gostos por comida italiana, eu recomendaria o prato Pasta Carbonara do nosso menu. É um prato clássico e delicioso que combina massa, ovos, queijo parmesão, pancetta e pimenta preta. Tenho certeza de que você vai adorar!
    

## 2 - Adicionar delimitadores
Adicionar delimitadores ajuda a estruturar melhor as instruções e os componentes gerais do *prompt*. Isso é benéfico para obter respostas mais confiáveis.


```python
prompt = """
Transforme o bloco de código apresentado na seção #### <code> #### para a linguagem Python:

####
strings2.push("um")
strings2.push("DOIS")
strings2.push("7")
strings2.push("4")
####
"""

message = [
    {
        "role": "user",
        "content": prompt
    }
]

IPython.display.Markdown("```python" + get_completion(message) + "\n```")
```




```pythonstrings2.append("um")
strings2.append("DOIS")
strings2.append("7")
strings2.append("4")
```



## 3 - Especifique o formato de saída
Se o formato das respostas for importante, isso deverá ser explicitamente declarado no *prompt* para obter os resultados desejados. No exemplo a seguir, gostaríamos de exportar os resultados como um objeto JSON.


```python
prompt = """
Sua tarefa é: dada a descrição do produto, retornar as informações solicitadas na seção delimitada por ### ###.
Formate a saída como um objeto JSON.

Descrição do produto: {description}

###
product_name: o nome do produto
product_brand: o nome da marca (se houver)
###
"""

description = """
Apresentando o Nike Air Max 270 React: um tênis confortável e estiloso que combina duas das melhores tecnologias
da Nike. Com um design preto elegante e uma sola em bolha exclusiva, esses sapatos são perfeitos
para o uso diário.
"""

message = [
    {
        "role": "user",
        "content": prompt.format(description=description)
    }
]

print(get_completion(message))
```

    {
        "product_name": "Nike Air Max 270 React",
        "product_brand": "Nike"
    }
    

## 4 – Pense passo a passo
Para obter raciocínio em LLMs, você pode fazer com que o modelo pense passo a passo. Solicitar ao modelo dessa forma permite que ele forneça as etapas detalhadas antes de fornecer uma resposta final que resolva o problema.


```python
prompt = """
Os números ímpares neste grupo, 15, 32, 5, 13, 82, 7, 1, somam um número par?

Resolva dividindo o problema em etapas. Identifique os números ímpares, depois some-os e, por fim, indique se
o resultado é ímpar ou par.
"""

messages = [
    {
        "role": "system",
        "content": prompt
    }
]

response= get_completion(messages)

print(response)
```

    1. Identificar os números ímpares no grupo: 15, 32, 5, 13, 82, 7, 1
       Números ímpares: 15, 5, 13, 7, 1
    
    2. Somar os números ímpares: 15 + 5 + 13 + 7 + 1 = 41
    
    3. Verificar se a soma dos números ímpares é par ou ímpar:
       41 é um número ímpar
    
    Portanto, a soma dos números ímpares neste grupo é um número ímpar.
    

## 5 - Interpretação de papéis
O exemplo abaixo mostra como aplicar a interpretação de papéis. Nesse caso, o modelo foi induzido a assumir o papel de um assistente de pesquisa, com o objetivo de responder às questões científicas dos usuários. Você pode também combinar diferentes mensagens para imitar ou iniciar o comportamento que deseja ou espera do modelo.


```python
system_message = """
Você é um assistente de pesquisa experiente. Suas respostas dever ter sempre um tom técnico e científico.
Estruture suas respostas em formato markdown. 
"""

prompt = """
Você pode me contar sobre a criação de buracos negros?
"""

messages = [
    {
        "role": "system",
        "content": system_message
    },

    {
        "role": "user",
        "content": prompt
    }
]

response = get_completion(messages)
print(response)
```

    ### Criação de Buracos Negros
    
    Os buracos negros são formados a partir do colapso gravitacional de uma estrela massiva no final de sua vida. Quando uma estrela esgota seu combustível nuclear, a pressão gerada pela fusão nuclear não consegue mais contrabalancear a força da gravidade, levando a um colapso gravitacional.
    
    Durante o colapso, a estrela massiva pode se tornar uma supernova, liberando uma quantidade imensa de energia. Se a massa remanescente após a explosão for grande o suficiente, a matéria restante pode colapsar ainda mais, formando um buraco negro.
    
    O buraco negro é caracterizado por uma região de espaço-tempo onde a gravidade é tão intensa que nem mesmo a luz consegue escapar, formando assim o horizonte de eventos. Toda a massa da estrela colapsada fica concentrada em um ponto de densidade infinita, chamado de singularidade.
    
    Os buracos negros são objetos extremamente fascinantes e desempenham um papel fundamental na astrofísica e na compreensão do universo. Suas propriedades únicas desafiam nossa compreensão da física e da natureza do espaço-tempo.
    
**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/prompt-engineering/blob/main/basic-of-prompt-engineering.ipynb)

Caso tenha interesse, você pode me encontrar no [GitHub](https://github.com/gallileugenesis) e [LinkedIn](https://www.linkedin.com/in/gallileugenesis/).