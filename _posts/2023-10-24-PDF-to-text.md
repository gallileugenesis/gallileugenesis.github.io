---
title:  "Como converter um arquivo PDF da web em texto"
date:   2023-10-24 12:00:00 -500
categories: [Blog]
tags: [data science, extração de texto]
---


**Nota:** Todo o código está disponível no [Github](https://github.com/gallileugenesis/PDF-to-text)


A capacidade de converter o conteúdo de um arquivo PDF da web em texto é uma habilidade fundamental para diversas aplicações, incluindo análise de texto, extração de informações e processamento de linguagem natural. Neste tutorial, mostraremos como realizar essa tarefa de forma eficiente usando Python.

## Passo 1: Baixar o arquivo PDF da web
O primeiro passo é baixar o arquivo PDF da web. Podemos usar a biblioteca requests para realizar o download do arquivo a partir de uma URL:


```python
import requests

# url do arquivo
pdf_url = "https://www.ic.unicamp.br/~stolfi/misc/2012-02-13-domine-casmurrus.pdf"

# Solicitação HTTP para baixar o PDF a partir do URL
response = requests.get(pdf_url)

# Abre um arquivo local chamado "Dom-Casmurro.pdf" 
# e escreve o conteúdo do PDF nesse arquivo
with open("Dom-Casmurro.pdf", "wb") as pdf_file:
   pdf_file.write(response.content)
```

## Passo 2: Converter o PDF em texto
Para converter o arquivo PDF em texto utilizável, usamos a biblioteca PyPDF2. A seguir, o código que realiza essa conversão:


```python
import PyPDF2

# Caminho o arquivo pdf
pdf_path = "Dom-Casmurro.pdf"

pdf_text = ""
with open(pdf_path, "rb") as pdf_file:
   # Usamos PyPDF2 para ler o PDF
   pdf_reader = PyPDF2.PdfReader(pdf_file)

   # Iteramos pelas páginas do PDF e extraímos o texto
   for page in pdf_reader.pages:
       pdf_text += page.extract_text()
```

O conteúdo do pdf já está na variável pdf_text


```python
# exibir os primeiros 1000 caracteres do texto 
pdf_text[0:1000]
```




    'Dom Casmurro\nMachado de Assis\n1899\nVers˜ao Preliminar\n2012-02-13\nI\nDo titulo.\nUma noite destas, vindo da cidade para o Engenho\nNovo, encontrei no trem da Central um rapaz aqui do\nbairro, que eu conhe¸ co de vista e de chap´ eo. Comprimentou-\nme, sentou-se ao p´ e de mim, falou da lua e dos ministros,\ne acabou recitando-me versos. A viagem era curta, e\nos versos p´ ode ser que n˜ ao fossem inteiramente maus.\nSuccedeu, por´ em, que como eu estava can¸ cado, fechei os\nolhos tres ou quatro vezes; tanto bastou para que elle\ninterrompesse a leitura e mettesse os versos no bolso.\n— Continue, disse eu accordando.\n— J´ a acabei, murmurou elle.\n— S˜ ao muito bonitos.\nVi-lhe fazer um gesto para tiral-os outra vez do bolso,\nmas n˜ ao passou do gesto; estava amuado. No dia seguinte\nentrou a dizer de mim nomes feios, e acabou alcunhando-\nmeDom Casmurro . Os visinhos, que n˜ ao gostam dos\nmeus h´ abitos reclusos e calados, deram curso ´ a alcu-\nnha, que aﬁnal pegou. Nem por isso me zanguei. Con-\ntei '



**Obs:** O caractere '\n' representa uma quebra de linha. Ele é chamado de caractere de escape de nova linha ou simplesmente um caractere de nova linha.


## Passo 3: Salvar o texto em um arquivo *.txt*

Adicionalmente, você pode querer salvar o texto em um arquivo *.txt*.


```python
# Abrimos um arquivo de texto chamado "Dom-Casmurro.txt" no modo de escrita ('w') # e escrevemos o texto extraído nele
with open("Dom-Casmurro.txt", "w", encoding="utf-8") as txt_file:
   txt_file.write(pdf_text)
```

## Passo 4: Ler o conteúdo do arquivo *.txt*

Para ler o conteúdo do arquivo *.txt* basta executar o código a seguir.


```python
with open("Dom-Casmurro.txt", "r", encoding="utf-8") as txt_file:
   # Lê o texto do arquivo de texto que foi convertido a partir do PDF
   pdf_text = txt_file.read()
```

Agora você tem o conteúdo do PDF convertido em texto, pronto para análises adicionais ou qualquer outra aplicação de processamento de texto que desejar, usando, por exemplo, bibliotecas como o NLTK.
