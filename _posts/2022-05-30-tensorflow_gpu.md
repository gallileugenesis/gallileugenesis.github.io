---
title: "Como instalar e configurar o Tensorflow/Keras com suporte para CPU e GPU no Windows"
date: 2022-05-24 12:00:00 -500
categories: [Blog]
tags: [tensorflow, cpu, gpu, deep learning, data science, machine learning]
layout: post
comments: true
---

## Introdução

Em um [artigo anterior](https://gallileugenesis.github.io/2022/CPUvsGPU.html) vimos uma visão geral sobre as CPUs e as GPUs. Nesse artigo, você verá como instalar e configurar o Tensorflow/Keras com suporte para CPU e GPU no Windows.

O processo é simples, mas é preciso ficar atento com a compatibilidade de versões de todos os pacotes e softwares necessários. 

## Pré-requisito: 
É necessário ter instalados na sua máquina: 

- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [cuDNN](https://developer.nvidia.com/cudnn)

<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2022-05-30-tensorflow_gpu/aanconda_prompt.png?raw=true" alt="Imagem do Anaconda Prompt" style="width:600px">
</div>

## 1º passo: baixar e instalar o Anacoda

A melhor opção para fazer esse processo é por meio do ambiente Anaconda. Você pode baixá-lo clicando no link a seguir:

- [Anaconda](https://www.anaconda.com/)

*Nota: Anaconda é uma plataforma de distribuição Python, de código aberto, e com uma série de ferramentas para o desenvolvimento de projetos em inteligência artificial integradas*.

## 2º passo: abra o Anaconda Prompt 

Feito isso, na barra de pesquisa do Windows digite "Anaconda Prompt".

## 3º passo: crie um ambiente Anaconda

Você deve garantir que o TensorFlow tenha a versão do Python compatível. A melhor maneira de fazer isso é criar um ambiente Anaconda. Cada ambiente que você cria pode ter sua própria versão Python, de drivers e bibliotecas Python.

O comando a seguir cria um ambiente chamado "tensorflow" para a versão Python 3.9. Você pode nomeá-lo como quiser. 

> conda create --name tensorflow python=3.9

Para entrar neste ambiente, você deve usar o seguinte comando:

> conda activate tensorflow

Vamos agora adicionar suporte ao Jupyter ao seu novo ambiente.

> conda install -c conda-forge nb_conda

## 4º passo: instale o TensorFlow para GPU e CPU

O comando a seguir instala o TensorFlow para suporte a GPU. Todas as instalações de driver complexas devem ser tratadas por este comando.

> conda install -c anaconda tensorflow-gpu

## 5º passo: registre seu ambiente

O comando a seguir registra seu ambiente "tensorflow". Novamente, certifique-se de "conda ativar" seu novo ambiente "tensorflow".

> python -m ipykernel install --user --name tensorflow --display-name "Python 3.9 (tensorflow)"

## 6º passo: teste seu ambiente

Agora você pode iniciar o notebook Jupyter. Use o seguinte comando.

> jupyter notebook

Em seguida, copie o seguinte código em uma das células do Jupyter Notebook

<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2022-05-30-tensorflow_gpu/jupyter_notebook_code.png?raw=true" alt="Código para teste do ambiente" style="width:600px">
</div>

Muito obrigado por ler esse artigo. 

Caso tenha interesse, você pode me encontrar no [GitHub](https://github.com/gallileugenesis) e [LinkedIn](https://www.linkedin.com/in/gallileugenesis/).
