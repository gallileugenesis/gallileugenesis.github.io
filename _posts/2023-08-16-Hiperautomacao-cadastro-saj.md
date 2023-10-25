---
title:  "Cadastro automático de processos no SAJ"
date:   2023-08-16 12:00:00 -500
categories: [Projetos]
tags: [hiperautomação, sistemas especialistas ]
---


Este projeto teve como objetivo automatizar o processo de cadastro de processos no [SAJ](https://www.softplan.com.br/produto/saj-tribunais/). O cadastro dos processos é realizado mediante o preenchimento automático de campos predeterminados, com valores conhecidos e campos cujas informações necessitam ser extraídas de documentos específicos. Além disso, os referidos documentos têm que ser anexados ao sistema para completar o cadastro.

A identificação dos documentos foi um desafio, uma vez que não havia dados rotulados para treinar um modelo de classificação supervisionado. A solução foi desenvolver um sistema especialista, baseado em regras. Basicamente, o sistema é suportado por diversas ferramentas de processamento de linguagem natural e um conjunto de regras composto por palavras-chaves e seus respectivos pesos, os quais podem ser negativos ou positivos.   

Já a automação do cadastro foi realizada com o [UiPath](https://www.uipath.com/).

O resultado final é um processo de [hiperautomação](https://www.sap.com/brazil/products/technology-platform/process-automation/what-is-hyperautomation.html#:~:text=A%20hiperautoma%C3%A7%C3%A3o%20refere%2Dse%20ao,poss%C3%ADvel%20%E2%80%93%20o%20mais%20r%C3%A1pido%20poss%C3%ADvel.), como mostra a figura abaixo.

 
<div style="text-align:center;">
  <img src="https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2023-08-16-Hiperautomacao-cadastro-saj/1-Hiperautoma%C3%A7%C3%A3o.png?raw=true" alt="Python Logo" width="600" height="400">
</div>