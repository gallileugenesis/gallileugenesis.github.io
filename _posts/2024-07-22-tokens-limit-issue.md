---
title:  "How to deal with token limit issues in large language models (LLMs)"
date:   2024-07-22 12:00:00 -500
categories: [Blog]
tags: [data science, machine learning, GenAI, NPL, LLM]
layout: post
comments: true
---

![png](https://github.com/gallileugenesis/gallileugenesis.github.io/blob/main/post-img/2024-07-22-tokens-limit-issue/header_image.png?raw=true)

## Introduction 

Every large language model (LLM) has limits on how many tokens it can process for each request, due due to computational constraints, such as memory and processing data. This limit involves the sum of the input and output number of tokens, and it define the model’s context window. The size of the context window impacts the amount of information the model can process and respond to in tasks such as translation, question answering, or text generation.

Different models have different context window sizes, as show the list below:

Open-source models:
- Llama 2 13B: 4.096 tokens
- Mistral 7B: 8.192 tokens
- Mixtral 8x7B: 32.000 tokens

Proprietary models:
- GPT 3.5 Turbo: 4.096 tokens
- GPT 4: 8.192 tokens
- GPT 4 32k: 32.768 tokens
- Gemini 1.5-pro: 2.105.344 tokens (2.097.152 for input and 8.192 for output)

When the model’s context window limit is overflowing, a token limit error occur. This usually happens when the input is too large or if the input plus output are too large.

Before we proceed with the description of some methods to overcome the token limit problem, it is important to understand what tokens are.

## What are tokens?

In the context of natural language processing (NLP) and LLMs, tokens can be described as fragments of language. They are the basic units of text that a model processes to understand and generate text. We tend to imagine that each word represents a token. But for most models this is almost never true. A token can be as short as one character or as long as one word. But tokens can also include trailing spaces, numbers, special characters, punctuation marks, and even sub-words. 

Examples:

- **Word-level tokenization:** "Hello, world!" -> ["Hello", ",", "world", "!"]
- **Subword tokenization:** "unhappiness" -> ["un", "happiness"]
- **Character-level tokenization:** "Hello" -> ["H", "e", "l", "l", "o"]

This means that token generation (tokenization) and tokens counting varies a lot for different languages and different models. For example, for [OpenAI](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) models and English language:

- 1 token ~= 4 chars in English
- 1 token ~= ¾ words
- 100 tokens ~= 75 words
- 1-2 sentence ~= 30 tokens
- 1 paragraph ~= 100 tokens

### Tokenization

Tokenization is the process of converting a string of text into tokens, and each token is assigned a numerical representation, or index, which can be used to feed into a model. Tokenization rules can vary based on the language and the specific NLP model being used. Some models, like Transformer-based models are based on WordPiece or Byte Pair Encoding (BPE), might break down words into smaller subwords or even individual characters to handle rare or unknown words better.

As define in [Mistral documentation](https://docs.mistral.ai/guides/tokenization/), a typical LLM workflow:

1. **Tokenization:** First, tokenize the input text using a tokenizer, assigning each unique token a specific index number from the tokenizer’s vocabulary.

2. **Embedding:** The tokenized text is then passed through the model, which typically includes an embedding layer and transformer blocks. The embedding layer converts the tokens into dense vectors that capture their semantic meanings. For more information, refer to our embedding guide.

3. **Contextual Processing:** Next, the transformer blocks process these embedding vectors to understand the context and generate results.

4. **Decoding:** Finally, in the decoding step, the output tokens are converted back into human-readable text by mapping the tokens to their corresponding words using the tokenizer’s vocabulary.
 
## How to Solve LLM Token Limits?

### Handle the maximum number of tokens (max_tokens) parameter:

The max_tokens parameter controls the maximum number of tokens generated in a single call to the LLM model. It limits the output size. So, the token count of the prompt plus `max_tokens` cannot exceed the model's context length, and you’ll get a tokens limit error.

Setting a suitable value for max_tokens can help avoid some (but not all) token limit errors. Let’s look at some examples of how this can be done:
- **Set max_tokens as None:** When set max_tokens as None, means that there is no explicit limit on the number of tokens the model can generate in its response.This leads to greater flexibility in the number of input tokens: if the number of input tokens is greater, there will be fewer tokens left for the model's response, and vice versa. 
- **Set max_tokens dynamically:** Create a function that counts the number of tokens in the input and, based on the model's context window, defines max_tokens. This alternative optimizes the model's response length.
- **Set a specically value to max_tokens:** This is the strictest option, as it previously defines what the maximum length of the model's response should be. It also restricts the maximum number of input tokens. However, if you have sufficient knowledge of the business and the model's behavior, it can be a good alternative.

**Note:** It is important to note that these options do not definitively prevent the token limit error. They give the model more flexibility to manage the number of response tokens, and thus prevent this error from occurring when it could be avoided.

### Effective prompt token management:

Keeping prompts concise and prioritizing essential information is an effective approach to minimizing the number of tokens submitted to the LLM. When submitting large amounts of text for analysis, try to include only the crucial and pertinent information that the model requires to process your response.

Let’s look at some good practices that may be useful:

- Using well-known acronyms like LLM instead of large language model.
- Eliminating filler words or phrases, such as "would you," "can you," "please," and "thank you," also aids in reducing token usage.
- Request the model to provide a concise response to prevent exceeding your context limits.
- Additionally, another useful strategy to reduce your prompt token usage is to ask the LLM itself for assistance, asking it to abbreviate your prompt to reduce token usage.

### Multiple-prompt approach:

For complex use cases where different types of input are expected and need to be handled in different ways, like retrieval different information kinds, apply a multiple-prompt approach can be a good option.

Instead of creating a unic and big prompt to handle all tasks at once, create several targeted prompts, each one designed for optimal effectiveness with one type of taks.

This is a efficient alternative to handle  of the tokens limit, but the processing time mybe a limiter. 

### Use a Model With a Bigger Context Window:
One of the simplest solutions is to find a model with a larger context window. For example, use Mixtral 8x7B (context window of 32.000 tokens) instead Llama 2 13B (context window of 4.096 tokens). 

But be careful, as models with a larger context window will not necessarily be the most suitable for your task.

### Prompt Truncation:
The easiest way to bring the prompt within the token limit is to remove parts from the prompt’s start or end. It is a simple fix, but it comes at the cost of loss of information. The model will not process the truncated prompt and might miss the important context.

Truncation can be done on a character or word level, depending on the requirement. 

### Chunk Processing
An alternative approach to handling lengthy prompt bodies involves dividing the text into smaller segments, called chunks. Each chunks is processed separately by the LLM, generating independent outputs. These outputs are then merged to create a single, comprehensive result. However, this method can introduce errors because each chunks only represents a portion of the complete information, and combining the outputs may result in gaps.

This approach is usually used in summarization tasks, but can also be applied in another's NLP task, as Q&A.

#### Summarizing: Map reduce
1. Split Documents into Chunks
2. Get Summary For Each Chunk
3. Get Summary of Summaries ​

- Pros: Scales to larger documents, can be parallelized
- Cons: Many API Calls. Loses information ​

#### Summarizing: Refine
1. Split Documents
2. Get Summary for first chunk
3. Refine the total summary with summary #1 and chunk #2
4. Refine the total summary with summary #N and chunk #N+1 ​

#### Q&A: Map-Rerank
1. Split Documents
2. Get & Rank Answer
3. Return the Top Answer

- Pros: Scales well, better for single-answer questions
- Cons: Cannot combine information between documents ​

### Retrieval Augmented Generation (RAG): 

RAG combines retrieval-based and generative models to enhance the quality and accuracy of text generation. The retrieval model is responsible for fetching relevant information from a large corpus of documents or a database. When given a query or a prompt, the retrieval model searches for and retrieves a set of documents or passages that are most relevant to the query.

Depending of the application, RAG can removes the need for the vast majority of the context in the prompt, decreasing your amount of tokens of input. However, the final results depend greatly on the search relevancy. 

  

https://github.com/lanchuhuong/Langchain-tutorials/blob/main/chains/DocumentChain%20Types.ipynb
https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
https://www.youtube.com/watch?v=f9_BWhCI4Zo
https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-context-length.html?context=wx
https://deepchecks.com/5-approaches-to-solve-llm-token-limits/
