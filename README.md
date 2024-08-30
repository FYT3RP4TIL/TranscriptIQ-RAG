# Transcript-IQ RAG (Retrieval Augmented Generation)

# RAG Project with OpenRouter, Pinecone, and OpenAI

This project implements a Retrieval-Augmented Generation (RAG) system using OpenRouter for model access, Pinecone for vector storage, and OpenAI for embeddings. The project is set up in Google Colab for easy access and execution.

## Table of Contents
1. Prerequisites
2. Setting Up API Keys
3. Pinecone Database Setup
4. Using Different Models for RAG

## Prerequisites

- Google account (for accessing Google Colab)
- OpenRouter account
- Pinecone account
- OpenAI account

## Setting Up API Keys

All these keys can also be setup in colab using key option and adding specified keys in the notebook.

### OpenRouter API Key

1. Go to [OpenRouter](https://openrouter.ai/) and create an account if you haven't already.
2. Navigate to your account settings or API section.
3. Generate a new API key.
4. In your Colab notebook, add the following code:

```python
import os
os.environ["OPENROUTER_API_KEY"] = "your_openrouter_api_key_here"
```

### Pinecone API Key

1. Log in to your [Pinecone console](https://www.pinecone.io/).
2. Create a new project if you haven't already.
3. In the API Keys section, create a new API key.
4. In your Colab notebook, add:

```python
os.environ["PINECONE_API_KEY"] = "your_pinecone_api_key_here"
os.environ["PINECONE_ENVIRONMENT"] = "your_pinecone_environment_here"
```

### OpenAI API Key

1. Go to [OpenAI](https://openai.com/) and sign up or log in.
2. Navigate to the API section in your account.
3. Generate a new API key.
4. In your Colab notebook, add:

```python
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
```

## Pinecone Database Setup

1. In your Pinecone console, create a new index:

```python
import pinecone

pinecone.init(api_key=os.environ["PINECONE_API_KEY"], 
              environment=os.environ["PINECONE_ENVIRONMENT"])

pinecone.create_index("your_index_name", dimension=1536, metric="cosine")
```

2. Connect to your index:

```python
index = pinecone.Index("your_index_name")
```

3. To add vectors to your index:

```python
index.upsert(vectors=[("id1", [1.0, 2.0, 3.0, ...], {"metadata": "value"})])
```

## Using Different Models for RAG

This project supports various models through OpenRouter. Here's how you can use different models:

1. Import the necessary libraries:

```python
from langchain.llms import OpenRouter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
```

2. Set up the OpenRouter LLM:

```python
llm = OpenRouter(model="openai/gpt-3.5-turbo")  # You can change the model here
```

3. Set up embeddings and vector store:

```python
embeddings = OpenAIEmbeddings()
vectorstore = Pinecone(index, embeddings.embed_query, "text")
```

4. Create a RetrievalQA chain:

```python
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
```

5. Use the chain for questions:

```python
question = "Your question here"
answer = qa.run(question)
print(answer)
```

To use a different model, simply change the `model` parameter when initializing the OpenRouter LLM. For example:

```python
llm = OpenRouter(model="anthropic/claude-2")
```

Available models may change over time, so check the OpenRouter documentation for the most up-to-date list of supported models.

Remember to adjust your prompts and parameters based on the specific requirements and capabilities of each model.
