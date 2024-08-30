# 🧠 Transcript-IQ RAG (Retrieval Augmented Generation)

## 🚀 RAG Project with OpenRouter, Pinecone, and OpenAI

This project implements a Retrieval-Augmented Generation (RAG) system using OpenRouter for model access, Pinecone for vector storage, and OpenAI for embeddings. The project is set up in Google Colab for easy access and execution.

### 🤔 What is RAG?

RAG stands for **Retrieval-Augmented Generation**. It is a framework that combines the strengths of both information retrieval and text generation techniques, particularly in the context of natural language processing (NLP) and machine learning.

### 🛠️ How Does RAG Work?

1. **Retrieval Step**: RAG starts by retrieving relevant documents or information from a large dataset or knowledge base. This is typically done using a retriever model (e.g., dense passage retrieval or traditional search techniques) that identifies the most relevant pieces of information based on a given query.

2. **Generation Step**: Once the relevant information is retrieved, a generative model (such as a transformer-based language model) takes this information as input and generates a coherent and contextually relevant response. The generation model uses the retrieved information to augment its ability to produce accurate and context-aware answers.

### 🌟 Why Use RAG?

- **Improved Accuracy**: By combining retrieval with generation, RAG can produce more accurate and factually grounded responses compared to using a generative model alone.
- **Scalability**: RAG systems can scale to large corpora, enabling them to use vast amounts of knowledge stored in databases, wikis, or other structured/unstructured data sources.
- **Flexibility**: RAG can be used for a variety of tasks, including question answering, summarization, and more, making it a versatile tool in the NLP toolkit.

### 🎯 Applications of RAG

- **Open-Domain Question Answering**: RAG is commonly used to answer questions where the answers are not contained in a predefined dataset but need to be retrieved from a broader knowledge base.
- **Customer Support Chatbots**: RAG models can help in generating responses to customer queries by retrieving relevant information from a company's internal documentation.
- **Content Generation**: RAG can assist in generating content that requires specific facts or references, such as writing reports or articles with up-to-date information.

## 📑 Table of Contents
1. Prerequisites
2. Setting Up API Keys
3. Pinecone Database Setup
4. Using Different Models for RAG
5. Video Example

## 📋 Prerequisites

- Google account (for accessing Google Colab)
- OpenRouter account
- Pinecone account
- OpenAI account

## 🔑 Setting Up API Keys

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

## 💾 Pinecone Database Setup

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

## 🤖 Using Different Models for RAG

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

## 🎥 Video Example - Lex Fridman and Preplixity CEO [Podcast](https://www.youtube.com/watch?v=e-gwvmhyU7A)

![](https://github.com/user-attachments/assets/03691841-d482-41ca-b7b7-a280327643c2)

![](https://github.com/user-attachments/assets/4943ce3d-8f6f-4280-b29a-7b776423ae6d)

### RAG Query 1:

```python
perform_rag("What does Aravind mention about pre-training and why it is important?")
```

**Output:**
```
Aravind Srinivas discusses the importance of pre-training in the development of effective AI models, highlighting its role in creating a foundation of general common sense that is crucial for the model's performance. Here are the key points he mentions about pre-training and its significance:

1. **Foundational Stage**: Pre-training is the stage where the raw scaling on compute happens. It involves training the model on a vast amount of data to develop a general understanding of language and common sense.

2. **General Common Sense**: Without substantial pre-training, the model lacks the baseline common sense necessary for effective reasoning. This foundational knowledge is critical because it equips the model with a broad understanding of language and concepts.

3. **Importance in Combination with Post-Training**: While post-training (which includes supervised fine-tuning and reinforcement learning from human feedback, or RLHF) is essential for refining and controlling the model's outputs, it builds upon the foundation laid during pre-training. The combination of extensive pre-training and targeted post-training is what leads to highly capable and controllable language models.

4. **Scale and Compute**: Aravind emphasizes that the pre-training stage is where the majority of the compute resources are used. This massive computational effort is necessary to build the model's foundational capabilities.

5. **Analogies to Human Learning**: He draws parallels between pre-training and human learning, suggesting that like humans, AI models need to develop a broad base of knowledge before they can effectively reason and respond to specific tasks or queries.

6. **Impact on Model Performance**: The quality and extent of pre-training significantly influence the model's overall performance, affecting its ability to understand context, generate coherent responses, and apply knowledge across various domains.

In summary, Aravind portrays pre-training as a crucial investment in building a model's fundamental capabilities, which then enables more sophisticated and targeted training techniques to produce highly effective AI systems.
```

### RAG Query 2:

```python
perform_rag("What advantages does Perplexity have over other AI companies?")
```

**Output:**
```
Perplexity differentiates itself from other AI companies by focusing on a few unique aspects:

1. **Answer-Centric Approach**: Unlike traditional search engines that display a list of URLs, Perplexity aims to provide direct, Wikipedia-like responses to queries. This method prioritizes giving users direct answers and relevant information over sending them to another webpage. This shifts the UI focus from a list of links to summarized answers, aiming to provide a more streamlined and valuable user experience.

2. **Factual Grounding (RAG - Retrieval-Augmented Generation)**: Perplexity ensures their answers are factually grounded by only generating responses based on documents retrieved from the internet. This principle aims to reduce hallucinations by sticking closely to the retrieved content, enhancing the trustworthiness and accuracy of the information provided.

3. **Knowledge-Centric Mission**: The company's mission goes beyond search and aims to make people smarter by helping them access and understand information more effectively. This focus on knowledge dissemination sets them apart from companies that might prioritize ad revenue or other metrics.

4. **Transparency and Attribution**: Perplexity provides clear attribution for the sources of information used in generating answers. This transparency allows users to verify information and delve deeper into topics if they choose.

5. **Customizable Experience**: Users can adjust the level of detail in responses, allowing for more personalized interactions based on their needs and preferences.

6. **Focus on Reducing Misinformation**: By grounding responses in factual information retrieved from the internet, Perplexity aims to combat the spread of misinformation, which is a growing concern in the AI and information retrieval space.

7. **Continuous Learning and Updating**: The system is designed to continuously update its knowledge base, ensuring that responses reflect the most current information available on the internet.

8. **Efficient Information Processing**: Perplexity's approach aims to save users time by condensing vast amounts of information into concise, relevant answers, potentially improving productivity for information-seeking tasks.

These features collectively position Perplexity as a company focused on revolutionizing how people interact with and consume information, potentially offering a more efficient and trustworthy alternative to traditional search engines and other AI-powered information retrieval systems.
```
