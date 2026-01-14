# university-based-llm-chatbot-with-rag-pipeline


This project is a university-based LLM chatbot built as a web application using a dual RAG architecture.
We started by scraping the entire official university website using BeautifulSoup, collecting unstructured data from multiple pages.
This raw corpus was then preprocessed by removing HTML tags, Unicode characters, duplicates, and irrelevant content, resulting in a clean dataset.
<br>
This polished corpus was used in two pipelines:
<br>
Pipeline 1: LLM Training
<br>

We trained a TinyLlama model using causal language modeling (CLM) on the cleaned university corpus.
Although the model learned the knowledge, it struggled to respond conversationally.
To fix this, we applied instruction-based fine-tuning using curated Q&A pairs, which taught the model how to answer user queries properly.
<br>

Pipeline 2: Dual RAG Retrieval
<br>

The same cleaned corpus was converted into embeddings and stored in two vector databases:
<br>

Qdrant → used for normal website-based and general university queries
<br>

Pinecone → used specifically for PDF-based queries such as regulations, syllabi, and official documents
<br>

During inference, the system decides which retriever to use based on query type.
The retrieved context is then passed to our custom fine-tuned CLM model, which acts as the generator.
<br>

This dual RAG setup significantly improved factual accuracy and reduced hallucinations compared to a standalone LLM.”
