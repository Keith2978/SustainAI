# SustainAI: A Conversational AI for Sustainable Finance

SustainAI is a software-as-a-service (SaaS) platform designed to help financial institutions navigate the complexities of sustainable finance. Using **Retrieval-Augmented Generation (RAG)** and **Large Language Models (LLMs)**, SustainAI provides a powerful AI-powered chatbot that delivers highly contextualized, actionable insights to clients. It integrates domain-specific knowledge bases and offers localized responses, enabling banks to offer personalized and regulatory-compliant sustainable finance advisory services.

---

## Key Features

- **Contextualized Responses**: Provides answers based on the specific knowledge base relevant to the client's region or market.
- **LLM Integration**: Utilizes advanced LLMs (e.g., GPT-4) to generate responses based on the retrieved context.
- **SaaS Model**: No need for in-house AI expertise; banks simply upload documents to curate knowledge for their clientele.
- **Persistent Storage**: Document data is stored persistently using **Render**'s cloud infrastructure for seamless operation, even during periods of inactivity.

---

## Components

This repository contains the **FastAPI backend** for serving the following frontend interfaces:

1. **RAG Chatbot Interface**: An interactive AI chatbot for clients to query sustainable finance-related questions.
2. **Document Upload Interface**: A UI for banks to upload documents to curate a custom knowledge base for their clientele.

The **SustainAI** app is designed for deployment on cloud platforms like **Render** and serves as a backend for client-facing services. The rest of the web application, including its other user-facing features and functionalities, is coded separately.

---

## Installation & Setup

### Prerequisites

1. **Python 3.8+**
2. **FastAPI** 
3. **LangChain**  
4. **FAISS**  
5. **OpenAI API Key** (For LLM usage)

### Steps to Get Started:

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/your-repository/sustainai.git
    cd sustainai
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your OpenAI API Key:
    ```plaintext
    OPENAI_API_KEY=your-openai-api-key
    ```

4. **Run the Application**:
    ```bash
    uvicorn app.main:app --reload
    ```

5. **Access the Interfaces**:  
    - **Chatbot**: `http://localhost:8000/`
    - **File Upload**: `http://localhost:8000/upload-ui`

---

## How It Works

1. **Document Upload**:  
    Banks upload sustainable finance-related documents. These documents form the knowledge base from which the chatbot will provide answers.

2. **Retrieval-Augmented Generation (RAG)**:  
    The platform uses **LangChain** and **FAISS** to retrieve the most relevant context from uploaded documents based on user queries. The context is then passed to an **LLM** (e.g., GPT-4) to generate a response.

3. **Interactive Chatbot**:  
    Clients can interact with the chatbot for contextualized advice based on the specific documents uploaded by their bank. If no documents are uploaded, the system will respond using broad knowledge from the LLM.

---

## API Endpoints

### 1. **/chat**:  
    **POST**  
    Sends a query to the chatbot and retrieves an answer based on the current context or LLM general knowledge if no context is available.  
    **Payload**:
    ```json
    {
      "query": "What are the latest sustainable finance regulations in the EU?"
    }
    ```

### 2. **/upload**:  
    **POST**  
    Uploads a PDF document to be added to the knowledge base.  
    **Payload**:  
    - The document file in the `multipart/form-data` format.

### 3. **/documents**:  
    **GET**  
    Lists the uploaded documents available in the system.

---

## Why We Chose This Architecture

- **FastAPI Backend**:  
   We use **FastAPI** for its simplicity, speed, and ease of integration with **LangChain** and **OpenAI's API**. It helps us serve endpoints like uploading documents, querying the chatbot, and retrieving the list of uploaded documents.

- **LangChain & OpenAI**:  
   **LangChain** facilitates building the **RAG** pipeline, connecting the document retrieval process with OpenAI’s **GPT-4** for contextualized response generation. 

- **FAISS for Vector Store**:  
   **FAISS** enables efficient similarity search for the most relevant context in large datasets, allowing for fast document retrieval.

- **Render Cloud Hosting**:  
   **Render** allows us to deploy the app with minimal configuration, and its persistent disk storage ensures documents and knowledge bases are maintained even when the service is inactive.

---

