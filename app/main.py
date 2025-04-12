from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import os

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load secrets
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTORSTORE_PATH = "vectorstore"

app = FastAPI()

# In-memory storage for uploaded file names
uploaded_files: List[str] = []

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Relax in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static HTML files (chatbot + upload iframe)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")

@app.get("/upload-ui")
def get_upload_page():
    return FileResponse("app/static/upload.html")

# Load existing vector store or create empty
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if os.path.exists(VECTORSTORE_PATH):
    try:
        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embedding,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        print(f"⚠️ Failed to load vectorstore: {e}")
        vectorstore = None  # fallback to empty state
else:
    vectorstore = None  # Lazy init until first upload

llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=OPENAI_API_KEY)
rag_chain = None

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant in sustainable finance. 
You must use the context below to help you answer the question. 
Only use your general knowledge if there is no provided context or the context is insufficient. 
Do not mention about this context in your reply to the user.

Context:
{context}

Question:
{question}

Answer in a professional and concise tone.
"""
)

if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )

# In-memory chat history
chat_history: List[tuple[str, str]] = []

class ChatQuery(BaseModel):
    query: str

@app.post("/chat")
async def chat(query: ChatQuery):
    # If there's no RAG chain (i.e., no documents uploaded), just use the LLM directly
    if rag_chain is None:
        # Convert the chat history to the correct format (list of dicts with 'role' and 'content')
        conversation = [{"role": "user", "content": query.query}]
        
        # Generate the model's response using the conversation history
        response = llm.generate(conversation)
        
        # Extract the answer from the response
        answer = response['text']
        
        # Add the assistant's response to the conversation history
        chat_history.append((query.query, answer))
        
        return {"answer": answer}
    
    # If there's a RAG chain, use it for context-based answers
    result = rag_chain({"question": query.query, "chat_history": chat_history})
    chat_history.append((query.query, result["answer"]))
    return {"answer": result["answer"]}





@app.get("/documents")
def list_documents():
    return {"documents": uploaded_files}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = f"temp_uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    loader = PyPDFLoader(file_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    global vectorstore, rag_chain
    
    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embedding)
    else:
        vectorstore.add_documents(chunks)

    vectorstore.save_local(VECTORSTORE_PATH)

    # ✅ Rebuild the chain with updated vectorstore
    retriever = vectorstore.as_retriever()
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": custom_prompt},
        return_source_documents=True
    )

    # ✅ Track uploaded file in memory
    uploaded_files.append(file.filename)

    os.remove(file_path)

    return {"message": f"'{file.filename}' added to the knowledge base."}
