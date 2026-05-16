import os
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_history_aware_retriever
from langchain_community.document_loaders import PyPDFium2Loader
from dotenv import load_dotenv

load_dotenv()  

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,  
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

def process_uploaded_files(uploaded_files):
    """Processes uploaded files, extracts text, and stores them in ChromaDB."""
    all_docs = []
    
    # Create a temporary directory to save uploaded files for LangChain loaders
    os.makedirs("temp_dir", exist_ok=True)
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join("temp_dir", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Select loader based on file extension
        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFium2Loader(file_path)
        elif uploaded_file.name.endswith(".docx") or uploaded_file.name.endswith(".doc"):
            loader = Docx2txtLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        #elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
            # Uses Unstructured to extract text/OCR from images
            #loader = UnstructuredImageLoader(file_path)
        else:
            continue
            
        all_docs.extend(loader.load())
        
        # Clean up temp file
        os.remove(file_path)

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(all_docs)
    
    # Create an in-memory vector store
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})

def get_conversational_chain(retriever):
    """Creates a chain that understands chat history."""
    
    # 1. Setup a prompt that rephrases the question if chat history exists
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    # This retriever is smart enough to use chat history to search vectors
    history_aware_retriever = create_history_aware_retriever(
        model, retriever, contextualize_q_prompt
    )

    # 2. Setup the final system prompt to answer the question
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    
    # This combines the smart retriever and the answering chain
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
