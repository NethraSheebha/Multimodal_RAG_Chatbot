import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
from rag_backend import process_uploaded_files, get_conversational_chain
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("📂 Multimodal RAG Assistant")

# Sidebar for configuration and file uploads
with st.sidebar:
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, Word docs, or Images", 
        type=["pdf", "docx", "png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    
    process_btn = st.button("Process Documents", type="primary")

# Initialize session state for retriever and chat history
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process files when button is clicked
if process_btn and uploaded_files:
    with st.spinner("Processing and indexing documents..."):
        try:
            st.session_state.retriever = process_uploaded_files(uploaded_files)
            st.success("Documents indexed successfully! Ask away.")
        except Exception as e:
            st.error(f"Error processing files: {e}")

# Main Chat Interface
st.subheader("Chat with your Docs")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if user_query := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        if st.session_state.retriever is None:
            response = "Please upload and process some documents first."
            st.markdown(response)
        else:
            with st.spinner("Thinking..."):
                try:
                    # 1. Convert Streamlit's message history to LangChain format
                    chat_history = []
                    for msg in st.session_state.messages[:-1]: # exclude the current query
                        if msg["role"] == "user":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        else:
                            chat_history.append(AIMessage(content=msg["content"]))
                    
                    # 2. Get the conversational chain
                    conversational_chain = get_conversational_chain(st.session_state.retriever)
                    
                    # 3. Invoke the chain with chat history
                    res = conversational_chain.invoke({
                        "input": user_query,
                        "chat_history": chat_history
                    })
                    
                    response = res["answer"]
                    st.markdown(response)
                except Exception as e:
                    response = f"An error occurred: {e}"
                    st.markdown(response)
                    
    st.session_state.messages.append({"role": "assistant", "content": response})