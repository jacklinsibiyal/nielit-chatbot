import asyncio
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import os
import time

# Load .env
load_dotenv()

# Ensure asyncio event loop exists (prevents "no current event loop in thread" for gRPC clients)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="NIELIT Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 NIELIT Chatbot")
st.markdown("---")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="meta-llama/llama-4-maverick-17b-128e-instruct"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are NIELIT AI, you help people with their queries regarding courses on NIELIT
    and help them solve their doubts related to the courses, exam, registration anything related to NIELIT.
    Your main goal is to assist them as best as you can. Be friendly and polite. Dont answer questions unrelated to NIELIT 
    or rather convince them to ask questions related to NIELIT. Try to be concise unless asked. Give some course details, 
    exam details, registration details, etc.Dont mention about the context but use the context to answer the
    questions asked by the user.(The context is got from the official webpage of NIELIT, so it will be correct.)
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Create the embeddings object once (outside the cached resource) after ensuring event loop exists.
if "embeddings" not in st.session_state:
    try:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    except Exception as e:
        # Provide a helpful error in the UI if embeddings init fails
        st.error(f"⚠️ Failed to initialize embeddings: {e}")
        st.stop()

# Cached resource to load vector store from disk (this should not re-create embeddings)
@st.cache_resource
def load_vector_store_cached():
    vector_store_path = "./vector_store/faiss_index"
    # Check for FAISS files presence
    index_pkl = os.path.join(vector_store_path, "index.pkl")
    index_faiss = os.path.join(vector_store_path, "index.faiss")

    if os.path.exists(index_pkl) and os.path.exists(index_faiss):
        try:
            # Use embeddings already created in session_state
            vectors = FAISS.load_local(vector_store_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
            return vectors
        except Exception as e:
            # If loading fails, return None and show error upstream
            st.error(f"⚠️ Failed to load FAISS vector store: {e}")
            return None
    else:
        st.error("⚠️ Vector store files not found! Ensure index.pkl and index.faiss are in ./vector_store/faiss_index")
        return None

# Load vector store into session state (only if not already loaded)
if "vectors" not in st.session_state:
    with st.spinner("🚀 Loading Vector Store..."):
        st.session_state.vectors = load_vector_store_cached()
    if st.session_state.vectors:
        st.success("✅ Vector Store Loaded!")
    else:
        # stop further execution if vectors couldn't be loaded
        st.stop()

prompt1 = st.text_input("💬 Enter Your Question")

if prompt1:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    with st.spinner("🔎 Retrieving..."):
        start = time.process_time()
        chain_input = {
            'input': prompt1,
            'chat_history': st.session_state.memory.chat_memory.messages
        }
        response = retrieval_chain.invoke(chain_input)
        response_time = time.process_time() - start

    st.markdown(f"**⏱️ Response Time:** {response_time} seconds")
    st.markdown(f"**🤖 NIELIT AI:** {response.get('answer', 'No answer returned.')}")

    with st.expander("📄 Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.markdown(doc.page_content)
            st.markdown("---")
