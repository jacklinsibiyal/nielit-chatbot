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

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="NIELIT Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 NIELIT Chatbot")
st.markdown("---")

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are NIELIT AI, you help people with their queries regarding courses on NIELIT
    and help them solve their doubts related to the courses, exam, registration anything related to NIELIT.
    Your main goal is to assist them as best as you can. Be friendly and polite. Dont answer questions unrelated to NIELIT 
    or rather convince them to ask questions related to NIELIT. Try to be concise unless asked. Give some course details, 
    exam details, registration details, etc.Dont mention about the context but use the context to answer the
    questions asked by the user.
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

# Function to create vector embeddings
@st.cache_resource
def load_vector_store():
    vector_store_path = "./vector_store/faiss_index"
    if os.path.exists(f"{vector_store_path}/index.pkl") and os.path.exists(f"{vector_store_path}/index.faiss"):
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.vectors = FAISS.load_local(vector_store_path, st.session_state.embeddings, allow_dangerous_deserialization=True)
        return st.session_state.vectors
    else:
        st.error("⚠️ Vector store files not found! Ensure index.pkl and index.faiss are in the specified folder.")
        return None
if "vectors" not in st.session_state:
    with st.spinner("🚀 Loading Vector Store..."):
        st.session_state.vectors = load_vector_store()
    if st.session_state.vectors:
        st.success("✅ Vector Store Loaded!")


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
    st.markdown(f"**🤖 NIELIT AI:** {response['answer']}")

    with st.expander("📄 Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.markdown(doc.page_content)
            st.markdown("---")
