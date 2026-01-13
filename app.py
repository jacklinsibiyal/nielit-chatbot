import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
import os
import time
import asyncio

# ✅ Ensure event loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="NIELIT AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Header Styling */
    .main-header {
        font-size: 2.5rem;
        color: #004e92;
        text-align: center;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    /* Chat Bubble Styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR1OsjkNcltbzjJ3qtopQck2ZPeDWmTjoYBpg&s", width=100)
    st.title("NIELIT Assistant")
    st.markdown("---")
    st.markdown("### 🛠️ Functionality")
    st.info(
        "I can answer questions about:\n"
        "- NIELIT Courses (O/A/B Level)\n"
        "- Exam Registrations\n"
        "- Syllabus & Certification\n"
        "- General Enquiries"
    )
    st.markdown("---")
    
    # Clear Chat Button
    if st.button("🗑️ Clear Conversation", type="primary"):
        st.session_state.messages = []
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        st.rerun()

# --- MAIN PAGE HEADER ---
st.markdown('<div class="main-header">🤖 NIELIT AI Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your 24/7 Guide for Courses, Exams, and Certifications</div>', unsafe_allow_html=True)

# --- BACKEND SETUP ---

# ✅ Create embeddings ONCE
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load LLM
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="meta-llama/llama-3.3-70b-versatile" # Switched to a standard stable model alias, revert if needed
)

# Prompt template
prompt_template = ChatPromptTemplate.from_template(
    """
    You are NIELIT AI, an official assistant for the National Institute of Electronics & Information Technology.
    
    Guidelines:
    1. Answer queries regarding NIELIT courses, exams, and registrations accurately using the context provided.
    2. Be polite, professional, and concise.
    3. If the user asks about something unrelated to NIELIT, politely steer them back to NIELIT topics.
    4. Do not mention "based on the context provided" explicitly; just answer naturally.
    
    <context>
    {context}
    </context>
    
    History: {chat_history}
    Question: {input}
    """
)

# Function to load vector store
@st.cache_resource
def load_vector_store():
    vector_store_path = "./vector_store/faiss_index"
    if os.path.exists(f"{vector_store_path}/index.pkl") and os.path.exists(f"{vector_store_path}/index.faiss"):
        vectors = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vectors
    else:
        return None

# Initialize Vector Store
if "vectors" not in st.session_state:
    with st.spinner("🚀 Booting up NIELIT AI Knowledge Base..."):
        st.session_state.vectors = load_vector_store()
    
    if st.session_state.vectors:
        st.toast("✅ Knowledge Base Loaded Successfully!", icon="🎓")
    else:
        st.error("⚠️ Vector store files not found! Please check your directory.")

# Initialize Chat Memory & History
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am the NIELIT AI Assistant. How can I help you with your courses or exams today?"}
    ]

# --- CHAT INTERFACE LOGIC ---

# 1. Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# 2. Handle User Input
if user_input := st.chat_input("Ask about NIELIT courses, exams, etc..."):
    
    # Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # 3. Generate Response
    if st.session_state.vectors:
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            start_time = time.process_time()
            
            with st.spinner("Searching NIELIT database..."):
                try:
                    chain_input = {
                        'input': user_input,
                        'chat_history': st.session_state.memory.chat_memory.messages
                    }
                    response = retrieval_chain.invoke(chain_input)
                    
                    # Update memory
                    st.session_state.memory.save_context({'input': user_input}, {'output': response['answer']})
                    
                    response_time = time.process_time() - start_time
                    
                    # Display Answer
                    full_response = response['answer']
                    message_placeholder.markdown(full_response)
                    
                    # Show sources/metadata in an expander
                    with st.expander("📚 View Source Documents (Context)"):
                        st.caption(f"⏱️ Response generated in {response_time:.2f} seconds")
                        for i, doc in enumerate(response["context"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(f"_{doc.page_content[:300]}..._") # Truncate for cleaner UI
                            st.divider()

                    # Save bot response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.error("Vector Store is not loaded. Cannot process query.")
