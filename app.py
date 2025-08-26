import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import nest_asyncio
import speech_recognition as sr
import uuid
from datetime import datetime

# Apply nest_asyncio to handle event loop in Streamlit
nest_asyncio.apply()

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, pdf_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    # Add metadata for source citation
    doc_chunks = [Document(page_content=chunk, metadata={"source": pdf_name}) for chunk in chunks]
    return doc_chunks

def get_vector_store(doc_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(doc_chunks, embeddings)
    vector_store.save_local("vector_store")

def get_conversational_chain():
    prompt_template = """You are a medical assistant specializing in diabetes. Answer the question as detailed as possible using only the provided context, focusing exclusively on diabetes-related information, including symptoms, management, treatments, complications, or preventive measures. If the answer is not in the provided context or is unrelated to diabetes, say "I am sorry, answer is not available in this context" and do not provide a wrong answer. Avoid speculation and ensure responses are concise, evidence-based, and relevant to diabetes.

    Context: {context}

    Question: {question}

    Answer in English:"""
    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

def process_user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    sources = set(doc.metadata.get("source", "Unknown") for doc in docs)
    source_text = f"Source: {', '.join(sources)}"
    return response["output_text"] + f"\n\n{source_text}"

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Écoute en cours... Parle maintenant !")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language="fr-FR")
            st.write(f"Texte reconnu : {text}")
            return text
        except sr.UnknownValueError:
            st.error("Désolé, je n'ai pas compris l'audio.")
            return None
        except sr.RequestError:
            st.error("Erreur avec le service de reconnaissance vocale.")
            return None
        except sr.WaitTimeoutError:
            st.error("Aucun son détecté dans le délai imparti.")
            return None

def tag_health_topic(message):
    message = message.lower()
    if any(keyword in message for keyword in ["diabetes", "glucose", "insulin", "a1c", "blood sugar"]):
        return "Diabetes"
    return "Non-Diabetes"

def main():
    st.set_page_config(page_title="Diabetes Q&A Bot", layout="wide")
    st.header("Diabetes Q&A Bot")

    # Initialize session state
    if 'conversations' not in st.session_state:
        st.session_state.conversations = {}
        st.session_state.current_conversation_id = str(uuid.uuid4())
        st.session_state.conversations[st.session_state.current_conversation_id] = []
    if 'show_history' not in st.session_state:
        st.session_state.show_history = False

    # Medical-themed CSS
    with st.sidebar:
        theme = st.selectbox("Theme", ["Light", "Dark"], key="theme")
        if theme == "Dark":
            css = """
            <style>
            .chat-container { 
                background-color: #2c2f33; 
                aria-label="Diabetes chat conversation area"; 
                padding: 15px; 
                border-radius: 12px; 
                max-height: 70vh; 
                overflow-y: auto; 
                border: 1px solid #4CAF50;
            }
            .user-bubble { 
                background: linear-gradient(135deg, #005670 0%, #007bff 100%); 
                color: white; 
                padding: 12px 18px; 
                border-radius: 15px 15px 5px 15px; 
                max-width: 75%; 
                align-self: flex-end; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.2); 
                margin: 8px; 
                aria-label="User diabetes question"; 
                font-family: 'Roboto', sans-serif;
            }
            .bot-bubble { 
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); 
                color: white; 
                padding: 12px 18px; 
                border-radius: 15px 15px 15px 5px; 
                max-width: 75%; 
                align-self: flex-start; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.2); 
                margin: 8px; 
                aria-label="Bot diabetes response"; 
                font-family: 'Roboto', sans-serif;
            }
            .sidebar .sidebar-content { 
                background-color: #343a40; 
                color: #e0e0e0; 
                padding: 20px; 
                border-radius: 12px; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.2); 
                border: 1px solid #005670;
            }
            .sidebar .sidebar-content .stButton > button { 
                background: linear-gradient(135deg, #005670 0%, #007bff 100%); 
                color: white; 
                border-radius: 10px; 
                padding: 12px; 
                margin: 8px 0; 
                width: 100%; 
                text-align: left; 
                font-family: 'Roboto', sans-serif;
            }
            .sidebar .sidebar-content .stButton > button:hover { 
                background: linear-gradient(135deg, #003d54 0%, #0056b3 100%); 
            }
            .sidebar .sidebar-content .history-button { 
                background-color: #495057; 
                color: #e0e0e0; 
                font-weight: bold; 
                padding: 12px; 
                border-radius: 10px; 
                font-family: 'Roboto', sans-serif;
            }
            .input-with-arrow { 
                display: flex; 
                align-items: center; 
                margin-top: 15px; 
                aria-label="Diabetes question input area"; 
            }
            .input-with-arrow input { 
                flex: 1; 
                padding: 12px; 
                border: 1px solid #005670; 
                border-radius: 10px 0 0 10px; 
                font-size: 16px; 
                background-color: #495057; 
                color: #e0e0e0; 
                font-family: 'Roboto', sans-serif;
            }
            .input-with-arrow button { 
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); 
                color: white; 
                padding: 12px 18px; 
                border-radius: 0 10px 10px 0; 
                font-size: 16px; 
                font-family: 'Roboto', sans-serif;
            }
            .input-with-arrow button:before { 
                content: '⚕️ '; 
                font-size: 14px;
            }
            .stButton > button[title="Ask a diabetes question using voice"]:before { 
                content: '🎙️ '; 
                font-size: 14px;
            }
            h1 { 
                color: #005670; 
                font-family: 'Roboto', sans-serif; 
                position: relative; 
                padding-left: 30px;
            }
            h1:before { 
                content: '🩺'; 
                position: absolute; 
                left: 0; 
                font-size: 24px;
            }
            </style>
            """
        else:
            css = """
            <style>
            .chat-container { 
                background-color: #F5F7FA; 
                aria-label="Diabetes chat conversation area"; 
                padding: 15px; 
                border-radius: 12px; 
                max-height: 70vh; 
                overflow-y: auto; 
                border: 1px solid #4CAF50;
            }
            .user-bubble { 
                background: linear-gradient(135deg, #005670 0%, #007bff 100%); 
                color: white; 
                padding: 12px 18px; 
                border-radius: 15px 15px 5px 15px; 
                max-width: 75%; 
                align-self: flex-end; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.2); 
                margin: 8px; 
                aria-label="User diabetes question"; 
                font-family: 'Roboto', sans-serif;
            }
            .bot-bubble { 
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); 
                color: white; 
                padding: 12px 18px; 
                border-radius: 15px 15px 15px 5px; 
                max-width: 75%; 
                align-self: flex-start; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.2); 
                margin: 8px; 
                aria-label="Bot diabetes response"; 
                font-family: 'Roboto', sans-serif;
            }
            .sidebar .sidebar-content { 
                background-color: #F5F7FA; 
                color: #333; 
                padding: 20px; 
                border-radius: 12px; 
                box-shadow: 0 3px 6px rgba(0,0,0,0.2); 
                border: 1px solid #005670;
            }
            .sidebar .sidebar-content .stButton > button { 
                background: linear-gradient(135deg, #005670 0%, #007bff 100%); 
                color: white; 
                border-radius: 10px; 
                padding: 12px; 
                margin: 8px 0; 
                width: 100%; 
                text-align: left; 
                font-family: 'Roboto', sans-serif;
            }
            .sidebar .sidebar-content .stButton > button:hover { 
                background: linear-gradient(135deg, #003d54 0%, #0056b3 100%); 
            }
            .sidebar .sidebar-content .history-button { 
                background-color: #e6f3ff; 
                color: #333; 
                font-weight: bold; 
                padding: 12px; 
                border-radius: 10px; 
                font-family: 'Roboto', sans-serif;
            }
            .input-with-arrow { 
                display: flex; 
                align-items: center; 
                margin-top: 15px; 
                aria-label="Diabetes question input area"; 
            }
            .input-with-arrow input { 
                flex: 1; 
                padding: 12px; 
                border: 1px solid #005670; 
                border-radius: 10px 0 0 10px; 
                font-size: 16px; 
                background-color: #ffffff; 
                color: #333; 
                font-family: 'Roboto', sans-serif;
            }
            .input-with-arrow button { 
                background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%); 
                color: white; 
                padding: 12px 18px; 
                border-radius: 0 10px 10px 0; 
                font-size: 16px; 
                font-family: 'Roboto', sans-serif;
            }
            .input-with-arrow button:before { 
                content: '⚕️ '; 
                font-size: 14px;
            }
            .stButton > button[title="Ask a diabetes question using voice"]:before { 
                content: '🎙️ '; 
                font-size: 14px;
            }
            h1 { 
                color: #005670; 
                font-family: 'Roboto', sans-serif; 
                position: relative; 
                padding-left: 30px;
            }
            h1:before { 
                content: '🩺'; 
                position: absolute; 
                left: 0; 
                font-size: 24px;
            }
            </style>
            """
        st.markdown(css, unsafe_allow_html=True)

        # Help section
        st.markdown("### How to Use")
        with st.expander("Help"):
            st.write("""
            - **Upload PDF**: Upload diabetes-related PDFs (e.g., medical guidelines on diabetes management) to query.
            - **Text Input**: Type diabetes-related questions (e.g., "What are the symptoms of type 2 diabetes?").
            - **Speech Input**: Click the microphone to ask diabetes questions verbally.
            - **History**: View past diabetes-related conversations or start a new one.
            """)

        # History filtering (diabetes only)
        st.write("### Diabetes Conversations")
        if st.button("📖 Historique", key="history_button", help="View diabetes conversation history"):
            st.session_state.show_history = not st.session_state.show_history
            st.rerun()
        if st.session_state.show_history:
            for conv_id, messages in st.session_state.conversations.items():
                if messages:
                    first_message = messages[0].split(": ", 1)[1][:20] + "..."
                    topic = tag_health_topic(messages[0])
                    if topic == "Diabetes":
                        if st.button(f"Chat {conv_id[:8]}: {first_message}", key=f"conv_{conv_id}"):
                            st.session_state.current_conversation_id = conv_id
                            st.session_state.show_history = False
                            st.rerun()

        # PDF upload
        pdf_docs = st.file_uploader("Upload Diabetes PDF", type=["pdf"], accept_multiple_files=True, help="Upload diabetes-related PDFs like guidelines or research papers.")
        if st.button("Submit", key="submit_pdf", help="Process uploaded PDFs for diabetes queries"):
            if pdf_docs:
                text = get_pdf_text(pdf_docs)
                # Use the first PDF's name for metadata (or customize as needed)
                doc_chunks = []
                for pdf in pdf_docs:
                    text = get_pdf_text([pdf])
                    doc_chunks.extend(get_text_chunks(text, pdf.name))
                get_vector_store(doc_chunks)
                st.success("PDF uploaded successfully")

    # Chat display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
    for message in current_conv:
        if message.startswith("Toi:") or message.startswith("Toi (vocal):"):
            st.markdown(f'<div class="user-bubble">{message[5:]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble">{message[4:]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input section
    st.markdown('<div class="input-with-arrow">', unsafe_allow_html=True)
    user_question = st.text_input("", key="text_input", placeholder="Ask a diabetes-related question (e.g., 'What is A1c?')...")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.write("")
    with col2:
        if st.button("▶", key="send_button", help="Send your diabetes question"):
            if user_question:
                with st.spinner("Processing your diabetes question..."):
                    current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
                    current_conv.append(f"Toi: {user_question} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                    response = process_user_input(user_question)
                    current_conv.append(f"Bot: {response} [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
                    st.rerun()
    with col3:
        if st.button("🗑️", key="clear_button", help="Clear input"):
            st.session_state.text_input = ""
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Speech input
    if st.button("🎙️ Parler", key="speak_button", help="Ask a diabetes question using voice"):
        with st.spinner("Enregistrement..."):
            text = recognize_speech()
            if text:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
                current_conv.append(f"Toi (vocal): {text} [{timestamp}]")
                response = process_user_input(text)
                current_conv.append(f"Bot: {response} [{timestamp}]")
                st.rerun()

if __name__ == '__main__':
    main()