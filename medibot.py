import os
import base64
import streamlit as st
import speech_recognition as sr
import time
from typing import List, Dict, Any
from gtts import gTTS
import io

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

from groq import Groq

import warnings 
warnings.filterwarnings('ignore') 

def describe_image(image_bytes):
    """
    Describe medical image using Groq's vision model
    """
    client = Groq()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Provide a precise medical description of this image. Focus ONLY on medical characteristics, potential medical conditions, anatomical features, symptoms, or abnormalities visible in the image. Use professional medical terminology and be as specific as possible about any observable medical details. If image is not related to medical just say that (I don't have Knowledge of it) No need to give extra context understand this"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
        }
    ]
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="meta-llama/llama-4-maverick-17b-128e-instruct"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error in image description: {e}")
        return None

def text_to_speech(text):
    """
    Convert text to speech and return audio bytes
    """
    try:
        tts = gTTS(text=text, lang='en')
        audio_bytes = io.BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        return audio_bytes.read()
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}")
        return None

def display_source_documents(source_docs: List[Dict[str, Any]]):
    if not source_docs:
        st.warning("No source documents found.")
        return

    st.subheader("📚 Source Documents")
    
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get('source', 'Unknown Source')
        page = doc.metadata.get('page', 'N/A')
        
        with st.expander(f"Source {i}: {source} (Page {page})", expanded=False):
            content_preview = doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            
            st.markdown(f"**Reference Book:** {source}")
            st.markdown(f"**Page Number:** {page}")
            st.markdown(f"**Content Excerpt:**")
            st.markdown(f"*{content_preview}*")

def add_to_chat_history(query, response, source_type='text'):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({
        'query': query,
        'response': response,
        'type': source_type,
        'timestamp': time.time()
    })
    
    if len(st.session_state.chat_history) > 20:
        st.session_state.chat_history = st.session_state.chat_history[-20:]

def display_chat_history():
    if not hasattr(st.session_state, 'chat_history'):
        st.session_state.chat_history = []
    
    st.sidebar.title("💬 Conversation History")
    
    if not st.session_state.chat_history:
        st.sidebar.info("No chat history yet.")
        return
    
    for idx, chat in enumerate(reversed(st.session_state.chat_history), 1):
        if chat['type'] == 'image':
            icon = "🖼️"
            color = "linear-gradient(to right, #ff7e5f, #feb47b)"
        else:
            icon = "💬"
            color = "linear-gradient(to right, #6a11cb 0%, #2575fc 100%)"
        
        with st.sidebar.expander(f"{icon} Conversation {len(st.session_state.chat_history) - idx + 1}", expanded=False):
            st.markdown(f"""
            <div style="background: {color}; 
                        color: white; 
                        padding: 10px; 
                        border-radius: 10px; 
                        margin-bottom: 10px;">
                <strong>Query:</strong><br>
                {chat['query']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.1); 
                        color: black; 
                        padding: 10px; 
                        border-radius: 10px;">
                <strong>Response:</strong><br>
                {chat['response']}
            </div>
            """, unsafe_allow_html=True)

def speech_to_text():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak your medical query")
        
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            try:
                text = recognizer.recognize_google(audio)
                st.success(f"🔊 Detected: {text}")
                return text
            except sr.UnknownValueError:
                st.warning("🚫 Could not understand audio. Please try again.")
                return None
            except sr.RequestError:
                st.error("🌐 Could not request results from speech recognition service")
                return None
        
        except sr.WaitTimeoutError:
            st.warning("⏰ No speech detected. Timeout occurred.")
            return None

def main():
    st.set_page_config(
        page_title="MediBot AI", 
        page_icon="🩺", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

    st.sidebar.title("MediBot 🩺")
    st.sidebar.markdown("""
    ### 🌟 About MediBot
    - **Advanced Medical AI Assistant**
    - Intelligent Medical Insights
    - Image & Text Analysis
    """)
    
    st.sidebar.markdown("### 🤖 Model Capabilities")
    st.sidebar.info("Vision Model: meta-llama/llama-4-maverick-17b-128e-instruct")
    st.sidebar.info("Medical LLM: Mistral-7B-Instruct")

    display_chat_history()

    @st.cache_resource
    def get_vectorstore():
        try:
            embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            db = FAISS.load_local("vectorstore/db_faiss", embedding_model, allow_dangerous_deserialization=True)
            return db
        except Exception as e:
            st.error(f"Could not load vector store: {str(e)}")
            return None

    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #2575fc; 
                   font-size: 3em; 
                   text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🩺 MediBot: Medical Insight Generator
        </h1>
        <p style="color: #6a11cb; 
                  font-style: italic;">
            AI-powered medical analysis and information retrieval
        </p>
    </div>
    """, unsafe_allow_html=True)

    CUSTOM_PROMPT_TEMPLATE = """
    You are a helpful medical AI assistant. Follow these strict guidelines:
    1. Use ONLY the information provided in the context to answer the question and how to cure it.
    2. If the context does NOT contain relevant information to answer the question or the user query is not related to any medical terms, 
       respond EXACTLY with: "I do not have any information about this topic in the provided context."
    3. Do not generate or invent information not present in the context.
    4. Provide a direct and concise answer if information is available.
    Context: {context}
    Question: {question}
    Answer:"""

    # Initialize session state for audio
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = {}

    tab1, tab2 = st.tabs(["💬 Text Query", "🖼️ Image Analysis"])

    with tab1:
        st.subheader("📋 Medical Text Query")
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            prompt = st.chat_input("Ask a medical question...", key="text_query")
        
        with col2:
            if st.button("🎤 Voice Input", key="voice_input_btn"):
                voice_prompt = speech_to_text()
                if voice_prompt:
                    prompt = voice_prompt

        if prompt:
            try:
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Vector store could not be loaded.")
                    return

                HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
                HF_TOKEN = os.environ.get("HF_TOKEN")

                qa_chain = RetrievalQA.from_chain_type(
                    llm=HuggingFaceEndpoint(
                        repo_id=HUGGINGFACE_REPO_ID,
                        temperature=0.5,
                        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
                    ),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={
                        'prompt': PromptTemplate(
                            template=CUSTOM_PROMPT_TEMPLATE, 
                            input_variables=["context", "question"]
                        )
                    }
                )

                with st.spinner('Analyzing your medical query...'):
                    time.sleep(1)
                    response = qa_chain.invoke({'query': prompt})

                add_to_chat_history(prompt, response["result"])
                
                st.markdown("""
                <div style="background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%); 
                            color: white; 
                            padding: 20px; 
                            border-radius: 15px; 
                            animation: fadeIn 1s;">
                    <h3>🤖 AI Medical Insights</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.2); 
                            color: black; 
                            padding: 15px; 
                            border-radius: 10px; 
                            margin-bottom: 15px;">
                    <strong>📝 Your Question:</strong><br>
                    {prompt}
                </div>
                """, unsafe_allow_html=True)
                
                st.write(response["result"])
                
                # Generate audio once and store it
                audio_key = "text_response"
                if audio_key not in st.session_state.audio_data:
                    audio_bytes = text_to_speech(response["result"])
                    if audio_bytes:
                        st.session_state.audio_data[audio_key] = audio_bytes
                
                # Display audio player
                if audio_key in st.session_state.audio_data:
                    st.audio(st.session_state.audio_data[audio_key], format="audio/mp3")
                
                # Speaker button to replay audio
                if st.button("🔊 Speak Response", key="speak_text_response"):
                    if audio_key in st.session_state.audio_data:
                        st.audio(st.session_state.audio_data[audio_key], format="audio/mp3", start_time=0)
                
                display_source_documents(response.get('source_documents', []))

            except Exception as e:
                st.error(f"An error occurred: {e}")

    with tab2:
        st.subheader("🔬 Medical Image Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a medical image", 
            type=['png', 'jpg', 'jpeg'], 
            help="Upload an image for medical condition analysis"
        )

        if uploaded_file is not None:
            st.image(uploaded_file, 
                     caption="Uploaded Medical Image", 
                     use_column_width=True, 
                     clamp=True)
            
            if st.button("🔍 Analyze Image", key="image_analyze_btn"):
                with st.spinner("Analyzing medical image..."):
                    try:
                        image_bytes = uploaded_file.getvalue()
                        
                        st.subheader("🤖 Vision Model Image Analysis")
                        image_description = describe_image(image_bytes)
                        
                        if image_description:
                            vectorstore = get_vectorstore()
                            if vectorstore is None:
                                st.error("Vector store could not be loaded.")
                                return

                            HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
                            HF_TOKEN = os.environ.get("HF_TOKEN")

                            qa_chain = RetrievalQA.from_chain_type(
                                llm=HuggingFaceEndpoint(
                                    repo_id=HUGGINGFACE_REPO_ID,
                                    temperature=0.5,
                                    model_kwargs={"token": HF_TOKEN, "max_length": "512"}
                                ),
                                chain_type="stuff",
                                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                                return_source_documents=True,
                                chain_type_kwargs={
                                    'prompt': PromptTemplate(
                                        template=CUSTOM_PROMPT_TEMPLATE, 
                                        input_variables=["context", "question"]
                                    )
                                }
                            )

                            response = qa_chain.invoke({'query': image_description})
                            
                            add_to_chat_history(
                                f"Image Analysis: {image_description}", 
                                response["result"], 
                                source_type='image'
                            )
                            
                            st.markdown("""
                            <div style="background: linear-gradient(to right, #ff7e5f, #feb47b); 
                                        color: white; 
                                        padding: 20px; 
                                        border-radius: 15px; 
                                        animation: fadeIn 1s;">
                                <h3>🔬 Vision Analysis Insights</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            st.write(image_description)
                            
                            st.markdown("""
                            <div style="background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%); 
                                        color: white; 
                                        padding: 20px; 
                                        border-radius: 15px; 
                                        animation: fadeIn 1s;">
                                <h3>🩺 Detailed Medical Insights</h3>
                            </div>
                            """, unsafe_allow_html=True)
                            st.write(response["result"])
                            
                            # Generate audio once and store it for medical insights
                            audio_key = "insights_response"
                            if audio_key not in st.session_state.audio_data:
                                audio_bytes_insights = text_to_speech(response["result"])
                                if audio_bytes_insights:
                                    st.session_state.audio_data[audio_key] = audio_bytes_insights
                            
                            # Display audio player
                            if audio_key in st.session_state.audio_data:
                                st.audio(st.session_state.audio_data[audio_key], format="audio/mp3")
                            
                            # Speaker button to replay audio
                            if st.button("🔊 Speak Medical Insights", key="speak_insights_response"):
                                if audio_key in st.session_state.audio_data:
                                    st.audio(st.session_state.audio_data[audio_key], format="audio/mp3", start_time=0)
                            
                            display_source_documents(response.get('source_documents', []))

                    except Exception as e:
                        st.error(f"Error in image analysis: {e}")

if __name__ == "__main__":
    main()