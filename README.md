<h1 align="center">🧠 MediBot – AI Medical Assistant</h1>
<p align="center"><b>Multilingual | RAG-Powered | Vision-Enabled | Voice Interaction</b></p>

<p>
MediBot is an advanced AI-powered medical assistant designed to deliver <b>accurate, explainable, and accessible medical insights</b> using state-of-the-art machine learning, natural language processing (NLP), and computer vision.
<br><br>
Built with a <b>Retrieval-Augmented Generation (RAG)</b> pipeline, MediBot integrates Hugging Face models, Groq vision, and the Gemini API to provide fast, reliable medical information through <b>text, image, and voice-based interactions</b>.
</p>

<h2>🚀 Key Highlights</h2>
<ul>
  <li><b>0.84 BERTScore F1</b> for medical response accuracy</li>
  <li>🌐 Multilingual RAG chatbot</li>
  <li>🖼️ Medical image analysis using <b>LLaMA Maverick 17B Vision (Groq)</b></li>
  <li>🎤 Voice input (Google Speech Recognition)</li>
  <li>🔊 Natural TTS output (gTTS)</li>
  <li>📚 FAISS vector store from <i>The Gale Encyclopedia of Medicine</i></li>
  <li>📈 Increased accessibility by <b>60%</b> with voice interaction</li>
  <li>🖥️ Clean Streamlit UI with chat history</li>
</ul>

<h2>📂 Table of Contents</h2>
<ul>
  <li><a href="#architecture">Architecture Overview</a></li>
  <li><a href="#features">Features</a></li>
  <li><a href="#tech">Tech Stack</a></li>
  <li><a href="#method">Methodology</a></li>
  <li><a href="#snapshots">Snapshots</a></li>
  <li><a href="#requirements">Hardware & Software Requirements</a></li>
  <li><a href="#advantages">Advantages</a></li>
  <li><a href="#applications">Applications</a></li>
  <li><a href="#future">Future Work</a></li>
  <li><a href="#run">How to Run</a></li>
  <li><a href="#references">References</a></li>
</ul>

<h2 id="architecture">🧩 Architecture Overview</h2>

<h3>Three-Phase Pipeline</h3>
<ol>
  <li><b>Phase 1 – Data Preparation</b><br>Extract PDF → Split into 500-character chunks.</li>
  <li><b>Phase 2 – Vector Store Creation</b><br>MiniLM-L6-v2 embeddings → stored in FAISS.</li>
  <li><b>Phase 3 – Query Processing</b><br>Text/Image/Voice → RAG → Answer + Sources.</li>
</ol>

<h2 id="features">⭐ Features</h2>

<h3>1. Medical Text Query Handling</h3>
<ul>
  <li>Mistral-7B-Instruct for text generation</li>
  <li>Top-k retrieval via LangChain RetrievalQA</li>
  <li>Answers with citation transparency</li>
</ul>

<h3>2. Medical Image Analysis</h3>
<ul>
  <li><b>LLaMA Maverick 17B Vision</b> (Groq accelerated)</li>
  <li>Extracts symptoms, patterns & medical features</li>
  <li>Rejects non-medical images gracefully</li>
</ul>

<h3>3. Voice & Audio Support</h3>
<ul>
  <li>🎤 Voice input via Google SpeechRecognition</li>
  <li>🔊 TTS audio output via gTTS</li>
</ul>

<h3>4. Streamlit UI</h3>
<ul>
  <li>Text tab & Image tab</li>
  <li>Audio playback</li>
  <li>Chat history stored & displayed</li>
</ul>

<h2 id="tech">🛠 Tech Stack</h2>

<h3>Machine Learning</h3>
<ul>
  <li>Mistral-7B-Instruct</li>
  <li>LLaMA-Maverick-17B Vision</li>
  <li>MiniLM-L6-v2 Embeddings</li>
</ul>

<h3>NLP & Retrieval</h3>
<ul>
  <li>LangChain RetrievalQA</li>
  <li>FAISS vector store</li>
  <li>HuggingFace Transformers</li>
</ul>

<h3>Libraries</h3>
<ul>
  <li>Streamlit</li>
  <li>LangChain</li>
  <li>FAISS</li>
  <li>speech_recognition</li>
  <li>gTTS</li>
  <li>groq</li>
</ul>

<h3>Development Tools</h3>
<ul>
  <li>Visual Studio Code</li>
  <li>Python 3.9+</li>
</ul>

<h2 id="method">🧬 Methodology</h2>
<img width="1066" height="517" alt="image" src="https://github.com/user-attachments/assets/3ba8c76a-0f93-4777-bd1d-028eb784e32a" />

<ul>
  <li><b>System Design:</b> Frontend (Streamlit) + Backend (Python RAG)</li>
  <li><b>Data Preprocessing:</b> Chunking, embeddings, FAISS storage</li>
  <li><b>Model Implementation:</b> Mistral text model + LLaMA vision model</li>
  <li><b>Results:</b> High accuracy & explainable medical insights</li>
</ul>

<h2 id="snapshots">📸 Snapshots</h2>

<ul>
  <li>Text query interface</li>
  <img width="1066" height="482" alt="image" src="https://github.com/user-attachments/assets/ddcdf2f3-d653-4d52-b4de-e129bbfd5739" />

  <li>Voice query usage</li>
  <img width="1073" height="467" alt="image" src="https://github.com/user-attachments/assets/0ca25857-6192-49dc-8b72-25ad8142fca9" />

  <li>Retrieved medical documents</li>
  <img width="1081" height="484" alt="image" src="https://github.com/user-attachments/assets/020f4452-2d70-4ae0-8d68-f1170c7cfdef" />

  <li>Medical image analysis output</li>
  <img width="1066" height="501" alt="image" src="https://github.com/user-attachments/assets/72dd629a-d5cb-4486-81f6-5eb167a62d2f" />
<img width="1066" height="471" alt="image" src="https://github.com/user-attachments/assets/2ef1cac8-fe1e-4390-9937-5adbe0e8f231" />

  <li>Non-medical image rejection</li>
  <img width="1066" height="468" alt="image" src="https://github.com/user-attachments/assets/3a7ff879-1517-4271-b85b-9483c50f2f59" />
<img width="1066" height="503" alt="image" src="https://github.com/user-attachments/assets/0dc96eac-3dcf-4cc5-ac32-388e310b7d2c" />

  <li>Chat history</li>
  <img width="395" height="531" alt="image" src="https://github.com/user-attachments/assets/da156bbb-e584-4ba3-bd3f-8ca62602ac57" />
<img width="356" height="741" alt="image" src="https://github.com/user-attachments/assets/07430af7-bcb7-489c-aa5b-100eca83fc4f" />

</ul>

<h2 id="requirements">💻 Hardware & Software Requirements</h2>

<h3>Hardware</h3>
<ul>
  <li>64-bit CPU</li>
  <li>8–16 GB RAM</li>
  <li>512 GB SSD</li>
  <li>Windows 11</li>
</ul>

<h3>Software</h3>
<ul>
  <li>VS Code 2022+</li>
  <li>Python 3.9+</li>
  <li>pip libraries (Streamlit, FAISS, LangChain, etc.)</li>
</ul>

<h2 id="advantages">⭐ Advantages</h2>
<ul>
  <li>Efficient medical information retrieval</li>
  <li>Voice + Audio accessibility</li>
  <li>Source-backed responses</li>
  <li>Supports text & image queries</li>
  <li>Educational & research-friendly</li>
</ul>

<h2 id="applications">🏥 Applications</h2>
<ul>
  <li>Medical education</li>
  <li>Symptom understanding</li>
  <li>Institutional knowledge bases</li>
  <li>Research support</li>
</ul>

<h2 id="future">🔮 Future Work</h2>
<ul>
  <li><b>Multi-Agent System</b>  
    <ul>
      <li>Agent 1: Medical insights</li>
      <li>Agent 2: Medicine booking</li>
      <li>Agent 3: Doctor appointment scheduling</li>
    </ul>
  </li>
  <li>Expanded medical datasets</li>
  <li>Full multilingual support</li>
  <li>Mobile app (iOS/Android)</li>
</ul>

<h2 id="run">▶️ How to Run</h2>

<pre>
git clone https://github.com/&lt;your-username&gt;/MediBot
cd MediBot
pip install -r requirements.txt
streamlit run app.py
</pre>

<p><b>Set environment variables:</b></p>

<pre>
export GROQ_API_KEY="YOUR_KEY"
export HF_API_KEY="YOUR_KEY"
</pre>

<h2 id="references">📚 References</h2>
<ul>
  <li>FAISS – Facebook AI Similarity Search</li>
  <li>Meta LLaMA Vision Models</li>
  <li>Groq AI Acceleration Platform</li>
  <li>gTTS – Google Text-to-Speech</li>
  <li>LangChain Documentation</li>
  <li>Medical mT5 Paper</li>
  <li>Apollo Medical LLM</li>
</ul>

<br><br>
<h3 align="center">🎉 MediBot – Making Medical Knowledge Accessible with AI</h3>

