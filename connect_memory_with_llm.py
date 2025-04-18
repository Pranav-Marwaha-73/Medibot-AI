import os
import base64
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# Environment Variables
HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Step 1: Setup LLM (Mistral with HuggingFace)
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    """Load Hugging Face LLM"""
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

# Step 2: Custom Prompt Template
CUSTOM_PROMPT_TEMPLATE = """ 
Use the pieces of information provided in the context to answer the user's question about the medical condition described. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.  
Provide a precise, informative response based strictly on the available context.

Context: {context} 
Question: {question}  

Respond directly and professionally, focusing on medical insights from the context.
"""

def set_custom_prompt(custom_prompt_template):
    """Create a custom prompt template"""
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Step 3: Image Encoding Function

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Step 4: Image Description Function using Vision Model
def describe_image(image_path):
    """Get descriptive text for an image using Groq's vision model"""
    client = Groq()
    
    # Encode the image
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Create a generic description query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Describe the medical condition or symptoms visible in this image in clear, precise medical terminology. Focus on specific visual characteristics, colors, textures, and any notable features."
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
    
    # Get image description
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama-3.2-90b-vision-preview"
    )
    
    return chat_completion.choices[0].message.content

# Main Execution Function
def main():
    # Load Database
    DB_FAISS_PATH = "vectorstore/db_faiss"
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    # Create QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k':3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt':set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )

    # User Interaction Loop
    while True:
        print("\nChoose interaction type:")
        print("1. Text Query")
        print("2. Image-based Medical Query")
        print("3. Exit")
        
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == '1':
            # Text-based Query
            user_query = input("Write Text Query Here: ")
            response = qa_chain.invoke({'query': user_query})
            print("\nRESULT: ", response["result"])
            print("\nSOURCE DOCUMENTS: ", response["source_documents"])
        
        elif choice == '2':
            # Image-based Medical Query
            image_path = input("Enter the path to the medical image: ")
            if not os.path.exists(image_path):
                print("Image file not found!")
                continue
            
            try:
                # Get image description from vision model
                image_description = describe_image(image_path)
                print("\nIMAGE DESCRIPTION: ", image_description)
                
                # Use image description as query for RAG system
                response = qa_chain.invoke({'query': image_description})
                print("\nMEDICAL INSIGHT: ", response["result"])
                print("\nSOURCE DOCUMENTS: ", response["source_documents"])
            
            except Exception as e:
                print(f"Error processing image: {e}")
        
        elif choice == '3':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()