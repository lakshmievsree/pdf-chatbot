# 📄 PDF Chatbot using LangChain, Hugging Face & Streamlit

Hi, 
This is a simple PDF chatbot application I built as part of learning from the **"Generative AI for Beginners"** course by **Aakriti E-Learning Academy** on Udemy.
----------------------------------------------------------------
## 📌 About This Project

In the course, the chatbot was created using **OpenAI API for embeddings and answer generation**.  
Since OpenAI API requires paid credits, I tried using **Hugging Face models locally** instead.

👉 **Note:** This is not my original idea — the PDF chatbot concept was part of the course.  
I just replaced the OpenAI API parts with open-source Hugging Face models and made some adjustments to run everything locally.

I’m very new to AI and ML, and this project helped me understand how AI chatbots process PDF text and answer user questions.

---------------------------------------------------------------------

## 📦 Tech Stack Used

- **Python 3**
- **Streamlit** — to create the web interface  
- **PyPDF2** — for PDF text extraction  
- **LangChain** — for chaining the chatbot workflow  
- **Hugging Face Transformers** — to run **FLAN-T5** model locally  
- **sentence-transformers (MiniLM)** — for text embeddings  
- **FAISS** — for similarity search and vector storage  

-----------------------------------------------------------------------

## 📌 How It Works

1. 📤 User uploads a PDF file  
2. 📑 The app extracts text from the PDF  
3. ✂️ Splits text into chunks  
4. 🔢 Converts text chunks to embeddings using **MiniLM**  
5. 📚 Stores embeddings in a FAISS vector store  
6. 🔍 Searches for the most relevant chunks for the user's question  
7. 🤖 Uses **FLAN-T5 model locally via HuggingFacePipeline** to generate an answer  
8. 💬 Displays the answer to the user  

-------------------------------------------------------------------------------------
## 📌 ⚙️ Installation
### 📥 Install packages one by one:
```bash
pip install streamlit
pip install PyPDF2
pip install langchain
pip install langchain-community
pip install sentence-transformers
pip install transformers
pip install faiss-cpu
pip install torch
pip install huggingface-hub
```

📥Or, install everything at once:
```bash
pip install streamlit PyPDF2 langchain langchain-community sentence-transformers transformers faiss-cpu torch huggingface-hub

````

📥Or using requirements.txt:
```bash
 pip install -r requirements.txt
```

----------------------------------------------------------------------------------------------------------

📌 🚀 How to Run the App
1️. Install the packages (use one of the installation methods above)
2️. Run the Streamlit app:
```bash
    streamlit run your_pythonfile.py
```
-----------------------------------------------------------------------------------------------------------

## 📸 Screenshot
![image](https://github.com/user-attachments/assets/32b64ed5-90ba-4a83-966f-76abd2fb0def)
