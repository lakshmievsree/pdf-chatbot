import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline

# Streamlit app header
st.header("📄 PDF Chatbot (Local FLAN-T5)")

# Sidebar for uploading PDF
with st.sidebar:
    st.title("📚 Your Documents")
    files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Process the uploaded PDF
if files:
    text = ""
    for file in files:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Generate embeddings using HuggingFace's MiniLM
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # User question input
    user_question = st.text_input("💬 Type your question about the PDF")

    # If user enters a question
    if user_question:
       with st.spinner("Answering your question..."):
        # Search for relevant chunks
        match = vector_store.similarity_search(user_question)

        # Load FLAN-T5 model & tokenizer locally
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

        # Create text2text-generation pipeline
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=1024)

        # Wrap pipeline in LangChain LLM
        llm = HuggingFacePipeline(pipeline=pipe)

        # Create QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Run QA chain on matching chunks
        response = chain.run(input_documents=match, question=user_question)

        # Display the answer
        st.write("📝 Answer:")
        st.write(response)
