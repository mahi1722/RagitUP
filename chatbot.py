import streamlit as st
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded from .env file.")

# Access the GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is not set. Please check your .env file.")
    st.error("GROQ_API_KEY is not set. Please check your .env file.")
else:
    logger.info("GROQ_API_KEY loaded successfully.")

# Streamlit UI
st.header("Chat with your PDF")

# Upload PDF file
with st.sidebar:
    st.title("Upload your PDF")
    file = st.file_uploader("Upload your PDF", type="pdf")

# Extract text from PDF
if file is not None:
    logger.info(f"Processing uploaded file: {file.name}")
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        logger.info(f"Extracted text from {len(pdf.pages)} pages.")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks.")
        st.write(f"Number of chunks: {len(chunks)}")

        # Create embeddings
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        logger.info(f"Creating embeddings using model: {model_name}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        logger.info("Vector store created successfully.")

        # Accept user questions/query
        user_query = st.text_input("Ask questions about your PDF file:")
        if user_query:
            logger.info(f"User query: {user_query}")
            try:
                # Retrieve relevant documents
                docs = vectorstore.similarity_search(query=user_query, k=3)
                logger.info(f"Retrieved {len(docs)} relevant documents.")

                # Initialize ChatGroq model
                llm = ChatGroq(
                    api_key=GROQ_API_KEY,
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                    max_retries=2
                )
                logger.info("ChatGroq model initialized.")

                # Initialize RetrievalQA chain
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )
                logger.info("RetrievalQA chain initialized.")

                # Get the answer from the chain
                response = qa_chain.invoke({"query": user_query})
                logger.info("Generated response from RetrievalQA chain.")

                # Extract and display the result
                if isinstance(response, dict) and "result" in response:
                    answer = response["result"]
                    logger.info("Extracted result from response.")
                else:
                    answer = "No valid response received."
                    logger.warning("Response does not contain a 'result' field.")

                # Display the answer in proper formatting
                st.subheader("Answer:")
                st.markdown(answer)
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                st.error(f"An error occurred: {e}")
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        st.error(f"An error occurred while processing the PDF: {e}")

# Add footer with credit
footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px 0;
            font-size: 14px;
            color: #555;
        }
    </style>
    <div class="footer">
        Created with ❤️ by Mahi
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
