from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
from dotenv import load_dotenv
load_dotenv()
import time
import logging
from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))  # For session management
CORS(app, supports_credentials=True)  # Enable credentials for session cookies

# Cache for document processing to improve performance
@lru_cache(maxsize=1)
def get_documents_from_pdf(pdf_path):
    """Loads and processes the PDF file to extract text with caching."""
    logger.info(f"Loading PDF from {pdf_path}")
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increased from 400
            chunk_overlap=100  # Increased from 20
        )
        splitDocs = splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in splitDocs]
        logger.info(f"Successfully processed PDF with {len(documents)} chunks")
        return documents
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise

def create_db(docs):
    """Creates a vector store from the documents."""
    try:
        start_time = time.time()
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorStore = FAISS.from_documents(docs, embedding=embedding)
        logger.info(f"Vector store created in {time.time() - start_time:.2f} seconds")
        return vectorStore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def create_chain(vectorStore):
    """Creates a history-aware retriever chain that will be used for chat."""
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.9
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the user's questions using only the relevant information from the context below. Pay special attention to personal details like education, location, work experience, and skills. Be clear and concise. Do not use phrases like 'Based on the information provided' or 'According to the context'. Just provide the answer directly.\nContext: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}")
        ])

        chain = create_stuff_documents_chain(
            llm=model,
            prompt=prompt
        )

        retriever = vectorStore.as_retriever(search_kwargs={"k": 5})  # Increased from 3

        retriever_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm=model,
            retriever=retriever,
            prompt=retriever_prompt
        )

        retrieval_chain = create_retrieval_chain(
            history_aware_retriever,
            chain
        )

        logger.info("Created history-aware retrieval chain")
        return retrieval_chain
    except Exception as e:
        logger.error(f"Error creating chain: {str(e)}")
        raise

# Initialize global variables
try:
    pdf_path = os.environ.get("PDF_PATH", 'document.pdf')
    docs = get_documents_from_pdf(pdf_path)
    vectorStore = create_db(docs)
    chain = create_chain(vectorStore)
    logger.info("Application initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize application: {str(e)}")
    # We'll continue and handle this in the routes

@app.route('/')
def home():
    return jsonify({
        'status': 'success',
        'message': 'Portfolio AI API is running',
        'version': '2.0'
    })

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    try:
        # Input validation
        data = request.get_json()
        if not data:
            return jsonify({
                'answer': None,
                'status': 'error',
                'message': 'Invalid request format. JSON required'
            }), 400
            
        if 'question' not in data:
            return jsonify({
                'answer': None,
                'status': 'error',
                'message': 'Invalid request. Please provide a question'
            }), 400
        
        question = data['question']
        
        # Get or initialize chat history from session
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        chat_history = []
        for msg in session['chat_history']:
            if msg['type'] == 'human':
                chat_history.append(HumanMessage(content=msg['content']))
            else:
                chat_history.append(AIMessage(content=msg['content']))
        
        # Process the chat message with history
        response = chain.invoke({
            "chat_history": chat_history,
            "input": question
        })
        
        # Update session with new messages
        session['chat_history'].append({'type': 'human', 'content': question})
        session['chat_history'].append({'type': 'ai', 'content': response["answer"]})
        
        # Limit history size to prevent session bloat
        if len(session['chat_history']) > 20:  # Keep last 10 exchanges (20 messages)
            session['chat_history'] = session['chat_history'][-20:]
        
        processing_time = time.time() - start_time
        logger.info(f"Processed chat in {processing_time:.2f} seconds")
        
        return jsonify({
            'answer': response["answer"],
            'status': 'success',
            'processing_time': processing_time
        })
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'answer': None,
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/reset', methods=['POST'])
def reset_chat():
    """Reset the chat history."""
    try:
        session.pop('chat_history', None)
        return jsonify({
            'status': 'success',
            'message': 'Chat history reset successfully'
        })
    except Exception as e:
        logger.error(f"Error resetting chat: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))  # Default to 8000 for local dev
    app.run(host="0.0.0.0", port=port)