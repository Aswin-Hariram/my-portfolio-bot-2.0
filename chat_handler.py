import logging
import time
from flask import request, jsonify, session
from functools import lru_cache
import PyPDF2

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

# Get logger from main application
logger = logging.getLogger(__name__)

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
     ("system", 
             "You are a AI assistant. Your task is to answer questions based on the Mr. Aswin H.\n"
             "1. Answer the user's questions using only the relevant information from the context.\n"
             "2. Focus on personal details such as:\n"
             "   a. Education\n"
             "   b. Location\n"
             "   c. Work experience\n"
             "   d. Skills\n"
             "3. Be clear and concise in your responses.\n"
             "4. Do not use phrases like:\n"
             "   a. 'Based on the information provided'\n"
             "   b. 'According to the context'\n"
             "5. Simply provide the direct answer.\n"
             "6. If the question is about aswin h photo or picture return '<img src=\"https://github.com/user-attachments/assets/570d7bae-1a6a-4eac-9936-d1f59f00ad6e\" alt=\"Aswin H Profile Picture\">' as response\n"
             "7. The response should be in markdown format using:\n"
             "   a. Should Compulsory to use Attractive emojis and headings \n"
             "   b. Bullet points and numbered lists\n" 
             "   c. Bold and italic text for emphasis\n"
             "   d. Code blocks where relevant\n"
             "   e. Tables for structured data\n"
             "   f. Horizontal rules for section breaks\n"
             "   g. Blockquotes for important highlights\n"
            "8. Should Compulsory to use Attractive emojis and headings\n"
             "\nContext: {context}"
            ),
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

def initialize_chat(pdf_path):
    """Initialize the chat components."""
    try:
        docs = get_documents_from_pdf(pdf_path)
        vectorStore = create_db(docs)
        chain = create_chain(vectorStore)
        logger.info("Chat components initialized successfully")
        return chain
    except Exception as e:
        logger.error(f"Failed to initialize chat components: {str(e)}")
        raise

def handle_chat(chain):
    """Handle chat interactions with the AI."""
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

def reset_chat_history():
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

# Add a new function to format responses for UI
def format_response_for_ui(text):
    """Format the response text for proper UI display."""
    if not text:
        return ""
    
    # Replace markdown formatting with HTML equivalents if needed
    formatted_text = text.strip()
    
    # Handle escaped newlines that might come from the model
    # Preserve actual newlines for multiline display
    formatted_text = formatted_text.replace("/n", "\n")
    formatted_text = formatted_text.replace("\\n", "\n")
    
    # Process each line separately to handle multiple spaces
    lines = formatted_text.split("\n")
    processed_lines = []
    for line in lines:
        # Remove multiple spaces within each line
        processed_line = " ".join(line.split())
        processed_lines.append(processed_line)
    
    # Rejoin with newlines
    formatted_text = "\n".join(processed_lines)
    
    # Remove markdown formatting characters
    markdown_chars = ["*", "_", "#", "`", ">", "-", "+"]
    for char in markdown_chars:
        formatted_text = formatted_text.replace(char, "")
    
    # Specifically handle "**" bold formatting
    formatted_text = formatted_text.replace("**", "")
    
    # Handle any special HTML entities if needed
    html_entities = {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
    }
    for entity, replacement in html_entities.items():
        formatted_text = formatted_text.replace(entity, replacement)
    
    return formatted_text