from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage, AIMessage

from pdfminer.high_level import extract_text

app = Flask(__name__)
CORS(app)

# In-memory chat history (for demo only)
chat_history = []

def get_documents_from_pdf(pdf_path):
    """Extracts text from a PDF file and splits it into documents."""
    text = extract_text(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    chunks = splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"source": f"Chunk {i + 1}"}) for i, chunk in enumerate(chunks)]
    return documents

def create_vector_store(docs):
    """Creates a vector store using Gemini embeddings."""
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(docs, embedding=embedding)
    return vector_store

def create_qa_chain(vector_store):
    """Creates a Gemini-based retrieval QA chain."""
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You're a helpful assistant. Answer the user's question strictly based on the context below.\n"
         "If the requested information is not found in the provided context, please respond with: 'I apologize, I cannot find specific information. However, I can assist you with other questions about Mr. Aswin H'\n"
         "Context:\n{context}"),
        ("user", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)

    return retrieval_chain

# Initialize the PDF and chain
PDF_PATH = "document.pdf"
docs = get_documents_from_pdf(PDF_PATH)
vector_store = create_vector_store(docs)
qa_chain = create_qa_chain(vector_store)

@app.route('/')
def index():
    return 'PDF Q&A Server is running!'

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({
                'answer': None,
                'status': 'error',
                'message': 'Please provide a valid question.'
            }), 400

        question = data['question']
        chat_history.append(HumanMessage(content=question))

        result = qa_chain.invoke({"input": question})
        answer = result.get("answer", "I couldn’t find that in the document.")
        
        chat_history.append(AIMessage(content=answer))

        return jsonify({
            'answer': answer,
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'answer': None,
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
