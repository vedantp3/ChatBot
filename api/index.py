import os
import re
import shutil
import traceback
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve

# --- Langchain and Model Imports ---
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_community.vectorstores import Chroma

# --- Web Scraping and API Imports ---
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Configuration ---
TOGETHER_API_KEY = os.environ.get('tgp_v1_P219VY9RYZhULscfC_wx7Vt9Q6ZYf5CpqU-3-Smxrps')
REQUESTS_HEADERS = { 'User-Agent': 'MyChatbotScraper/1.0 (mycontact@example.com)' }
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_LOCAL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"

# --- Global Variables ---
embeddings = None
# This dictionary will now hold the status and the database for each session
active_sessions = {}

# --- Helper Function for API Calls ---
def get_embeddings_from_api(texts: list[str]) -> list[list[float]]:
    """Gets embeddings for a list of texts using the Together.ai API."""
    api_url = "https://api.together.xyz/v1/embeddings"
    headers = { "Authorization": f"Bearer {TOGETHER_API_KEY}" }
    
    # The API might have a limit on the number of texts per request, so we batch them.
    all_embeddings = []
    batch_size = 50 # A reasonable batch size
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        payload = { "model": "togethercomputer/m2-bert-80M-8k-retrieval", "input": batch }
        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        all_embeddings.extend([data['embedding'] for data in result['data']])
    return all_embeddings

# --- Core Functions ---

def discover_all_site_links(root_url: str) -> list[str]:
    urls_to_visit, visited_urls, all_site_links = {root_url}, set(), set()
    max_pages = 30 # Reduced limit to speed up the process
    while urls_to_visit and len(all_site_links) < max_pages:
        url = urls_to_visit.pop()
        if url in visited_urls: continue
        visited_urls.add(url)
        try:
            response = requests.get(url, headers=REQUESTS_HEADERS, timeout=10)
            if 'text/html' in response.headers.get('Content-Type', ''):
                soup = BeautifulSoup(response.content, 'lxml')
                all_site_links.add(url)
                for link_tag in soup.find_all('a', href=True):
                    full_url = urljoin(url, link_tag['href']).split("#")[0]
                    if full_url.startswith(root_url) and full_url not in visited_urls:
                        urls_to_visit.add(full_url)
        except requests.RequestException as e:
            print(f"Could not fetch {url}: {e}")
    return list(all_site_links)

def extract_text_from_url(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'lxml')
        for tag_id in ['chatbot-toggle-button', 'chatbot-popup']:
            if (tag := soup.find(id=tag_id)): tag.decompose()
        for tag in soup(['nav', 'footer', 'aside', 'script', 'style', 'header']):
            tag.decompose()
        main_content = soup.find('main') or soup.find('article') or soup.body
        return re.sub(r'\s\s+', ' ', main_content.get_text(separator=' ', strip=True)), soup.title.string if soup.title else "Unknown"
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, None

def background_load_and_process(url, session_id):
    """This function runs in a background thread to avoid timeouts."""
    try:
        print(f"[{session_id}] Background processing started for {url}")
        url_safe_name = re.sub(r'[^a-zA-Z0-9]', '_', url)
        persist_dir = os.path.join(PERSIST_DIRECTORY, url_safe_name)

        if os.path.exists(persist_dir):
            print(f"[{session_id}] Loading existing vector store.")
            db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        else:
            urls = discover_all_site_links(url)
            docs = [Document(page_content=text, metadata={"source": doc_url}) for doc_url in urls if (text := extract_text_from_url(doc_url, REQUESTS_HEADERS)[0])]
            if not docs: raise RuntimeError("No documents scraped.")
            
            texts = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(docs)
            print(f"[{session_id}] Getting embeddings for {len(texts)} chunks via API...")
            
            text_contents = [doc.page_content for doc in texts]
            text_embeddings = get_embeddings_from_api(text_contents)
            text_metadatas = [doc.metadata for doc in texts]

            print(f"[{session_id}] Creating and persisting vector store.")
            db = Chroma.from_texts(texts=text_contents, embeddings=text_embeddings, metadatas=text_metadatas, persist_directory=persist_dir)

        active_sessions[session_id]['db'] = db
        active_sessions[session_id]['status'] = 'ready'
        print(f"[{session_id}] Background processing finished successfully.")

    except Exception as e:
        print(f"[{session_id}] Error in background task: {e}")
        traceback.print_exc()
        active_sessions[session_id]['status'] = 'error'

# --- Flask API Endpoints ---

@app.route("/")
def home():
    return jsonify({"message": "RAG Chatbot API (Async)", "status": "Ready"})

@app.route("/api/load_url", methods=["POST"])
def load_url_endpoint():
    data = request.get_json()
    url, session_id = data.get("url"), data.get("session_id")
    if not all([url, session_id]): return jsonify({"error": "URL and session_id required."}), 400

    active_sessions[session_id] = {'status': 'loading', 'db': None}
    
    # Start the long-running task in a background thread
    thread = threading.Thread(target=background_load_and_process, args=(url, session_id))
    thread.start()
    
    # Immediately return a response to the client
    return jsonify({"message": "Processing started.", "session_id": session_id}), 202

@app.route("/api/load_status/<session_id>", methods=["GET"])
def load_status_endpoint(session_id):
    session = active_sessions.get(session_id)
    if not session: return jsonify({"status": "not_found"}), 404
    return jsonify({"status": session['status']})

@app.route("/api/get_response", methods=["POST"])
def get_bot_response_api():
    data = request.json
    query, session_id = data.get("query"), data.get("session_id")
    if not all([query, session_id]): return jsonify({"error": "Query and session_id required."}), 400

    session = active_sessions.get(session_id)
    if not session or session['status'] != 'ready':
        return jsonify({"error": "Session not ready or invalid."}), 404

    try:
        db = session['db']
        query_embedding = get_embeddings_from_api([query])[0]
        docs = db.similarity_search_by_vector(embedding=query_embedding, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        system_prompt = "You are a helpful AI assistant. Use the context provided by the user to answer their question. If the answer is not in the context, say so."
        user_prompt = f"Context:\n---\n{context}\n---\nQuestion: {query}"

        api_url = "https://api.together.xyz/v1/chat/completions"
        headers = { "Authorization": f"Bearer {TOGETHER_API_KEY}" }
        payload = {
            "model": "meta-llama/Llama-3-8B-Instruct-Turbo", # Switched to a faster model
            "max_tokens": 1024, "temperature": 0.7,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        api_response = response.json()
        response_text = api_response['choices'][0]['message']['content'].strip()
        return jsonify({"response": response_text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error during API call: {str(e)}"}), 500

# Initialize the embedding model once on startup
print("Initializing embedding model...")
try:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_LOCAL, model_kwargs={'device': 'cpu'})
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"FATAL: Could not load embedding model. {e}")
