import os
import re
import shutil
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

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
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Configuration ---
# MODIFICATION: Read the API key from environment variables for security.
TOGETHER_API_KEY = os.environ.get('tgp_v1_P219VY9RYZhULscfC_wx7Vt9Q6ZYf5CpqU-3-Smxrps')

REQUESTS_HEADERS = {
    'User-Agent': 'MyChatbotScraper/1.0 (mycontact@example.com)'
}
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_MODEL_LOCAL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"

# --- Global Variables ---
embeddings = None
models_loaded_successfully = False
active_sessions = {}

# --- Core Functions ---

def discover_internal_links(root_url: str) -> list[str]:
    """Crawls a website starting from the root_url to discover all internal links."""
    urls_to_visit = {root_url}
    visited_urls = set()
    all_site_links = set()
    max_pages_to_scrape = 50 

    while urls_to_visit and len(all_site_links) < max_pages_to_scrape:
        url = urls_to_visit.pop()
        if url in visited_urls:
            continue
        
        print(f"Discovering links on: {url}")
        visited_urls.add(url)
        
        try:
            response = requests.get(url, headers=REQUESTS_HEADERS, timeout=10)
            response.raise_for_status()
            
            if 'text/html' not in response.headers.get('Content-Type', ''):
                continue

            soup = BeautifulSoup(response.content, 'lxml')
            all_site_links.add(url)

            for link_tag in soup.find_all('a', href=True):
                href = link_tag['href']
                full_url = urljoin(url, href).split("#")[0]

                if full_url.startswith(root_url) and full_url not in visited_urls:
                    urls_to_visit.add(full_url)

        except requests.RequestException as e:
            print(f"Could not fetch {url}: {e}")
        except Exception as e:
            print(f"An error occurred on {url}: {e}")

    return list(all_site_links)


def extract_text_from_url(url, headers):
    """Smarter scraper that targets main content and removes the chatbot's own UI."""
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'lxml')

        for element_id in ['chatbot-toggle-button', 'chatbot-popup']:
            if (found := soup.find(id=element_id)):
                found.decompose()
        
        for tag in soup(['nav', 'footer', 'aside', 'script', 'style', 'header']):
            tag.decompose()
        
        main_content = soup.find('main') or soup.find('article') or soup.body
        if not main_content:
            return None, "Could not find main content or body."
            
        text = main_content.get_text(separator=' ', strip=True)
        text = re.sub(r'\s\s+', ' ', text)
        page_title = soup.title.string if soup.title else "Unknown Title"
        return text, page_title

    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, f"Processing Error: {e}"

def initialize_embedding_model():
    """Loads only the embedding model into memory."""
    global embeddings, models_loaded_successfully
    
    if models_loaded_successfully:
        return

    print("Initializing embedding model...")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_LOCAL,
            model_kwargs={'device': 'cpu'}
        )
        models_loaded_successfully = True
        print("Embedding model initialized successfully!")

    except Exception as e:
        print(f"FATAL: Error initializing embedding model: {e}")
        traceback.print_exc()
        models_loaded_successfully = False

def load_data_and_create_retriever(root_url: str):
    """Scrapes data, creates embeddings, and returns a retriever."""
    if not models_loaded_successfully:
        raise RuntimeError("Embedding model is not loaded.")

    url_safe_name = re.sub(r'[^a-zA-Z0-9]', '_', root_url)
    specific_persist_dir = os.path.join(PERSIST_DIRECTORY, url_safe_name)

    if os.path.exists(specific_persist_dir):
        print(f"Loading existing vector store from: {specific_persist_dir}")
        db = Chroma(persist_directory=specific_persist_dir, embedding_function=embeddings)
    else:
        print(f"No existing vector store found. Starting crawl for: {root_url}")
        
        discovered_urls = discover_all_site_links(root_url)
        if not discovered_urls:
            raise RuntimeError(f"Failed to discover any links from {root_url}.")
        
        print(f"Discovered {len(discovered_urls)} pages to scrape.")
        all_docs = []
        for url in discovered_urls:
            print(f"  Scraping: {url}")
            scraped_text, page_title = extract_text_from_url(url, REQUESTS_HEADERS)
            if scraped_text:
                all_docs.append(Document(page_content=scraped_text, metadata={"source": url, "title": page_title}))
        
        if not all_docs:
            raise RuntimeError("No documents were scraped successfully.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents(all_docs)
        
        print(f"Creating and persisting vector store at: {specific_persist_dir}")
        db = Chroma.from_documents(texts, embeddings, persist_directory=specific_persist_dir)
    
    return db.as_retriever(search_kwargs={"k": 3})

# --- Flask API Endpoints ---

@app.route("/")
def home():
    status = "Ready" if models_loaded_successfully else "Initialization Failed"
    return jsonify({"message": "RAG Chatbot API (Together.ai)", "status": status, "active_sessions": len(active_sessions)})

@app.route("/api/load_url", methods=["POST"])
def load_url_endpoint():
    data = request.get_json()
    url = data.get("url")
    session_id = data.get("session_id")

    if not all([url, session_id]):
        return jsonify({"error": "URL and session_id parameters are required."}), 400

    print(f"Received request to load data for URL: {url} (Session: {session_id})")
    
    try:
        new_retriever = load_data_and_create_retriever(url)
        active_sessions[session_id] = new_retriever
        print(f"Session {session_id} created successfully. Total active sessions: {len(active_sessions)}")
        return jsonify({"message": f"Successfully loaded data for {url}"}), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route("/api/get_response", methods=["POST"])
def get_bot_response_api():
    data = request.json
    query = data.get("query")
    session_id = data.get("session_id")

    if not all([query, session_id]):
        return jsonify({"error": "Query and session_id parameters are required."}), 400

    retriever = active_sessions.get(session_id)
    if not retriever:
        return jsonify({"error": "Invalid session ID or session has expired."}), 404

    try:
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])

        system_prompt = "You are a helpful AI assistant. Use the context provided by the user to answer their question. If the answer is not in the context, say so."
        user_prompt = f"Context:\n---\n{context}\n---\nQuestion: {query}"

        api_url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "max_tokens": 1024,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()

        api_response = response.json()
        response_text = api_response['choices'][0]['message']['content'].strip()

        return jsonify({"response": response_text}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Error during API call: {str(e)}"}), 500

print("Starting application initialization...")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)
initialize_embedding_model()

if models_loaded_successfully:
    print("-" * 50)
    print("Embedding model loaded. API is ready for multiple sessions.")
    print("Responses will be generated by Together.ai.")
    print("-" * 50)
else:
    print("-" * 50)
    print("MODEL INITIALIZATION FAILED. The API will not be functional.")
    print("-" * 50)
