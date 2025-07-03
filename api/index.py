import os
import re
import shutil
import traceback
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
# MODIFICATION: Changed to a more general CORS configuration to ensure all origins are allowed.
CORS(app)

# --- Configuration ---
TOGETHER_API_KEY = os.environ.get('tgp_v1_P219VY9RYZhULscfC_wx7Vt9Q6ZYf5CpqU-3-Smxrps')

REQUESTS_HEADERS = {
    'User-Agent': 'MyChatbotScraper/1.0 (mycontact@example.com)'
}
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_LOCAL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIRECTORY = "chroma_db"

# --- Global Variables ---
embeddings = None
models_loaded_successfully = True
active_sessions = {}

# --- Helper Function for API Calls ---
def get_embeddings_from_api(texts: list[str]) -> list[list[float]]:
    """Gets embeddings for a list of texts using the Together.ai API."""
    api_url = "https://api.together.xyz/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "togethercomputer/m2-bert-80M-8k-retrieval",
        "input": texts
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return [data['embedding'] for data in result['data']]

# --- Core Functions ---

def discover_all_site_links(root_url: str) -> list[str]:
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

    return list(all_site_links)


def extract_text_from_url(url, headers):
    """Smarter scraper that targets main content."""
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
        if not main_content: return None, "No main content."
            
        text = main_content.get_text(separator=' ', strip=True)
        return re.sub(r'\s\s+', ' ', text), soup.title.string if soup.title else "Unknown Title"

    except Exception as e:
        print(f"Error processing {url}: {e}")
        return None, f"Processing Error: {e}"

def load_data_and_create_db(root_url: str):
    """Scrapes data, gets embeddings via API, and creates a Chroma DB."""
    url_safe_name = re.sub(r'[^a-zA-Z0-9]', '_', root_url)
    specific_persist_dir = os.path.join(PERSIST_DIRECTORY, url_safe_name)

    if os.path.exists(specific_persist_dir):
        print(f"Loading existing vector store from: {specific_persist_dir}")
        dummy_embed_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={'device':'cpu'})
        db = Chroma(persist_directory=specific_persist_dir, embedding_function=dummy_embed_func)
    else:
        print(f"No existing vector store found. Starting crawl for: {root_url}")
        
        discovered_urls = discover_all_site_links(root_url)
        if not discovered_urls: raise RuntimeError(f"Failed to discover links from {root_url}.")
        
        print(f"Discovered {len(discovered_urls)} pages to scrape.")
        all_docs = []
        for url in discovered_urls:
            print(f"  Scraping: {url}")
            scraped_text, page_title = extract_text_from_url(url, REQUESTS_HEADERS)
            if scraped_text:
                all_docs.append(Document(page_content=scraped_text, metadata={"source": url, "title": page_title}))
        
        if not all_docs: raise RuntimeError("No documents scraped.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents(all_docs)
        
        print(f"Getting embeddings for {len(texts)} chunks via API...")
        text_contents = [doc.page_content for doc in texts]
        text_embeddings = get_embeddings_from_api(text_contents)
        text_metadatas = [doc.metadata for doc in texts]

        print(f"Creating and persisting vector store at: {specific_persist_dir}")
        db = Chroma.from_texts(
            texts=text_contents,
            embedding=None,
            embeddings=text_embeddings,
            metadatas=text_metadatas,
            persist_directory=specific_persist_dir
        )
    
    return db

# --- Flask API Endpoints ---

@app.route("/")
def home():
    return jsonify({"message": "RAG Chatbot API (Together.ai)", "status": "Ready"})

@app.route("/api/load_url", methods=["POST"])
def load_url_endpoint():
    data = request.get_json()
    url = data.get("url")
    session_id = data.get("session_id")

    if not all([url, session_id]):
        return jsonify({"error": "URL and session_id parameters are required."}), 400

    print(f"Received request to load data for URL: {url} (Session: {session_id})")
    
    try:
        new_db = load_data_and_create_db(url)
        active_sessions[session_id] = new_db
        print(f"Session {session_id} created. Total active sessions: {len(active_sessions)}")
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

    db = active_sessions.get(session_id)
    if not db:
        return jsonify({"error": "Invalid session ID or session has expired."}), 404

    try:
        query_embedding = get_embeddings_from_api([query])[0]
        docs = db.similarity_search_by_vector(embedding=query_embedding, k=3)
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



