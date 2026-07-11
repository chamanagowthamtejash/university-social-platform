from flask import Flask, render_template, request, jsonify
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os
import uuid
import time

# ==========================================
# FLASK APP
# ==========================================

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==========================================
# LOAD MISTRAL
# ==========================================

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY environment variable not set")

client = MistralClient(api_key=api_key)

# ==========================================
# LOAD PINECONE
# ==========================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

PINECONE_INDEX_NAME = "demo-app"
pc = Pinecone(api_key=PINECONE_API_KEY)

# ==========================================
# EMBEDDINGS MODEL
# ==========================================

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"token": HF_TOKEN},
    encode_kwargs={"normalize_embeddings": True}
)

EMBED_DIM = 768
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Check if index exists, if not create it
existing_indexes = [i.name for i in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"Creating index: {PINECONE_INDEX_NAME}")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    time.sleep(5)

index = pc.Index(PINECONE_INDEX_NAME)

# ==========================================
# GLOBAL VARIABLES
# ==========================================

current_pdf_name = ""
current_namespace = ""
pdf_uploaded = False

# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_vector_db(pdf_path):
    global current_pdf_name, current_namespace, pdf_uploaded

    current_pdf_name = os.path.basename(pdf_path)
    current_namespace = str(uuid.uuid4())
    pdf_uploaded = False

    print("=" * 60)
    print(f"Processing PDF: {current_pdf_name}")
    print(f"Namespace: {current_namespace}")
    print("=" * 60)

    try:
        # Load PDF
        print("Loading PDF...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        if not documents:
            raise ValueError("No content found in PDF")

        print(f"Loaded {len(documents)} pages")

        # Split into chunks
        print("Splitting PDF into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
        )
        docs = splitter.split_documents(documents)
        print(f"Created {len(docs)} chunks")

        # Generate embeddings
        print("Generating embeddings...")
        texts = [doc.page_content for doc in docs]

        # Process in batches
        batch_size = 50
        all_vectors = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_vectors = embeddings.embed_documents(batch_texts)
            all_vectors.extend(batch_vectors)
            print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        # Verify vector dimensions
        if all_vectors:
            print(f"Vector dimension: {len(all_vectors[0])}")
            if len(all_vectors[0]) != EMBED_DIM:
                raise ValueError(f"Vector dimension {len(all_vectors[0])} does not match index dimension {EMBED_DIM}")

        # Prepare for Pinecone upsert
        print("Uploading to Pinecone...")
        upserts = []
        for i, (doc, vec) in enumerate(zip(docs, all_vectors)):
            upserts.append({
                "id": f"{current_namespace}-{i}",
                "values": vec.tolist() if hasattr(vec, 'tolist') else vec,
                "metadata": {
                    "text": doc.page_content,
                    "page": doc.metadata.get("page", 0),
                    "source": current_pdf_name
                }
            })

        # Upsert in batches
        batch_size = 100
        for i in range(0, len(upserts), batch_size):
            batch = upserts[i:i+batch_size]
            index.upsert(vectors=batch, namespace=current_namespace)
            print(f"Upserted batch {i//batch_size + 1}/{(len(upserts)-1)//batch_size + 1}")

        # Verify upload
        stats = index.describe_index_stats()
        print(f"Index stats: {stats}")

        pdf_uploaded = True
        print("=" * 60)
        print(f"✅ PDF Processing Complete!")
        print(f"Total chunks uploaded: {len(upserts)}")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        pdf_uploaded = False
        raise e

# ==========================================
# RETRIEVE RELEVANT CONTEXT
# ==========================================

def retrieve_info(query):
    global current_namespace, pdf_uploaded

    if not pdf_uploaded or current_namespace == "":
        return ""

    try:
        # Format query with BGE prefix
        query_text = BGE_QUERY_PREFIX + query
        query_vector = embeddings.embed_query(query_text)

        # Query Pinecone
        results = index.query(
            vector=query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector,
            top_k=5,
            namespace=current_namespace,
            include_metadata=True
        )

        # Debug output
        print(f"Query: {query}")
        matches = results.get('matches', [])
        print(f"Matches found: {len(matches)}")

        # Extract context
        context = []
        seen = set()

        for match in matches:
            metadata = match.get("metadata", {})
            text = metadata.get("text", "").strip()
            score = match.get("score", 0)

            if text and score > 0.3 and text not in seen:
                seen.add(text)
                page = metadata.get("page", "Unknown")
                context.append(f"[Page {page + 1}]\n{text}")

        if not context:
            return ""

        return "\n\n".join(context)

    except Exception as e:
        print(f"Error in retrieve_info: {str(e)}")
        return ""

# ==========================================
# GREETING CHECK
# ==========================================

def is_greeting(message):
    greetings = [
        "hi", "hello", "hey", "good morning",
        "good afternoon", "good evening", "hi there",
        "hello there", "hey there"
    ]
    return message.lower().strip() in greetings

# ==========================================
# HELP QUESTIONS
# ==========================================

def is_help_question(message):
    message = message.lower()
    keywords = [
        "what can you do", "help", "how can you help",
        "what kind of questions", "example questions",
        "capabilities", "features"
    ]
    return any(word in message for word in keywords)

# ==========================================
# AGENT
# ==========================================

def agent(message):
    global current_pdf_name, pdf_uploaded

    # Check if PDF is uploaded
    if not pdf_uploaded:
        return "Please upload a PDF first before asking questions."

    # Greeting
    if is_greeting(message):
        return f"Hello! I am ready to answer questions from the uploaded PDF '{current_pdf_name}'. Ask me anything about its contents."

    # Help
    if is_help_question(message):
        return """
You can ask me anything that is present in the uploaded PDF.

Examples:
- Summarize this document
- Explain this topic
- What is the conclusion?
- Who are the authors?
- List the important points
- What are the key findings?
- Define a term
- Compare two concepts
- Give a short summary
- What is the purpose of this document?

I'll answer only from the uploaded PDF.
"""

    # Retrieve context
    context = retrieve_info(message)

    if context.strip() == "":
        return "I couldn't find relevant information in the uploaded PDF. Please try rephrasing your question or ask about a different topic."

    # Prepare prompt for Mistral
    prompt = f"""
You are an AI assistant for Sasi Institute of Technology and Engineering.

Your task is to answer user questions ONLY using the information retrieved from the provided documents.

Retrieved Context:
{context}

User Question: {message}

Instructions:
1. Answer ONLY using the retrieved context above
2. If the answer is not available in the context, respond: "I couldn't find that information in the available college documents."
3. Do not make assumptions or generate information not present in the context
4. Keep answers clear, concise, and well-structured
5. For lists, use bullet points with dashes (-) or numbers
6. If multiple relevant sections are retrieved, combine them into one complete answer
7. Be polite and professional
8. Mention the page numbers when referencing information
9. IMPORTANT: Do NOT use bold text, asterisks (*), or any markdown formatting in your response
10. Use plain text only - no bold, no italics, no markdown

Answer:
"""

    # Get response from Mistral
    try:
        response = client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(role="user", content=prompt)
            ],
            temperature=0.1,
            max_tokens=1000
        )

        # Clean response of any markdown/bold formatting
        answer = response.choices[0].message.content.strip()
        # Remove asterisks (bold/italic)
        answer = answer.replace('*', '')
        # Remove other markdown characters if needed
        answer = answer.replace('_', '')
        answer = answer.replace('`', '')

        return answer

    except Exception as e:
        print(f"Error in Mistral API: {str(e)}")
        return f"Error generating response: {str(e)}"

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")

# ==========================================
# UPLOAD PDF
# ==========================================

@app.route("/upload", methods=["POST"])
def upload_pdf():
    global pdf_uploaded, current_pdf_name, current_namespace

    if "pdf" not in request.files:
        return jsonify({
            "status": "error",
            "message": "Please select a PDF."
        })

    file = request.files["pdf"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Please choose a PDF."
        })

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({
            "status": "error",
            "message": "Please upload a PDF file."
        })

    # Reset previous document
    pdf_uploaded = False
    current_namespace = ""
    current_pdf_name = ""

    # Save and process
    pdf_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(pdf_path)

    try:
        create_vector_db(pdf_path)

        return jsonify({
            "status": "success",
            "message": f"{file.filename} uploaded and processed successfully. You can now ask questions!"
        })

    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing PDF: {str(e)}"
        })

# ==========================================
# CHAT
# ==========================================

@app.route("/chat", methods=["POST"])
def chat():
    global pdf_uploaded

    if not pdf_uploaded:
        return jsonify({
            "response": "Please upload a PDF first before asking questions."
        })

    data = request.get_json()
    if not data:
        return jsonify({
            "response": "Please enter a question."
        })

    user_message = data.get("message", "").strip()
    if user_message == "":
        return jsonify({
            "response": "Please enter a question."
        })

    bot_response = agent(user_message)

    return jsonify({
        "response": bot_response
    })

# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "pdf_uploaded": pdf_uploaded,
        "pdf_name": current_pdf_name if pdf_uploaded else None,
        "namespace": current_namespace if pdf_uploaded else None,
        "index_dimension": EMBED_DIM,
        "model": "BAAI/bge-base-en-v1.5"
    })

# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
