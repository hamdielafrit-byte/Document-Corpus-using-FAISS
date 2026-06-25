import os
import faiss
import numpy as np
import ollama
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

MODEL_NAME = "nomic-embed-text"

# 1. Our Corpus (The database users will search through)
corpus = [
    "Artificial Intelligence is transforming industries like healthcare and finance.",
    "FAISS is an open-source library built by Meta for efficient vector similarity search.",
    "Machine learning models require high-quality, cleaned datasets to perform accurately.",
    "Python is widely used for data science, web development, and AI engineering tasks.",
    "Large Language Models (LLMs) understand context by turning sentences into numerical embeddings.",
    "Vector databases allow applications to remember long-term interactions with AI.",
]

# 2. Build the FAISS Index globally when the server starts
def initialize_faiss():
    try:
        print("Generating embeddings with Ollama... please wait.")
        response = ollama.embed(model=MODEL_NAME, input=corpus)
        embeddings = np.array(response["embeddings"], dtype="float32")

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings)
        print("FAISS Index successfully built!")
        return index
    except Exception as e:
        print(f"Error initializing FAISS: {e}")
        print("Make sure Ollama is running (`ollama serve`)!")
        return None


index = initialize_faiss()


# 3. Web Route: Serves the HTML frontend interface
@app.route("/")
def home():
    return render_template("index.html", corpus=corpus)


# 4. API Route: Handles incoming search queries from the webpage
@app.route("/search", methods=["POST"])
def search():
    if index is None:
        return (
            jsonify({"error": "FAISS index not loaded. Is Ollama running?"}),
            500,
        )

    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"results": []})

    try:
        # Vectorize the user's natural language input
        query_response = ollama.embed(model=MODEL_NAME, input=query)
        query_vector = np.array(query_response["embeddings"], dtype="float32")

        # Ask FAISS to find the top 2 closest matching blocks
        D, I = index.search(query_vector, k=2)

        # Build clean JSON response
        results = []
        for rank, idx in enumerate(I[0]):
            results.append(
                {
                    "text": corpus[idx],
                    "score": float(
                        D[0][rank]
                    ),  # L2 distance (lower score = closer match)
                }
            )

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Runs a local server on http://127.0.0.1:5000
    app.run(debug=True)