from flask import Flask, jsonify, request
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

pinecone = Pinecone(api_key="55e1c7a7-6374-4a46-9650-b075f74e2158", environment="us-west1-gcp")
index = pinecone.Index('youtube-search')

retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')


@app.route("/search", methods=["POST"])
def search():
    query = request.json.get("query")

    if query:
        xq = retriever.encode([query]).tolist()
        xc = index.query(vector=xq, top_k=5, include_metadata=True)
        xc_dict = xc.to_dict()
        
        return jsonify(xc_dict), 200
    else:
        return jsonify({"error": "Query is required"}), 400

if __name__ == "__main__":
    app.run(debug=True)
   