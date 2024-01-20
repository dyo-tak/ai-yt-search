from flask import Flask, jsonify, request
from flask_cors import CORS
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from Sentiment import get_youtube_sentiments
from Summary import get_video_summary

app = Flask(__name__)
CORS(app)

pinecone = Pinecone(api_key="55e1c7a7-6374-4a46-9650-b075f74e2158", environment="us-west1-gcp")
index = pinecone.Index('youtube-search')

retriever = SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

@app.route("/")
def hello():
    return "Hello, World!"

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
    
@app.route("/sentiment/<video_id>")
def sentiment(video_id):
    if video_id:
        sentiments = get_youtube_sentiments(video_id)
        
        return jsonify(sentiments), 200
    else:
        return jsonify({"error": "Video ID is required"}), 400
    
@app.route("/summary/<video_id>")
def summary(video_id):
    if video_id:
        summ = get_video_summary(video_id)
        
        return jsonify({"summary":summ}), 200
    else:
        return jsonify({"error": "Video ID is required"}), 400

if __name__ == "__main__":
    app.run(debug=True)
   