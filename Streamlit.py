import streamlit as st
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

PINECONE_API = os.getenv("PINECONE_API")

@st.cache_resource
def init_pinecone():
    pinecone = Pinecone(api_key=PINECONE_API, environment="us-west1-gcp")

    return pinecone.Index('youtube-search')
    
@st.cache_resource
def init_retriever():
    return SentenceTransformer('flax-sentence-embeddings/all_datasets_v3_mpnet-base')

index = init_pinecone()
retriever = init_retriever()

import streamlit as st

def card(thumbnail, title, url, context):
    return st.markdown(f"""
    <div class="container-fluid">
        <div class="row align-items-start">
            <div class="col-md-4 col-sm-4">
                 <div class="position-relative">
                     <a href={url}><img src={thumbnail} class="img-fluid" style="width: 192px; height: 106px"></a>
                 </div>
             </div>
             <div  class="col-md-8 col-sm-8">
                 <a href={url}>{title}</a>
                 <br>
                 <span style="color: #808080;">
                     <small>{context[:200].capitalize()+"...."}</small>
                 </span>
             </div>
        </div>
     </div>
        """, unsafe_allow_html=True)

    
st.title('YourTube Search')

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.0.0/dist/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
""", unsafe_allow_html=True)


query = st.text_input("Search!", "")

if query != "":
    xq = retriever.encode([query]).tolist()
    xc = index.query(vector=xq, top_k=5, include_metadata=True)
    
    for context in xc['matches']:
        card(
            context['metadata']['thumbnail'],
            context['metadata']['title'],
            context['metadata']['url'],
            context['metadata']['text']
        )