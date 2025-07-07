from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings
import os
import time

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY')
)


index_name = "rag-getting-started"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec=spec
    )

# See that it is empty
print("Index before upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")
