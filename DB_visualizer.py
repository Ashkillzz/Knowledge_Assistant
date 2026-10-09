import chromadb
from chromaviz import visualize_collection

# Connect to your Chroma persistent directory
client = chromadb.PersistentClient(path="./chroma_db")  # or your collection path
collection = client.get_collection("langchain")    # default collection name used by LangChain

# Launch the 3D visualizer web interface
visualize_collection(collection, port=5000)