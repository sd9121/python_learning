# Import required modules
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeEmbeddings, PineconeVectorStore
import os
import time

# Set a namespace for organizing vectors inside Pinecone
namespace = "wondervector5000"

# Define a markdown document that we want to split and upload
markdown_document = "## Introduction\n\nWelcome to the whimsical world of the WonderVector5000, an astonishing leap into the realms of imaginative technology. This extraordinary device, borne of creative fancy, promises to revolutionize absolutely nothing while dazzling you with its fantastical features. Whether you're a seasoned technophile or just someone looking for a bit of fun, the WonderVector5000 is sure to leave you amused and bemused in equal measure. Let's explore the incredible, albeit entirely fictitious, specifications, setup process, and troubleshooting tips for this marvel of modern nonsense.\n\n## Product overview\n\nThe WonderVector5000 is packed with features that defy logic and physics, each designed to sound impressive while maintaining a delightful air of absurdity:\n\n- Quantum Flibberflabber Engine: The heart of the WonderVector5000, this engine operates on principles of quantum flibberflabber, a phenomenon as mysterious as it is meaningless. It's said to harness the power of improbability to function seamlessly across multiple dimensions.\n\n- Hyperbolic Singularity Matrix: This component compresses infinite possibilities into a singular hyperbolic state, allowing the device to predict outcomes with 0% accuracy, ensuring every use is a new adventure.\n\n- Aetherial Flux Capacitor: Drawing energy from the fictional aether, this flux capacitor provides unlimited power by tapping into the boundless reserves of imaginary energy fields.\n\n- Multi-Dimensional Holo-Interface: Interact with the WonderVector5000 through its holographic interface that projects controls and information in three-and-a-half dimensions, creating a user experience that's simultaneously futuristic and perplexing.\n\n- Neural Fandango Synchronizer: This advanced feature connects directly to the user's brain waves, converting your deepest thoughts into tangible actions—albeit with results that are whimsically unpredictable.\n\n- Chrono-Distortion Field: Manipulate time itself with the WonderVector5000's chrono-distortion field, allowing you to experience moments before they occur or revisit them in a state of temporal flux.\n\n## Use cases\n\nWhile the WonderVector5000 is fundamentally a device of fiction and fun, let's imagine some scenarios where it could hypothetically be applied:\n\n- Time Travel Adventures: Use the Chrono-Distortion Field to visit key moments in history or glimpse into the future. While actual temporal manipulation is impossible, the mere idea sparks endless storytelling possibilities.\n\n- Interdimensional Gaming: Engage with the Multi-Dimensional Holo-Interface for immersive, out-of-this-world gaming experiences. Imagine games that adapt to your thoughts via the Neural Fandango Synchronizer, creating a unique and ever-changing environment.\n\n- Infinite Creativity: Harness the Hyperbolic Singularity Matrix for brainstorming sessions. By compressing infinite possibilities into hyperbolic states, it could theoretically help unlock unprecedented creative ideas.\n\n- Energy Experiments: Explore the concept of limitless power with the Aetherial Flux Capacitor. Though purely fictional, the notion of drawing energy from the aether could inspire innovative thinking in energy research.\n\n## Getting started\n\nSetting up your WonderVector5000 is both simple and absurdly intricate. Follow these steps to unleash the full potential of your new device:\n\n1. Unpack the Device: Remove the WonderVector5000 from its anti-gravitational packaging, ensuring to handle with care to avoid disturbing the delicate balance of its components.\n\n2. Initiate the Quantum Flibberflabber Engine: Locate the translucent lever marked “QFE Start” and pull it gently. You should notice a slight shimmer in the air as the engine engages, indicating that quantum flibberflabber is in effect.\n\n3. Calibrate the Hyperbolic Singularity Matrix: Turn the dials labeled 'Infinity A' and 'Infinity B' until the matrix stabilizes. You'll know it's calibrated correctly when the display shows a single, stable “∞”.\n\n4. Engage the Aetherial Flux Capacitor: Insert the EtherKey into the designated slot and turn it clockwise. A faint humming sound should confirm that the aetherial flux capacitor is active.\n\n5. Activate the Multi-Dimensional Holo-Interface: Press the button resembling a floating question mark to activate the holo-interface. The controls should materialize before your eyes, slightly out of phase with reality.\n\n6. Synchronize the Neural Fandango Synchronizer: Place the neural headband on your forehead and think of the word “Wonder”. The device will sync with your thoughts, a process that should take just a few moments.\n\n7. Set the Chrono-Distortion Field: Use the temporal sliders to adjust the time settings. Recommended presets include “Past”, “Present”, and “Future”, though feel free to explore other, more abstract temporal states.\n\n## Troubleshooting\n\nEven a device as fantastically designed as the WonderVector5000 can encounter problems. Here are some common issues and their solutions:\n\n- Issue: The Quantum Flibberflabber Engine won't start.\n\n    - Solution: Ensure the anti-gravitational packaging has been completely removed. Check for any residual shards of improbability that might be obstructing the engine.\n\n- Issue: The Hyperbolic Singularity Matrix displays “∞∞”.\n\n    - Solution: This indicates a hyper-infinite loop. Reset the dials to zero and then adjust them slowly until the display shows a single, stable infinity symbol.\n\n- Issue: The Aetherial Flux Capacitor isn't engaging.\n\n    - Solution: Verify that the EtherKey is properly inserted and genuine. Counterfeit EtherKeys can often cause malfunctions. Replace with an authenticated EtherKey if necessary.\n\n- Issue: The Multi-Dimensional Holo-Interface shows garbled projections.\n\n    - Solution: Realign the temporal resonators by tapping the holographic screen three times in quick succession. This should stabilize the projections.\n\n- Issue: The Neural Fandango Synchronizer causes headaches.\n\n    - Solution: Ensure the headband is properly positioned and not too tight. Relax and focus on simple, calming thoughts to ease the synchronization process.\n\n- Issue: The Chrono-Distortion Field is stuck in the past.\n\n    - Solution: Increase the temporal flux by 5%. If this fails, perform a hard reset by holding down the “Future” slider for ten seconds."

# Define which markdown headers we want to split on (H2 headers in this case)
headers_to_split_on = [
    ("##", "Header 2")
]

# Create a markdown splitter that uses the above rule
markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False  # Keep the header text in the output
)

# Split the markdown into chunks based on headers
md_header_splits = markdown_splitter.split_text(markdown_document)

# Set up the embedding model to convert text chunks into vector embeddings
model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key=os.environ.get('PINECONE_API_KEY')  # Load Pinecone API key from environment
)

# Initialize the Pinecone client
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Get cloud and region info (or use defaults)
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

# Define your index name
index_name = "rag-getting-started"

# Create the index if it doesn’t exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,  # Must match the embedding model's output size
        metric="cosine",  # Use cosine similarity to compare vectors
        spec=spec  # Cloud and region settings
    )

# Upload (upsert) the document chunks to Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=md_header_splits,      # Your split markdown sections
    index_name=index_name,           # Index name to insert into
    embedding=embeddings,            # Embedding model to convert text to vectors
    namespace=namespace              # Namespace for logical separation
)

# Wait for upsert to complete
time.sleep(5)

# Show how many vectors were inserted
print("Index after upsert:")
print(pc.Index(index_name).describe_index_stats())
print("\n")
time.sleep(2)

# === Querying the data we just inserted === #

# Connect to the Pinecone index
index = pc.Index(index_name)

# Define namespace (same as used during upload)
namespace = "wondervector5000"

# Loop through all vector IDs in the namespace
for ids in index.list(namespace=namespace):
    # Query each vector by its ID and get the top-1 most similar vector
    query = index.query(
        id=ids[3],                # Use the first vector ID from the list
        namespace=namespace,      # Search within our namespace
        top_k=1,                  # Get only the top 1 match
        include_values=True,      # Include vector values in the result
        include_metadata=True     # Include text and header metadata
    )
    # Print the query result
    print(query)
    print("\n")

#When you no longer need the rag-getting-started index, use the delete_index operation to delete it:

# pc.delete_index(index_name)