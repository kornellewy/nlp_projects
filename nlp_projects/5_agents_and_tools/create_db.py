from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Define the directory containing the text file and the persistent directory
current_dir = Path(__file__).parent
file_path = current_dir /  "books"/ "odyssey.txt"
persistent_directory = current_dir / "db" / "chroma_db"

if not persistent_directory.exists():
    if file_path.exists():
        # Load the text from the file
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        # Split the text into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # Display information about the split documents
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")
        print(f"Sample chunk:\n{docs[0].page_content}\n")

        # Create embeddings
        print("\n--- Creating embeddings ---")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )  # Update to a valid embedding model if needed
        print("\n--- Finished creating embeddings ---")

         # Create the vector store and persist it automatically
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=str(persistent_directory))
        db.persist()
        print("\n--- Finished creating vector store ---")