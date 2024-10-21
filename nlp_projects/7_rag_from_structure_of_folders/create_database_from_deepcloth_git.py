"""
sources:
chatgpt graphs
https://www.anthropic.com/news/contextual-retrieval

"""

from pathlib import Path
import os
import time
import hashlib
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from directory_tree import DisplayTree


load_dotenv()

SUFFIXES_OF_FILES_TO_COUNT = [".md", ".txt", ".py"]


def hash_file_combined(file_path):
    """Compute both SHA-256 and SHA-512 hashes of a file and concatenate them."""
    sha256_hash = hashlib.sha256()
    sha512_hash = hashlib.sha512()

    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
            sha512_hash.update(byte_block)

    # Concatenate both hashes
    return sha256_hash.hexdigest() + sha512_hash.hexdigest()


def init_vector_db(persist_dir):
    """
    Initializes a vector database using the specified persist directory.

    Args:
        persist_dir (str): The directory path where the vector database will be persisted.

    Returns:
        Chroma: An instance of the Chroma class with the specified embedding model and persist directory.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )  # Replace with your embedding model
    return Chroma(embedding_function=embeddings, persist_directory=persist_dir)


def count_tokens_in_path(path: str) -> int:
    """
    Counts the number of tokens in a directory and its subdirectories.

    """
    files_paths = [
        str(path)
        for path in Path(path).rglob("*")
        if path.is_file() and path.suffix in SUFFIXES_OF_FILES_TO_COUNT
    ]

    return sum([count_tokens_in_file(path) for path in files_paths])


def count_tokens_in_file(file_path: str) -> int:
    """
    Counts the number of tokens in a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        int: The number of tokens in the file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        tokens = content.split()  # Split by whitespace (space, newline, etc.)
        return len(tokens)


def get_file_metadata(file_path):
    """
    Retrieves the metadata of a file.

    Args:
        file_path (str): The path to the file.

    Returns:
        dict: A dictionary containing the file's metadata, including the file path and last modified time.
    """
    file_info = os.stat(file_path)
    metadata = {
        "file_path": str(file_path),
        "last_modified": time.ctime(file_info.st_mtime),  # Get last modified time
    }
    return metadata


def group_files_by_hash(file_paths):
    """Takes a list of file paths, hashes them, and groups paths by hash."""
    hash_dict = defaultdict(list)
    path_to_hash_dict = {}

    for file_path in file_paths:
        try:
            # Compute the hash for the file
            file_hash = hash_file_combined(file_path)
            # Add the file path to the corresponding hash in the dictionary
            hash_dict[file_hash].append(file_path)
            path_to_hash_dict[file_path] = file_hash
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return dict(hash_dict), path_to_hash_dict


def generate_chunk_context(document: str, chunk: str, path: str) -> str:
    """
    Generates a succinct context for a chunk within a document using an LLM,
    given the document's content, chunk, and file path.

    :param document: The entire content of the document.
    :param chunk: The specific chunk that needs to be situated within the document.
    :param path: The file path where the document resides.
    :return: Succinct context that situates the chunk within the document.
    """
    # Define the template prompt
    prompt_template = """
    <document> 
    {WHOLE_DOCUMENT}
    </document> 
    Here is the chunk we want to situate within the whole document: 
    <chunk> 
    {CHUNK_CONTENT}
    </chunk> 
    The document comes from this file: {FILE_PATH}.
    Please give a short succinct context to situate this chunk within the overall document 
    for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
    """
    # Initialize the LLM (e.g., OpenAI's GPT model)
    llm = ChatOpenAI(model="gpt-3.5-turbo")  # Use your preferred LLM here
    # Create the prompt template
    prompt = PromptTemplate.from_template(prompt_template)
    # Format the prompt by injecting the document, chunk, and path
    formatted_prompt = prompt.format(
        WHOLE_DOCUMENT=document, CHUNK_CONTENT=chunk, FILE_PATH=path
    )
    # Get the result from the LLM
    response = llm.invoke(formatted_prompt)
    return response.content


def add_file_to_database(file_path, metadata, vector_db):
    # Load the document from the text file
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    # Split the document into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=900, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    whole_text = "\n".join([doc.page_content for doc in documents])
    if len(docs) == 0:
        return vector_db
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    for doc in docs:
        # metadata["context"] = generate_chunk_context(
        #     whole_text, doc.page_content, metadata["file_path"]
        # )
        doc.metadata = metadata  # Add the file metadata to each document chunk
    # Add documents to the vector database
    vector_db.add_documents(docs)
    print(f"Uploaded {file_path} with metadata to the vector store.")
    return vector_db


def process_files_and_upload(directory, vector_db):
    file_graph = nx.DiGraph()

    files_paths = [
        str(path)
        for path in Path(directory).rglob("*")
        if path.is_file() and path.suffix in [".md", ".txt", ".py"]
    ]

    for dirpath, dirnames, filenames in os.walk(directory):
        file_graph.add_node(dirpath, type="directory", path=dirpath)
        for subdir in dirnames:
            file_graph.add_node(
                os.path.join(dirpath, subdir),
                type="directory",
                path=os.path.join(dirpath, subdir),
            )
            file_graph.add_edge(
                dirpath, os.path.join(dirpath, subdir), relationship="contains"
            )
    # pos = nx.spring_layout(file_graph)
    # nx.draw_networkx_nodes(file_graph, pos)
    # # nx.draw_networkx_labels(file_graph, pos)
    # nx.draw_networkx_edges(file_graph, pos, edge_color="r", arrows=True)

    # plt.show()

    stringRepresentation: str = DisplayTree(directory, stringRep=True)
    print(stringRepresentation)

    print(f"Processing files in {directory}..., {len(files_paths)}")
    # hash_to_paths_dict, path_to_hash_dict = group_files_by_hash(files_paths)
    # print(f"Processing files after hash in {directory}..., {len(hash_to_paths_dict)}")
    # hash_to_data_and_metadata = {}
    for file_path in tqdm(files_paths[:20]):
        print(f"Processing {file_path}...")
        # file_hash = path_to_hash_dict[file_path]
        metadata = get_file_metadata(file_path)
        vector_db = add_file_to_database(file_path, metadata, vector_db)
    return vector_db, file_graph


if __name__ == "__main__":
    # Define the directory containing the text file and the persistent directory
    current_dir = Path(__file__).parent
    target_dir_path = str(
        Path("C:\\Users\\LVY-BACKUP_2\\Desktop\\deepcloth\\deepcloth_data_procesing\\")
    )
    persistent_directory = str(current_dir / "db" / "deepcloth_db")

    # Initialize the vector database (Chroma) and embeddings
    vector_db = init_vector_db(persistent_directory)

    print(count_tokens_in_path(target_dir_path))

    # Process all eligible files and upload them to the vector store with metadata
    vector_db, file_graph = process_files_and_upload(target_dir_path, vector_db)

    # Persist the vector store
    vector_db.persist()
    nx.write_graphml(
        file_graph,
        str(current_dir / "db" / "deepcloth_db" / "file_system_graph.graphml"),
    )
