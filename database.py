# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import BSHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
import os   
from tqdm import tqdm
import shutil
import argparse

DATA_PATH = "./data/"
CHROMA_PATH = "./chroma/"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

def load_documents():
    # doc_loader = PyPDFDirectoryLoader(DATA_PATH)
    doc_loader = BSHTMLLoader("./data/Japanese_Fairy_Tales_by_Yei_Theodora_Ozaki.html", open_encoding="utf8")
    return doc_loader.load()

def split_documents(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(docs)

def add_to_chroma(chunks: list[Document]):
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
        )
    except ValueError as e:
        print(f"Failed to initialize Chroma: {e}")
        return

    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new data to the Database: {len(new_chunks)} chunks")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        for i in tqdm(range(len(new_chunks)), desc="Adding the datas to DB"):
            db.add_documents([new_chunks[i]], ids=[new_chunk_ids[i]])
    else:
        print("✅ No new data to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
