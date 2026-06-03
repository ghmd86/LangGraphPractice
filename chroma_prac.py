from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import pytesseract
from pdf2image import convert_from_path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

embedding_model = OllamaEmbeddings(model="all-minilm")

def load_pdf_with_ocr(path: str) -> list[Document]:
    images = convert_from_path(path)
    docs = []
    for i, image in enumerate(images):
        text = pytesseract.image_to_string(image)
        docs.append(Document(page_content=text, metadata={"page": i, "source": path}))
    return docs

def persist_chroma():
    global vector_store
    chroma_dir = "./chroma_db_dir/"
    poa = "./powerofAttoreny.pdf"
    docs = load_pdf_with_ocr(poa)

    print(f"length {len(docs)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    split_doc = splitter.split_documents(docs)

    print(split_doc)
    print(split_doc[-1].page_content)
    print(split_doc[-1].type)
    vector_store = Chroma.from_documents(
        documents=split_doc,
        embedding=embedding_model,
        persist_directory=chroma_dir,
    )
    count = vector_store._collection.count()

    print(f"Persisted {count}")
    print(f"Located at: {chroma_dir}")

    del vector_store

    reloaded = Chroma(
        embedding_function=embedding_model,
        persist_directory=chroma_dir
    )

    reloaded_count = reloaded._collection.count()

    print(f"Reloaded count {reloaded_count}")


    results = reloaded.similarity_search("Ghulam Mohammed", k=2)
    for result in results:

        print(f"Search result: {result.page_content[:50]} ...")

if __name__ == "__main__":
    persist_chroma()