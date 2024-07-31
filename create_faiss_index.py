import json
import argparse
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def combine_title_details(data):
    combined_docs = []
    for section in data["manual"]["sections"]:
        title = section["title"]
        for detail in section["details"]:
            subtitle = detail.get("subtitle", "")
            description = detail.get("description", "")
            full_text = f"{title}-{subtitle}\n{description}"
            combined_docs.append(full_text)
    return combined_docs


def create_faiss_index(json_path, output_path="db/"):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = combine_title_details(data)

    embedding_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        encode_kwargs={"normalize_embeddings": True},
    )
    document_objects = [Document(page_content=text) for text in documents]

    vectorstore = FAISS.from_documents(
        documents=document_objects, embedding=embedding_model
    )

    vectorstore.save_local(output_path)
    print(f"FAISS index saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create FAISS index from JSON")
    parser.add_argument("json_path", type=str, help="Path to the JSON file")
    parser.add_argument(
        "--output_path", type=str, default="db/", help="Path to save the FAISS index"
    )

    args = parser.parse_args()

    create_faiss_index(args.json_path, args.output_path)
