import os
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Optional: BLIP captioning
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

HTML_DIR = "data/raw"
IMAGES_DIR = "data/raw/_images"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
USE_BLIP = True  # Toggle this if you want image captions

# Initialize BLIP
if USE_BLIP:
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)


def blip_caption(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"[BLIP error] {img_path}: {e}")
        return None


def extract_text_and_images_from_html(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

        for tag in soup(["script", "style", "head", "noscript"]):
            tag.decompose()

        images = []
        for img in soup.find_all("img"):
            src = img.get("src", "").strip()
            alt = img.get("alt", "").strip()
            if src:
                images.append({"src": src, "alt": alt})

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)

        return clean_text, images


def find_all_html_files(directory):
    html_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".html"):
                html_files.append(os.path.join(root, fname))
    return html_files


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def create_documents():
    html_files = find_all_html_files(HTML_DIR)
    print(f"Found {len(html_files)} HTML files.")

    html_docs = []
    image_docs = []

    for html_path in tqdm(html_files, desc="Processing HTML files"):
        rel_path = os.path.relpath(html_path, start=HTML_DIR)
        text, images = extract_text_and_images_from_html(html_path)

        if text:
            html_docs.append(Document(
                page_content=text,
                metadata={"source": str(Path(html_path).resolve())}
            ))

        for img in tqdm(images, desc=f"Captions in {rel_path}", leave=False):
            img_src = img["src"]
            alt_text = img["alt"]
            img_path = os.path.join(HTML_DIR, img_src)
            if not os.path.isfile(img_path):
                continue

            caption = alt_text or ""
            if USE_BLIP and not caption:
                caption = blip_caption(img_path) or ""

            if caption:
                image_docs.append(Document(
                    page_content=caption,
                    metadata={
                        "type": "image",
                        "source": str(Path(img_path).resolve()),
                        "origin_html": str(Path(html_path).resolve())
                    }
                ))

    print(f"Parsed {len(html_docs)} HTML docs and {len(image_docs)} image captions.")
    return html_docs, image_docs


def save_documents(docs, path):
    data = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_chunked_documents(docs, path):
    data = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in docs
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_chunked_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in data]


def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in data]


def main():
    raw_cache_path = "./processed/processed_docs.json"
    chunked_cache_path = "./chunked/chunked_docs.json"

    if os.path.exists(chunked_cache_path):
        print("🔁 Loading chunked documents from cache...")
        chunked = load_chunked_documents(chunked_cache_path)

    else:
        if os.path.exists(raw_cache_path):
            print("🔁 Loading raw documents from cache...")
            all_docs = load_documents(raw_cache_path)
        else:
            print("🚀 Processing HTML and image documents...")
            html_docs, image_docs = create_documents()
            all_docs = html_docs + image_docs
            save_documents(all_docs, raw_cache_path)
            print(f"✅ Saved raw documents to {raw_cache_path}")

        chunked = chunk_documents(all_docs)
        save_chunked_documents(chunked, chunked_cache_path)
        print(f"✅ Saved {len(chunked)} chunked documents to {chunked_cache_path}")

    # Preview
    print("\n--- Example Chunk ---")
    print(chunked[0].page_content[:400])

    return chunked




if __name__ == "__main__":
    docs = main()
