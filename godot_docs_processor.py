import os
import json
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
from PIL import Image
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

class GodotDocsProcessor:
    def __init__(
        self,
        html_dir="data/raw",
        chunk_dir="data/chunked",
        processed_dir="data/processed",
        chunk_size=500,
        chunk_overlap=50,
        use_blip=True,
    ):
        self.html_dir = html_dir
        self.images_dir = os.path.join(html_dir, "_images")
        self.chunk_path = os.path.join(chunk_dir, "chunked_docs.json")
        self.raw_path = os.path.join(processed_dir, "processed_docs.json")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_blip = use_blip

        if use_blip:
            print("⚙️ Loading BLIP model...")
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)

    def blip_caption(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"[BLIP error] {img_path}: {e}")
            return None

    def extract_text_and_images_from_html(self, file_path):
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

    def find_all_html_files(self):
        html_files = []
        for root, _, files in os.walk(self.html_dir):
            for fname in files:
                if fname.lower().endswith(".html"):
                    html_files.append(os.path.join(root, fname))
        return html_files

    def chunk_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return splitter.split_documents(documents)

    def create_documents(self):
        html_files = self.find_all_html_files()
        print(f"Found {len(html_files)} HTML files.")

        html_docs = []
        image_docs = []

        for html_path in tqdm(html_files, desc="Processing HTML files"):
            rel_path = os.path.relpath(html_path, start=self.html_dir)
            text, images = self.extract_text_and_images_from_html(html_path)

            if text:
                html_docs.append(Document(
                    page_content=text,
                    metadata={"source": str(Path(html_path).resolve())}
                ))

            for img in tqdm(images, desc=f"Captions in {rel_path}", leave=False):
                img_src = img["src"]
                alt_text = img["alt"]
                img_path = os.path.join(self.html_dir, img_src)
                if not os.path.isfile(img_path):
                    continue

                caption = alt_text or ""
                if self.use_blip and not caption:
                    caption = self.blip_caption(img_path) or ""

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

    def save_documents(self, docs, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in docs
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_documents(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in data]

    def prepare(self):
        if os.path.exists(self.chunk_path):
            print("🔁 Loading chunked documents from cache...")
            return self.load_documents(self.chunk_path)

        if os.path.exists(self.raw_path):
            print("🔁 Loading raw documents from cache...")
            all_docs = self.load_documents(self.raw_path)
        else:
            print("🚀 Extracting HTML + image documents...")
            html_docs, image_docs = self.create_documents()
            all_docs = html_docs + image_docs
            self.save_documents(all_docs, self.raw_path)
            print(f"✅ Saved raw documents to {self.raw_path}")

        chunked = self.chunk_documents(all_docs)
        self.save_documents(chunked, self.chunk_path)
        print(f"✅ Saved {len(chunked)} chunked documents to {self.chunk_path}")
        return chunked
