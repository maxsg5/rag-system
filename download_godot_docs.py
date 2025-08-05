#!/usr/bin/env python3
"""
Script to download Godot documentation from nightly builds,
unzip it, and save the raw data to data/raw directory.
"""

import os
import sys
import requests
import zipfile
import shutil
from pathlib import Path

# URL for the Godot documentation zip file
GODOT_DOCS_URL = "https://nightly.link/godotengine/godot-docs/workflows/build_offline_docs/master/godot-docs-html-stable.zip"

# Target directory for raw data
RAW_DATA_DIR = Path("data/raw")
TEMP_ZIP_FILE = "godot-docs-html-stable.zip"

def create_directories():
    """Create necessary directories if they don't exist."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {RAW_DATA_DIR}")

def download_file(url, filename):
    """Download a file from the given URL."""
    print(f"Downloading {url}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    
                    # Show progress
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\nDownload completed: {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        return False

def unzip_file(zip_filename, extract_to):
    """Unzip the downloaded file to the specified directory."""
    print(f"Extracting {zip_filename} to {extract_to}...")
    
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"Extraction completed to: {extract_to}")
        return True
        
    except zipfile.BadZipFile as e:
        print(f"Error: Invalid zip file - {e}")
        return False
    except Exception as e:
        print(f"Error extracting file: {e}")
        return False

def cleanup_temp_files():
    """Remove temporary files."""
    if os.path.exists(TEMP_ZIP_FILE):
        os.remove(TEMP_ZIP_FILE)
        print(f"Cleaned up temporary file: {TEMP_ZIP_FILE}")

def main():
    """Main function to orchestrate the download and extraction process."""
    print("Starting Godot documentation download...")
    
    # Create necessary directories
    create_directories()
    
    # Download the zip file
    if not download_file(GODOT_DOCS_URL, TEMP_ZIP_FILE):
        print("Failed to download the documentation.")
        sys.exit(1)
    
    # Extract the zip file
    if not unzip_file(TEMP_ZIP_FILE, RAW_DATA_DIR):
        print("Failed to extract the documentation.")
        cleanup_temp_files()
        sys.exit(1)
    
    # Clean up temporary files
    cleanup_temp_files()
    
    print(f"\nGodot documentation successfully downloaded and extracted to: {RAW_DATA_DIR}")
    
    # List the contents of the extracted directory
    extracted_items = list(RAW_DATA_DIR.iterdir())
    if extracted_items:
        print(f"\nExtracted contents:")
        for item in extracted_items:
            print(f"  - {item.name}")
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
