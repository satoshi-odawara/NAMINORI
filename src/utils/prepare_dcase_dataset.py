import requests
import zipfile
import os
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import re

def download_file(url: str, file_full_path: Path, allow_resume: bool = False):
    """
    Downloads a file from a URL with a progress bar and optional resume capability.
    """
    file_full_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directory exists

    filename = file_full_path.name
    file_path = file_full_path # Directly use file_full_path

    headers = {}
    mode = 'wb'
    total_size_in_bytes = 0
    downloaded_size = 0

    try:
        with requests.head(url, allow_redirects=True) as r:
            r.raise_for_status()
            total_size_in_bytes = int(r.headers.get('content-length', 0))
    except requests.exceptions.RequestException as e:
        print(f"Warning: Could not get file size from {url}: {e}. Proceeding without size check.")

    if file_path.exists():
        downloaded_size = file_path.stat().st_size
        if total_size_in_bytes > 0 and downloaded_size < total_size_in_bytes:
            print(f"Resuming download of {filename} from {downloaded_size} bytes...")
            headers = {'Range': f'bytes={downloaded_size}-'}
            mode = 'ab'
        elif downloaded_size == total_size_in_bytes and total_size_in_bytes > 0:
            print(f"File {filename} already downloaded and complete: {file_path}")
            return
        else: # Partially downloaded file is larger than expected or total_size_in_bytes is 0 (unknown size)
            print(f"Partial file {filename} found with unexpected size ({downloaded_size} bytes). Re-downloading...")
            os.remove(file_path)
            downloaded_size = 0
    else:
        print(f"Starting fresh download of {filename}...")

    try:
        with requests.get(url, headers=headers, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            if total_size_in_bytes == 0: # If size was unknown before, try to get it now or use current stream length
                 total_size_in_bytes = int(r.headers.get('content-length', 0)) + downloaded_size
            
            block_size = 1024 # 1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, initial=downloaded_size, desc=f"Downloading {filename}")
            with open(file_path, mode) as f:
                for chunk in r.iter_content(block_size):
                    progress_bar.update(len(chunk))
                    f.write(chunk)
            progress_bar.close()
        print(f"Download complete: {file_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error during download of {filename}: {e}")
        print(f"Partial download saved to {file_path}. Please try running the script again to resume.")
        raise # Re-raise the exception to stop the script

def download_and_extract_dcase_pump(target_dir: Path):
    """
    Downloads the DCASE 2020 Challenge Task 2 "pump" development dataset,
    extracts it, and restructures it into the format expected by our
    benchmarking framework. Also downloads the separate metadata file.
    
    Args:
        target_dir: The base directory where the DCASE pump dataset will be
                    stored (e.g., 'data/benchmarks/dcase2020/pump').
                    The data will be restructured under this directory.
    """
    
    ZENODO_URL_ZIP = "https://zenodo.org/record/3678171/files/dev_data_pump.zip"
    ZIP_FILENAME = "dev_data_pump.zip"
    METADATA_URL = "https://raw.githubusercontent.com/daisukelab/dcase2020_task2_variants/master/file_info.csv"
    METADATA_FILENAME = "file_info.csv"
    
    # Create target directory if it doesn't exist
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Path for raw extracted data
    raw_data_dir = target_dir / "raw"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    # --- Download ZIP file ---
    zip_path = raw_data_dir / ZIP_FILENAME # Define zip_path here
    download_file(ZENODO_URL_ZIP, zip_path, allow_resume=True)

    # --- Download Metadata file ---
    meta_file_path = raw_data_dir / METADATA_FILENAME
    download_file(METADATA_URL, meta_file_path, allow_resume=False) # Metadata files are small, no resume needed

    print(f"Extracting {zip_path} to {raw_data_dir}...")
    # Check if raw data directory already contains content, avoid re-extracting
    # This check now ensures 'pump' directory and not just the meta file is present
    if not (raw_data_dir / "pump").is_dir(): 
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)
            print("Extraction complete.")
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is a bad zip file. It might be corrupted. Deleting and forcing re-download on next run.")
            if zip_path.exists():
                os.remove(zip_path)
            if raw_data_dir.exists():
                shutil.rmtree(raw_data_dir) # Clear partial extraction
            raise # Re-raise the exception
    else:
        print(f"Raw audio data already extracted to {raw_data_dir}, skipping extraction.")

    # Clean up the zip file (only if it exists and extraction was successful)
    if zip_path.exists():
        os.remove(zip_path)
        print(f"Removed temporary zip file: {zip_path}")


    print("Restructuring dataset...")
    
    # The DCASE 2020 "pump" zip extracts to a single folder named "pump" inside raw_data_dir
    extracted_audio_source_dir = raw_data_dir / "pump"
    if not extracted_audio_source_dir.is_dir():
        raise FileNotFoundError(f"Expected extracted audio folder 'pump' not found inside {raw_data_dir}. "
                                f"Please verify the zip content structure. Content in raw_data_dir: {[d.name for d in raw_data_dir.iterdir()]}")

    print(f"Using metadata file: {meta_file_path}")
    meta_df = pd.read_csv(meta_file_path)

    # Filter for 'pump' machine type using the 'type' column
    meta_df = meta_df[meta_df['type'] == 'pump'].copy() # Use .copy() to avoid SettingWithCopyWarning

    # Extract machine_id (e.g., 'id_00') from the 'file' column
    # Example: '/data/task2/dev/pump/train/normal_id_00_0000_0.wav'
    def extract_machine_id_from_path(file_path_in_csv: str) -> str:
        filename = Path(file_path_in_csv).name
        match = re.search(r'(id_\d{2})', filename)
        if match:
            return match.group(1)
        return "unknown_id" # Fallback if pattern not found

    meta_df['machine_id'] = meta_df['file'].apply(extract_machine_id_from_path)
    # Handle cases where machine_id might not be found
    if 'unknown_id' in meta_df['machine_id'].unique():
         print("Warning: Some machine_ids could not be extracted. Check 'file' column format.")

    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Restructuring files"):
        machine_id = row['machine_id']
        split_type = row['split'] # 'train' or 'test'

        # Extract 'normal' or 'anomaly' from the filename part of the 'file' column
        filename_part = Path(row['file']).name
        if "normal" in filename_part:
            label = "normal"
        elif "anomaly" in filename_part:
            label = "anomaly"
        else:
            print(f"Warning: Could not determine label for file: {row['file']}. Skipping.")
            continue

        # The 'file' column in the CSV is like '/data/task2/dev/pump/train/normal_id_00_0000_0.wav'
        # We need the path relative to `extracted_audio_source_dir` which is `data/benchmarks/dcase2020/pump/raw/pump`
        # So we need to remove `/data/task2/dev/pump/` prefix.

        csv_file_path_parts = Path(row['file']).parts
        try:
            # Find the index of 'pump' in the absolute path from the CSV
            pump_idx = csv_file_path_parts.index('pump')
            # The relative path starts from the element *after* 'pump'
            relative_path_from_extracted_pump_root = Path(*csv_file_path_parts[pump_idx + 1:])
        except ValueError:
            print(f"Error: 'pump' not found in file path: {row['file']}. Skipping.")
            continue # Skip this row

        original_file_path = extracted_audio_source_dir / relative_path_from_extracted_pump_root
        
        if not original_file_path.exists():
            print(f"Warning: Original file not found: {original_file_path}. Skipping.")
            continue

        # Construct new destination path
        dest_dir = target_dir / machine_id / split_type / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy the file
        shutil.copy(original_file_path, dest_dir / original_file_path.name)
        
    print("Restructuring complete. Cleaning up raw data directory...")
    shutil.rmtree(raw_data_dir) # Clean up the entire raw directory
    print(f"Removed raw data directory: {raw_data_dir}")
    print(f"DCASE 'pump' dataset prepared at: {target_dir}")

if __name__ == "__main__":
    BASE_BENCHMARK_DIR = Path("data/benchmarks/dcase2020/pump")
    download_and_extract_dcase_pump(BASE_BENCHMARK_DIR)