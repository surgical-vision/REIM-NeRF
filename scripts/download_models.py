import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

def download_file(url):
    # https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192), desc=f"Downloading {local_filename}"): 
                f.write(chunk)
    return local_filename

def main():
    dst_dir = Path(__file__).parent.parent.absolute()
    # raise SystemError("Replace url with the one from the UCL data storage")
    downloaded_file_path = Path(download_file('https://rdr.ucl.ac.uk/ndownloader/files/42853327'))
    if not downloaded_file_path.exists():
        raise SystemError("Could not download the file")
    error_msg = ''
    try:
        with zipfile.ZipFile(downloaded_file_path, 'r') as zip_ref:
            print(f"extracting the models to {dst_dir}")
            zip_ref.extractall(dst_dir)
    except zipfile.BadZipFile as e:
        error_msg = str(e)
    finally:
        downloaded_file_path.unlink()
        if error_msg:
            raise SystemError(f"Could not unzip the file {error_msg}")
    
if __name__ == "__main__":
    main()