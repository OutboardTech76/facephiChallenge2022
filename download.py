import os
import tqdm
import argparse
from urllib.request import urlretrieve
import tarfile
import zipfile
import os


midv500_links = [
    # "ftp://smartengines.com/midv-500/dataset/01_alb_id.zip",
    "ftp://smartengines.com/midv-500/dataset/05_aze_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/21_esp_id_old.zip",
    "ftp://smartengines.com/midv-500/dataset/22_est_id.zip",
    "ftp://smartengines.com/midv-500/dataset/24_fin_id.zip",
    "ftp://smartengines.com/midv-500/dataset/25_grc_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/32_lva_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/39_rus_internalpassport.zip",
    "ftp://smartengines.com/midv-500/dataset/41_srb_passport.zip",
    "ftp://smartengines.com/midv-500/dataset/42_svk_id.zip",
]

midv2019_links = [
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/01_alb_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/05_aze_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/21_esp_id_old.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/22_est_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/24_fin_id.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/25_grc_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/32_lva_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/39_rus_internalpassport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/41_srb_passport.zip",
    "ftp://smartengines.com/midv-500/extra/midv-2019/dataset/42_svk_id.zip",
]

midv2020_links = ["ftp://smartengines.com//midv-2020/dataset/photo.tar"]

def extract(path):
    out_path, extension = os.path.splitext(path)

    if extension == ".tar":
        with tarfile.open(path, "r:") as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, out_path)
    elif extension == ".zip":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(out_path)
    else:
        raise NotImplementedError()

class tqdm_upto(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download(url: str, save_dir: str):
    # Creates save_dir if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Downloads the file
    with tqdm_upto(unit="B", unit_scale=True, miniters=1) as t: 
        urlretrieve(
            url,
            filename=os.path.join(save_dir, url.split("/")[-1]),
            reporthook=t.update_to,
            data=None,
        )

def download_and_extract(links_set, download_dir: str = './data'):
    out_path = os.path.join(download_dir)
    for i, link in enumerate(links_set):
        # download zip file
        link = link.replace("\\", "/")
        filename = os.path.basename(link)
        print()
        print(f"Downloading {i+1}/{len(links_set)}:", filename)
        download(link, out_path)

        # unzip zip file
        print("Unzipping:", filename)
        zip_path = os.path.join(out_path, filename)
        extract(zip_path)

        # remove zip file
        os.remove(zip_path)

# download_and_extract(midv500_links, download_dir='data/midv500')
download_and_extract(midv2019_links, download_dir='data/midv2019')

