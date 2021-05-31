""" Modifified from torchvision.datasets.utils due to download issues using built-in version """
import hashlib
import os
from pathlib import Path

import requests
from tqdm.auto import tqdm


def download_url(url: str, root: Path, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = os.path.basename(url)
    fpath = str(root / filename)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        print("Using downloaded and verified file: {}".format(fpath))
    else:  # download the file
        print("Downloading " + str(url) + " to " + str(fpath))
        response = requests.get(url, stream=True)
        chunk_size = 4096
        total = int(response.headers.get("content-length", 0))
        with tqdm(total=total) as pbar, open(fpath, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(chunk_size)

        # check integrity of downloaded file
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath):
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def _is_tarxz(filename):
    return filename.endswith(".tar.xz")


def _is_tar(filename):
    return filename.endswith(".tar")


def _is_targz(filename):
    return filename.endswith(".tar.gz")


def _is_tgz(filename):
    return filename.endswith(".tgz")


def _is_gzip(filename):
    return filename.endswith(".gz") and not filename.endswith(".tar.gz")


def _is_zip(filename):
    return filename.endswith(".zip")


def _is_rar(filename):
    return filename.endswith(".rar")


def extract_archive(from_path, to_path=None, remove_finished=False):
    from_path = str(from_path)
    if to_path is None:
        to_path = os.path.dirname(from_path)
    else:
        to_path = str(to_path)

    print("Extracting {} to {}".format(from_path, to_path))

    if _is_tar(from_path):
        import tarfile

        with tarfile.open(from_path, "r") as tar:
            tar.extractall(path=to_path)
    elif _is_targz(from_path) or _is_tgz(from_path):
        import tarfile

        with tarfile.open(from_path, "r:gz") as tar:
            tar.extractall(path=to_path)
    elif _is_tarxz(from_path):
        import tarfile

        # .tar.xz archive only supported in Python 3.x
        with tarfile.open(from_path, "r:xz") as tar:
            tar.extractall(path=to_path)
    elif _is_gzip(from_path):
        import gzip

        to_path = os.path.join(
            to_path, os.path.splitext(os.path.basename(from_path))[0]
        )
        with open(to_path, "wb") as out_f, gzip.GzipFile(from_path) as zip_f:
            out_f.write(zip_f.read())
    elif _is_zip(from_path):
        import zipfile

        with zipfile.ZipFile(from_path, "r") as z:
            z.extractall(to_path)
    elif _is_rar(from_path):
        from unrar import rarfile

        with rarfile.RarFile(from_path, "r") as z:
            z.extractall(to_path)
    else:
        raise ValueError("Extraction of {} not supported".format(from_path))

    if remove_finished:
        os.remove(from_path)


def download_and_extract_archive(
    url,
    download_root,
    extract_root=None,
    filename=None,
    md5=None,
    remove_finished=False,
):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    extract_archive(archive, extract_root, remove_finished)
