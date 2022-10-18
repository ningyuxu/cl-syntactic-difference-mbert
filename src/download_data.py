import os
import tarfile
import wget
import zipfile

from datasets.constant import DATA_PATH

CORPUS_INFO = [
    {
        "name": "UD",
        "description": "The Corpus of Universal Dependencies",
        "file": "ud-treebanks-v2.8.tgz",
        "url": "https://lindat.mff.cuni.cz/repository/xmlui/bitstream"
               "/handle/11234/1-3687/ud-treebanks-v2.8.tgz?sequence=1"
               "&isAllowed=y"
    },
]

if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

for corpus in CORPUS_INFO:
    file = DATA_PATH + '/' + corpus["file"]
    if not os.path.exists(file):
        print(f"Downloading {corpus['description']} ...")
        wget.download(corpus["url"], file)
        print(f"\nExtracting file: {corpus['file']} ...")
        if file.endswith("zip"):
            with zipfile.ZipFile(file, 'r') as f:
                f.extractall(DATA_PATH)
        elif file.endswith("tgz"):
            with tarfile.open(file, "r:gz") as f:
                f.extractall(DATA_PATH)
        elif file.endswith("tar"):
            with tarfile.open(file, "r:") as f:
                f.extractall(DATA_PATH)
print("Done!")
