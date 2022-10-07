"""
Downloads the following:
- Glove vectors
- Stanford Natural Language Inference (SNLI) Corpus
- WordNet 
- CoreNLP tools
"""

import sys
import os
import zipfile
import gzip
import tarfile

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    os.system('wget {} -O {}'.format(url, filepath))
    return filepath

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def ungzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with tarfile.open(filepath) as zf:
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
            
        
        safe_extract(zf, dirpath)
    os.remove(filepath)

def download_wordvecs(dirpath):
    if os.path.exists(dirpath):
        print('Found Glove vectors - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://www-nlp.stanford.edu/data/glove.840B.300d.zip'
    unzip(download(url, dirpath))

def download_snli(dirpath):
    if os.path.exists(dirpath):
        print('Found SNLI dataset - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
    unzip(download(url, dirpath))

def download_wordnet(dirpath):
    if os.path.exists(dirpath):
        print('Found WordNet 3.0 - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://wordnetcode.princeton.edu/3.0/WNprolog-3.0.tar.gz'
    ungzip(download(url, dirpath))

def download_corenlp(dirpath):
    if os.path.exists(dirpath):
        print('Found Stanford CoreNLP - skip')
        return
    else:
        os.makedirs(dirpath)
    url = 'http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip'
    unzip(download(url, dirpath))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.realpath(__file__))
    snli_dir = os.path.join(base_dir, 'snli')
    wordvec_dir = os.path.join(base_dir, 'glove')
    wordnet_dir = os.path.join(base_dir, 'wordnet')
    corenlp_dir = os.path.join(base_dir, 'corenlp')
    download_snli(snli_dir)
    download_wordvecs(wordvec_dir)
    download_wordnet(wordnet_dir)
    download_corenlp(corenlp_dir)

