import urllib.request
import os

os.makedirs('./data/ptb', exist_ok=True)

# Download Penn Treebank dataset
ptb_data = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
for f in ['train.txt', 'test.txt', 'valid.txt']:
    file_path = os.path.join('./data/ptb', f)
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(ptb_data + f, file_path)