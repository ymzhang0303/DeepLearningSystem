import urllib.request
import os

# Create directory if not exists
os.makedirs('./data/ptb', exist_ok=True)

# Download Penn Treebank dataset
ptb_data = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
for f in ['train.txt', 'test.txt', 'valid.txt']:
    file_path = os.path.join('./data/ptb', f)
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(ptb_data + f, file_path)

# Download CIFAR-10 dataset
cifar_path = "./data/cifar-10-batches-py"
if not os.path.isdir(cifar_path):
    cifar_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar_tar_path = "./data/cifar-10-python.tar.gz"
    urllib.request.urlretrieve(cifar_url, cifar_tar_path)
    os.system(f"tar -xvzf {cifar_tar_path} -C ./data")
