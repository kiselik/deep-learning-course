from os import mkdir
from urllib.request import urlretrieve
try:
    mkdir('train-images')
except:
    pass
urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'train-images/train-images-idx3-ubyte.gz')
urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'train-images/train-labels-idx1-ubyte.gz')
print('train-images downloaded')
urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'train-images/t10k-images-idx3-ubyte.gz')
urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'train-images/t10k-labels-idx1-ubyte.gz')
print('t10k-images downloaded')
