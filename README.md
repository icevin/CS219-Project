# CS219-Project
Cross-Device Machine Learning Pipeline for Video and Image Tasks on IoT

## How to install:

### Install modified transformers library:

Python 3.11.3 - you can make a new env with
> conda create -n cs219 python=3.11.3

Clone this repo:
```
git clone https://github.com/icevin/CS219-Project.git
cd CS219-Project
```

Clone from git:
```
git submodule init  
git submodule update
```

Install transformers (might have to uninstall default version of transformers)
```
cd ./inference/transformers  
pip install -e .
```

Install other dependencies
```
pip install torch
```

Start server or client
```
python server.py [PORT]
```



Dependencies:
torch

