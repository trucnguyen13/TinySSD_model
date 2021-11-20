#python -m venv venv & venv\Scripts\activate.bat
# !pip install d2l==0.17.0
# !pip install -U mxnet-cu101==1.7.0

#lib.py
#prepare_data.py
#train.py
#predict.py

from mxnet import autograd, gluon, image, init, np, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()

import os
from io import BytesIO
from zipfile import ZipFile
import urllib.request


