from lib import *

zipurl = 'http://d2l-data.s3-accelerate.amazonaws.com/pikachu.zip'

with urllib.request.urlopen(zipurl) as zipresp:
    with ZipFile(BytesIO(zipresp.read())) as zfile:
        zfile.extractall('data')