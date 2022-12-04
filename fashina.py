import os
import glob
import click
import pandas as pd

from PIL import Image
from jina import Document, DocumentArray

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from annlite import AnnLite

os.environ['JINA_LOG_LEVEL'] = 'DEBUG'
MAX_NUM_DOCS = 200

df = pd.read_csv('/home/aswin/data/archive/styles.csv', warn_bad_lines=True, error_bad_lines=False)
df = df.dropna()
df['year'] = df['year'].astype(int)
def get_product_docs(max_num: int = MAX_NUM_DOCS):
    da = DocumentArray()
    for index, row in df.iterrows():
        doc_id = row.pop('id')
        doc_uri = f'/home/aswin/data/archive/images/{doc_id}.jpg'
        if not os.path.exists(doc_uri):
            continue

        doc = Document(id=str(doc_id), uri=doc_uri, tags=dict(row))
        da.append(doc)
        if len(da) == max_num:
            break
    
    return da
docs = get_product_docs(500)
def preproc(d: Document):
    return (d.load_uri_to_image_tensor()  # load
             .set_image_tensor_normalization()  # normalize color 
             .set_image_tensor_channel_axis(-1, 0))  # switch color axis

docs.apply(preproc)
# import torchvision
# model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
# docs.embed(model, batch_size=8, device='cpu', to_numpy=True)
from jina import Flow

f = Flow().add(uses='jinahub://PQLiteIndexer',
    uses_with={
        'dim': 1000,
        'metric': 'cosine',
        'columns': [
            ('year', 'int'), 
            ('baseColour', 'str'), 
            ('masterCategory', 'str')
        ],
        'include_metadata': True
    },
    uses_metas={'workspace': './workspace'}, 
    install_requirements=True
)
def index():
    with f:
        f.index(inputs=docs, show_progress=True)
        f.block()
def serve():
    with f:
        f.block()
@click.command()
@click.option(
    "--task",
    "-t",
    type=click.Choice(
        ["index", "serve"], case_sensitive=False
    ),
)
def main(task: str):
    # if task == "index":
        # index(CSV_FILE, num_docs=num_docs)
    if task == "index":
        index()
    else: serve()


if __name__ == "__main__":
    main()