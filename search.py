from http import server
from jina import Flow
import pandas as pd
from docarray import DocumentArray, Document
import os
import torchvision
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image 
df = pd.read_csv('/home/aswin/Data/archive/styles.csv', warn_bad_lines=True, error_bad_lines=False)
df = df.dropna()
MAX_NUM_DOCS = 20
df['year'] = df['year'].astype(int)
def get_product_docs(max_num: int = MAX_NUM_DOCS):
    da = DocumentArray()
    for index, row in df.iterrows():
        doc_id = row.pop('id')
        doc_uri = f'/home/aswin/Data/archive/images/{doc_id}.jpg'
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
model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
docs.embed(model, batch_size=8, device='cpu', to_numpy=True)
query = docs[0:1]
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
before_year = "2017" #@param [2017, 2018, 2019]
category = "Apparel" #@param ["Apparel", "Footwear"]
color = "Brown" #@param ["White", "Black", "Brown"]
with f:
    resp = f.search(inputs=query, server="0.0.0.0:49888",
                    return_results=True, 
                    parameters={
                        'filter': {
                            'year': {'$lte': before_year},
                            'masterCategory': {'$eq': category},
                            'baseColour': {'$eq': color}
                        },
                        'limit': 5
                    })

for match in resp[0].matches:
    print(f'({match.scores["cosine"].value}) id: {match.id} tags: {dict(match.tags)}')
    img = Image.open(match.uri)
    imshow(img)
    plt.title(f'{match.tags["productDisplayName"]}')
    plt.show()