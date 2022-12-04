from docarray import Document, DocumentArray
from jina import Client
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
import torchvision
input = DocumentArray.from_files(['/home/aswin/Data/archive/images/11110.jpg'])
# model = torchvision.models.resnet50(pretrained=True)  # load ResNet50
# input =img.embed(model, batch_size=8, device='cpu', to_numpy=True)
def get_matches_from_image(input, server="0.0.0.0:49888", limit=3, filters=None):
    # data = input.read()
    # query_doc = Document(blob=data)

    client = Client(host=server)
    response = client.search(
        input,
        return_results=True,
        parameters={"limit": limit, "filter": filters},
        show_progress=True,
    )
    for match in response[0].matches:
        print(f'({match.scores["cosine"].value}) id: {match.id} tags: {dict(match.tags)}')
        img = Image.open(match.uri)
        imshow(img)
        plt.title(f'{match.tags["productDisplayName"]}')
        plt.show()
get_matches_from_image(input)