from settings import *
from flask import Flask, request, jsonify
from io import BytesIO
from pdf2image import convert_from_path
import torch,torchvision 
import base64
from PIL import Image

#насройки для torch модели
device='cpu'
model=torch.load(PATH_MODEL)
model.eval()
test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                      torchvision.transforms.ToTensor(),
                                     ])

#функция обработки jpg изображений
def preprocess_jpg(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))

#функция обработки pdf изображений
def preprocess_pdf(b64):
    pdf=base64.b64decode(b64)
    file = open('file.pdf', 'wb')
    file.write(pdf)
    file.close()
    pages = convert_from_path(r'file.pdf',strict=True)
    return pages[0]

#функция прогноза модели
def predict_image(image):
    image=test_transforms(image).float()
    image=image.unsqueeze_(0)
    image=image.to(device)
    output=model(image)
    return output.data.cpu().numpy().argmax()

app = Flask(__name__)

@app.route("/")
def hello():
    return "Image classification example\n"
@app.route('/predict', methods=['GET','POST'])
def predict():
    response =request.get_json()
    try:
        formt=response["format"].split('.')[-1]
        if formt=='pdf' or formt=='PDF':
            img=preprocess_pdf(response["b64"])
        else:
            img=preprocess_jpg(response["b64"])

        idx=predict_image(img)
    except Exception:
        idx=1
    return jsonify({'index':int(idx),'class':classes.get(int(idx))})
if __name__ == '__main__':
    app.run(host=server_ip, debug=True, port=PORT)