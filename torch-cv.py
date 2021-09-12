# PyTorch imports
import torch
import torchvision
import numpy as np
import time

def sync_e():
    e = torch.cuda.Event()
    e.record()
    e.synchronize()

sync_e()
start = time.time()

model_path = './model_dir/resnet50.pt'
downloaded_model = True

if downloaded_model:
    scripted_model = torch.jit.load(model_path)
else:
    model_name = "resnet50"
    model = getattr(torchvision.models, model_name)(pretrained=True)
    d = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(d)
    # model = model.cuda()
    model = model.eval()

    input_shape = [1, 3, 224, 224]
    input_data = torch.randn(input_shape).cuda(0)
    scripted_model = torch.jit.trace(model, input_data).eval()
    torch.jit.save(scripted_model, model_path)

# import the image
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = './data-dir/cat.png'
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)
# print(img.shape)
input_img = torch.from_numpy(img)
input_img = input_img.cuda()
# print(input_img)
print(scripted_model(input_img))
sync_e()
end = time.time()
print("Time cost: ", end - start)