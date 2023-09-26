from fastapi import FastAPI, UploadFile, File
import torch
from model import initialize_model
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import cv2
import numpy as np

app = FastAPI()

t = []

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
])

label2num = {'Mèo lông dài Mỹ': 0,
 'Mèo Xiêm': 1,
 'Mèo Bengal': 2,
 'Mèo Ragdoll': 3,
 'Mèo Ai Cập': 4,
 'Mèo tai cụp': 5,
 'Mèo Nga mắt xanh': 6,
 'Mèo Anh lông ngắn': 7,
 'Mèo Devon Rex': 8,
 'Mèo thần Miến Điện': 9,
 'Mèo lông ngắn phương Đông': 10,
 'Mèo Mau Ai Cập': 11,
 'Mèo lông ngắn Hoa Kỳ': 12,
 'Mèo Rex Cornwall': 13,
 'Mèo Angora Thổ Nhĩ Kỳ': 14,
 'Mèo Ragamuffin': 15,
 'Mèo Manx': 16,
 'Mèo Chartreux': 17,
 'Mèo Bắc Kỳ': 18,
 'Mèo Bali': 19,
 'Mèo rừng Na Uy': 20,
 'Mèo Van Thổ Nhĩ Kỳ': 21,
 'Mèo cộc đuôi Nhật Bản': 22,
 'Mèo lông ngắn Ba Tư': 23,
 'Mèo Scottish Straight': 24,
 'Mèo lông xoăn': 25,
 'Mèo Peterbald': 26,
 'Mèo LaPerm': 27,
 'Mèo Mỹ tai xoắn': 28,
 'Mèo Munchkin': 29,
 'Mèo Singapura': 30,
 'Mèo Bombay': 31,
 'Mèo Chausie': 32,
 'Mèo cộc đuôi Kuril': 33,
 'Mèo Burmilla': 34,
 'Mèo Toyger': 35,
 'Mèo mướp': 36,
 'Mèo tam thể': 37,
 'Mèo Ba Tư': 38,
 'Mèo Abyssinian': 39,
 'Mèo Miến Điện': 40}

num2label = {v:k for k, v in label2num.items()}

@app.post("/predict")
async def upload_file(file: UploadFile):
    model, input_size = initialize_model('mobilenetv2', 41, False, use_pretrained=False)
    model.to('cpu')
    checkpoint = torch.load('work_dirs/mobilenetv2/best_checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    image = cv2.imdecode(np.fromstring(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_transform = transform(img)
    img_transform = img_transform.unsqueeze(0)
    pred = model(img_transform.to('cpu'))
    softmax = torch.nn.Softmax(dim=1)
    pred = softmax(pred)
    label = torch.argmax(pred, axis=1)
    print(f'Image: {num2label[label[0].item()]} - Acc: {pred[0][label[0].item()]}')
    return {"result": num2label[label[0].item()]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)