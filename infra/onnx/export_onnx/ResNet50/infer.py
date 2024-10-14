import torch as tr
from PIL import Image
import numpy as np
import onnxruntime as ort
import time
from onnx import numpy_helper
from torchvision.models.resnet import ResNet50_Weights
from torchvision import models,transforms as T



def log(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        print("{} took {} seconds".format(func.__name__,format(time.time() - start,'.2f')))
        return out
    return wrapper

def get_label(path):
    with open(path, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    return categories

def get_top5(output, labels):
    top5 = np.argsort(output)[-5:][::-1]
    print("Top 5:")
    print("class"," "*5,"probability")
    for i in top5:
        print("{}:\t{}%".format(labels[i], format(output[i]*100, '.2f')))

def load_image(image_path):
    img = Image.open(image_path)
    img = T.functional.resize(img, (224, 224))
    img = T.functional.to_tensor(img)
    img = T.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img = img.unsqueeze(0)
    return img

@log
def torch_infer(model, img):
    if tr.cuda.is_available():
        img = img.cuda()
        model = model.cuda()
    model.eval()
    with tr.no_grad():
        torch_out = model(img)
        torch_out = tr.nn.functional.softmax(torch_out, dim=1)
        torch_out = torch_out.cpu().numpy().flatten()
    
    return torch_out

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

@log
def onnx_infer(onnx_path, img):
    ort_session = ort.InferenceSession(onnx_path,providers=["CUDAExecutionProvider"] if tr.cuda.is_available() else ["CPUExecutionProvider"])
    ort_inputs = {ort_session.get_inputs()[0].name: img.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)[0]
    ort_outs = ort_outs.flatten()
    ort_outs = softmax(ort_outs)


    return ort_outs

def main():
    resnet50 = models.resnet50(weights = ResNet50_Weights.DEFAULT)
    image_path = "/home/wyq/hobby/model_deploy/snake.jpeg"
    img = load_image(image_path)
    labels = get_label("imagenet_classes.txt")

    print("If use GPU:",tr.cuda.is_available())
    print("*"*50)
    torch_out = torch_infer(resnet50, img)
    get_top5(torch_out, labels)
    print("*"*50)
    ort_outs = onnx_infer("resnet50.onnx", img)
    get_top5(ort_outs, labels)


if __name__ == "__main__":
    main()