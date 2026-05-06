from ultralytics import YOLO
from paddleocr import PaddleOCR
from flask import Flask, request, jsonify
import cv2
import numpy as np

app=Flask(__name__)

model=YOLO("runs/detect/train2/weights/best.pt")
pdl=PaddleOCR(use_angle_cls=True, lang='en',show_log=False)

def detectYolo(img_path):
    results=model(img_path,verbose=False)
    return results[0].boxes.xyxy, results[0].boxes.cls.cpu().numpy()     #bbox cords and class labels

def crop_box(img,boxes,classes):
    crops=[]       # List to store the cropped images as array
    
    for (x1,y1,x2,y2),cls in zip(boxes,classes):
        crop=img[int(y1):int(y2), int(x1):int(x2)]
        crops.append({"image": crop, "class": int(cls)})    #stores img & its class
    return crops

def run_ocr(crops):
    all_text=[]

    for i in crops:
        crop_img=i["image"]     #cropped image from the detected box
        cls=i["class"]      #class of the detected object
        result=pdl.ocr(crop_img,cls=True)     
        texts=[]

        if result and result[0]:
            for line in result[0]:
                if line and len(line)>1:
                    texts.append(line[1][0])      #appends only text [[cords], [text, confidence]]
        all_text.append({
            "class": cls,
            "text": texts                  # all_text=[ {"class": 0, "text": ["detected text"]} ]
        })

    return all_text

def to_json(all_texts,model):
    output={}

    for i in all_texts:
        label=model.names[i["class"]]
        text=" ".join(i["text"])
    
        if label not in output:
            output[label]=text
    
    return output

@app.route('/', methods=['POST'])
def output():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file=request.files["file"]
    npimg=np.frombuffer(file.read(), np.uint8)
    img=cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    boxes,classes=detectYolo(img)
    crops=crop_box(img,boxes,classes)
    texts=run_ocr(crops)
    final_output=to_json(texts,model)

    return jsonify(final_output)


if __name__=="__main__":
     app.run(host="0.0.0.0", port=5000)

                                           