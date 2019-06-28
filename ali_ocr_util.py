import base64
import json
import os

import cv2
import requests

def ali_ocr(img,url='http://192.168.2.13:80/ocrapidocker/ocrservice.json'):

    retval, bytes_data=cv2.imencode('.jpg',img)
    img_base64=base64.encodebytes(bytes_data.tobytes())
    data = {"method": "ocrService",
            "img": img_base64}
    req = requests.post(url=url, data=data)

    res_json = json.loads(req.text)

    code=res_json['code']
    data=res_json['data']
    text=''
    if code==200:
        text=data['content']
    return text

if __name__=='__main__':
    dataset='./cell_images'
    for name in os.listdir(dataset)[:]:
        name='102.tif'
        jpg_path = os.path.join(dataset, name)
        img=cv2.imread(jpg_path,0)
        ret=ali_ocr(img)
        print(ret)
        break