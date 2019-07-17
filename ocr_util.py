import base64
import json
import os
from multiprocessing.pool import Pool
from time import time

import  cv2
import pytesseract
import requests


ip=os.getenv('ALI_OCR_IP','192.168.2.13')
ALI_OCR_URL = f'http://{ip}:80/ocrapidocker/ocrservice.json'

def ali_ocr(img,url=ALI_OCR_URL):
    if  img is None:
        return ""
    retval, encode_img=cv2.imencode('.jpg',img)

    if not retval:
        print("can't encode image")
        return ""
    data = {"method": "ocrService",
            "prob": "true",
            "charInfo":"true",
            "img": base64.encodebytes(encode_img.tobytes())}
    req = requests.post(url=url, data=data)

    res_json = json.loads(req.text)
    print(res_json)
    code=res_json['code']
    if code ==200:
        data=res_json['data']
        prism_wordsInfo=data['prism_wordsInfo']
        content=''
        for word_info in prism_wordsInfo:
            word=word_info['word']
            prob=word_info['prob']
            content+=word
        return content
    else:
        print(res_json)
        return ""



def tesseract_ocr(img,conf_th=50):
    if  img is None:
        return ""
    retval,img_bin=cv2.threshold(img,128,255,cv2.THRESH_OTSU)
    # cv2.imshow('img',img_bin);cv2.waitKey()

    img_bin_inv=255-img_bin
    retval, labels, stats, centroids=cv2.connectedComponentsWithStats(img_bin_inv)
    border_labels=set()
    H,W=img.shape[:2]
    for row in range(H):
        for col in range(W):
            if labels[row,col] !=0 and (row<2 or col <2 or row >H-2 or col > W-2):
                border_labels.add(labels[row,col])

    for row in range(H):
        for col in range(W):
            if labels[row, col] in border_labels:
                img_bin[row,col]=255

    # cv2.imshow('img',img_bin);cv2.waitKey()

    start=time()
    ocr_str=pytesseract.image_to_data(img_bin,lang='chi_sim+eng',config="--psm 6 -c tessedit_write_images=true")
    end=time()
    print(f'tesseract time: {end-start}')
    print(ocr_str)
    ret=''
    lines=ocr_str.split('\n')
    for line in lines[1:]:
        value=line.split('\t')
        if len(value)==11:
            conf = int(value[10])
            level=int(value[0])
            if conf == -1:
                text='\n'
                ret+=text
        if len(value)==12:
            conf = int(value[10])
            level=int(value[0])
            text=value[11]
            if conf>conf_th:
                # print(level,conf,text)
                ret += text
    # print(ret)
    return ret



def multi_process_ocr(chars_cell_imgs,ocr_fun,processes=None):
    if not processes:
        processes=os.cpu_count()
    print(f"processes: {processes}")
    if chars_cell_imgs:
        with Pool(processes) as p:
            ret=p.map(ocr_fun,chars_cell_imgs)
            return ret



if __name__=='__main__':
    for i in range(0, 112):
        i=109
        print(f'img {i}')
        img = cv2.imread(f"./cell_images/{i}.tif",0)

        ret=tesseract_ocr(img)
        print(ret)
        break
    img = cv2.imread("/home/ubuntu/PycharmProjects/extract_tables_by_hough/tessinput.tif", 0)
    print(ali_ocr(img))
    # (ali_ocr(img))
    # imgs=[cv2.imread(f"./cell_images/{i}.tif",0) for i in range(0, 112)]
    # print(multi_process_ocr(imgs,ali_ocr))
    # print(multi_process_ocr( imgs,tesseract_ocr))
