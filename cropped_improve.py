import  cv2
import pytesseract

for i in range(2,114):
    # i=61

    img=cv2.imread(f"./Cropped/{i}.jpg",0)
    # img = img[2:-2, 2:-2]

    ocr_str=pytesseract.image_to_data(img,lang='chi_sim',config="--psm 7 -c tessedit_write_images=true")

    lines=ocr_str.split('\n')
    for line in lines[1:]:
        value=line.split('\t')
        if len(value)==12:
            conf = int(value[10])
            level=int(value[0])
            text=value[11]
            if conf>80:
                print(level,conf,text)
    # break

