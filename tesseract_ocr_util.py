import  cv2
import numpy as np
import pytesseract




def ocr(img,cut_down_th=30,conf_th=50):

    # # 判断是否有字
    # laplacian_img = cv2.Laplacian(img,cv2.CV_64F)
    # cut_dowm=laplacian_img<cut_down_th+10
    # laplacian_img[cut_dowm]=0
    # # cv2.imshow('img',laplacian_img);cv2.waitKey()
    # n = np.sum(~cut_dowm)
    # if n<30 or  np.sum(np.abs(laplacian_img))/n<30:
    #     return ''

    retval, img = cv2.threshold(img, 128, 255, cv2.THRESH_OTSU)
    H,W=img.shape[:2]

    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    (success, saliencyMap) = saliency.computeSaliency(img)
    saliencyMap = (saliencyMap * 255).astype("uint8")
    retval, saliencyMap = cv2.threshold(saliencyMap, 100, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy=cv2.findContours(saliencyMap,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    chars_cnt=[]
    for cnt in contours:
        x,y,w,h =cv2.boundingRect(cnt)
        if x<5 or y <5 or x+w>W-5 or y+h>H-5:
            continue
        chars_cnt.append(cnt)

        # bg=np.zeros((H,W,3),dtype=np.uint8)
        # cv2.drawContours(bg,[cnt],-1,(255,255,0),10)
        # cv2.imshow("Contours", bg)
        # cv2.waitKey(0)
    if not chars_cnt:
        return ""
    concat_cnt=np.concatenate(chars_cnt,axis=0)

    # bg = np.zeros((H, W, 3), dtype=np.uint8)
    # cv2.drawContours(bg, [concat_cnt], -1, (255, 255, 0), 10)
    # cv2.imshow("concat_cnt", bg)
    # cv2.waitKey(0)


    x,y,w,h=cv2.boundingRect(concat_cnt)

    chars_img=img[x-5:x+w+5,y-5:y+h+5]
    # cv2.rectangle(img,(x,y),(x+w,y+h),128,5)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    # saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    # (success, saliencyMap) = saliency.computeSaliency(img)
    #
    # # if we would like a *binary* map that we could process for contours,
    # # compute convex hull's, extract bounding boxes, etc., we can
    # # additionally threshold the saliency map
    # threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,
    #                           cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # show the images
    # cv2.imshow("Image", img)
    # cv2.imshow("Output", saliencyMap)
    # cv2.imshow("Thresh", threshMap)
    # cv2.waitKey(0)



    ocr_str=pytesseract.image_to_data(chars_img,lang='chi_sim+eng',config="--psm 6 -c tessedit_write_images=true")

    # print(ocr_str)
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
    print(ret)
    return ret

if __name__=='__main__':
    for i in range(28, 112):
        # i=102
        print(f'img {i}')
        img = cv2.imread(f"./cell_images/{i}.tif",0)

        ocr(img)
        # break