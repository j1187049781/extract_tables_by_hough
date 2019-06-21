from operator import itemgetter

import cv2
import numpy as np

from PIL import Image
import pytesseract
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
def box_extraction(img_for_box_extraction_path):
    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)  # Thresholding the image

    # cv2.imshow("img", img_bin);cv2.waitKey()

    # Defining a kernel length
    kernel_length = np.array(img).shape[1] // 60

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)
    # cv2.imshow("img", verticle_lines_img);cv2.waitKey()

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)
    # cv2.imshow("img", horizontal_lines_img);cv2.waitKey()


    img_final_bin = cv2.bitwise_or(verticle_lines_img, horizontal_lines_img)
    # cv2.imshow("img", img_final_bin);cv2.waitKey()

    # joining points
    # joining_points=cv2.bitwise_and(verticle_lines_img, horizontal_lines_img)
    # cv2.imshow("img", joining_points);cv2.waitKey()


    #Bold table edges
    img_final_bin=cv2.dilate(img_final_bin,kernel)

    # Find contours for image, which will detect all the tables
    im2, contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    for c in contours:
        # Returns the location and width,height for every contour
        area=cv2.contourArea(c)
        area_rate=cv2.contourArea(c) / (img.shape[0] * img.shape[1])
        print(f"area: {area_rate}")
        # pass small tables
        if  area_rate>0.1:
            cnt_ploy=cv2.approxPolyDP(c,10,True)
            x, y, w, h = cv2.boundingRect(cnt_ploy)
            img_tab_bin = img_final_bin[y:y + h, x:x + w]
            cv2.imshow("img",img_tab_bin);cv2.waitKey()

            #lines
            lines=cv2.HoughLinesP(img_final_bin,1,np.pi/360,200)
            lines_img=np.zeros(img.shape,np.uint8)
            for line in lines:
                x_start,y_start,x_end,y_end=line[0]
                cv2.line(lines_img,(x_start,y_start),(x_end,y_end),255)
            # cv2.imshow("img", lines_img);
            # cv2.waitKey()
            # find cell
            cell_info=[]
            im2, cnts_cell, hierarchy=cv2.findContours(img_tab_bin,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE,offset=(x,y))
            for index,cell in enumerate(cnts_cell):
                if hierarchy[index][3]==1:
                    continue
                x, y, w, h=cv2.boundingRect(cell)
                cell_info.append((x, y, w, h))

                # img_cp = img.copy()
                # cv2.drawContours(img_cp, [cell], -1, 0, thickness=5)
                # cv2.imshow("img", img_cp)
                # cv2.waitKey()

            # Projection in X (row) direction

            sorted_cell_info=sorted(cell_info,key=itemgetter(1,2))
            for cell in sorted_cell_info:
                cell_img = np.zeros(img.shape, np.uint8)
                x_start,y_start,w,h=cell
                cv2.rectangle(cell_img,(x_start,y_start),(x_start+w,y_start+h),255)
                cv2.imshow("img", cell_img);
                cv2.waitKey()

if __name__=='__main__':
    for i in range(1,2):
        box_extraction(f"data1/1000{i}.jpg")
