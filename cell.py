import os
from operator import itemgetter
from functools import cmp_to_key, reduce
import cv2
import numpy as np
import xlsxwriter

from PIL import Image
import pytesseract

from tesseract_ocr_util import ocr

W_TOLERANCE_PIXELS=5
H_TOLERANCE_PIXELS=5
MIN_CELL_AREA=1000
MIN_CELL_RESPECT_RATIO=0.05
cv2.namedWindow('img', cv2.WINDOW_NORMAL)


def is_overlay_point(x1,y1,x2,y2):

    return (x1-x2)*(x1-x2)+(y1-y2)*(y1-y2) <W_TOLERANCE_PIXELS*H_TOLERANCE_PIXELS*4

def find_overlay_point(x,y,points):
    for point in points:
        if is_overlay_point(x,y,point[0],point[1]):
            return point
def cmp_rect(r1):
    assert len(r1)==4
    x1, y1,w,h=r1


    return x1*x1+y1*y1


def rotate_bound(image, angle):
    rows, cols = image.shape[:2]

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(image, M, (cols, rows),borderMode=cv2.BORDER_REFLECT)
    return dst

def box_extraction(img_for_box_extraction_path):
    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    if  img is None:
        return
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)  # Thresholding the image

    # cv2.imshow("img", img_bin);cv2.waitKey()

    #纠正倾斜
    # lines
    lines = cv2.HoughLinesP(img_bin, rho=1, theta=np.pi / 360, threshold=500,minLineLength=500,maxLineGap=5)

    # lines_img = np.zeros(img.shape, np.uint8)
    # for line in lines:
    #     x_start, y_start, x_end, y_end = line[0]
    #     cv2.line(lines_img, (x_start, y_start), (x_end, y_end), 255,5)
    #     cv2.imshow("img", lines_img);
    #     cv2.waitKey()

    angle_sum=0
    num=0
    for line in lines:
        x_start, y_start, x_end, y_end = line[0]
        if(x_end-x_start==0):
            continue
        k=(y_end-y_start)/(x_end-x_start)
        if (k>1 or k < -1):
            angle_sum+=np.arctan(k)*(180/np.pi)
            num+=1
    rotated_img=img
    if num!=0:
        angle_avg=angle_sum/num
        sign=1 if angle_avg>0 else -1
        angle=sign*(abs(angle_avg)-90)
        rotated_img=rotate_bound(img,angle)
        print(f"rotate img {angle_avg} degree")

    # cv2.imshow('img',rotated_img);cv2.waitKey()

    (thresh, img_bin) = cv2.threshold(rotated_img, 128, 255,
                                      cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)  # Thresholding the image

    # cv2.imshow('img',img_bin);cv2.waitKey()

    # Defining a kernel length
    w_kernel_length = np.array(rotated_img).shape[1] // 30
    h_kernel_length = np.array(rotated_img).shape[0] // 30

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h_kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w_kernel_length, 1))
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

    # # Bold table edges
    # img_bin = cv2.dilate(img_bin, kernel)
    # cv2.imshow("img", img_bin);cv2.waitKey()

    # joining points
    joining_points_img=cv2.bitwise_and(verticle_lines_img, horizontal_lines_img)

    # cv2.imshow("img", joining_points_img);cv2.waitKey()



    # Find contours for image, which will detect all the tables
    im2, contours, hierarchy = cv2.findContours(
        img_final_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    tabs=[]
    for c in contours:

        area_rate = cv2.contourArea(c) / (rotated_img.shape[0] * rotated_img.shape[1])
        # print(f"area: {area_rate}")
        # pass small tables
        if area_rate > 0.1:
            cnt_ploy = cv2.approxPolyDP(c, 10, True)
            x, y, w, h = cv2.boundingRect(cnt_ploy)
            img_tab_bin = img_final_bin[y:y + h, x:x + w]
            # cv2.imshow("img", img_tab_bin);
            # cv2.waitKey()


            # find cell
            cell_info = []
            im2, cnts_cell, hierarchy = cv2.findContours(img_tab_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                         offset=(x, y))

            if not cnts_cell:
                continue
            hierarchy=hierarchy[0]


            for index, cell in enumerate(cnts_cell):
                # print(hierarchy[index])
                # print(cv2.contourArea(cell))
                #
                # img_cp = rotated_img.copy()
                # cv2.drawContours(img_cp, [cell], -1, 0, thickness=5)
                # cv2.imshow("img", img_cp)
                # cv2.waitKey()
                # 验证继承关系
                if hierarchy[index][3] != 0:
                    print(f"fliter  {index}  hierarchy {hierarchy[index][3]} ")
                    continue
                if cv2.contourArea(cell)<MIN_CELL_AREA:
                    print(f"fliter  {index}  contourArea {cv2.contourArea(cell)} ")
                    continue
                x, y, w, h = cv2.boundingRect(cell)
                if min(w,h)/max(w,h)<MIN_CELL_RESPECT_RATIO:
                    print(f"fliter  {index}  ratio {min(w,h)} / {max(w,h)}  ")
                    continue


                if abs(cv2.contourArea(cell)-w*h)>(2*w*W_TOLERANCE_PIXELS+2*h*H_TOLERANCE_PIXELS):
                    print(f"fliter  {index}  abs area ratio  {abs(cv2.contourArea(cell)-w*h)}  ")
                    continue
                cell_info.append((x, y, w, h))



            #合并单元格
            #find all joints
            if not cell_info:
                continue
            joints = []
            for cell in cell_info:
                x, y, w, h=cell

                joints.append((x, y))
                joints.append((x+w, y))
                joints.append((x, y+h))
                joints.append((x+w, y+h))

            # points_img = np.zeros(rotated_img.shape, np.uint8)
            # for x,y in joints:
            #         cv2.circle(points_img,(x,y),3,255)
            # cv2.imshow('img',points_img);cv2.waitKey()
            # 在Y方向投影连接点
            round_x_points = []
            x_projection_joints = sorted(joints, key=itemgetter(0))
            near_x_points = [x_projection_joints[0]]
            for i in range(1, len(x_projection_joints)):
                w = x_projection_joints[i][0] - x_projection_joints[i - 1][0]
                if w > W_TOLERANCE_PIXELS * 2:
                    round_x_points.append(near_x_points)
                    near_x_points = [x_projection_joints[i]]
                else:
                    near_x_points.append(x_projection_joints[i])
            round_x_points.append(near_x_points)
            # 在X方向投影连接点
            round_y_points = []
            y_projection_joints = sorted(joints, key=itemgetter(1))
            near_y_points = [y_projection_joints[0]]
            for i in range(1, len(y_projection_joints)):
                h = y_projection_joints[i][1] - y_projection_joints[i - 1][1]
                if h > H_TOLERANCE_PIXELS * 2:
                    round_y_points.append(near_y_points)
                    near_y_points = [y_projection_joints[i]]
                else:
                    near_y_points.append(y_projection_joints[i])
            round_y_points.append(near_y_points)
            #近似化提取表格框架信息
            avg_spilt_xs=[]
            for near_round_x_points in round_x_points:
                sum=reduce(lambda s,p2:s+p2[0],near_round_x_points,0)
                avg_spilt_x=np.around(sum/len(near_round_x_points))
                avg_spilt_xs.append(avg_spilt_x)

            avg_spilt_ys=[]
            for near_round_y_points in round_y_points:
                sum=reduce(lambda s,p2:s+p2[1],near_round_y_points,0)
                avg_spilt_y=np.around(sum/len(near_round_y_points))
                avg_spilt_ys.append(avg_spilt_y)

            round_cell_joints=[]
            for xstart,x_axis in enumerate(avg_spilt_xs):
                for ystart,y_axis in enumerate(avg_spilt_ys):
                    round_cell_joints .append((x_axis,y_axis,xstart+1,ystart+1))

            # 确定单元格的起始信息
            print(f"tables : {len(avg_spilt_xs)-1} x {len(avg_spilt_ys)-1}")
            tab_cell_info=[]
            for cell in cell_info:

                # cell_img = cv2.cvtColor(rotated_img.copy(), cv2.COLOR_GRAY2BGR)
                # x_start, y_start, w, h= cell
                # color=np.random.randint(0,256,(3,))
                # color=tuple((int(c) for c in color))
                # cv2.rectangle(cell_img, (x_start, y_start), (x_start + w, y_start + h),color,5)
                # cv2.imshow("img", cell_img);
                # cv2.waitKey()



                x,y,w,h =cell
                top_left=find_overlay_point(x,y,round_cell_joints)
                top_right=find_overlay_point(x+w,y,round_cell_joints)
                bottom_left=find_overlay_point(x,y+h,round_cell_joints)
                bottom_right=find_overlay_point(x+w,y+h,round_cell_joints)

                if top_left and top_right and bottom_left and bottom_right:
                    _,_,xsc,ysc=top_left
                    _,_,xec,yec=bottom_right
                    if xec==top_right[2] and ysc==top_right[3] and xsc==bottom_left[2] and yec==bottom_left[3]:
                        tab_cell_info.append((x,y,w,h,xsc,ysc,xec-1,yec-1))
                    else:
                        print(f"fliter  {cell} : four joints not overlay")
                else:
                    print(f"fliter {cell}: can't find four joints")


            # cell_img = cv2.cvtColor(rotated_img.copy(),cv2.COLOR_GRAY2BGR)
            # for i,cell in enumerate(tab_cell_info):
            #     # cell_img = np.zeros(rotated_img.shape, np.uint8)
            #     x_start, y_start, w, h ,xsc,ysc,xec,yec= cell
            #     # print(xsc,ysc,xec,yec)
            #     color=np.random.randint(0,256,(3,))
            #     color=tuple((int(c) for c in color))
            #     cv2.rectangle(cell_img, (x_start, y_start), (x_start + w, y_start + h),color,5)
            # cv2.imshow("img", cell_img);
            # cv2.waitKey()


    tabs.append(tab_cell_info)
    return rotated_img,tabs


def img_to_tab(rotated_img,tabs_info):
    tabs=[]
    for tab_cell_info in tabs_info:
        tab = []
        for cell in tab_cell_info:
            x, y, w, h, xsc, ysc, xec, yec = cell
            chars_cell_img=rotated_img[y:y+h,x:x+w]
            ret=ocr(chars_cell_img)
            print(ret)
            cell_info={'xsc':xsc,
                       'ysc': ysc,
                       'xec': xec,
                       'yec': yec,
                       'word': ret,
                       }
            tab.append(cell_info)
        tabs.append(tab)
    return tabs

def write_xlsx(path,tabs):
    with  xlsxwriter.Workbook(path) as workbook:
        for tab_id,tab in enumerate(tabs):
            worksheet = workbook.add_worksheet(str(tab_id))
            for cell in tab:
                xsc = cell["xsc"]-1
                xec = cell["xec"]-1
                ysc = cell["ysc"]-1
                yec = cell["yec"]-1
                if (xsc < 0 or xec < 0 or ysc < 0 or yec < 0):
                    continue
                word = cell["word"]
                # 行列需要合并
                if (xsc < xec or ysc < yec):
                    worksheet.merge_range(ysc, xsc, yec, xec, word)
                else:
                    worksheet.write(ysc, xsc, word)
if __name__ == '__main__':
    dataset='./data'
    for name in os.listdir(dataset)[:1]:
        print(name)
        jpg_path=os.path.join(dataset, name)
        # jpg_path='/home/ubuntu/PycharmProjects/extract_tables_by_hough/data/4.jpg'
        rotated_img,tabs_info=box_extraction(jpg_path)
        tabs=img_to_tab(rotated_img,tabs_info)
        write_xlsx('1.xlsx',tabs)
