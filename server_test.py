import os
import requests
tables_path='/home/ubuntu/PycharmProjects/extract_tables_by_hough/data/'

# for jpg_name in os.listdir(tables_path)[:]:
# with open(os.path.join(tables_path,"1.jpg"),'rb') as fp:
#     # url='http://182.150.24.15:7000/extractTable'
#     # url = 'http://127.0.0.1:6060/extractTable'
#     url = 'http://192.168.2.13:6060/extractTable'
#     # data={"imgBase64":base64.encodebytes(fp.read())}
#     data={"Img":fp}
#     req=requests.post(url=url,files=data,data={'return_json':'True'})
#     print(req.text)
with open(os.path.join(tables_path,"1.jpg"),'rb') as fp:
    # url='http://182.150.24.15:7000/extractTable'
    # url = 'http://127.0.0.1:6060/extractTable'
    url = 'http://192.168.2.13:6060/extractTable'
    # data={"imgBase64":base64.encodebytes(fp.read())}
    data={"Img":fp}
    req=requests.post(url=url,files=data,data={'return_json':'false'})
    with open('test.xlsx','wb') as f:
        f.write(req.content)