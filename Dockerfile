FROM ubuntu:disco
#change pip source

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y  nano libopencv-dev  python3.7-dev python3-pip tesseract-ocr tesseract-ocr-chi-sim
RUN mkdir /root/.pip &&  echo "[global]\nindex-url = https://pypi.douban.com/simple\ntrusted-host = pypi.douban.com" > /root/.pip/pip.conf
RUN pip3 install  xlsxwriter requests 'opencv-contrib-python==3.4.2.17' pytesseract numpy flask

COPY ./*.py   /app/
COPY ./scaneData   /app/data
RUN mkdir -p /app/excel
#expose port

WORKDIR /app
ENV PYTHONUNBUFFERED 1
ENV LANG C.UTF-8

CMD ["python3.7","server.py"]
