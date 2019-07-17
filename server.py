


from flask import Flask, request, jsonify
import json
import os
from flask import send_file


from extractTabFromImage import extract_ocr, write_xlsx

app = Flask(__name__)

@app.route("/")
def hello():
    return "passed!"

@app.route('/extractTable', methods=['POST'])
def extractTable():
    result = {}
    result['status']=1
    try:
        fp = request.files['Img']
        return_json=request.form['return_json']
        fname = fp.name
        tabs=extract_ocr(fp)
        if return_json is not None and (return_json.lower()=='true'):
            result['data']=tabs
            return jsonify(result)
        else:
            path=os.path.join('./excel', fname + '.xlsx')
            write_xlsx(path,tabs)
            return send_file(path,attachment_filename=fname + '.xlsx')
    except BaseException as e:
        print(e)
        result['err'] = "{}".format(e)
        type=0 # error
        result['status'] = type
        return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6060)

