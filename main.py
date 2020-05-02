import pandas as pd
from model.predictor import predictor
from model import preprocess as pr
from flask import *
import io
import os
import csv

UPLOAD_FOLDER = './upload/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=['GET'])
def home():

    return render_template('home.html')


@app.route('/upload', methods=['POST'])
def csv_upload():
    file = request.files['csv_file']
    filename = file.filename

    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) #uploadフォルダに保存
    output = predictor(filename)
    output.to_csv("./output_csv/predict.csv",index = None)
    return render_template("result.html")
    #ここに推論しているなんかページをreturnさせて, 終わったらcsvダウンロードボタンをだす


#ダウンロードボタンが押されたら実行
@app.route("/export")
def export_action():
    # 現在のディレクトリを取得
    path = os.path.abspath(__file__)[:-7]
    print(path)
    return send_from_directory(
        directory= path + "/output_csv",
        filename='predict.csv',
        as_attachment=True,
        attachment_filename='predict.csv',
    )

if __name__ == "__main__":
    app.run(debug = True)