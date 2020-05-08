
import pandas as pd
from model.predictor import predictor
from model import preprocess as pr
from flask import Flask,request, send_file,render_template, send_from_directory,redirect,url_for
import os
import subprocess

UPLOAD_FOLDER = './upload/'
app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
NAME = None



@app.route('/', methods=["GET",'POST'])
def csv_upload():
    global NAME
    NAME = None
    if request.method == "GET":
        return render_template('home.html')
    else:
        file = request.files['csv_file']
        NAME = file.filename
        file = pd.read_csv(file)
        #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) #uploadフォルダに保存
        output = predictor(file)

        output.to_csv(f"./output_csv/predict_{NAME[:-4]}.csv",index = None)
        #subprocess.call(["rm",f"./upload/{filename}"])
        return render_template("result.html")
        #ここに推論しているなんかページをreturnさせて, 終わったらcsvダウンロードボタンをだす
@app.route('/redirect/')
def redirect_example():
    # url_for で index() に紐付いた URL を生成
    # 生成された URL にリダイレクト
    return redirect(url_for('csv_upload'))

#ダウンロードボタンが押されたら実行
@app.route("/export")
def export_action():
    # 現在のディレクトリを取得
    path = os.path.abspath(__file__)[:-7]

    return send_from_directory(
        directory= path + "/output_csv",
        filename=f'predict_{NAME[:-4]}.csv',
        as_attachment=True,
        attachment_filename=f"predict_{NAME[:-4]}.csv",
    )


if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8000, debug = True)