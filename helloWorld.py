from flask import Flask,url_for,request,render_template,redirect,session,json,Response,jsonify

import vgg16_predicted_dog


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('/login.html')

@app.route('/test')
def test():
    return 'test'

@app.route('/upload')
def upload():
    return render_template('/Upload.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        if request.form.get('user') == 'admin':
            return 'Admin login successfully!'
        else:
            return 'No such user!'
    return "login fail"


@app.route('/api1/upload', methods=['GET', 'POST'])
def editorData():
    # 获取图片文件 name = upload
    img = request.files.get('uploadfile')

    # 定义一个图片存放的位置 存放在static下面
    path = "static/img/"

    # 图片名称
    imgName = img.filename

    # 图片path和名称组成图片的保存路径
    file_path = path + imgName

    # 保存图片
    img.save(file_path)

    # url是图片的路径
    url = 'static/img/' + imgName

    result = vgg16_predicted_dog.VGG16_predict_breed(url)
    # print(result)
    return jsonify({'name':result})
    # return jsonify({'url': url})




if __name__ == "__main__":
    app.run(threaded=False,host = '0.0.0.0')

