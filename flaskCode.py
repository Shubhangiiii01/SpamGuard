import pickle
from flask import Flask, request
from flask_cors import CORS

app = Flask(_name_)
CORS(app)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('newVectorizer.pkl', 'rb') as f:
    newtfidf = pickle.load(f)

with open('newECModel.pkl', 'rb') as f:
    newModel = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return 'Hello, World!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    processed_text = tfidf.transform([text])
    extraProcesses = newtfidf.transform([text])
    result1 = newModel.predict(extraProcesses)
    result = model.predict(processed_text)
    flag = ""
    if result.tolist()[0] == 0 and result1.tolist()[0] == 0:
        flag = "Not Spam"
    else:
        flag = "Spam"
    return {'result': flag}

if _name_ == "_main_":
    app.run(host='0.0.0.0', port=6000)
