from flask import Flask, request, jsonify
# import model
# import your_ml_module
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
@app.route('/predictparkinsons', methods=['POST'])
def predict():
    data = request.json
    prediction =data #model.predict(data)
    print("Handled the data" , data)
    print(data)
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(port=5001) 
