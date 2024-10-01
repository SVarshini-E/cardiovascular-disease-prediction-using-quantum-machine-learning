from flask import Flask, request, jsonify,render_template
import numpy as np
import pickle


# Initialize Flask app
app = Flask(__name__)
model=pickle.load(open('qsvc_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('cardio.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the POST request
    v1=request.form['f1']
    v2=request.form['f2']
    v3=request.form['f3']
    v4=request.form['f4']
    v5=request.form['f5']
    v6=request.form['f6']
    input_data=[[int(v1),int(v2),int(v3),int(v4),int(v5),int(v6)]]
    reshaped_data=np.array(input_data).reshape(1,-1)
    prediction=model.predict(reshaped_data)

    output= prediction[0]

    if output == 0:
        message = "The person does not have cardiovascular disease symptoms."
    else:
        message = "The person has cardiovascular disease symptoms."

    

    return render_template('cardio.html',prediction_text=message)


if __name__ == "__main__":
    app.run(debug=True)
