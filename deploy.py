import pickle
import pandas as pd
import numpy as np
from flask import Flask
from flask import request

app = Flask('predict')
file = open('titanic_model.pkl', "rb")
CLF = pickle.load(file)
file.close()

FEATURES = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age_cut', 'family',
       'Title_Capt', 'Title_Col', 'Title_Countess', 'Title_Dr', 'Title_Lady',
       'Title_Major', 'Title_Master', 'Title_Miss', 'Title_Mlle', 'Title_Mme',
       'Title_Mr', 'Title_Mrs', 'Title_Ms', 'Title_Rev', 'Embarked_C',
       'Embarked_Q', 'Embarked_S']

@app.route('/predict_survived')
def predict_survived():
    keys = []
    count = 0
    for f in FEATURES:
        try:
            keys.append(float(request.args.get(f)))
            count+=1
        except TypeError:
            return f'No valid arguments, you should return value for {list(FEATURES)}. ' \
                   f'\n{len(FEATURES)-count} arguments missing'

    data = pd.DataFrame(np.array(keys).reshape(1, -1))
    data.columns = FEATURES
    y_pred_api = CLF.predict(data)
    return f'{int(y_pred_api)}'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)