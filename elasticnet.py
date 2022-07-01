import argparse
from re import M
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn


def evaluate(actual, pred):
    rmse= np.sqrt(mean_squared_error(actual, pred))
    mae= mean_absolute_error(actual, pred)
    r2= r2_score(actual, pred)

    return rmse, mae, r2

def get_data():
    local_path= 'C:/Users/acreddy/Desktop/wine/winequality.csv'
    try:
        df = pd.read_csv(local_path, sep= ';')
        return df
    except Exception as e:
        raise e

def main(alpha, l1_ratio):
    df= get_data()
    train,test= train_test_split(df)
    target= 'quality'

    train_x= train.drop([target], axis=1)
    test_x= test.drop([target], axis=1)
    train_y= train[[target]]
    test_y= test[[target]]

    # mlflow starts here:

    with mlflow.start_run():
       
        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)

        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=17)
        model.fit(train_x, train_y)

        pred= model.predict(test_x)

        rmse, mae, r2=  evaluate(test_y, pred)

        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2', r2) 

        print(f'params- alpha = {alpha}, l1_ratio = {l1_ratio}')
        print(f'eval metrics-  rmse= {rmse}, mae = {mae}, r2 = {r2}')

        mlflow.sklearn.log_model(model, 'model')


if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--alpha', '-a', type= float, default= 0.6)
    parser.add_argument('--l1_ratio', '-l1', type= float, default= 0.4)
    parsed_args = parser.parse_args()

    main(alpha=parsed_args.alpha, l1_ratio=parsed_args.l1_ratio)


