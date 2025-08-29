import mlflow
import mlflow.sklearn
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib
if __name__ =="__main__":
    
    x,y=make_classification(n_samples=1000,n_features=10,n_informative=5,
                            n_redundant=5,random_state=42)
    x=pd.DataFrame(x,columns=["feature_{}".format (i)for i in range (10)])
    y=pd.DataFrame(y,columns=["target"])

    _,X_test,_,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    model_url="file:///F:/playing/flask_mlflow/mlruns/183401935329498046/models/m-f004fb08f5a9494e89f94cb389a9883b/artifacts/model.pkl"
    rfc=mlflow.sklearn.load_model(model_url)
    y_pred=rfc.predict(X_test)
    y_pred=pd.DataFrame(y_pred,coloums=["prediction"])
    print(y_pred.head())
