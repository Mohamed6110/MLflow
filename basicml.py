import random
from re import M
import mlflow
from datetime import datetime
import pandas as pd
import mlflow.sklearn
from sqlalchemy import column
from mlflow.models.signature import infer_signature
from mlflow_utils import get_mlflow_experiment
from sklearn import experimental
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import PrecisionRecallDisplay,RocCurveDisplay,ConfusionMatrixDisplay
from sklearn.metrics import r2_score, recall_score,roc_auc_score,roc_curve,recall_score
import matplotlib.pyplot as plt
import mlflow.sklearn
if __name__ =="__main__":
    experiment=mlflow.set_experiment(experiment_name="forest_classification")
    # experiment=get_mlflow_experiment(experiment_name="beasic_ml")
    print(f"experiment id :{experiment.experiment_id}")
    with mlflow.start_run(run_name="model signature",experiment_id=experiment.experiment_id)as run:
        x,y=make_classification(n_samples=1000,n_features=10,
                                n_informative=5,n_redundant=5,random_state=42)
        X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

        X_train=pd.DataFrame(X_train,columns=['feature_{}'.format(i)for i in range(10)])
        X_test=pd.DataFrame(X_test,columns=['feature_{}'.format(i)for i in range(10)])
        y_train=pd.DataFrame(y_train,columns=['target'])
        y_test=pd.DataFrame(y_test,columns=['target'])

        pram1=mlflow.log_param("n_estimators",100)
        # pram2=mlflow.log_param("max_depth",100)
        
        rfc=RandomForestClassifier(pram1,random_state=42)
        rfc.fit(X_train,y_train)
        y_pred=rfc.predict(X_test)
        y_pred1=pd.DataFrame(y_pred,columns=["prediction"])

        model_signature=infer_signature(model_input=X_train,model_output=y_pred1)

        mlflow.sklearn.log_model(sk_model=rfc, signature=model_signature,name='random_forest_model')

        # mlflow.sklearn.save_model(sk_model=rfc,path="RandomForestClassifier_model",signature=model_signature)

        # fig_pr=plt.figure()
        # pr_display=PrecisionRecallDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        # plt.title("PrecisionRecall curve")
        # plt.legend()

        # mlflow.log_figure(figure=fig_pr,artifact_file="metrics/percision_recall_curve.png")

        # fig_roc=plt.figure()
        # roc_display=RocCurveDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        # plt.title("Roc curve")
        # plt.legend()

        # mlflow.log_figure(figure=fig_roc,artifact_file="metrics/roc_curve.png")

        # fig_cm=plt.figure()
        # roc_display=ConfusionMatrixDisplay.from_predictions(y_test,y_pred,ax=plt.gca())
        # plt.title("ConfusionMatrix")
        # plt.legend()

        # mlflow.log_figure(figure=fig_cm,artifact_file="metrics/cm.png")
        
        # metrics={
        #       "r2_score":float(r2_score(y_test,y_pred=y_pred)),
        #       "recall_score":float(recall_score(y_test,y_pred)),
        #       "roc_auc_score":float(roc_auc_score(y_test,y_pred)),
        #     #   "roc_curve":float(roc_curve(y_test,y_pred))

        # }

        # mlflow.log_metrics(metrics=metrics)

        

        print("run_id: {}".format(run.info.run_id))
        print("artifact_uri: {}".format(run.info.artifact_uri))
        print("experiment id: {}".format(run.info.experiment_id))
        print("status : {}".format(run.info.status))
        unix=run.info.start_time
        dt=datetime.fromtimestamp(unix/1000)
        # unixend=run.info.end_time
        # dtend=datetime.fromtimestamp(unixend/1000)
        print(f"start time :{dt}")
        print(f"end time : {run.info.end_time}")
        print(f"lifecycle_stage : {run.info.lifecycle_stage}")




