from matplotlib import table
import mlflow
import os
import argparse
import time
def eval(pram1,pram2):
    return (pram1+pram2)/2
table_dict={
    "a":[1,2,3,4],
    "b":[5,6,7,8]
}
def main(pram1,pram2):
    mlflow.set_experiment(experiment_name="demo_example")
    with mlflow.start_run()as run:
        mlflow.set_tag("version","1.0.0")
        mlflow.log_param("pram1",pram1)
        mlflow.log_param("pram2",pram2)
        mlflow.log_metric("mean",eval(pram1=pram1,pram2=pram2))

    os.makedirs("dummy_folder",exist_ok=True)
    with open ("./dummy_folder/example.txt",'w')as f:
        f.write(f"created at:{time.asctime()}")
    mlflow.log_artifact(local_path="dummy_folder")
    mlflow.log_table(data=table_dict,artifact_file="table.json")
if __name__=="__main__":
    praser=argparse.ArgumentParser()
    praser.add_argument("--pram1","-p1",default=10,type=int)
    praser.add_argument("--pram2","-p2",default=20,type=int)
    args=praser.parse_args()

    main(pram1=args.pram1,pram2=args.pram2)   