import mlflow 
import openai
import os
import pandas as pd
import dagshub

dagshub,inti(repo_owner='karthik01', repo_name='MLFLOW', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/karthik10/MLflow.mlflow")
eval_data = pd.DataFrame(
    {
        "inputs":[
            "what is MLflow?",
            "what is Spark?",
        ],
        "ground_truth":[
            "MLflow is an open_source platform for managing the end_to_end machine learning(ML)"
            "lifecycle. It was developed by Databricks, a company that specalizes in big data and"
            "machine learning solutins. MLflow is designed to address the challenges ehat data"
            "machine learning models,",
            "Apache spark is an open-source, distrubuted in response to limitations of the Hadoop"
            "MapReduce computing model, offerning imporvements in speed and ease of use.Spark"
            "provides libraries for various tasks such as data ingestion, processing, and analysis"
            "through its components like Spark SQL for structured data, Spark streaming for"
        ],
    }
)
mlflow.set_expreiment("LLM Evakucation")
with mlflow.start_run() as run:
    system_prompt = "Answer the following question in two sentences"
    logged_model_info = mlflow.openai.log_model(
        model="gpt-4",
        task = openai.chat.completions,
        artifact_path = "model",
        messages=[
            {"role": "system","content":system_prompt},
            {"role": "user", "content": "{question}"},
        ],
    )

    #use predicated question-amswering metrics to evaluate our model.
    results = mlflow,evaluate(
        logged_model_inflow.model_uri,
        eval_data,
        targets="ground_truth"
        model_type="question-answering",
        extra_metrics=[mlflow.metrics.toxicity(), mlflow.metrics.latency(),mlflow.metrics.genai.answer_similarity()]
    )
    print(f"see aggregated evaluation results below: \n{result.metrics}")

    #evaluation result for each data record is available in "results.tables"
    eval_table= results.tables["eval_results_table"]
    df=pd.DataFrame(eval_table)
    df.to_csv("eval.csv")
    print(f"see evaluation table below: \n{eval_table}")

# export OPENAI_API_KEY=""

