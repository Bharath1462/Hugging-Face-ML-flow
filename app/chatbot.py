import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import mlflow
import dagshub
import time

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='Bharath1462', repo_name='Hugging-Face-ML-Flow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/Bharath1462/Hugging-Face-ML-Flow.mlflow")
mlflow.set_experiment("aws_devops_chatbot")

# Load dataset
df = pd.read_csv("data/questions_answers.csv")

# Load HuggingFace QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Load SentenceTransformer model for semantic similarity
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Enhanced chatbot logic
def get_best_answer(user_question):
    best_score = 0
    best_answer = "క్షమించండి, నాకు సమాధానం తెలియదు."
    top_answers = []

    run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        question_embedding = sentence_model.encode(user_question, convert_to_tensor=True)

        # Loop through all Q&A pairs and evaluate
        for _, row in df.iterrows():
            context = row["answer"]
            context_embedding = sentence_model.encode(context, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(question_embedding, context_embedding).item()

            result = qa_pipeline(question=user_question, context=context)
            top_answers.append({
                "context": context,
                "qa_score": result["score"],
                "similarity_score": similarity_score
            })

            if similarity_score > best_score:
                best_score = similarity_score
                best_answer = context

        # Sort top answers by similarity score
        top_answers = sorted(top_answers, key=lambda x: x["similarity_score"], reverse=True)[:3]

        # Log info to MLflow
        mlflow.log_param("user_question", user_question)
        mlflow.log_metric("top_similarity_score", best_score)
        mlflow.set_tag("model_used", "distilbert-base-cased-distilled-squad + all-MiniLM-L6-v2")

        for i, ans in enumerate(top_answers):
            mlflow.set_tag(f"top{i+1}_answer", ans["context"])
            mlflow.log_metric(f"top{i+1}_similarity_score", ans["similarity_score"])
            mlflow.log_metric(f"top{i+1}_qa_score", ans["qa_score"])

    return best_answer