import gradio as gr
from chatbot import get_best_answer

def ask_bot(question):
    return get_best_answer(question)

gr.Interface(
    fn=ask_bot,
    inputs="text",
    outputs="text",
    title="AWS & DevOps Q&A Chatbot",
    description="Get answers to questions related to AWS, DevOps, Docker, CI/CD, and Cloud!",
).launch()
