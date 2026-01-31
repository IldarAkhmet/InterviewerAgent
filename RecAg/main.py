from interviewer import Interviewer
from answer_evaluator import AnswerEvaluator
from agent_memory import VectorMemory
from question_creator import QuestionCreator
from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv(".local_env")
api_key = os.getenv("MISTRAL_API_KEY")

client = Mistral(api_key=api_key)
interviewer = Interviewer(
    client=client,
    initial_data="Привет. Я Алекс, претендую на позицию Junior Backend Developer. Знаю Python, SQL и Git",
)

interviewer.run_interview()
