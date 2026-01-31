from utils import json_parser
from agent_memory import VectorMemory

from mistralai import UserMessage, SystemMessage


class QuestionCreator:
    def __init__(self, client, vector_memory: VectorMemory, initial_data):
        self.client = client
        self.vector_memory = vector_memory
        self.initial_data = initial_data

    def create_question(self, n=5):
        last_qa = self.vector_memory.get_last_n(n)

        context_messages = []
        avg_score = 0
        nan_scores_cnt = 0
        for e in last_qa:
            context_messages.append(
                SystemMessage(
                    role="system",
                    content=f"Previous question: {e['question']}"
                )
            )
            if e["review"]["score"] is None:
                nan_scores_cnt += 1
            else:
                avg_score += e["review"]["score"]

        if len(last_qa) - nan_scores_cnt != 0:
            avg_score /= (len(last_qa) - nan_scores_cnt)
            last_review = last_qa[-1]["review"]
            last_score = [last_qa[x]["review"]["score"] for x in range(len(last_qa) - 1, -1, -1) if last_qa[x]["review"]["score"] is not None][0]
            if avg_score < 2 and (len(last_qa) - nan_scores_cnt) > 3:
                return None

        system_prompt = """
            You are a professional technical interviewer.
    
            Goal:
            Generate the NEXT interview question for the candidate.
    
            Inputs you will receive in the user message:
            - candidate_background
            - last_review (may be empty)
            - last_score (may be null)
            - average_score (may be null)
            - difficulty (explicit)
            - knowledge_gaps (may be empty)
            - llm_answer (may be empty): assistant answer to the candidate's meta/job question from the evaluator step
    
            Process (must follow in this order):
            1) If llm_answer is non-empty: return it in the output field "llm_answer" (do NOT merge it into the question).
            2) Generate exactly ONE new technical interview question in Russian, aligned with:
               - explicit difficulty,
               - candidate_background,
               - last_review feedback and knowledge_gaps.
            3) Adapt difficulty:
               - If last_score is not null and last_score < 7: make the next question easier and target 1-2 knowledge_gaps.
               - If average_score is not null and average_score < 5: make the next question easier.
               - If last_review is good and last_score >= 7: do NOT repeat or paraphrase previous questions; move forward.
    
            Rules:
            - Ask EXACTLY ONE question (one question mark).
            - No explanations, no formatting, no multiple sub-questions.
            - Do not repeat or paraphrase previous questions when performance is good.
            - Think internally in English; output in Russian ONLY.
    
            Output:
            Return ONLY valid JSON with exactly these keys:
            {
              "question": "…",
              "llm_answer": "…"  // empty string if none
            }
        """
        if len(last_qa) - nan_scores_cnt != 0:
            user_prompt = (
                f"Candidate background:\n{self.initial_data}\n\n"
                f"Review last answer:\n{str(last_review)}\n\n"
                f"Last score:\n{last_score}\n\n"
                f"Average score:\n{avg_score}\n\n"
            )
        else:
            user_prompt = (
                f"Candidate background:\n{self.initial_data}\n\n"
            )

        messages = [
            SystemMessage(role="system", content=system_prompt),
            *context_messages,
            UserMessage(role="user", content=user_prompt),
        ]

        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        return json_parser(response.choices[0].message.content.strip())

