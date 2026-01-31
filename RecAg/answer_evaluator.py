from utils import json_parser
from agent_memory import VectorMemory

from mistralai import UserMessage, SystemMessage


def build_llm_messages(entries, last_n=3):
    messages = []

    for turn in entries[-last_n:]:
        if "review" not in turn:
            messages.append({
                "role": "system",
                "content": turn["question"]["content"]
            })
        messages.append({
            "role": "user",
            "content": turn["answer"]["content"]
        })

        if "review" in turn:
            messages.append({
                "role": "system",
                "content": turn["review"]["content"]
            })

    return messages


class AnswerEvaluator:
    def __init__(self, client, memory: VectorMemory, initial_data):
        self.client = client
        self.memory = memory
        self.initial_data = initial_data

    def evaluate(self, question, answer):
        history = build_llm_messages(self.memory.entries, last_n=3)

        messages = [
            SystemMessage(
                role="system",
                content=(
                    """
                        You are a senior technical interviewer and evaluator.
                        
                        Task:
                        Given:
                        - interview_question (the question asked to the candidate)
                        - candidate_message (the candidate's latest message)
                        - candidate_answer (the candidate's answer, if present; may be empty)
                        - interview_history (past Q/A and your past reviews; context only)
                        - candidate_profile (initial data about the person)
                        Return an evaluation in STRICT JSON.
                        
                        Core principle:
                        First, classify the candidate_message intent:
                        A) TECHNICAL_ANSWER: the candidate attempts to answer the interview_question (fully or partially).
                        B) META_JOB_QUESTION: the candidate asks about the job, role, company, interview process, expectations, compensation, logistics, feedback, or similar meta topics; they may also include some info about themselves.
                        C) END_INTERVIEW: the candidate explicitly wants to stop/exit/end the interview.
                        
                        Rules (global):
                        - Be concise, objective, and professional.
                        - Do not ask follow-up questions.
                        - Do not explain the correct answer.
                        - Use interview_history only as context for consistency; do not “double-penalize” repeated gaps.
                        - Check the answer against candidate_profile constraints when relevant (seniority, stack, experience).
                        - Increase difficulty adjustment over time if performance is strong (reflect this only in review phrasing; do not add new fields).
                        - If intent is END_INTERVIEW -> stop_interview = true.
                        
                        Scoring rules:
                        - If intent is TECHNICAL_ANSWER:
                          - Provide score as an integer 0-10 based on technical correctness, clarity, and depth.
                        - If intent is META_JOB_QUESTION:
                          - DO NOT penalize.
                          - score MUST be null.
                          - Provide a concise review describing what happened (e.g., “Candidate asked about the role; no technical evaluation this turn.”)
                          - Provide a helpful, direct response to their meta/job question in llm_answer.
                        - If intent is END_INTERVIEW:
                          - score MUST be null.
                          - stop_interview MUST be true.
                          - llm_answer may contain a brief closing or next steps.
                        
                        knowledge_gaps ru_
                    """
                )
            ),
        ]

        messages.extend(history)

        messages.append(
            UserMessage(
                role="user",
                content=(
                    f"Candidate initial data: \n{self.initial_data}\n\n"
                    f"Question:\n{question}\n\n"
                    f"Candidate answer:\n{answer}\n\n"
                    "Evaluate the answer strictly."
                )
            )
        )

        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content

        return json_parser(content)
