from agent_memory import VectorMemory
from question_creator import QuestionCreator
from answer_evaluator import AnswerEvaluator
from end_control import StopControl

import json


class Interviewer:
    def __init__(self, client, initial_data):
        self.client = client
        self.memory = VectorMemory()
        self.question_creator = QuestionCreator(client, self.memory, initial_data)
        self.answer_evaluator = AnswerEvaluator(client, self.memory, initial_data)
        self.feedback = StopControl(client)
        self.history_log = []

    def run_interview(self):
        idx = 0
        while True:
            question = self.question_creator.create_question()
            if question is None:
                feedback = self.feedback.stop_interview("interview_log.json")
                print(feedback)
                self.history_log.append({"finally_feedback": feedback})
                with open("interview_log.json", "w", encoding="utf-8") as f:
                    json.dump(self.history_log, f, ensure_ascii=False, indent=4)

                return None
            if len(question["llm_answer"]) > 0:
                print(f"Ответ: {question["llm_answer"]}")
            print(f"Вопрос: {question["question"]}")

            answer = input("Ответ кандидата: ")
            review = self.answer_evaluator.evaluate(question, answer)
            stop_interview = review["stop_interview"]
            if stop_interview:
                feedback = self.feedback.stop_interview("interview_log.json")
                print(feedback)
                self.history_log.append({"finally_feedback": feedback})
                with open("interview_log.json", "w", encoding="utf-8") as f:
                    json.dump(self.history_log, f, ensure_ascii=False, indent=4)

                return None

            try:
                score = review["score"]
                knowledge_gaps = review["review"]["knowledge_gaps"]
            except Exception as e:
                knowledge_gaps = review["knowledge_gaps"]

            self.memory.add_entry(
                turn_id=idx,
                question=question["question"],
                answer=answer,
                review=review["review"],
                llm_answer=review["llm_answer"],
                score=score,
                knowledge_gaps=knowledge_gaps,
            )

            self.history_log.append({
                "turns": [{
                    "turn_id": idx + 1,
                    "agent_visible_message": question["question"],
                    "user_message": answer,
                    "review": review["review"],
                    "internal_thoughts": [
                        {
                            "agent_evaluator": {
                                "llm_answer": review["llm_answer"],
                                "score": score,
                                "knowledge_gaps": knowledge_gaps
                            }
                        }
                    ]
                }]
            })

            idx += 1
            with open("interview_log.json", "w", encoding="utf-8") as f:
                json.dump(self.history_log, f, ensure_ascii=False, indent=4)
