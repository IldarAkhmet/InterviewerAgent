import json
from mistralai import SystemMessage


class StopControl:
    def __init__(self, client):
        self.client = client

    def stop_interview(self, log_path):
        try:
            with open(log_path, encoding='utf-8') as f:
                logs = json.load(f)
        except Exception as e:
            print(e)
            return 'Нет информации для анализа собеседования. Необходимо ответить хотя бы на один вопрос интервьюера.'

        system_message = SystemMessage(
            role="system",
            content="""
                You are a senior technical interviewer and hiring analyst.

                Your task is to analyze the FULL interview log and produce a final evaluation report.
                The interview log is provided as structured JSON and contains:
                - questions
                - candidate answers
                - hints given
                - answer reviews
                - intermediate evaluations
                
                Internal reasoning and analysis must be done in English.
                The final report MUST be written in Russian.
                
                Rules:
                1. Analyze ONLY the information present in the log.
                2. Do NOT assume knowledge that was not explicitly demonstrated.
                3. Evasive, vague, or incorrect answers count as knowledge gaps.
                4. Honest "I don't know" answers indicate honesty but still count as technical gaps.
                5. If multiple questions were asked on the same topic, evaluate overall understanding.
                6. Be strict and realistic, as in a real technical hiring interview.
                
                ========================
                A. Decision
                ========================
                Provide:
                - Grade: Junior / Middle / Senior
                - Hiring Recommendation: No Hire / Hire / Strong Hire
                - Confidence Score: integer from 0 to 100 indicating confidence in the assessment
                
                Evaluation criteria:
                - Junior: fragmented knowledge, frequent hints, weak confidence
                - Middle: solid fundamentals, partial advanced knowledge, rare hints
                - Senior: deep understanding, confident answers, architectural thinking
                
                ========================
                B. Hard Skills Analysis (Technical Review)
                ========================
                1. List all technical topics covered in the interview.
                2. For each topic specify:
                   - ✅ Confirmed Skills — if answers were correct
                   - ❌ Knowledge Gaps — if answers were incorrect, vague, or missing
                
                For each ❌ topic:
                - Briefly explain what the candidate misunderstood
                - Provide the correct technical explanation (concise and accurate)
                
                ========================
                C. Soft Skills & Communication
                ========================
                Rate each as: Low / Medium / High
                
                - Clarity: how clearly and structurally the candidate explained ideas
                - Honesty: whether the candidate admitted lack of knowledge or tried to bluff
                - Engagement: whether the candidate asked clarifying or follow-up questions (if applicable)
                
                Support each rating with brief observations from the log.
                
                ========================
                D. Personal Learning Roadmap (Next Steps)
                ========================
                Provide a concrete list of topics and technologies the candidate should improve.
                
                For each item:
                - What to study
                - Why it is important for the target role
                
                Optional:
                - Include links to official documentation or high-quality articles if appropriate.
                
                ========================
                The output must be a structured professional interview report in Russian.
                No motivational phrases. No generic praise. Facts and analysis only.
            """
        )

        messages = [
            system_message,
            SystemMessage(
                role="system",
                content=str(logs)
            )
        ]


        response = self.client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.1,
        )

        return response.choices[0].message.content.strip()