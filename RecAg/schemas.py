from enum import Enum
from dataclasses import dataclass

class InterviewState(Enum):
    ACTIVE = 1
    TERMINATED_WEAK = 0
    TERMINATED_BY_USER = 0
    COMPLETED = 0

@dataclass
class InterviewStats:
    bad_answer_streak: int = 0
    hint_failures: int = 0
    total_questions: int = 0
    evasive_answers: int = 0

