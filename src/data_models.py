from pydantic import BaseModel, Field, model_validator
import uuid
from typing import Self


class MinimalSource(BaseModel):
    """Single source of information.

    text is optional: it is filled by the indexer, but the sources of the
    reference datasets do not have it.
    """
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str = ""

    @model_validator(mode="after")
    def check_range(self) -> Self:
        first = self.first_character_index
        last = self.last_character_index
        if not first <= last:
            raise ValueError(
                f"The first index is superior to the last in {self!r}")
        if self.text and last - first != len(self.text):
            raise ValueError(
                f"The length of the text does not correspond in {self!r}")
        return self


class UnansweredQuestion(BaseModel):
    """Representation of an unanswered question."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Representation of an answered question."""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Representation of a RAG dataset."""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Minimal representation of search results for a question."""
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Minimal representation of an answer to a question"""
    answer: str


class StudentSearchResults(BaseModel):
    """Representation of a student's search results for a question."""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Representation of a student's search results and answer"""
    search_results: list[MinimalAnswer]
    k: int
