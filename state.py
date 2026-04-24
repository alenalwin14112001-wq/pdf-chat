from typing import TypedDict, Optional

class SectionData(TypedDict):
    title: str
    description: str
    sub_query: str
    chunks: list[str]
    content: str
    score: float
    feedback: str

class GlobalOutline(TypedDict):
    main_topic: str
    key_themes: list[str]
    writing_tone: str

class ReportState(TypedDict):
    # Input
    user_query: str
    template_raw: str

    # Planner output
    global_outline: Optional[GlobalOutline]
    sections: list[SectionData]

    # Control flow
    retry_count: int
    validation_passed: bool

    # Final output
    final_report: str