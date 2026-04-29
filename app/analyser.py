import json
import os
import re

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

_MODEL = "models/gemini-2.5-flash"

TOPICS = [
    "Relations & Functions", "Inverse Trigonometry", "Matrices", "Determinants",
    "Continuity & Differentiability", "Applications of Derivatives",
    "Integrals", "Applications of Integrals", "Differential Equations",
    "Vectors", "3D Geometry", "Linear Programming", "Probability",
]


def analyse_submission(questions: list[dict]) -> dict:
    """
    Takes all questions with marks + transcriptions.
    Returns structured strengths/weaknesses analysis.
    """
    q_summary = [
        {
            "q": q["question_number"],
            "marks": q["marks_obtained"],
            "max": q["max_marks"],
            "answer": (q["answer_text"] or "")[:400],
        }
        for q in questions
    ]

    prompt = f"""You are analysing a Class 12 Mathematics answer sheet.

Questions with student answers and marks:
{json.dumps(q_summary, indent=2)}

Topics covered in this exam:
{", ".join(TOPICS)}

Identify which questions belong to which topic based on the answer content.
Then analyse performance patterns.

Output ONLY valid JSON:
{{
  "overall_percentage": 42.5,
  "topic_performance": {{
    "Integrals": {{"scored": 6, "total": 10, "verdict": "weak"}},
    "Matrices": {{"scored": 7, "total": 8, "verdict": "strong"}}
  }},
  "question_topics": {{"1": "Relations & Functions", "2": "Inverse Trigonometry"}},
  "strengths": [
    {{"topic": "Matrices", "observation": "Correctly applied cofactor expansion"}}
  ],
  "weaknesses": [
    {{"topic": "Integrals", "observation": "Consistently forgets +C", "frequency": 3}}
  ],
  "error_patterns": [
    "Sign errors in determinant calculation (Q15, Q16)"
  ],
  "recommendations": [
    "Revise integration by parts and substitution rules"
  ]
}}"""

    response = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
    )
    text = (response.text or "").strip()
    # Strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        return {}
