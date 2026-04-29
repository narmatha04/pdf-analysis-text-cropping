import json
import os

import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

_FLASH = "gemini-2.0-flash"

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

    model = genai.GenerativeModel(_FLASH)
    response = model.generate_content(prompt)
    return json.loads(response.text.strip())
