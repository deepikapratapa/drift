"""
drift/serving/persona.py

GenAI persona layer for Drift.
Uses Groq's free API (LLaMA 3) to generate plain-English
persona reports from behavioral data and SHAP risk factors.

Usage (standalone test):
    python drift/serving/persona.py
"""

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"


def generate_persona_report(
    archetype: str,
    churn_probability: float,
    risk_level: str,
    risk_factors: list[dict],
    recommendation: str,
    user_stats: dict,
) -> str:
    """
    Generate a plain-English persona report for a user segment.

    Args:
        archetype: behavioral archetype name e.g. "The Cart Abandoner"
        churn_probability: float between 0 and 1
        risk_level: "low" | "medium" | "high" | "critical"
        risk_factors: list of risk factor dicts from the API
        recommendation: intervention recommendation string
        user_stats: dict of key feature values for context

    Returns:
        Plain-English persona narrative as a string.
    """

    risk_factor_text = "\n".join([
        f"- {rf['factor']}: {rf['detail']} (impact: {rf['impact']})"
        for rf in risk_factors
    ])

    stats_text = "\n".join([
        f"- {k.replace('_', ' ').title()}: {v}"
        for k, v in user_stats.items()
    ])

    prompt = f"""You are a behavioral data analyst writing a persona report for a product team.
You have analyzed a user segment and need to write a concise, insightful report.
Write in plain English — no jargon, no bullet points, no headers.
Three paragraphs maximum. Be specific and actionable.
Do not start with "This user" — start with the archetype name.

USER SEGMENT DATA:
Archetype: {archetype}
Churn probability: {churn_probability:.1%}
Risk level: {risk_level}

Key behavioral signals:
{stats_text}

Risk factors identified:
{risk_factor_text}

Recommended intervention: {recommendation}

Write a 3-paragraph persona report:
1. Who this user is and how they behave on the platform
2. Why they are at risk of churning (or not)
3. What the team should do and when
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


def generate_cluster_summary(profiles: dict) -> str:
    """
    Generate a high-level summary of all behavioral archetypes.

    Args:
        profiles: cluster_profiles.json content

    Returns:
        Plain-English executive summary of all archetypes.
    """
    profiles_text = ""
    for cluster_id, profile in profiles.items():
        if int(cluster_id) == -1:
            continue
        profiles_text += f"\n{profile['archetype']}:\n"
        profiles_text += f"  - {profile['user_count']:,} users ({profile['pct_of_sample']}% of sample)\n"
        if profile.get("churn_rate"):
            profiles_text += f"  - Churn rate: {profile['churn_rate']:.1%}\n"
        for feat, val in list(profile.get("features", {}).items())[:4]:
            profiles_text += f"  - {feat.replace('_', ' ')}: {val}\n"

    prompt = f"""You are a behavioral data scientist presenting findings to a product team.
Below are behavioral user segments discovered through machine learning clustering.
Write a 2-paragraph executive summary of what these segments reveal about the user base.
Be specific, insightful, and actionable. No bullet points or headers.

SEGMENTS:
{profiles_text}

Write an executive summary of these behavioral archetypes and what they mean for the business.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print("Testing Groq persona layer...\n")

    report = generate_persona_report(
        archetype="The Cart Abandoner",
        churn_probability=0.87,
        risk_level="critical",
        risk_factors=[
            {"factor": "Cart abandonment", "detail": "78% of cart sessions not purchased", "impact": "high"},
            {"factor": "Declining activity", "detail": "Session frequency dropping over time", "impact": "medium"},
            {"factor": "High recency", "detail": "Last seen 22 days ago", "impact": "high"},
        ],
        recommendation="Trigger abandoned cart email within 2 hours with limited-time offer.",
        user_stats={
            "total_sessions": 14,
            "sessions_per_week": 1.2,
            "total_purchases": 1,
            "cart_abandonment_rate": 0.78,
            "avg_price_point": 89.50,
            "weekend_activity_ratio": 0.65,
            "recency_days": 22,
        },
    )

    print("PERSONA REPORT:")
    print("=" * 60)
    print(report)
    print("=" * 60)
