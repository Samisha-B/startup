import os
import pandas as pd

STAGES = ["Pre-seed", "Seed", "Series A", "Series B", "Series C+"]
STAGE_ROUND = {"Pre-seed": 1, "Seed": 1, "Series A": 2, "Series B": 3, "Series C+": 4}
BUSINESS_MODELS = ["B2B", "B2C", "B2B2C", "Marketplace", "SaaS", "D2C", "Other"]
EMPLOYEE_BANDS = ["1-10", "11-50", "51-200", "201-500", "501-1000", "1000+"]
METRICS = ["Funding Amount in $", "No. of Investors", "Funding Round", "Age (years)"]


def sector_options(df):
    found = set()
    for value in df["Industries"].dropna():
        for part in str(value).split(","):
            if part.strip():
                found.add(part.strip())
    return sorted(found)


def find_peers(df, sector):
    mask = df["Industries"].fillna("").str.contains(sector, case=False, regex=False)
    peers = df.loc[mask].copy()
    peers["Age (years)"] = pd.Timestamp.now().year - peers["Starting Year"]
    return peers


def startup_metrics(profile):
    return {
        "Funding Amount in $": float(profile["funding"]),
        "No. of Investors": float(profile["investors"]),
        "Funding Round": float(STAGE_ROUND[profile["stage"]]),
        "Age (years)": float(pd.Timestamp.now().year - profile["year"]),
    }


def compare_to_peers(profile, peers):
    mine = startup_metrics(profile)
    rows = []
    for metric in METRICS:
        series = peers[metric].dropna() if metric in peers else pd.Series(dtype=float)
        if series.empty:
            rows.append({"Metric": metric, "This startup": mine[metric], "Peer median": None, "Percentile": None})
            continue
        rows.append({
            "Metric": metric,
            "This startup": mine[metric],
            "Peer median": float(series.median()),
            "Percentile": round(float((series < mine[metric]).mean() * 100 + (series == mine[metric]).mean() * 50), 1),
        })
    return pd.DataFrame(rows)


def score_startup(profile, comparison):
    score, reasons, risks = 0, [], []
    pct = comparison.set_index("Metric")["Percentile"]
    for metric, weight, label in [("Funding Amount in $", 20, "funding raised"), ("No. of Investors", 10, "investor count")]:
        value = pct.get(metric)
        if value is None or pd.isna(value):
            risks.append(f"No peer data to benchmark {label}.")
            continue
        score += weight * value / 100
        (reasons if value >= 50 else risks).append(f"{label.capitalize()} is at the {value:.0f}th percentile of peers.")
    traction = {"No revenue": 0, "Early revenue": 8, "Growing revenue": 15, "Profitable": 20}[profile["revenue_status"]]
    score += traction
    (reasons if traction >= 15 else risks).append(f"Revenue status: {profile['revenue_status']}.")
    team = min(profile["founder_experience"], 10) * 1.5 + (5 if profile["team_size"] >= 10 else 0)
    score += team
    (reasons if team >= 10 else risks).append(f"Founder experience {profile['founder_experience']} yrs, team size {profile['team_size']}.")
    market = {"Small": 3, "Medium": 8, "Large": 12, "Very large": 15}[profile["market_size"]]
    score += market
    (reasons if market >= 12 else risks).append(f"Self-reported market size: {profile['market_size']} (unverified).")
    runway = min(profile["runway_months"], 24) / 24 * 10
    score += runway
    (reasons if profile["runway_months"] >= 12 else risks).append(f"Runway of {profile['runway_months']} months.")
    if profile["has_competitive_moat"]:
        score += 5
        reasons.append("Founders claim a defensible moat (IP, network effects, regulation or similar).")
    else:
        risks.append("No defensible moat claimed.")
    score = round(min(score, 100))
    verdict = "Invest" if score >= 70 else "Consider with due diligence" if score >= 50 else "Do not invest yet"
    return score, verdict, reasons, risks


def profile_text(profile):
    return "\n".join(f"{k}: {v}" for k, v in profile.items())


def ai_summary(profile, comparison, score, verdict, reasons, risks, peer_count):
    fallback = (
        f"Rule-based verdict: {verdict} (score {score}/100, {peer_count} peers).\n\n"
        "Strengths:\n" + "\n".join(f"- {r}" for r in reasons) +
        "\n\nRisks:\n" + "\n".join(f"- {r}" for r in risks)
    )
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return "AI summary disabled (OPENAI_API_KEY not set). " + fallback
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=30, max_retries=1)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            messages=[
                {"role": "system", "content": (
                    "You are an analyst helping an investor screen a startup. Use only the founder-submitted profile, "
                    "the peer comparison table and the rule-based score provided. Founder data is self-reported and unverified. "
                    "Give a clear INVEST / CONSIDER WITH DUE DILIGENCE / DO NOT INVEST YET recommendation, 3 strengths, 3 risks, "
                    "and 3 diligence questions. Do not invent numbers. State that this is decision support, not financial advice."
                )},
                {"role": "user", "content": (
                    f"PROFILE:\n{profile_text(profile)}\n\nPEER COMPARISON ({peer_count} peers):\n{comparison.to_string(index=False)}\n\n"
                    f"RULE-BASED SCORE: {score}/100 -> {verdict}"
                )},
            ],
        )
        return response.choices[0].message.content or fallback
    except Exception:
        return "AI summary unavailable (check API key, quota, network). " + fallback
