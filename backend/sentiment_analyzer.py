"""
Keyword-based Sentiment Analyzer
---------------------------------
Free, offline sentiment scoring for financial news headlines.
No API calls, no cost - pure keyword matching.
"""

POSITIVE_WORDS = [
    "surge", "surges", "rally", "rallies", "gain", "gains", "gained",
    "rise", "rises", "rising", "jump", "jumps", "jumped", "soar", "soars",
    "profit", "profits", "growth", "grow", "grows", "beat", "beats",
    "upgrade", "upgraded", "outperform", "record high", "boost", "boosts",
    "strong", "positive", "bullish", "expansion", "buy", "recommend",
    "approval", "approved", "partnership", "deal", "acquire", "acquisition"
]

NEGATIVE_WORDS = [
    "fall", "falls", "fell", "falling", "drop", "drops", "dropped",
    "decline", "declines", "declined", "plunge", "plunges", "plunged",
    "crash", "crashes", "loss", "losses", "lose", "loses",
    "downgrade", "downgraded", "underperform", "record low", "cut", "cuts",
    "weak", "negative", "bearish", "contraction", "sell", "sued", "lawsuit",
    "fraud", "investigation", "probe", "penalty", "fine", "scam",
    "resign", "resigns", "resigned", "layoff", "layoffs", "bankruptcy"
]


def analyze_sentiment(text):
    if not text:
        return {"label": "neutral", "score": 0, "positive_hits": [], "negative_hits": []}

    text_lower = text.lower()

    positive_hits = [word for word in POSITIVE_WORDS if word in text_lower]
    negative_hits = [word for word in NEGATIVE_WORDS if word in text_lower]

    score = len(positive_hits) - len(negative_hits)

    if score > 0:
        label = "positive"
    elif score < 0:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "score": score,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits
    }


if __name__ == "__main__":
    test_headlines = [
        "Reliance shares surge after strong quarterly profit beat",
        "TCS shares fall after weak guidance and analyst downgrade",
        "NSE files draft papers with Sebi",
        "Company under investigation for fraud, shares plunge"
    ]

    for headline in test_headlines:
        result = analyze_sentiment(headline)
        print(f"\nHeadline: {headline}")
        print(f"  Label: {result['label']} (score: {result['score']})")
        print(f"  Positive hits: {result['positive_hits']}")
        print(f"  Negative hits: {result['negative_hits']}")
