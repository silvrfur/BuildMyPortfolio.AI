"""
Keyword-based NLP signals.
Used for signals where transformer models are unnecessary or inefficient.
"""

#!!! needs a better way to know the keywords
KEYWORD_SIGNALS = {

    # Urgency / Immediacy
    "urgency_score": {
        "keywords": [
            "urgent", "immediately", "asap", "right now",
            "quick", "quickly", "fast", "without delay"
        ],
        "description": "Indicates perceived need for immediate action"
    },

    # Time Horizon Orientation
    "time_horizon_bias": {
        "short_term": [
            "now", "today", "this week", "short term",
            "quick return", "immediate profit"
        ],
        "long_term": [
            "long term", "years", "hold", "patient",
            "compounding", "future growth"
        ],
        "description": "Indicates short-term vs long-term investment orientation"
    },

    # Risk Salience (Density)
    "risk_language_density": {
        "keywords": [
            "risk", "safe", "safety", "protect",
            "hedge", "loss", "downside", "secure"
        ],
        "description": "Frequency of risk-related language"
    },

    
    # #already covering this with transformer model, but keeping here for reference
    # #Herding / Social Proof
    # "herding_marker": {
    #     "keywords": [
    #         "everyone is", "most people", "trending",
    #         "fomo", "my friend", "people are buying",
    #         "social media says", "popular stock"
    #     ],
    #     "description": "Indicates social imitation or crowd-following behavior"
    # },
    
    
    # Internal Locus of Control
    "internal_locus_score": {
        "keywords": [
            "my strategy", "i decided", "i analyzed",
            "i believe", "i can manage", "my plan",
            "based on my analysis"
        ],
        "description": "Indicates personal agency and self-attribution"
    },

    # External Locus of Control
    "external_locus_score": {
        "keywords": [
            "market decides", "luck", "fate",
            "nothing i can do", "depends on market",
            "out of my control", "unpredictable market"
        ],
        "description": "Indicates attribution to external forces"
    },

    # Uncertainty 
    "uncertainty_score": {
        "keywords": [
            "not sure", "confused", "uncertain",
            "maybe", "i don't know", "unsure",
            "hard to decide"
        ],
        "description": "Indicates doubt, ambiguity, or lack of confidence"
    }
}
