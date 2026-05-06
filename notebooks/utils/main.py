import pandas as pd
from pathlib import Path



from recommendation_system import (
    generate_recommendations,
    display_recommendations
)


def main():
    
    BASE_DIR = Path(__file__).resolve().parents[2]
    data_path = BASE_DIR / "data" / "sample_synthetic_posts.csv"
    df = pd.read_csv(data_path)

    positive_rules = pd.read_pickle(
        BASE_DIR
        / "data"
        / "positive_rules.pkl"
)

    negative_rules = pd.read_pickle(
        BASE_DIR
        / "data"
        / "negative_rules.pkl"
)

    user_post = {

    "language": "Arabic",

    "post_type": "image",

    "posting_hour": 10,

    "hashtags_count": 3,

    "caption_length": 70,

    "discount_percent": 0,

    "arabic_dialect_style": True,

    "CTA_present": True,

    "day_of_week": "Saturday"
}

    recommendations = generate_recommendations(
        user_post=user_post,
        positive_rules=positive_rules,
        negative_rules=negative_rules
    )

    display_recommendations(
        recommendations,
        top_n=10
    )


if __name__ == "__main__":
    main()