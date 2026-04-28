# Synthetic Social Media Post Generator (Palestinian / Local SMEs)

This folder contains a complete, runnable Python synthetic data generator for a social media engagement mining project focused on Palestinian/local small businesses.

The generator creates **one row per post** with **logically connected** columns (captions, hashtags, emojis, CTA, promo/discounts, dates, and engagement metrics are all consistent and correlated).

## Files
- `generate_synthetic_social_media_data.py`: main script
- `synthetic_social_media_posts.csv`: base output dataset (raw base columns)
- `synthetic_social_media_posts_with_kpis.csv`: base dataset + derived KPIs
- `data_dictionary.csv`: documentation for each column

## How to run
From this folder:
```powershell
python .\generate_synthetic_social_media_data.py
```

To change dataset size, edit:
```python
N_POSTS = 1000
```

## Base raw dataset columns (CSV 1)
The output `synthetic_social_media_posts.csv` contains **exactly** these base columns:

`business_name, sector, followers_count, post_date, posting_hour, day_of_week, month, post_type, caption_text, caption_length, hashtags_count, emoji_count, likes_count, comments_count, views_count, language, CTA_present, promo_post, discount_percent, mentions_location, religious_theme, patriotic_theme, arabic_dialect_style`

Even though some are *derived* from other fields (e.g. `day_of_week` from `post_date`), they are included as base “raw columns” because they resemble typical scraped/exported datasets.

## Derived KPI dataset (CSV 2)
The output `synthetic_social_media_posts_with_kpis.csv` includes all base columns plus:
- `engagement_count = likes_count + comments_count`
- `engagement_rate_by_followers = engagement_count / followers_count`
- `like_rate_by_views = likes_count / views_count`
- `comment_rate_by_likes = comments_count / likes_count`
- `views_per_follower = views_count / followers_count`
- `is_high_engagement` (above the 75th percentile of engagement rate)
- `posting_time_category` (morning/afternoon/evening/night)
- `caption_length_category` (short/medium/long)
- `hashtag_intensity` (none/low/medium/high)
- `content_localization_score` (0–10 score from: location mention, dialect, patriotic, religious cues)

These KPI columns are **not** raw scraped fields; they are derived for analytics, clustering, and dashboards.

## Realism rules implemented
The generator intentionally avoids “independent random columns”. Key realism links:
- `day_of_week` and `month` are derived from `post_date`.
- `caption_length` is derived from `caption_text`.
- `discount_percent` is **0 when** `promo_post=False`.
- `hashtags_count` is derived from hashtags in `caption_text`.
- `emoji_count` is derived from emojis in `caption_text`.
- `CTA_present` is derived from CTA phrases in `caption_text`.
- `mentions_location` is derived from city mentions in `caption_text`.
- `arabic_dialect_style` is usually true only for Arabic/Mixed posts.
- `views_count` depends on followers, post type, time-of-day, promo/CTA/themes, and controlled noise.
- `likes_count` is generated as a fraction of views (sector-adjusted).
- `comments_count` is generated as a fraction of likes (boosted by CTA/promo; carousels slightly higher).

## How to use for data mining
This dataset is suitable for:
- KPI engineering and dashboarding
- Clustering posts and businesses (KMeans)
- PCA / dimensionality reduction on encoded features
- Association rules on transaction-like items (sector, post_type, CTA, promo, dialect, themes, time bucket)
- Trend analysis by week/month
- Anomaly detection (viral spikes vs weak big-account posts)

The data includes real signals (sector and post type differences, CTA/promo effects, time-of-day effects), plus noise to avoid looking deterministic.

