from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# =========================
# User-tunable parameters
# =========================
N_POSTS = 1000
RANDOM_SEED = 42

DATE_START = date(2025, 1, 1)
DATE_END = date(2025, 12, 31)

POST_TYPE_WEIGHTS = {
    "image": 0.35,
    "reel": 0.35,
    "carousel": 0.20,
    "video": 0.10,
}

LANGUAGE_WEIGHTS = {
    "Arabic": 0.55,
    "Mixed": 0.35,
    "English": 0.10,
}

SECTORS = [
    "restaurant",
    "cafe",
    "clinic",
    "store",
    "gym",
    "beauty_salon",
    "bakery",
    "pharmacy",
    "education_center",
    "electronics_store",
]

CITIES = [
    "Ramallah",
    "Nablus",
    "Hebron",
    "Bethlehem",
    "Jenin",
    "Tulkarm",
    "Qalqilya",
    "Jericho",
    "Gaza",
    "Rafah",
]

CTA_PHRASES_AR = ["اطلب الآن", "احجز الآن", "ابعتولنا DM", "راسلونا", "زورونا", "اتصل فينا"]
CTA_PHRASES_EN = ["DM us", "Book now", "Order now", "Message us", "Visit us"]
CTA_PHRASES_MIXED = ["DM us", "Order now", "ابعتولنا DM", "اطلب الآن", "زورونا"]
CTA_PHRASES_ALL = CTA_PHRASES_AR + CTA_PHRASES_EN + CTA_PHRASES_MIXED

PROMO_PHRASES_AR = ["خصم", "عرض خاص", "لفترة محدودة", "الكمية محدودة"]
PROMO_PHRASES_EN = ["sale", "limited offer", "special deal", "discount"]
PROMO_PHRASES_MIXED = ["خصم special", "limited offer لفترة محدودة", "sale اليوم"]

RELIGIOUS_PHRASES = ["رمضان كريم", "عروض رمضان", "عيد مبارك", "كل عام وأنتم بخير"]
PATRIOTIC_PHRASES = ["صنع محلي", "منتج فلسطيني", "ادعم المحلي", "من قلب فلسطين"]

EMOJIS = ["🔥", "😍", "✨", "☕", "🍕", "💪", "🛍️", "📍", "🎉", "✅", "🥙", "🍰", "📚", "🧴", "🩺"]

HASHTAGS_BASE = [
    "#Palestine",
    "#SupportLocal",
    "#مطاعم",
    "#عروض",
    "#تسوق",
    "#جيم",
    "#عيادة",
    "#قهوة",
    "#مخبز",
    "#صيدلية",
    "#تعليم",
    "#الكترونيات",
]


def _weighted_choice(weights: Dict[str, float]) -> str:
    keys = list(weights.keys())
    vals = list(weights.values())
    return random.choices(keys, weights=vals, k=1)[0]


def _clamp_int(x: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(float(x)))))


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return float(a) / float(b)


@dataclass(frozen=True)
class BusinessProfile:
    business_name: str
    sector: str
    city: str
    followers_base: int
    size_label: str  # small / medium / large


def create_business_profiles() -> List[BusinessProfile]:
    """
    Create at least 20 businesses across required sectors.

    Followers count is stable per business with small per-post noise later.
    """
    # Deterministic, readable fake brands. Mix Arabic-ish and English names.
    catalog: Dict[str, List[str]] = {
        "restaurant": ["Al Balad Grill", "Hebron Table", "Nablus Bites", "Gaza Seaside Kitchen"],
        "cafe": ["Qahwa Al-Quds", "Ramallah Roastery", "Nablus Coffee Corner", "Bethlehem Brew Bar"],
        "clinic": ["Shifa Family Clinic", "Al Amal Dental", "Jericho Physio Hub", "Nablus Skin Center"],
        "store": ["Canaan Threads", "Al Karmel Boutique", "Tulkarm Streetwear", "Qalqilya Home Store"],
        "gym": ["Strong Roots Gym", "Ramallah Fit Hub", "Iron Club Nablus", "Gaza Power House"],
        "beauty_salon": ["Sama Beauty Studio", "Laila Hair & Nails", "Gaza Glow Salon", "Bethlehem Lash Bar"],
        "bakery": ["Nablus Sweet Oven", "Hebron Fresh Bakery", "Saj & Manakish House", "Jericho Date Bakery"],
        "pharmacy": ["Al Hayat Pharmacy", "Ramallah Care Pharmacy", "Nablus Health Point", "Gaza Family Pharmacy"],
        "education_center": ["Olive Tree Learning", "Ramallah English Corner", "Nablus Coding Kids", "Hebron Study Hub"],
        "electronics_store": ["TechSouq Ramallah", "Nablus Mobile Center", "Gaza Gadget Shop", "Hebron Electronics"],
    }

    # City assignments (spread them out)
    city_cycle = [
        "Ramallah",
        "Nablus",
        "Hebron",
        "Bethlehem",
        "Jenin",
        "Tulkarm",
        "Qalqilya",
        "Jericho",
        "Gaza",
        "Rafah",
    ]

    profiles: List[BusinessProfile] = []
    city_idx = 0

    # Sector follower tendencies (visual retail tends to grow bigger pages)
    sector_size_bias = {
        "restaurant": ("medium", "large"),
        "cafe": ("small", "medium"),
        "clinic": ("small", "medium"),
        "store": ("medium", "large"),
        "gym": ("small", "medium"),
        "beauty_salon": ("small", "medium"),
        "bakery": ("small", "medium"),
        "pharmacy": ("small", "medium"),
        "education_center": ("small", "medium"),
        "electronics_store": ("medium", "large"),
    }

    def sample_followers(size_label: str) -> int:
        if size_label == "small":
            return random.randint(500, 3000)
        if size_label == "medium":
            return random.randint(3000, 15000)
        return random.randint(15000, 80000)

    for sector in SECTORS:
        names = catalog[sector]
        for name in names:
            city = city_cycle[city_idx % len(city_cycle)]
            city_idx += 1

            low, high = sector_size_bias[sector]
            # Encourage some large pages but not too many
            if random.random() < 0.18:
                size = "large"
            else:
                size = random.choice([low, high])
            followers = sample_followers(size)

            profiles.append(
                BusinessProfile(
                    business_name=name,
                    sector=sector,
                    city=city,
                    followers_base=followers,
                    size_label=size,
                )
            )

    # Ensure at least 20 businesses (we create 40 here: 10 sectors x 4 businesses)
    assert len(profiles) >= 20
    return profiles


def generate_random_date(start: date, end: date) -> date:
    days = (end - start).days
    return start + timedelta(days=random.randint(0, days))


def generate_posting_hour() -> int:
    """
    Most posts between 9 and 23, with peaks at 12-14 and 18-22.
    """
    # Three-mode distribution
    r = random.random()
    if r < 0.15:
        # off-hours (small probability)
        return random.randint(0, 8)
    if r < 0.45:
        # midday peak
        return random.randint(12, 14)
    if r < 0.85:
        # evening peak
        return random.randint(18, 22)
    return random.randint(9, 23)


def _sector_phrase(sector: str, language: str) -> str:
    if language == "English":
        mapping = {
            "restaurant": "fresh flavors today",
            "cafe": "coffee vibes all day",
            "clinic": "care you can trust",
            "store": "new arrivals just landed",
            "gym": "train stronger this week",
            "beauty_salon": "glow up season",
            "bakery": "fresh from the oven",
            "pharmacy": "your health matters",
            "education_center": "learn with confidence",
            "electronics_store": "latest gadgets in store",
        }
        return mapping.get(sector, "available now")
    # Arabic / Mixed (keep a Palestinian small-business tone)
    mapping_ar = {
        "restaurant": "اليوم عنا أكل طيب",
        "cafe": "أجواء ولا أروع",
        "clinic": "صحتكم أولويتنا",
        "store": "وصلت تشكيلة جديدة",
        "gym": "جاهزين للتمرين",
        "beauty_salon": "دلعي حالك",
        "bakery": "طازة من الفرن",
        "pharmacy": "صحتك بتهمنا",
        "education_center": "استنونا بدورة جديدة",
        "electronics_store": "وصلت أجهزة جديدة",
    }
    return mapping_ar.get(sector, "جاهزين لطلباتكم")


def _pick_cta(language: str) -> str:
    if language == "Arabic":
        return random.choice(CTA_PHRASES_AR)
    if language == "English":
        return random.choice(CTA_PHRASES_EN)
    return random.choice(CTA_PHRASES_MIXED)


def _pick_promo_phrase(language: str) -> str:
    if language == "Arabic":
        return random.choice(PROMO_PHRASES_AR)
    if language == "English":
        return random.choice(PROMO_PHRASES_EN)
    return random.choice(PROMO_PHRASES_MIXED)


def _maybe_hashtags(sector: str, city: str, language: str, promo_post: bool) -> List[str]:
    # Start with a few semantic tags, then sprinkle a couple from base pool.
    tags = []
    if city in {"Ramallah", "Nablus", "Hebron", "Bethlehem"} and random.random() < 0.45:
        tags.append(f"#{city}")
    if promo_post and random.random() < 0.55:
        tags.append("#عروض")
    if sector in {"restaurant", "cafe"} and random.random() < 0.55:
        tags.append("#مطاعم" if sector == "restaurant" else "#قهوة")
    if sector == "gym" and random.random() < 0.45:
        tags.append("#جيم")
    if sector == "clinic" and random.random() < 0.40:
        tags.append("#عيادة")
    if random.random() < 0.25:
        tags.append("#Palestine")

    # Add 0-4 random extras
    extras = random.randint(0, 4)
    tags += random.sample(HASHTAGS_BASE, k=min(extras, len(HASHTAGS_BASE)))

    # Normalize and dedupe while preserving order
    out = []
    seen = set()
    for t in tags:
        t2 = t.strip()
        if not t2.startswith("#"):
            t2 = "#" + t2
        if t2 not in seen:
            out.append(t2)
            seen.add(t2)
    return out


def _maybe_emojis(sector: str, post_type: str) -> List[str]:
    # Emojis: more likely for restaurants/cafes/retail; also for reels.
    base_p = 0.30
    if sector in {"restaurant", "cafe", "store", "beauty_salon", "bakery"}:
        base_p += 0.20
    if post_type in {"reel", "video"}:
        base_p += 0.10
    if random.random() > base_p:
        return []
    k = random.randint(1, 4)
    return random.sample(EMOJIS, k=k)


def generate_caption(
    profile: BusinessProfile,
    post_type: str,
    language: str,
    promo_post: bool,
    discount_percent: int,
    CTA_present_target: bool,
    mentions_location_target: bool,
    religious_theme: bool,
    patriotic_theme: bool,
    arabic_dialect_style: bool,
) -> str:
    """
    Build caption text from templates.

    We treat CTA_present and mentions_location as *targets*; final booleans
    are derived from caption content later for strict consistency checks.
    """
    city = profile.city
    sector = profile.sector

    # Opening line
    opener = _sector_phrase(sector, language)

    # Dialect flavor (only when Arabic/Mixed)
    dialect_bit = ""
    if arabic_dialect_style and language in {"Arabic", "Mixed"} and random.random() < 0.75:
        dialect_bit = random.choice(["شو رأيكم؟", "يا جماعة", "أهلا بالناس الحلوة", "احنا جاهزين"])  # light, non-sensitive

    # Promo line
    promo_bit = ""
    if promo_post:
        promo_phrase = _pick_promo_phrase(language)
        if language == "English":
            promo_bit = f"{promo_phrase}: {discount_percent}% off."
        elif language == "Mixed":
            promo_bit = f"{promo_phrase} {discount_percent}% off اليوم!"
        else:
            promo_bit = f"{promo_phrase} {discount_percent}% خصم لفترة محدودة"

    # Location line
    loc_bit = ""
    if mentions_location_target:
        if language == "English":
            loc_bit = f"Available now in {city}."
        elif language == "Mixed":
            loc_bit = f"موجودين في {city} today."
        else:
            loc_bit = f"زورونا بفرعنا في {city}"

    # Themes
    relig_bit = random.choice(RELIGIOUS_PHRASES) if religious_theme else ""
    patri_bit = random.choice(PATRIOTIC_PHRASES) if patriotic_theme else ""

    # CTA line
    cta_bit = ""
    if CTA_present_target:
        cta = _pick_cta(language)
        if language == "English":
            cta_bit = f"{cta}."
        elif language == "Mixed":
            cta_bit = f"{cta} وخلينا نجهزلك طلبك."
        else:
            cta_bit = f"{cta} وخليها علينا"

    # Post-type hints
    if language == "English":
        format_bit = {"image": "Photo drop.", "carousel": "Swipe to see more.", "reel": "Watch the reel.", "video": "New video up."}[post_type]
    elif language == "Mixed":
        format_bit = {"image": "صورة جديدة.", "carousel": "Swipe وشوفوا الباقي.", "reel": "شوفوا الـ reel.", "video": "فيديو جديد."}[post_type]
    else:
        format_bit = {"image": "صورة جديدة.", "carousel": "سوايب وشوفوا التفاصيل.", "reel": "شوفوا الريل.", "video": "فيديو جديد."}[post_type]

    # Compose lines with natural variability
    lines = []
    if language == "English":
        lines.append(f"{opener.capitalize()} {format_bit}")
    elif language == "Mixed":
        lines.append(f"{opener} {format_bit}")
    else:
        lines.append(f"{opener} {format_bit}")

    for bit in [dialect_bit, promo_bit, loc_bit, relig_bit, patri_bit, cta_bit]:
        bit2 = str(bit).strip()
        if bit2:
            lines.append(bit2)

    # Add hashtags and emojis occasionally
    tags = _maybe_hashtags(sector=sector, city=city, language=language, promo_post=promo_post)
    emojis = _maybe_emojis(sector=sector, post_type=post_type)

    caption = " ".join(lines).strip()
    if emojis:
        caption = f"{caption} {' '.join(emojis)}"
    if tags and random.random() < 0.75:
        if len(tags) == 1:
            caption = f"{caption} {tags[0]}"
        else:
            k = random.randint(2, min(len(tags), 8))
            caption = f"{caption} {' '.join(tags[:k])}"

    # Keep it reasonably sized and variable
    if len(caption) > 320 and random.random() < 0.7:
        caption = caption[:320].rsplit(" ", 1)[0]
    return caption


def count_emojis(text: str) -> int:
    # Count occurrences of emojis from our controlled pool.
    return sum(text.count(e) for e in EMOJIS)


def _extract_hashtags(text: str) -> List[str]:
    # Works for Arabic/English hashtags and underscores.
    return re.findall(r"(?:^|\s)(#\w+)", text)


def _infer_CTA_present(text: str) -> bool:
    t = text.lower()
    for p in CTA_PHRASES_ALL:
        if p.lower() in t:
            return True
    return False


def _infer_mentions_location(text: str, city: str) -> bool:
    return city.lower() in text.lower()


def _choose_discount(promo_post: bool) -> int:
    if not promo_post:
        return 0
    return random.choice([5, 10, 15, 20, 25, 30, 40, 50])


def _choose_themes(post_dt: date) -> Tuple[bool, bool]:
    """
    religious_theme: usually False; higher chance in March/April
    patriotic_theme: small chance all year
    """
    m = int(post_dt.month)
    relig_p = 0.04
    if m in (3, 4):
        relig_p = 0.18
    religious = random.random() < relig_p

    patriotic = random.random() < 0.08
    return religious, patriotic


def _choose_arabic_dialect(language: str) -> bool:
    if language == "English":
        return random.random() < 0.03
    # Arabic / Mixed
    return random.random() < 0.70


def _sector_engagement_bias(sector: str) -> Tuple[float, float]:
    """
    Return (like_rate_multiplier, comment_rate_multiplier)
    """
    if sector in {"restaurant", "cafe", "store", "beauty_salon", "bakery"}:
        return 1.15, 1.10
    if sector in {"electronics_store"}:
        return 1.05, 1.05
    if sector in {"gym", "education_center"}:
        return 1.00, 1.10
    # clinic/pharmacy: more informational, often lower like-rate
    return 0.85, 0.95


def _post_type_view_range(post_type: str) -> Tuple[float, float]:
    if post_type == "image":
        return 0.08, 0.80
    if post_type == "carousel":
        return 0.15, 1.10
    if post_type == "video":
        return 0.35, 1.80
    # reel
    return 0.50, 3.00


def _boost_factor(
    posting_hour: int,
    promo_post: bool,
    CTA_present: bool,
    hashtags_count: int,
    emoji_count: int,
    language: str,
    religious_theme: bool,
    patriotic_theme: bool,
    arabic_dialect_style: bool,
) -> float:
    boost = 1.0

    # Time of day boosts
    if 18 <= posting_hour <= 22:
        boost *= 1.0 + random.uniform(0.15, 0.25)
    elif 12 <= posting_hour <= 14:
        boost *= 1.0 + random.uniform(0.08, 0.15)
    elif 0 <= posting_hour <= 6:
        boost *= random.uniform(0.85, 0.95)

    if promo_post:
        boost *= 1.0 + random.uniform(0.05, 0.15)
    if CTA_present:
        boost *= 1.0 + random.uniform(0.03, 0.08)

    # Hashtags/emoji mild effects
    if 4 <= hashtags_count <= 10:
        boost *= 1.0 + random.uniform(0.02, 0.06)
    if hashtags_count > 12:
        boost *= 1.0 - random.uniform(0.01, 0.04)
    if emoji_count >= 2:
        boost *= 1.0 + random.uniform(0.01, 0.03)

    if language == "English":
        boost *= random.uniform(0.92, 0.99)
    elif language == "Mixed":
        boost *= random.uniform(0.99, 1.05)
    else:
        boost *= random.uniform(1.00, 1.06)

    # Themes
    if religious_theme:
        boost *= 1.0 + random.uniform(0.02, 0.06)
    if patriotic_theme:
        boost *= 1.0 + random.uniform(0.02, 0.05)
    if arabic_dialect_style:
        boost *= 1.0 + random.uniform(0.02, 0.05)

    return boost


def _generate_engagement(
    followers_count: int,
    post_type: str,
    sector: str,
    posting_hour: int,
    promo_post: bool,
    CTA_present: bool,
    hashtags_count: int,
    emoji_count: int,
    language: str,
    religious_theme: bool,
    patriotic_theme: bool,
    arabic_dialect_style: bool,
) -> Tuple[int, int, int]:
    """
    Generate views, likes, comments with logical dependencies:
    - views depends on followers, post type, hour, and other boosts
    - likes depends on views
    - comments depends on likes (and CTA/promo/post_type)
    """
    lo, hi = _post_type_view_range(post_type)
    base_view_mult = random.uniform(lo, hi)

    boost = _boost_factor(
        posting_hour=posting_hour,
        promo_post=promo_post,
        CTA_present=CTA_present,
        hashtags_count=hashtags_count,
        emoji_count=emoji_count,
        language=language,
        religious_theme=religious_theme,
        patriotic_theme=patriotic_theme,
        arabic_dialect_style=arabic_dialect_style,
    )

    # Sector-level view bias (visual sectors do better)
    sector_view_bias = {
        "restaurant": 1.12,
        "cafe": 1.10,
        "store": 1.08,
        "beauty_salon": 1.10,
        "bakery": 1.06,
        "electronics_store": 1.02,
        "gym": 1.00,
        "education_center": 0.98,
        "clinic": 0.92,
        "pharmacy": 0.90,
    }.get(sector, 1.0)

    # Random multiplicative noise (avoid deterministic looking)
    noise = float(np.random.lognormal(mean=0.0, sigma=0.22))
    views = followers_count * base_view_mult * boost * sector_view_bias * noise
    views_count = max(1, int(round(views)))

    like_mult, comment_mult = _sector_engagement_bias(sector)

    # Likes are a fraction of views
    like_rate = random.uniform(0.03, 0.15)
    like_rate *= like_mult
    # Promo can slightly increase likes
    if promo_post:
        like_rate *= random.uniform(1.03, 1.10)
    # Clinic/pharmacy are typically lower like ratio
    if sector in {"clinic", "pharmacy"}:
        like_rate *= random.uniform(0.85, 0.95)
    likes_count = int(max(0, round(views_count * like_rate)))

    # Comments are a fraction of likes
    comment_rate = random.uniform(0.02, 0.12)
    comment_rate *= comment_mult
    if CTA_present:
        comment_rate *= random.uniform(1.10, 1.35)
    if promo_post:
        comment_rate *= random.uniform(1.05, 1.20)
    if post_type == "carousel":
        comment_rate *= random.uniform(1.08, 1.25)

    comments_count = int(max(0, round(likes_count * comment_rate)))

    # Enforce ordering and plausible minimums
    comments_count = min(comments_count, likes_count)
    likes_count = min(likes_count, views_count)
    if likes_count < comments_count:
        likes_count = comments_count
    if views_count < likes_count:
        views_count = likes_count

    return int(views_count), int(likes_count), int(comments_count)


def generate_single_post(profile: BusinessProfile) -> Dict:
    post_dt = generate_random_date(DATE_START, DATE_END)
    post_type = _weighted_choice(POST_TYPE_WEIGHTS)
    language = _weighted_choice(LANGUAGE_WEIGHTS)

    promo_post = random.random() < 0.30
    discount_percent = _choose_discount(promo_post)

    # CTA target around 55-65%, slight sector bias
    base_cta = 0.60
    if profile.sector in {"clinic", "education_center"}:
        base_cta = 0.63
    if profile.sector in {"pharmacy"}:
        base_cta = 0.55
    CTA_present_target = random.random() < base_cta

    # Location mention: moderate chance, higher for stores/services
    base_loc = 0.38 + (0.08 if profile.sector in {"store", "electronics_store"} else 0.0)
    mentions_location_target = random.random() < min(max(base_loc, 0.15), 0.65)

    religious_theme, patriotic_theme = _choose_themes(post_dt)

    arabic_dialect_style = _choose_arabic_dialect(language)
    if language == "English":
        # Almost always false
        arabic_dialect_style = arabic_dialect_style and (random.random() < 0.2)

    posting_hour = generate_posting_hour()

    # Followers: stable per business with small noise (no big jumps)
    followers_noise = np.random.normal(loc=0.0, scale=0.02)  # 2% std
    followers_count = int(max(0, round(profile.followers_base * (1.0 + float(followers_noise)))))
    followers_count = max(50, followers_count)

    caption_text = generate_caption(
        profile=profile,
        post_type=post_type,
        language=language,
        promo_post=promo_post,
        discount_percent=discount_percent,
        CTA_present_target=CTA_present_target,
        mentions_location_target=mentions_location_target,
        religious_theme=religious_theme,
        patriotic_theme=patriotic_theme,
        arabic_dialect_style=arabic_dialect_style,
    )

    # Derive certain columns from caption to keep strict consistency.
    hashtags = _extract_hashtags(caption_text)
    hashtags_count = len(hashtags)
    emoji_count = count_emojis(caption_text)
    CTA_present = _infer_CTA_present(caption_text)
    mentions_location = _infer_mentions_location(caption_text, profile.city)

    # Ensure promo consistency (discount must be 0 if not promo)
    if not promo_post:
        discount_percent = 0

    views_count, likes_count, comments_count = _generate_engagement(
        followers_count=followers_count,
        post_type=post_type,
        sector=profile.sector,
        posting_hour=posting_hour,
        promo_post=promo_post,
        CTA_present=CTA_present,
        hashtags_count=hashtags_count,
        emoji_count=emoji_count,
        language=language,
        religious_theme=religious_theme,
        patriotic_theme=patriotic_theme,
        arabic_dialect_style=arabic_dialect_style,
    )

    # Derivations from post_date (must be exact)
    post_date_iso = post_dt.isoformat()
    day_of_week = post_dt.strftime("%A")
    month = int(post_dt.month)

    return {
        "business_name": profile.business_name,
        "sector": profile.sector,
        "followers_count": int(followers_count),
        "post_date": post_date_iso,
        "posting_hour": int(posting_hour),
        "day_of_week": day_of_week,
        "month": int(month),
        "post_type": post_type,
        "caption_text": caption_text,
        "caption_length": int(len(caption_text)),
        "hashtags_count": int(hashtags_count),
        "emoji_count": int(emoji_count),
        "likes_count": int(likes_count),
        "comments_count": int(comments_count),
        "views_count": int(views_count),
        "language": language,
        "CTA_present": bool(CTA_present),
        "promo_post": bool(promo_post),
        "discount_percent": float(discount_percent),
        "mentions_location": bool(mentions_location),
        "religious_theme": bool(religious_theme),
        "patriotic_theme": bool(patriotic_theme),
        "arabic_dialect_style": bool(arabic_dialect_style),
    }


def generate_dataset(n_posts: int = N_POSTS) -> pd.DataFrame:
    profiles = create_business_profiles()

    # Sample businesses with a slight preference for larger pages (more content volume)
    weights = []
    for p in profiles:
        if p.size_label == "large":
            weights.append(1.6)
        elif p.size_label == "medium":
            weights.append(1.2)
        else:
            weights.append(1.0)

    rows = []
    for _ in range(n_posts):
        prof = random.choices(profiles, weights=weights, k=1)[0]
        rows.append(generate_single_post(prof))

    df = pd.DataFrame(rows)

    # Enforce column order and "exact base columns"
    base_cols = [
        "business_name",
        "sector",
        "followers_count",
        "post_date",
        "posting_hour",
        "day_of_week",
        "month",
        "post_type",
        "caption_text",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "likes_count",
        "comments_count",
        "views_count",
        "language",
        "CTA_present",
        "promo_post",
        "discount_percent",
        "mentions_location",
        "religious_theme",
        "patriotic_theme",
        "arabic_dialect_style",
    ]
    df = df[base_cols].copy()
    return df


def add_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["engagement_count"] = out["likes_count"] + out["comments_count"]
    out["engagement_rate_by_followers"] = out.apply(lambda r: _safe_div(r["engagement_count"], r["followers_count"]), axis=1)
    out["like_rate_by_views"] = out.apply(lambda r: _safe_div(r["likes_count"], r["views_count"]), axis=1)
    out["comment_rate_by_likes"] = out.apply(lambda r: _safe_div(r["comments_count"], r["likes_count"]), axis=1)
    out["views_per_follower"] = out.apply(lambda r: _safe_div(r["views_count"], r["followers_count"]), axis=1)

    q75 = float(out["engagement_rate_by_followers"].quantile(0.75))
    out["is_high_engagement"] = out["engagement_rate_by_followers"] > q75

    def time_cat(h: int) -> str:
        if 5 <= h <= 11:
            return "morning"
        if 12 <= h <= 16:
            return "afternoon"
        if 17 <= h <= 21:
            return "evening"
        return "night"

    out["posting_time_category"] = out["posting_hour"].astype(int).apply(time_cat)

    def cap_cat(n: int) -> str:
        if n < 90:
            return "short"
        if n < 180:
            return "medium"
        return "long"

    out["caption_length_category"] = out["caption_length"].astype(int).apply(cap_cat)

    def hash_intensity(c: int) -> str:
        if c <= 0:
            return "none"
        if c <= 3:
            return "low"
        if c <= 8:
            return "medium"
        return "high"

    out["hashtag_intensity"] = out["hashtags_count"].astype(int).apply(hash_intensity)

    # Simple explainable score: localization cues add up
    out["content_localization_score"] = (
        out["mentions_location"].astype(int) * 3
        + out["arabic_dialect_style"].astype(int) * 3
        + out["patriotic_theme"].astype(int) * 2
        + out["religious_theme"].astype(int) * 2
    ).astype(int)

    return out


def validate_dataset(df: pd.DataFrame) -> None:
    required_cols = [
        "business_name",
        "sector",
        "followers_count",
        "post_date",
        "posting_hour",
        "day_of_week",
        "month",
        "post_type",
        "caption_text",
        "caption_length",
        "hashtags_count",
        "emoji_count",
        "likes_count",
        "comments_count",
        "views_count",
        "language",
        "CTA_present",
        "promo_post",
        "discount_percent",
        "mentions_location",
        "religious_theme",
        "patriotic_theme",
        "arabic_dialect_style",
    ]
    extra = sorted([c for c in df.columns if c not in required_cols])
    missing = sorted([c for c in required_cols if c not in df.columns])
    if extra or missing:
        raise ValueError(f"Columns mismatch. missing={missing} extra={extra}")

    if df.isna().any().any():
        bad = df.isna().sum()
        raise ValueError(f"Nulls found: {bad[bad>0].to_dict()}")

    # Discount rules
    bad_discount = df.loc[(df["promo_post"] == False) & (df["discount_percent"] != 0)]  # noqa: E712
    if len(bad_discount) > 0:
        raise ValueError("discount_percent must be 0 when promo_post is False.")

    # Derivations from post_date
    parsed = pd.to_datetime(df["post_date"], errors="raise").dt.date
    derived_dow = parsed.apply(lambda d: d.strftime("%A"))
    derived_month = parsed.apply(lambda d: d.month)
    if not (derived_dow.values == df["day_of_week"].values).all():
        raise ValueError("day_of_week does not match post_date.")
    if not (derived_month.values == df["month"].astype(int).values).all():
        raise ValueError("month does not match post_date.")

    # Engagement ordering
    if not (df["views_count"] >= df["likes_count"]).all():
        raise ValueError("views_count must be >= likes_count.")
    if not (df["likes_count"] >= df["comments_count"]).all():
        raise ValueError("likes_count must be >= comments_count.")

    # Hour and categorical validity
    if not df["posting_hour"].between(0, 23).all():
        raise ValueError("posting_hour out of range.")

    valid_post_types = set(POST_TYPE_WEIGHTS.keys())
    if not df["post_type"].isin(valid_post_types).all():
        raise ValueError("Invalid post_type values.")

    valid_langs = set(LANGUAGE_WEIGHTS.keys())
    if not df["language"].isin(valid_langs).all():
        raise ValueError("Invalid language values.")

    # Derived consistency checks from caption
    # - caption_length equals len(caption_text)
    if not (df["caption_length"].astype(int) == df["caption_text"].astype(str).apply(len)).all():
        raise ValueError("caption_length must be derived from caption_text.")

    # - hashtags_count matches hashtags in caption_text
    derived_hash = df["caption_text"].astype(str).apply(lambda t: len(_extract_hashtags(t)))
    if not (derived_hash.values == df["hashtags_count"].astype(int).values).all():
        raise ValueError("hashtags_count must match hashtags extracted from caption_text.")

    # - emoji_count matches emoji occurrences in caption_text (from our controlled emoji pool)
    derived_emoji = df["caption_text"].astype(str).apply(count_emojis)
    if not (derived_emoji.values == df["emoji_count"].astype(int).values).all():
        raise ValueError("emoji_count must match emojis found in caption_text.")

    # - CTA_present matches CTA phrase presence in caption_text
    derived_cta = df["caption_text"].astype(str).apply(_infer_CTA_present)
    if not (derived_cta.values == df["CTA_present"].astype(bool).values).all():
        raise ValueError("CTA_present must match CTA phrases found in caption_text.")

    # - mentions_location matches whether ANY known city appears in caption_text
    #   (We treat this as \"mentions any location\" since city is not a stored column.)
    city_pattern = "|".join(re.escape(c) for c in CITIES)
    derived_loc = df["caption_text"].astype(str).str.contains(city_pattern, case=False, regex=True)
    if not (derived_loc.values == df["mentions_location"].astype(bool).values).all():
        raise ValueError("mentions_location must match presence of city name in caption_text.")

    # - arabic_dialect_style should almost always be False for English (we enforce always False)
    bad_dialect = df.loc[(df["language"] == "English") & (df["arabic_dialect_style"] == True)]  # noqa: E712
    if len(bad_dialect) > 0:
        raise ValueError("arabic_dialect_style must be False when language is English.")

    # - discount percent only from allowed list when promo_post True
    allowed_discounts = {0, 5, 10, 15, 20, 25, 30, 40, 50}
    if not set(df["discount_percent"].astype(int).unique()).issubset(allowed_discounts):
        raise ValueError("discount_percent contains unexpected values.")


def save_outputs(
    df: pd.DataFrame,
    df_kpis: pd.DataFrame,
    out_dir: Path,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = out_dir / "synthetic_social_media_posts.csv"
    kpi_path = out_dir / "synthetic_social_media_posts_with_kpis.csv"
    df.to_csv(base_path, index=False, encoding="utf-8")
    df_kpis.to_csv(kpi_path, index=False, encoding="utf-8")
    return base_path, kpi_path


def print_summary(df: pd.DataFrame) -> None:
    print("Dataset shape:", df.shape)
    print("\nFirst 5 rows:\n", df.head(5).to_string(index=False))
    print("\nSector value counts:\n", df["sector"].value_counts().to_string())
    print("\nPost type value counts:\n", df["post_type"].value_counts().to_string())

    df2 = df.copy()
    df2["engagement"] = df2["likes_count"] + df2["comments_count"]
    print("\nAverage engagement by post_type:\n", df2.groupby("post_type")["engagement"].mean().round(2).to_string())
    print("\nAverage engagement by sector:\n", df2.groupby("sector")["engagement"].mean().round(2).sort_values(ascending=False).to_string())


def main() -> None:
    # Avoid Windows console encoding crashes when captions include Arabic/emoji.
    # This does not change the CSV encoding (we always write UTF-8).
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    out_dir = Path(__file__).resolve().parent

    df = generate_dataset(n_posts=N_POSTS)
    validate_dataset(df)

    df_kpis = add_kpis(df)

    base_path, kpi_path = save_outputs(df=df, df_kpis=df_kpis, out_dir=out_dir)

    print("Wrote:", str(base_path))
    print("Wrote:", str(kpi_path))
    print_summary(df)


if __name__ == "__main__":
    main()
