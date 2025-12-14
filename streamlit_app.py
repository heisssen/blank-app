#!/usr/bin/env python3
"""
TG News Bot (Strict + Official-First + Better Dedup/Clustering) v3.5

ENV (GitHub Environments / Secrets):
- GOOGLE_API_KEY (secret)          required
- TG_BOT_TOKEN (secret)            required
- TG_CHAT_ID (var/secret)          required  e.g. -1001234567890
- DIGEST_CHANNEL (var)             optional  e.g. @oximets_digest (default)

Optional tuning:
- REQUIRE_OFFICIAL_FOR_ATTACKS=1   (default 1)
- CASUALTY_DIGEST_THRESHOLD=4      (default 4)
- POST_WATCHLIST=1                 (default 1)
- MAX_AGE_SECONDS=3800
- NEWS_TO_PUBLISH=10
- WATCH_TO_PUBLISH=5
- AI_CONTEXT_LIMIT=200
- DEEP_READ_LIMIT=25
- STATE_DIR=.d7_state
- SOURCES_FILE=./news_sources.json
- SOURCES_JSON='{"rss_ua":[...], ...}'   (optional override)
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import re
import html
import logging
import hashlib
import traceback
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from dateutil import parser as date_parser
from urllib.parse import urlparse, urlunparse
from zoneinfo import ZoneInfo

import requests
import numpy as np
import feedparser
import trafilatura
import google.generativeai as genai
from bs4 import BeautifulSoup
from difflib import SequenceMatcher

# --- ML (optional but recommended) ---
try:
    from sklearn.metrics.pairwise import cosine_distances  # noqa: F401
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

# =========================
# ENV CONFIG
# =========================
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "").strip()
TG_BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "").strip()

TG_CHAT_ID = os.environ.get("TG_CHAT_ID", "").strip()
DIGEST_CHANNEL = os.environ.get("DIGEST_CHANNEL", "@oximets_digest").strip()

TARGET_CHATS: List[str] = []
if TG_CHAT_ID:
    TARGET_CHATS.append(TG_CHAT_ID)
if DIGEST_CHANNEL and DIGEST_CHANNEL not in TARGET_CHATS:
    TARGET_CHATS.append(DIGEST_CHANNEL)

TG_ADMIN_ID = os.environ.get("TG_ADMIN_ID", "").strip()

GEN_MODEL_NAME = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash").strip()
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "models/text-embedding-004").strip()

REQUIRE_OFFICIAL_FOR_ATTACKS = os.environ.get("REQUIRE_OFFICIAL_FOR_ATTACKS", "1").strip() == "1"
CASUALTY_DIGEST_THRESHOLD = int(os.environ.get("CASUALTY_DIGEST_THRESHOLD", "4").strip() or "4")
POST_WATCHLIST = os.environ.get("POST_WATCHLIST", "1").strip() == "1"

MAX_AGE_SECONDS = int(os.environ.get("MAX_AGE_SECONDS", "3800").strip() or "3800")
AI_CONTEXT_LIMIT = int(os.environ.get("AI_CONTEXT_LIMIT", "200").strip() or "200")
DEEP_READ_LIMIT = int(os.environ.get("DEEP_READ_LIMIT", "25").strip() or "25")

NEWS_TO_PUBLISH = int(os.environ.get("NEWS_TO_PUBLISH", "10").strip() or "10")
WATCH_TO_PUBLISH = int(os.environ.get("WATCH_TO_PUBLISH", "5").strip() or "5")

CLUSTER_TIME_WINDOW_HOURS = float(os.environ.get("CLUSTER_TIME_WINDOW_HOURS", "8").strip() or "8")
SEM_SIM_THRESHOLD = float(os.environ.get("SEM_SIM_THRESHOLD", "0.86").strip() or "0.86")
SEM_SIM_SECONDARY = float(os.environ.get("SEM_SIM_SECONDARY", "0.80").strip() or "0.80")
LEX_SIM_THRESHOLD = float(os.environ.get("LEX_SIM_THRESHOLD", "0.72").strip() or "0.72")
LEX_SIM_OFFICIAL_THRESHOLD = float(os.environ.get("LEX_SIM_OFFICIAL_THRESHOLD", "0.68").strip() or "0.68")

HISTORY_RETENTION_DAYS = int(os.environ.get("HISTORY_RETENTION_DAYS", "3").strip() or "3")
HISTORY_TITLE_SIM_THRESHOLD = float(os.environ.get("HISTORY_TITLE_SIM_THRESHOLD", "0.80").strip() or "0.80")
HISTORY_SIMHASH_STRICT = int(os.environ.get("HISTORY_SIMHASH_STRICT", "3").strip() or "3")
HISTORY_SIMHASH_LOOSE = int(os.environ.get("HISTORY_SIMHASH_LOOSE", "6").strip() or "6")

STATE_DIR = os.environ.get("STATE_DIR", ".d7_state").strip() or ".d7_state"
os.makedirs(STATE_DIR, exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCES_FILE = os.environ.get("SOURCES_FILE", os.path.join(BASE_DIR, "news_sources.json")).strip()
SOURCES_JSON = os.environ.get("SOURCES_JSON", "").strip()

HISTORY_FILE = os.path.join(STATE_DIR, "published_news_history.json")
CACHE_FILE = os.path.join(STATE_DIR, "cache_db.json")

KYIV_TZ = ZoneInfo("Europe/Kyiv")

# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M"
)
log = logging.getLogger("tg_news_bot")

# =========================
# STOPWORDS
# =========================
STOPWORDS = {
    "і","й","та","але","аби","щоб","який","яка","яке","які","це","ці","цей","ця","на","у","в","до","з","із","за","про",
    "не","так","як","що","вже","ще","від","по","для","при","під","між","над","після","перед","зараз","сьогодні","вчора",
    "the","a","an","and","or","to","of","in","on","for","with","from","by","as","at","is","are","was","were","be","been",
    "но","и","а","что","это","этот","эта","эти","как","уже","еще","для","при","над","после","перед","сегодня","вчера",
}

# =========================
# AI PROMPT
# =========================
SYSTEM_PROMPT = """
You are the Editorial AI of a Ukrainian Telegram news digest.

INPUT: ranked list of candidates with metadata (ID, time, source, category, cluster size, content).
Return STRICT JSON.

CRITICAL FILTERS:
1) WAR COVERAGE — NO ROUTINE REPORTS:
   - REJECT: daily General Staff summaries, village captures, routine shelling statistics, tactical movements
   - REJECT: vague "explosions heard" without confirmed damage/casualties
   - ACCEPT: significant attacks with confirmed damage/casualties, major infrastructure strikes, new weapon deployments,
             high-casualty events, command decisions, major breakthroughs

2) POLITICS — NO EMPTY STATEMENTS:
   - REJECT: "held meeting", "expressed concern", "discussed plans", "urged", "called for", "expert says" without new facts
   - ACCEPT: laws signed, budgets approved, weapons delivered, sanctions imposed, arrests, concrete actions with numbers/dates

3) FUNDRAISING / CHARITY / GIVEAWAYS — HARD REJECT:
   - REJECT: any "збір", "донат", "банка", "розіграш", "подарунки", "підтримайте", "fundraising", "donate", "charity"

4) INTERNATIONAL — IMPACT FOCUS:
   - REJECT: generic statements of support
   - ACCEPT: military aid packages (amounts), policy changes, sanctions details, deployments, confirmed incidents

5) DEDUPLICATION:
   - If multiple candidates cover the same event, select ONLY the most detailed/official one
   - Discard duplicates

WRITING RULES (Output in Ukrainian):
Headline:
- factual, specific, informative; no clickbait, no questions, no vague phrases
Body:
- 1–3 concise sentences with essential facts (what/where/when/who/numbers)
- no speculation

TONE REQUIREMENTS:
War/attacks/casualties — strictly neutral/serious:
- use precise terms: "обстріл", "удар", "атака", "жертви", "поранені", "пошкоджено"
- forbidden: slang/euphemisms/irony about attacks on Ukraine

CATEGORY ASSIGNMENT:
- war, intl, ua, society, meme

VERIFICATION NOTES:
- official (government/OVA/DSNS/forces), major_media, unverified

SELECTION TARGETS:
- Select up to {NEWS_TARGET} items for NEWS
- Select up to {WATCH_TARGET} items for WATCH (international)

OUTPUT FORMAT (JSON only):
{{
  "editorial_chat": "short explanation",
  "publish": [
    {{
      "id": <int>,
      "headline": "...",
      "body": "...",
      "category": "war|intl|ua|society|meme",
      "verification_note": "official|major_media|unverified"
    }}
  ]
}}
"""

# =========================
# HTTP SESSION
# =========================
def get_session() -> requests.Session:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    s = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.35,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"])
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0 (compatible; TGNewsBot/3.5)"})
    return s

SESSION = get_session()

# =========================
# JSON IO
# =========================
def load_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path: str, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log.warning(f"Failed to save {path}: {e}")

def load_sources_cfg() -> dict:
    if SOURCES_JSON:
        try:
            return json.loads(SOURCES_JSON)
        except Exception as e:
            log.warning(f"SOURCES_JSON invalid, fallback to file: {e}")
    return load_json(SOURCES_FILE, {}) or {}

# =========================
# TEXT UTILS
# =========================
def normalize_url(url: str) -> str:
    if not url:
        return ""
    try:
        p = urlparse(url)
        q = p.query or ""
        drop_prefixes = ("utm_",)
        drop_exact = {
            "fbclid", "gclid", "yclid", "mc_cid", "mc_eid", "cmpid",
            "mkt_tok", "sr_share", "ref", "referrer", "rss", "rssfeed"
        }
        if q:
            kept = []
            for part in q.split("&"):
                if not part:
                    continue
                k = part.split("=", 1)[0].strip().lower()
                if any(k.startswith(px) for px in drop_prefixes):
                    continue
                if k in drop_exact:
                    continue
                kept.append(part)
            q = "&".join(kept)
        return urlunparse((p.scheme, p.netloc, p.path, "", q, ""))
    except Exception:
        return url

def clean_text(text: str) -> str:
    if not text:
        return ""
    soup = BeautifulSoup(text, "html.parser")
    t = soup.get_text(separator=" ")
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"\s+", " ", t).strip()

    junk_phrases = [
        "читайте також", "підписуйтесь", "всі права захищені", "копіювання заборонено",
        "долучайтесь", "новини партнерів", "більше новин", "наш телеграм",
        "джерело:", "photo by", "getty images"
    ]
    low = t.lower()
    for junk in junk_phrases:
        if junk in low:
            idx = low.find(junk)
            if idx > 150:
                t = t[:idx].strip()
                break
    return t

def normalize_title(title: str) -> str:
    return re.sub(r"[^\w\s'’іїєґІЇЄҐ-]", "", (title or "").lower()).strip()

def tokenize(text: str) -> List[str]:
    t = re.sub(r"[^\w\s'’іїєґІЇЄҐ-]", " ", (text or "").lower())
    parts = [x.strip("-'’") for x in t.split() if x]
    parts = [x for x in parts if len(x) > 2 and x not in STOPWORDS]
    return parts[:160]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))

def sim_title(a: str, b: str) -> float:
    an = normalize_title(a)
    bn = normalize_title(b)
    if not an or not bn:
        return 0.0
    ratio = SequenceMatcher(None, an, bn).ratio()
    jac = jaccard(tokenize(an), tokenize(bn))
    return max(ratio, jac)

def simhash64(text: str) -> int:
    toks = tokenize(text)
    if not toks:
        return 0
    v = [0] * 64
    for tok in toks:
        h = hashlib.md5(tok.encode("utf-8")).digest()
        x = int.from_bytes(h[:8], "big", signed=False)
        for i in range(64):
            v[i] += 1 if (x >> i) & 1 else -1
    out = 0
    for i in range(64):
        if v[i] > 0:
            out |= (1 << i)
    return out

def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()

# =========================
# HARD FILTERS
# =========================
FUNDRAISING_WORDS = [
    "збір", "збори", "донат", "донати", "задонать", "підтримайте", "підтримати",
    "банка", "монобанк", "mono", "розіграш", "подарунки", "лотерея", "raffle", "giveaway",
    "fundraising", "donate", "charity"
]
JUNK_TOPICS = [
    "гороскоп", "астролог", "сонник", "курс валют", "погода", "зодіак",
    "скандал", "п'яний", "сварка"
]

def has_any(text: str, words: List[str]) -> bool:
    low = (text or "").lower()
    return any(w in low for w in words)

ATTACK_KEYWORDS = [
    "обстріл", "атака", "удар", "вибух", "ракета", "ракет", "дрон", "дрони", "shahed",
    "пошкоджено", "зруйновано", "влучання", "загинул", "загиблих", "поранен", "жертв"
]

def is_attack_like(title: str, body: str) -> bool:
    blob = f"{title} {body}".lower()
    return any(k in blob for k in ATTACK_KEYWORDS)

# =========================
# HISTORY
# =========================
def load_history() -> dict:
    data = load_json(HISTORY_FILE, {"items": []})
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=HISTORY_RETENTION_DAYS)

    fresh = []
    for it in data.get("items", []):
        try:
            ts = date_parser.parse(it.get("ts", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts > cutoff:
                fresh.append(it)
        except Exception:
            pass

    data["items"] = fresh
    return data

def add_to_history(items: list, history: dict):
    now_iso = datetime.now(timezone.utc).isoformat()
    for it in items:
        title = it.get("title", "")
        body = clean_text(it.get("full_text", ""))[:800]
        fp = simhash64(f"{title} {body}")

        history["items"].append({
            "ts": now_iso,
            "link": it.get("link", ""),
            "url_norm": normalize_url(it.get("link", "")),
            "title": title,
            "title_norm": normalize_title(title),
            "source": it.get("source", ""),
            "simhash": str(fp),
            "bucket": it.get("bucket", ""),
            "is_official": bool(it.get("is_official", False)),
        })
    save_json(HISTORY_FILE, history)

def is_duplicate(item: dict, history: dict) -> bool:
    new_url_norm = normalize_url(item.get("link", ""))

    if new_url_norm:
        for old in history.get("items", []):
            if normalize_url(old.get("link", "")) == new_url_norm:
                return True
            if old.get("url_norm") == new_url_norm:
                return True

    new_title = item.get("title", "")
    new_title_norm = normalize_title(new_title)
    if len(new_title_norm) < 10:
        return False

    new_body = clean_text(item.get("full_text", ""))[:800]
    new_fp = simhash64(f"{new_title} {new_body}")

    new_dt = item.get("dt")
    if new_dt and new_dt.tzinfo is None:
        new_dt = new_dt.replace(tzinfo=timezone.utc)

    for old in history.get("items", []):
        old_title_norm = old.get("title_norm") or normalize_title(old.get("title", ""))

        if len(new_title_norm) > 14 and new_title_norm == old_title_norm:
            return True

        try:
            old_fp = int(old.get("simhash", "0"))
        except Exception:
            old_fp = 0

        if old_fp and new_fp:
            dist = hamming64(new_fp, old_fp)
            if dist <= HISTORY_SIMHASH_STRICT:
                return True
            if dist <= HISTORY_SIMHASH_LOOSE:
                try:
                    old_ts = date_parser.parse(old.get("ts", ""))
                    if old_ts.tzinfo is None:
                        old_ts = old_ts.replace(tzinfo=timezone.utc)
                    if new_dt and abs((new_dt - old_ts).total_seconds()) < 10 * 3600:
                        return True
                except Exception:
                    pass

        if abs(len(new_title_norm) - len(old_title_norm)) < 30:
            if SequenceMatcher(None, new_title_norm, old_title_norm).ratio() > HISTORY_TITLE_SIM_THRESHOLD:
                return True

    return False

# =========================
# COLLECTION
# =========================
def fetch_telegram(username: str, cache: dict) -> list:
    url = f"https://t.me/s/{username}"
    if url in cache and (time.time() - cache[url].get("ts", 0) < 180):
        return []
    try:
        r = SESSION.get(url, timeout=8)
        cache[url] = {"ts": time.time()}
        if r.status_code != 200:
            return []

        soup = BeautifulSoup(r.text, "html.parser")
        posts = []

        for wrap in soup.select(".tgme_widget_message_wrap"):
            if wrap.select_one(".tgme_widget_message_service"):
                continue
            msg = wrap.select_one(".tgme_widget_message_text")
            if not msg:
                continue

            meta = wrap.select_one(".tgme_widget_message_date")
            link = meta.get("href", "") if meta else ""

            dt = None
            if meta and meta.select_one("time"):
                raw_dt = meta.select_one("time").get("datetime")
                try:
                    dt = date_parser.parse(raw_dt).astimezone(timezone.utc)
                except Exception:
                    dt = None

            if not dt:
                continue

            text_content = msg.get_text(separator="\n")
            title = (text_content.split("\n")[0] if text_content else "").strip()
            if not title:
                title = "Telegram пост"

            posts.append({
                "title": title[:110],
                "full_text": text_content,
                "link": link,
                "dt": dt,
                "type": "tg",
            })

        return posts
    except Exception:
        return []

def fetch_rss(url: str, cache: dict) -> list:
    if url in cache and (time.time() - cache[url].get("ts", 0) < 300):
        return []
    try:
        r = SESSION.get(url, timeout=10)
        cache[url] = {"ts": time.time()}
        if r.status_code != 200:
            return []

        d = feedparser.parse(r.content)
        posts = []

        for e in d.entries:
            dt = None
            if getattr(e, "published_parsed", None):
                try:
                    dt = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
                except Exception:
                    dt = None
            if not dt and getattr(e, "published", None):
                try:
                    dt = date_parser.parse(e.published)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                except Exception:
                    dt = None
            if not dt:
                continue

            full_text = e.get("summary", "") or e.get("description", "") or ""
            link = e.get("link", "") or ""
            posts.append({
                "title": e.get("title", "") or "Новина",
                "full_text": full_text,
                "link": link,
                "dt": dt,
                "type": "rss",
            })
        return posts
    except Exception:
        return []

def classify_source_bucket(key: str) -> Tuple[str, bool]:
    key = key.lower()
    if "intl" in key or "geo" in key or "factcheck" in key or "watch" in key:
        return "watch", True
    return "news", False

def is_official_group(group_name: str) -> bool:
    return group_name in {"top_state", "defense_security", "law_justice", "energy", "ova_heads", "mayors"}

def collect_all(cfg: dict, history: dict) -> list:
    cache = load_json(CACHE_FILE, {})
    now = datetime.now(timezone.utc)
    data = []

    source_names = cfg.get("source_names", {}) or {}
    source_weights = cfg.get("source_weights", {}) or {}

    def get_weight_for_host(host: str, default_w: int) -> int:
        host = host.replace("www.", "")
        w = default_w
        for k, v in source_weights.items():
            if k and k in host:
                w = int(v)
        return w

    def process_item(it: dict, bucket: str, src: str, is_intl: bool, w: int, is_official: bool):
        age = (now - it["dt"]).total_seconds()
        if age > MAX_AGE_SECONDS or age < -300:
            return

        it["bucket"] = bucket
        it["source"] = src
        it["is_intl"] = bool(is_intl)
        it["weight"] = float(w)
        it["is_official"] = bool(is_official)

        blob = f"{it.get('title','')} {it.get('full_text','')}"
        if has_any(blob, FUNDRAISING_WORDS):
            return
        if has_any(blob, JUNK_TOPICS):
            return

        if is_duplicate(it, history):
            return

        data.append(it)

    # RSS groups
    for k, urls in cfg.items():
        if not k.startswith("rss_"):
            continue

        bucket, is_intl = classify_source_bucket(k)
        rss_is_official = k in {"rss_gov", "rss_defense", "rss_regional"}

        for url in (urls or []):
            items = fetch_rss(url, cache)
            host = urlparse(url).netloc.replace("www.", "")
            src = source_names.get(host, host.capitalize() if host else "RSS")

            base = 9 if rss_is_official else (6 if k in {"rss_ua", "rss_intl", "rss_geo", "rss_factcheck"} else 5)
            w = get_weight_for_host(host, base)

            for it in items:
                process_item(it, bucket, src, is_intl, w, rss_is_official)

    # Telegram groups
    tg = cfg.get("telegram_official", {}) or {}
    for group_name, chans in tg.items():
        group_official = is_official_group(group_name)
        for ch in (chans or []):
            is_off = group_official and group_name != "intl_watch"
            bucket = "watch" if group_name == "intl_watch" else "news"
            is_intl = group_name == "intl_watch"

            src = cfg.get("source_names", {}).get(ch, ch)
            w = 14 if is_off else (6 if is_intl else 5)
            w = int(cfg.get("source_weights", {}).get(ch, w))

            items = fetch_telegram(ch, cache)
            for it in items:
                process_item(it, bucket, src, is_intl, w, is_off)

    save_json(CACHE_FILE, cache)
    return data

# =========================
# OFFICIAL-FIRST FOR ATTACKS
# =========================
def prefer_official_for_attacks(items: list) -> list:
    if not items:
        return items

    officials = [x for x in items if x.get("is_official")]
    if not officials:
        return items

    for it in officials:
        if "fp" not in it:
            body = clean_text(it.get("full_text", ""))[:1200]
            it["fp"] = simhash64(f"{it.get('title','')} {body}")

    out = []
    for it in items:
        title = it.get("title", "")
        body = clean_text(it.get("full_text", ""))[:1200]

        if not is_attack_like(title, body):
            out.append(it)
            continue

        if it.get("is_official"):
            out.append(it)
            continue

        dt = it.get("dt")
        if dt and dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        it_fp = simhash64(f"{title} {body}")

        best = None
        best_score = -1.0

        for off in officials:
            odt = off.get("dt")
            if odt and odt.tzinfo is None:
                odt = odt.replace(tzinfo=timezone.utc)

            if dt and odt and abs((dt - odt).total_seconds()) > CLUSTER_TIME_WINDOW_HOURS * 3600:
                continue

            off_fp = off.get("fp", 0)
            dist = hamming64(it_fp, off_fp) if (it_fp and off_fp) else 64
            s1 = max(0.0, 1.0 - dist / 24.0)
            s2 = sim_title(title, off.get("title", ""))
            score = 0.65 * s1 + 0.35 * s2

            if score > best_score:
                best_score = score
                best = off

        if best and best_score >= 0.62:
            out.append(best)
        else:
            if REQUIRE_OFFICIAL_FOR_ATTACKS:
                continue
            it["ai_verification"] = "unverified"
            out.append(it)

    return out

# =========================
# EMBEDDINGS
# =========================
def get_embeddings(texts: List[str]) -> Optional[np.ndarray]:
    if not texts:
        return None
    try:
        res = genai.embed_content(model=EMBED_MODEL_NAME, content=texts, task_type="clustering")
        emb = res.get("embedding")
        if not emb or len(emb) != len(texts):
            return None
        return np.array(emb, dtype=np.float32)
    except Exception:
        return None

# =========================
# CLUSTERING (union-find)
# =========================
def semantic_cluster_best(items: list) -> list:
    if len(items) < 2:
        return items

    items = items[:400]
    texts = [f"{it.get('title','')} {clean_text(it.get('full_text',''))[:450]}" for it in items]

    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def time_close(i, j) -> bool:
        a, b = items[i].get("dt"), items[j].get("dt")
        if not a or not b:
            return True
        if a.tzinfo is None:
            a = a.replace(tzinfo=timezone.utc)
        if b.tzinfo is None:
            b = b.replace(tzinfo=timezone.utc)
        return abs((a - b).total_seconds()) <= CLUSTER_TIME_WINDOW_HOURS * 3600

    vectors = get_embeddings(texts) if ML_AVAILABLE else None
    if vectors is not None and len(vectors) == len(items):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        V = vectors / norms
        S = V @ V.T

        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                if not time_close(i, j):
                    continue
                sem = float(S[i, j])
                if sem >= SEM_SIM_THRESHOLD:
                    union(i, j)
                    continue

                ls = sim_title(items[i].get("title", ""), items[j].get("title", ""))
                off = bool(items[i].get("is_official") or items[j].get("is_official"))
                if ls >= (LEX_SIM_OFFICIAL_THRESHOLD if off else LEX_SIM_THRESHOLD) and sem >= SEM_SIM_SECONDARY:
                    union(i, j)
    else:
        n = len(items)
        for i in range(n):
            for j in range(i + 1, n):
                if not time_close(i, j):
                    continue
                ls = sim_title(items[i].get("title", ""), items[j].get("title", ""))
                off = bool(items[i].get("is_official") or items[j].get("is_official"))
                if ls >= (LEX_SIM_OFFICIAL_THRESHOLD if off else LEX_SIM_THRESHOLD):
                    union(i, j)

    clusters: Dict[int, List[dict]] = {}
    for idx in range(len(items)):
        clusters.setdefault(find(idx), []).append(items[idx])

    def rank_key(x: dict):
        return (
            1 if x.get("is_official") else 0,
            float(x.get("weight", 0)),
            1 if x.get("is_deep_read") else 0,
            len(x.get("full_text", "") or ""),
        )

    unique = []
    for group in clusters.values():
        group.sort(key=rank_key, reverse=True)
        best = group[0]
        best["cluster_size"] = len(group)
        unique.append(best)

    log.info(f"🧩 Clustered: {len(items)} -> {len(unique)} events.")
    return unique

# =========================
# SCORING
# =========================
CRITICAL_KEYWORDS = {
    "загинул": 5.0,
    "загиблих": 5.0,
    "поранен": 3.0,
    "жертв": 4.0,
    "ракета": 4.0,
    "дрон": 3.0,
    "обстріл": 3.0,
    "атака": 3.0,
    "удар": 3.0,
    "пошкоджено": 2.5,
    "зруйновано": 3.0,
    "критичн": 3.0,
    "інфраструктур": 3.0,
}
IMPORTANT_KEYWORDS = {
    "схвалено": 4.0,
    "підписано": 4.0,
    "санкц": 3.5,
    "арешт": 3.5,
    "млрд": 5.0,
    "мільярд": 5.0,
    "пакет": 3.0,
    "допомог": 3.0,
    "збро": 4.0,
    "patriot": 5.0,
    "himars": 5.0,
    "f-16": 5.0,
}
VIP_KEYWORDS = {
    "нато": 2.5,
    "білий дім": 2.5,
    "пентагон": 2.5,
    "трамп": 2.5,
    "байден": 2.5,
    "зеленськ": 2.0,
}
EMPTY_TALK = [
    "стурбован", "обговорили", "планують", "закликав",
    "висловив", "підкреслив", "зазначив", "наголосив",
    "експерт", "аналітик", "вважає", "припускає"
]
ROUTINE_PHRASES = ["за добу", "за минулу добу", "генштаб", "оперативна інформація", "станом на"]

def calculate_priority_score(item: dict) -> float:
    score = float(item.get("weight", 0))
    size = int(item.get("cluster_size", 1))
    score += math.log(size + 1) * 4.0

    if item.get("is_intl"):
        score += 6.0
    if item.get("is_deep_read"):
        score += 2.0
    if item.get("is_official"):
        score += 3.0

    now = datetime.now(timezone.utc)
    dt = item.get("dt")
    if dt:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_h = (now - dt).total_seconds() / 3600
        score += max(0.0, (1.5 - age_h) * 5.0)

    blob = (item.get("title", "") + " " + item.get("full_text", "")).lower()

    if has_any(blob, FUNDRAISING_WORDS):
        return -999.0

    for k, b in CRITICAL_KEYWORDS.items():
        if k in blob:
            score += b
    for k, b in IMPORTANT_KEYWORDS.items():
        if k in blob:
            score += b
    for k, b in VIP_KEYWORDS.items():
        if k in blob:
            score += b

    if any(p in blob for p in ROUTINE_PHRASES):
        score -= 8.0
    if any(p in blob for p in EMPTY_TALK) and float(item.get("weight", 0)) < 8:
        score -= 6.0

    return score

# =========================
# DEEP READ
# =========================
def deep_read_item(item: dict) -> dict:
    if item.get("type") != "rss":
        return item
    url = item.get("link", "")
    if not url:
        return item
    try:
        r = SESSION.get(url, timeout=12)
        if r.status_code != 200 or not r.text:
            return item
        extracted = trafilatura.extract(r.text, include_comments=False, include_tables=False)
        if extracted and len(extracted) > 120:
            item["full_text"] = extracted[:4500]
            item["is_deep_read"] = True
    except Exception:
        pass
    return item

def enrich_content_with_deepread(items: list) -> list:
    if not items:
        return items
    log.info(f"🔍 Deep reading top {len(items)} candidates...")
    out = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(deep_read_item, it) for it in items]
        for f in as_completed(futures):
            out.append(f.result())
    return out

# =========================
# AI SELECT + WRITE
# =========================
def ai_select_and_write(candidates: list) -> Tuple[List[dict], Optional[str]]:
    if not candidates:
        return [], None

    for it in candidates:
        it["final_score"] = calculate_priority_score(it)

    candidates = [x for x in candidates if x.get("final_score", 0) > 0]
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    top = candidates[:DEEP_READ_LIMIT]
    top = enrich_content_with_deepread(top)
    pool = top + candidates[DEEP_READ_LIMIT:AI_CONTEXT_LIMIT]

    txt_input = ""
    now_utc = datetime.now(timezone.utc)

    for i, it in enumerate(pool):
        limit = 2500 if it.get("is_deep_read") else 700
        body = clean_text(it.get("full_text", ""))[:limit]

        dt = it.get("dt", now_utc)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        t_str = dt.strftime("%H:%M")

        src_tag = "[INTL]" if it.get("is_intl") else "[UA]"
        off_tag = "✅OFFICIAL" if it.get("is_official") else ""
        deep_tag = "📄FULL" if it.get("is_deep_read") else ""
        clust = f"×{it.get('cluster_size', 1)}"
        score = f"{it.get('final_score', 0):.1f}"

        flags = []
        low = f"{it.get('title','')} {body}".lower()
        if any(w in low for w in ["загинул", "загиблих", "поранен", "жертв"]):
            flags.append("⚠️CASUALTIES")
        if any(w in low for w in ["млрд", "мільярд", "$", "€"]):
            flags.append("💰MONEY")
        if any(w in low for w in ROUTINE_PHRASES):
            flags.append("⚠️ROUTINE?")
        if any(w in low for w in FUNDRAISING_WORDS):
            flags.append("⛔FUNDRAISING?")
        flag_str = " ".join(flags)

        txt_input += (
            f"═══ ID:{i} ═══\n"
            f"📊 {src_tag} {off_tag} {deep_tag} | ⏰{t_str} | {clust} | Score:{score}\n"
            f"📰 SOURCE: {it.get('source','')}\n"
            f"🔖 {flag_str}\n"
            f"📌 TITLE: {it.get('title','')}\n"
            f"📝 BODY:\n{body}\n"
            f"{'─'*60}\n\n"
        )

    prompt = SYSTEM_PROMPT.format(NEWS_TARGET=NEWS_TO_PUBLISH, WATCH_TARGET=WATCH_TO_PUBLISH)
    prompt += f"\n\nCANDIDATES:\n\n{txt_input}"

    try:
        model = genai.GenerativeModel(
            GEN_MODEL_NAME,
            generation_config={"response_mime_type": "application/json"}
        )
        resp = model.generate_content(prompt)
        raw = (resp.text or "").replace("```json", "").replace("```", "").strip()
        data = json.loads(raw)

        chat_log = data.get("editorial_chat", "")
        final_posts = []

        for obj in data.get("publish", []) or []:
            idx = obj.get("id")
            if not isinstance(idx, int) or idx < 0 or idx >= len(pool):
                continue
            orig = pool[idx]

            orig["ai_title"] = obj.get("headline", "") or orig.get("title", "")
            orig["ai_body"] = obj.get("body", "") or ""
            orig["ai_category"] = obj.get("category", "ua")
            orig["ai_verification"] = obj.get("verification_note", "major_media")

            final_posts.append(orig)

        log.info(f"✅ AI selected {len(final_posts)} posts.")
        return final_posts, chat_log

    except Exception as e:
        log.error(f"❌ AI Generation Error: {e}")
        traceback.print_exc()
        return [], None

# =========================
# TELEGRAM PUBLISH
# =========================
def tg_send(text: str, disable_preview: bool = True) -> bool:
    if not text:
        return True
    if not TG_BOT_TOKEN:
        print("--- SIMULATED SEND ---\n" + text)
        return True

    url = f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage"
    ok_all = True

    for chat_id in TARGET_CHATS:
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": bool(disable_preview),
        }

        sent = False
        for attempt in range(3):
            try:
                r = SESSION.post(url, json=payload, timeout=14)
                if r.status_code == 200:
                    sent = True
                    break
                if r.status_code == 429:
                    try:
                        data = r.json()
                        wait_s = int(data.get("parameters", {}).get("retry_after", 2))
                    except Exception:
                        wait_s = 2
                    time.sleep(min(wait_s, 10))
                    continue
                log.warning(f"TG send failed [{r.status_code}] to {chat_id}: {r.text[:200]}")
                time.sleep(0.6 + attempt * 0.6)
            except Exception as ex:
                log.warning(f"TG send exception to {chat_id}: {ex}")
                time.sleep(0.6 + attempt * 0.6)

        ok_all = ok_all and sent

    return ok_all

def esc(s: str) -> str:
    return html.escape(s or "", quote=False)

def format_post_line(it: dict, headlines_only: bool = False) -> str:
    title = it.get("ai_title") or it.get("title") or ""
    body = it.get("ai_body") or ""
    src = it.get("source") or "Джерело"
    link = it.get("link") or ""

    cat = it.get("ai_category", "ua")
    icon = "🔹"
    if cat == "war":
        icon = "🛡️"
    elif cat == "intl":
        icon = "🌍"
    elif cat == "meme":
        icon = "🐸"

    # ЛІНК — НА СЛОВІ ДЖЕРЕЛА
    src_link = f"<a href=\"{link}\"><i>{esc(src)}</i></a>" if link else f"<i>{esc(src)}</i>"

    if headlines_only:
        return f"{icon} <b>{esc(title)}</b> — {src_link}\n"

    b = esc(body).strip()
    if b:
        return f"{icon} <b>{esc(title)}</b>\n{b}\n🔗 {src_link}\n\n"
    return f"{icon} <b>{esc(title)}</b>\n🔗 {src_link}\n\n"

def publish_and_save(items: list, history: dict) -> int:
    if not items:
        return 0

    to_send = []
    for it in items:
        if not is_duplicate(it, history):
            to_send.append(it)
    if not to_send:
        return 0

    news_items = [x for x in to_send if x.get("bucket") != "watch"]
    watch_items = [x for x in to_send if x.get("bucket") == "watch"] if POST_WATCHLIST else []

    casualty_like = []
    normal_news = []

    for it in news_items:
        txt = f"{it.get('ai_title','')} {it.get('ai_body','')} {it.get('title','')} {it.get('full_text','')}".lower()
        if any(w in txt for w in ["загинул", "загиблих", "поранен", "жертв"]) or it.get("ai_category") == "war":
            if is_attack_like(it.get("ai_title","") or it.get("title",""), it.get("ai_body","") or it.get("full_text","")):
                casualty_like.append(it)
            else:
                normal_news.append(it)
        else:
            normal_news.append(it)

    do_casualty_digest = len(casualty_like) >= CASUALTY_DIGEST_THRESHOLD

    ts_title = datetime.now(KYIV_TZ).strftime("%H:%M")

    parts: List[str] = [f"⚡️ <b>Головне на {ts_title}</b>\n\n"]
    current_len = len(parts[0])
    part_no = 1

    def flush_new_part():
        nonlocal parts, current_len, part_no
        text = "".join(parts).strip()
        if text:
            tg_send(text, disable_preview=True)
        part_no += 1
        parts = [f"⚡️ <b>Головне на {ts_title} (Частина {part_no})</b>\n\n"]
        current_len = len(parts[0])

    def add_block(block: str):
        nonlocal current_len
        if current_len + len(block) > 3900:
            flush_new_part()
        parts.append(block)
        current_len += len(block)

    for it in normal_news:
        add_block(format_post_line(it, headlines_only=False))

    if do_casualty_digest:
        add_block("<b>🛑 Обстріли та жертви: коротко</b>\n")
        for it in casualty_like:
            add_block(format_post_line(it, headlines_only=True))
        add_block("\n")
    else:
        for it in casualty_like:
            add_block(format_post_line(it, headlines_only=False))

    if watch_items:
        add_block("<b>👀 Watchlist</b>\n\n")
        for it in watch_items:
            add_block(format_post_line(it, headlines_only=False))

    final_text = "".join(parts).strip()
    if final_text:
        tg_send(final_text, disable_preview=True)

    add_to_history(to_send, history)
    return len(to_send)

# =========================
# STATS
# =========================
def log_analytics_report(stats: dict, chat_log: Optional[str] = None):
    report = (
        f"📊 <b>Stats</b>\n"
        f"⏱️ {stats.get('duration',0):.1f}s | 📥 {stats.get('raw',0)} | 🧽 {stats.get('after_official',0)}\n"
        f"🧩 Events: {stats.get('clusters',0)} | ✅ Posted: {stats.get('published',0)}"
    )
    log.info(report.replace("\n", " | ").replace("<b>", "").replace("</b>", ""))

    if TG_BOT_TOKEN and TG_ADMIN_ID:
        try:
            full = report
            if chat_log:
                full += f"\n\n💬 <b>AI Log:</b>\n{esc(str(chat_log)[:3500])}"
            SESSION.post(
                f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TG_ADMIN_ID,
                    "text": full,
                    "parse_mode": "HTML",
                    "disable_notification": True
                },
                timeout=12
            )
        except Exception:
            pass

# =========================
# MAIN
# =========================
def safe_main():
    start_time = time.time()
    stats = {"raw": 0, "after_official": 0, "clusters": 0, "published": 0, "duration": 0}

    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY is missing")
        sys.exit(1)
    if not TG_BOT_TOKEN:
        print("❌ TG_BOT_TOKEN is missing")
        sys.exit(1)
    if not TARGET_CHATS:
        print("❌ TG_CHAT_ID and/or DIGEST_CHANNEL is missing")
        sys.exit(1)

    genai.configure(api_key=GOOGLE_API_KEY)

    try:
        cfg = load_sources_cfg()
        if not cfg:
            log.error("❌ Sources config empty (news_sources.json or SOURCES_JSON).")
            return

        history = load_history()

        raw = collect_all(cfg, history)
        stats["raw"] = len(raw)

        if not raw:
            log.info("💤 No fresh news found.")
            return

        raw2 = prefer_official_for_attacks(raw)
        stats["after_official"] = len(raw2)

        clustered = semantic_cluster_best(raw2)
        stats["clusters"] = len(clustered)

        final, chat_log = ai_select_and_write(clustered)

        posted = publish_and_save(final, history)
        stats["published"] = posted

        stats["duration"] = time.time() - start_time
        log_analytics_report(stats, chat_log)

    except Exception as e:
        log.critical(f"🔥 FAIL: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    safe_main()
