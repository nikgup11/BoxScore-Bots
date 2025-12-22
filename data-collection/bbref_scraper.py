import os
import time
import datetime
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

# =========================
# CONFIG
# =========================

BASE_INDEX = "https://www.basketball-reference.com/players/{letter}/"
BASE_GAMELOG = "https://www.basketball-reference.com/players/{letter}/{slug}/gamelog/{season_end}"

OUT_DIR = "./raw_data/"
CACHE_DIR = "./cache/"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

S = requests.Session()


# =========================
# HELPERS
# =========================

def rotate_headers():
    S.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml"
    })


def polite_sleep(base=4, jitter=3):
    time.sleep(base + random.random() * jitter)


def make_out_path(slug, name):
    name_safe = "".join(
        c for c in name.replace(" ", "_")
        if c.isalnum() or c in ("_", "-")
    )
    return os.path.join(OUT_DIR, f"{slug}_{name_safe}_last3.csv")


def last_n_season_ends(n=3):
    today = datetime.date.today()
    last_end = today.year if today.month < 7 else today.year + 1
    return [last_end - i for i in range(n)]


# =========================
# PLAYER INDEX (CACHED)
# =========================

def get_players_for_letter(letter):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"players_{letter}.html")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            html = f.read()
    else:
        rotate_headers()
        r = S.get(BASE_INDEX.format(letter=letter), timeout=20)

        if r.status_code == 429:
            print("429 on index page – cooling down 5 minutes")
            time.sleep(300)
            return []

        r.raise_for_status()
        html = r.text

        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(html)

        polite_sleep(8, 5)

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="players")
    if table is None:
        return []

    players = []
    for tr in table.select("tbody tr"):
        if "thead" in (tr.get("class") or []):
            continue

        name_tag = tr.find("th").find("a") if tr.find("th") else None
        if not name_tag:
            continue

        href = name_tag.get("href", "")
        slug = href.split("/")[-1].replace(".html", "")
        name = name_tag.get_text(strip=True)

        year_max = tr.find("td", {"data-stat": "year_max"})
        year_max = year_max.get_text(strip=True) if year_max else ""

        players.append({
            "name": name,
            "slug": slug,
            "year_max": year_max
        })

    return players


def is_active(player, recent_season):
    ym = player.get("year_max", "").lower()
    if ym in ("", "active"):
        return True
    try:
        return int(ym) >= recent_season
    except ValueError:
        return False


# =========================
# GAMELOG FETCHING
# =========================

def fetch_player_gamelogs(letter, slug, season_ends, max_retries=3):
    frames = []

    for season in season_ends:
        url = BASE_GAMELOG.format(
            letter=letter,
            slug=slug,
            season_end=season
        )

        for attempt in range(max_retries):
            rotate_headers()

            r = S.get(url, timeout=20)

            if r.status_code == 429:
                print(f"429 on {slug} {season} – hard cooldown")
                time.sleep(300)
                return None

            if r.status_code != 200:
                polite_sleep(10, 5)
                continue

            soup = BeautifulSoup(r.text, "html.parser")
            table = soup.find("table", id="pgl_basic")
            if table is None:
                break

            df = pd.read_html(str(table))[0]

            if df.empty:
                break

            df = df.loc[~df.iloc[:, 0].isna()].copy()
            df["SEASON_END"] = season
            frames.append(df)

            polite_sleep(6, 4)
            break

    if frames:
        return pd.concat(frames, ignore_index=True, sort=False)

    return None


# =========================
# SAVE / MERGE
# =========================

def save(df, slug, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = make_out_path(slug, name)

    if os.path.exists(path):
        old = pd.read_csv(path)
        combined = pd.concat([old, df], ignore_index=True, sort=False)

        if "Date" in combined.columns:
            combined["Date"] = pd.to_datetime(
                combined["Date"], errors="coerce"
            )
            combined = combined.drop_duplicates(
                subset=["SEASON_END", "Date"],
                keep="last"
            ).sort_values(["SEASON_END", "Date"])
        else:
            combined = combined.drop_duplicates()

        combined.to_csv(path, index=False)
        print(f"updated: {path}")
    else:
        df.to_csv(path, index=False)
        print(f"saved: {path}")


# =========================
# MAIN RUNNER
# =========================

def run(letters=("a",), seasons_n=3):
    seasons = last_n_season_ends(seasons_n)
    recent = seasons[0]

    print("Fetching seasons:", seasons)

    for letter in letters:
        players = get_players_for_letter(letter)
        active_players = [
            p for p in players if is_active(p, recent)
        ]

        for p in active_players:
            path = make_out_path(p["slug"], p["name"])

            if os.path.exists(path):
                try:
                    old = pd.read_csv(path)
                    existing = set(
                        pd.to_numeric(
                            old.get("SEASON_END", []),
                            errors="coerce"
                        ).dropna().astype(int)
                    )
                except Exception:
                    existing = set()
            else:
                existing = set()

            fetch_seasons = [
                s for s in seasons
                if s == seasons[0] or s not in existing
            ]

            if not fetch_seasons:
                continue

            print(f"Fetching {p['name']} – {fetch_seasons}")
            df = fetch_player_gamelogs(
                letter, p["slug"], fetch_seasons
            )

            if df is not None and not df.empty:
                df["PLAYER_NAME"] = p["name"]
                save(df, p["slug"], p["name"])

            polite_sleep(8, 6)


# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    # Start SMALL — do NOT scrape entire alphabet in one run
    run(letters=("a",), seasons_n=3)