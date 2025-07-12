#!/usr/bin/env python3
"""Scrape myrient directories to populate missing URLs in platforms.csv.

This script checks both the No-Intro and Redump listings on myrient.erista.me
and tries to match each platform's directory name to a subfolder.

Usage: python scrape_platform_urls.py [-w]
 -w, --write    Overwrite platforms.csv with discovered URLs.
The default behavior only prints the updated CSV to stdout.
"""
import os
import re
import argparse
import pandas as pd
import requests
from bs4 import BeautifulSoup

NO_INTRO = "https://myrient.erista.me/files/No-Intro/"
REDUMP = "https://myrient.erista.me/files/Redump/"
PLATFORMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "platforms.csv")


def fetch_directories(url):
    """Return mapping of lower-case directory names -> absolute URLs."""
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    dirs = {}
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.endswith("/"):
            continue
        name = a.get_text(strip=True) or href
        dirs[name.lower()] = requests.compat.urljoin(url, href)
    return dirs


def match_directory(name, dir_map):
    """Return URL from dir_map that best matches name, else None."""
    lname = name.lower()
    # exact match first
    if lname in dir_map:
        return dir_map[lname]
    # relaxed matching: strip non-alphanumerics
    canon = re.sub(r"[^a-z0-9]", "", lname)
    for key, val in dir_map.items():
        if canon == re.sub(r"[^a-z0-9]", "", key):
            return val
    for key, val in dir_map.items():
        if canon in re.sub(r"[^a-z0-9]", "", key):
            return val
    return None


def scrape_urls():
    no_intro_dirs = fetch_directories(NO_INTRO)
    redump_dirs = fetch_directories(REDUMP)

    df = pd.read_csv(PLATFORMS_CSV)
    df.fillna("", inplace=True)
    updated = df.copy()

    for idx, row in df.iterrows():
        if row.get("URL"):
            continue
        name = row.get("Directory") or row.get("Platform")
        url = match_directory(name, no_intro_dirs)
        if not url:
            url = match_directory(name, redump_dirs)
        if url:
            updated.at[idx, "URL"] = url
            print(f"Matched {row['Platform']} -> {url}")
        else:
            print(f"No match for {row['Platform']}")
    return updated


def main():
    ap = argparse.ArgumentParser(description="Populate platform URLs by scraping myrient")
    ap.add_argument("-w", "--write", action="store_true", help="overwrite platforms.csv with results")
    args = ap.parse_args()

    updated = scrape_urls()
    if args.write:
        updated.to_csv(PLATFORMS_CSV, index=False)
        print(f"Updated {PLATFORMS_CSV}")
    else:
        print(updated.to_csv(index=False))


if __name__ == "__main__":
    main()
