#!/usr/bin/env python3
"""
Generate a CSV and text list of download URLs for the top N unmatched games.
Place this file in "..\RetroBat\scripts\results" alongside your snapshot folders.

Behavior changes:
 - Prompts user for snapshot directory name (with TAB-completion).
 - Guides user to input:
     • Number of top games per platform (default: 10)
     • Fuzzy-match score threshold (0-100; default: 85)
 - Reads CSVs from the specified snapshot directory:
     - summary.csv
     - unmatched_summary.csv
     - matched_summary.csv (optional, not used here)
"""
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import unicodedata
import re
from rapidfuzz import process, fuzz
from urllib.parse import urljoin, unquote

# for interactive directory tab-completion
import readline
import glob

# determine script directory (base for snapshots)
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_tab_completion():
    """Enable TAB-completion for directory names in input()."""
    def complete_dir(text, state):
        # Expand partial glob and append slash for directories
        base = text or '.'
        matches = glob.glob(base + '*')
        matches = [m + ('/' if os.path.isdir(m) else '') for m in matches]
        try:
            return matches[state]
        except IndexError:
            return None

    readline.set_completer(complete_dir)
    readline.parse_and_bind('tab: complete')

# ——— Title normalization helper ———————————————————————————————————
_ROMAN = {'ix':9,'viii':8,'vii':7,'vi':6,'iv':4,'iii':3,'ii':2,'i':1}
_ROMAN_RE = re.compile(r'\b(' + '|'.join(sorted(_ROMAN, key=len, reverse=True)) + r')\b')

def norm(title):
    s = unicodedata.normalize('NFKD', str(title))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r'[\(\[].*?[\)\]]', '', s)
    s = re.sub(r'^the\s+', '', s)
    s = s.replace('&', ' and ')
    s = _ROMAN_RE.sub(lambda m: str(_ROMAN[m.group(1)]), s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

# ——— Region priority preference ———————————————————————————————————
def region_priority(filename):
    lf = filename.lower()
    if re.search(r'\b(eu|eur|europe)\b', lf):
        return 3
    if re.search(r'\b(us|usa|ntsc-?u)\b', lf):
        return 2
    if re.search(r'\b(jp|japan|ntsc-?j)\b', lf):
        return 1
    return 0

# ——— Mappings to dataset codes ———————————————————————————————————
READABLE_NAMES = {
    '2600':'Atari 2600','5200':'Atari 5200','7800':'Atari 7800','AST':'Atari ST',
    # ... (other mappings unchanged) ...
    'XB':'Xbox','X360':'360',
}
INVERT_MAP = {v:k for k,v in READABLE_NAMES.items()}

# ——— Explicit category URLs for missing-ROM platforms —————————————
PLATFORM_URLS = {
    'PS3': 'https://myrient.erista.me/files/Redump/Sony%20-%20PlayStation%203/',
    'Wii': 'https://myrient.erista.me/files/Redump/Nintendo%20-%20Wii%20-%20NKit%20RVZ%20%5Bzstd-19-128k%5D/',
    'X360': 'https://myrient.erista.me/files/Redump/Microsoft%20-%20Xbox%20360/',
    'PSP': 'https://myrient.erista.me/files/Redump/Sony%20-%20PlayStation%20Portable/',
    'XB': 'https://myrient.erista.me/files/Redump/Microsoft%20-%20Xbox/',
    'GC': 'https://myrient.erista.me/files/Redump/Nintendo%20-%20GameCube%20-%20NKit%20RVZ%20%5Bzstd-19-128k%5D/',
    '3DS': 'https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Nintendo%203DS%20%28Decrypted%29/',
    'PSV': 'https://myrient.erista.me/files/No-Intro/Sony%20-%20PlayStation%20Vita%20%28PSN%29%20%28Content%29/',
    'WiiU': 'https://myrient.erista.me/files/No-Intro/Nintendo%20-%20Wii%20U%20%28Digital%29%20%28CDN%29/'
}

# ——— Main routine —————————————————————————————————————————————————
def main():
    # 1) setup interactive tab-completion
    setup_tab_completion()

    # 2) ask for snapshot directory
    while True:
        snap = input("Enter snapshot directory name: ").strip()
        snapshot_path = os.path.join(RESULTS_DIR, snap)
        if os.path.isdir(snapshot_path):
            break
        print(f"Directory '{snap}' not found. Please try again.")

    # 3) prompt for number of top games
    default_count = 10
    while True:
        cnt_in = input(f"Number of top games per platform [{default_count}]: ").strip()
        if not cnt_in:
            count = default_count
            break
        try:
            count = int(cnt_in)
            break
        except ValueError:
            print("Please enter a valid integer.")

    # 4) prompt for fuzzy-match threshold
    default_threshold = 85.0
    while True:
        thr_in = input(f"Fuzzy-match score threshold (0-100) [{default_threshold}]: ").strip()
        if not thr_in:
            threshold = default_threshold
            break
        try:
            threshold = float(thr_in)
            break
        except ValueError:
            print("Please enter a valid number.")

    # 5) load CSV data
    summary_df   = pd.read_csv(os.path.join(snapshot_path, 'summary.csv'))
    unmatched_df = pd.read_csv(os.path.join(snapshot_path, 'unmatched_summary.csv'))
    # matched_df  = pd.read_csv(os.path.join(snapshot_path, 'matched_summary.csv'))  # if needed

    # initial platforms with zero ROMs
    zero_roms = summary_df[summary_df['ROMs'] == 0]['Platform'].tolist()
    active = set(zero_roms)
    print("Platforms flagged as empty:", ", ".join(sorted(active)))

    # interactive add/remove
    print("Enter platform codes to toggle inclusion (e.g. PS3), or 'run' to continue:")
    while True:
        ans = input('> ').strip()
        if ans.lower() == 'run':
            break
        code = ans
        if code in active:
            active.remove(code)
            print(f"Removed {code}")
        elif code in zero_roms:
            active.add(code)
            print(f"Added {code}")
        else:
            print(f"Unknown or ineligible: {code}")
    platforms = sorted(active)
    print(f"Running for platforms: {', '.join(platforms)}")

    rows = []
    links = []

    for platform_name in platforms:
        ds_code = INVERT_MAP.get(platform_name, platform_name)
        url = PLATFORM_URLS.get(ds_code)
        if not url:
            print(f"Skipping {platform_name}: no URL configured.")
            continue
        print(f"Processing {platform_name} ({ds_code})")

        # scrape zips
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        zips = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.zip')]
        file_map = {norm(os.path.splitext(unquote(h))[0]): h for h in zips}

        # filter unmatched with exclusions
        subset = unmatched_df[unmatched_df['Platform'] == ds_code]
        if ds_code in ('PS3', 'X360'):
            pc_names = set(unmatched_df[unmatched_df['Platform'] == 'PC']['Name'])
            subset = subset[~subset['Name'].isin(pc_names)]
        if ds_code == 'X360':
            ps3_names = set(unmatched_df[unmatched_df['Platform'] == 'PS3']['Name'])
            subset = subset[~subset['Name'].isin(ps3_names)]

        if subset.empty:
            print(f" No unmatched entries for {platform_name}.")
            continue
        # select top rows with sales
        top_df = subset.sort_values('Global_Sales', ascending=False).head(count)

        for _, row in top_df.iterrows():
            game = row['Name']
            sales = row['Global_Sales']
            normed = norm(game)
            ts_results = process.extract(normed, list(file_map.keys()), scorer=fuzz.token_sort_ratio, limit=None)
            if not ts_results:
                print(f"  [SKIP] {game} (no candidates)")
                continue
            max_ts = max(score for _, score, _ in ts_results)
            best_keys = [key for key, score, _ in ts_results if score == max_ts]
            best_key = max(best_keys, key=lambda k: region_priority(unquote(file_map[k])))
            pr_score = fuzz.partial_ratio(normed, best_key)
            score = min(max_ts, pr_score)
            if score < threshold:
                print(f"  [SKIP] {game} (score below threshold: {score})")
                continue
            href = file_map[best_key]
            download_url = urljoin(url, href)
            matched_title = unquote(href)

            rows.append({
                'Search_Term': game,
                'Platform': ds_code,
                'Matched_Title': matched_title,
                'Score': score,
                'Global_Sales': sales,
                'URL': download_url
            })
            links.append(download_url)
            print(f"  Matched {game} -> {matched_title} (score {score}, sales {sales})")

    # write outputs
    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(snapshot_path, 'download_list.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"Wrote download list CSV ({len(rows)} matches) to {csv_path}")

    txt_path = os.path.join(snapshot_path, 'download_links.txt')
    with open(txt_path, 'w') as f:
        f.write("\n".join(links))
    print(f"Wrote download links TXT with {len(links)} URLs to {txt_path}")

    print(f"Summary: {len(rows)} games matched across {len(platforms)} platforms.")

if __name__ == '__main__':
    main()
