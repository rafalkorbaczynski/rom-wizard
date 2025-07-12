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
import sys
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

def read_single_key(prompt=''):
    """Read a single keypress without requiring ENTER."""
    print(prompt, end='', flush=True)
    try:
        import termios, tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except (ImportError, AttributeError):  # Windows fallback
        import msvcrt
        ch = msvcrt.getch()
        if isinstance(ch, bytes):
            ch = ch.decode('utf-8', errors='ignore')
    print(ch)
    return ch

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

# ——— Platform URL lookup ———————————————————————————————————————
PLATFORMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'platforms.csv')

def load_platform_urls():
    if not os.path.exists(PLATFORMS_CSV):
        return {}
    df = pd.read_csv(PLATFORMS_CSV)
    df['URL'] = df['URL'].fillna('')
    return dict(zip(df['Platform'], df['URL']))

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

    manual_mode = input("Enable manual mode for matching? [y/N]: ").strip().lower() == 'y'

    # 5) load CSV data
    summary_df   = pd.read_csv(os.path.join(snapshot_path, 'summary.csv'))
    unmatched_df = pd.read_csv(os.path.join(snapshot_path, 'unmatched_summary.csv'))
    # matched_df  = pd.read_csv(os.path.join(snapshot_path, 'matched_summary.csv'))  # if needed

    blacklist_path = os.path.join(snapshot_path, 'blacklist.csv')
    if os.path.exists(blacklist_path):
        blacklist_df = pd.read_csv(blacklist_path)
        # ignore legacy columns that may be present
        blacklist_df = blacklist_df[['Search_Term', 'Platform']]
    else:
        blacklist_df = pd.DataFrame(columns=['Search_Term','Platform'])
    blacklisted_pairs = set(zip(blacklist_df.get('Search_Term', []), blacklist_df.get('Platform', [])))

    # available platform codes from unmatched CSV
    all_codes = sorted(unmatched_df['Platform'].unique())
    zero_names = summary_df[summary_df['ROMs'] == 0]['Platform'].tolist()
    zero_codes = sorted(INVERT_MAP.get(n, n) for n in zero_names)

    print("Available platforms:")
    print(", ".join(all_codes))
    if zero_codes:
        print("Platforms with zero ROMs:", ", ".join(zero_codes))

    active = set()
    print("Enter platform codes separated by spaces.\n"
          "Use 'empty' to add empty platforms, 'all' to toggle all platforms,\n"
          "or 'run' to continue:")
    while True:
        ans = input('> ').strip()
        if ans.lower() == 'run':
            break
        tokens = re.split(r'[\s,]+', ans)
        for tok in filter(None, tokens):
            low = tok.lower()
            if low == 'empty':
                active.update(zero_codes)
                continue
            if low == 'all':
                if active >= set(all_codes):
                    active.clear()
                    continue
                active.update(all_codes)
                continue
            if tok not in all_codes and tok not in zero_codes:
                print(f"Unknown platform: {tok}")
                continue
            if tok in active:
                active.remove(tok)
                print(f"Removed {tok}")
            else:
                active.add(tok)
                print(f"Added {tok}")
        if active:
            print("Currently selected:", ", ".join(sorted(active)))
        else:
            print("No platforms selected")
    platforms = sorted(active)
    print(f"Running for platforms: {', '.join(platforms)}")

    rows = []
    links = []

    platform_urls = load_platform_urls()

    for platform_name in platforms:
        ds_code = INVERT_MAP.get(platform_name, platform_name)
        url = platform_urls.get(ds_code)
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
        subset = subset[~subset['Name'].apply(lambda n: (n, ds_code) in blacklisted_pairs)]
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
            if (game, ds_code) in blacklisted_pairs:
                continue
            normed = norm(game)
            ts_results = process.extract(normed, list(file_map.keys()), scorer=fuzz.token_sort_ratio, limit=None)
            if not ts_results:
                print(f"  [SKIP] {game} (no candidates)")
                continue

            if manual_mode:
                options = []
                print(f"Options for {game}:")
                for idx, (cand, ts_score, _) in enumerate(ts_results[:3], start=1):
                    pr_score = fuzz.partial_ratio(normed, cand)
                    opt_score = min(ts_score, pr_score)
                    options.append((cand, opt_score))
                    print(f"  [{idx}] {unquote(file_map[cand])} (score {opt_score})")
                choice = read_single_key("Press 1-3, 'b' to blacklist, or ENTER to skip: ").lower()
                if choice == 'b':
                    blacklist_df.loc[len(blacklist_df)] = {
                        'Search_Term': game,
                        'Platform': ds_code
                    }
                    blacklisted_pairs.add((game, ds_code))
                    print(f"  Blacklisted {game}")
                    continue
                if choice in {'1','2','3'}:
                    idx = int(choice) - 1
                    if idx >= len(options):
                        continue
                    best_key, score = options[idx]
                    if score < threshold:
                        print(f"  [SKIP] {game} (score below threshold: {score})")
                        continue
                else:
                    continue
            else:
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

    if not blacklist_df.empty:
        blacklist_df.to_csv(blacklist_path, index=False)
        print(f"Updated blacklist written to {blacklist_path}")

    print(f"Summary: {len(rows)} games matched across {len(platforms)} platforms.")

if __name__ == '__main__':
    main()
