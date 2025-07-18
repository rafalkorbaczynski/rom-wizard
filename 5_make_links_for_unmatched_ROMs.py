#!/usr/bin/env python3
"""
Generate a CSV of download URLs for the top N unmatched games.
Place this file in "..\RetroBat\scripts" alongside your snapshot folders.

Behavior changes:
 - Prompts user for snapshot directory name (with TAB-completion).
 - Guides user to input:
     • Number of top games per platform (default: 10)
     • Fuzzy-match score threshold (0-100; default: 85)
 - Reads CSVs from the specified snapshot directory:
     - summary.csv
     - unmatched_summary.csv
     - matched_summary.csv (optional, not used here)
 - Saves and updates `blacklist.csv` next to this script so it can be reused
   across multiple snapshots
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

# terminal colors
RESET = "\033[0m"
BOLD = "\033[1m"
HEADER_COLOR = "\033[95m"
DIFF_COLOR = "\033[93m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def progress_bar(current, total, width=20):
    if total == 0:
        return '[----------] 0/0'
    filled = int(width * current / total)
    return '[' + '#' * filled + '-' * (width - filled) + f'] {current}/{total}'

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

# --- Multi-disc helpers -----------------------------------------------------
DISC_RE = re.compile(r'(?:disc|disk|cd)\s*([0-9]+)', re.I)

def remove_disc(text: str) -> str:
    """Return string with disc information removed."""
    return re.sub(r'\s*[\[(]?(?:disc|disk|cd)\s*[0-9]+[^\])]*[\])]?', '', text, flags=re.I).strip()

def disc_number(text: str) -> int:
    match = DISC_RE.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0

from difflib import SequenceMatcher

def color_diff(candidate: str, search: str) -> str:
    """Return candidate string with differences highlighted."""
    sm = SequenceMatcher(None, candidate.lower(), search.lower())
    out = []
    for op, a1, a2, _, _ in sm.get_opcodes():
        segment = candidate[a1:a2]
        if op == 'equal':
            out.append(BOLD + segment + RESET)
        else:
            out.append(DIFF_COLOR + segment + RESET)
    return ''.join(out)

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
    '3DO':'3DO','3DS':'Nintendo 3DS','DS':'Nintendo DS','GB':'Game Boy','GBC':'Game Boy Color',
    'GBA':'Game Boy Advance','NES':'NES','SNES':'SNES','N64':'Nintendo 64','GC':'GameCube',
    'Wii':'Wii','WiiU':'Wii U','NS':'Switch','NGage':'N-Gage','VB':'Virtual Boy',
    'PS':'PlayStation','PS2':'PS2','PS3':'PS3','PS4':'PS4','PS5':'PS5','PSP':'PSP','PSV':'Vita',
    'SAT':'Saturn','SCD':'Sega CD','S32X':'32X','GG':'Game Gear','GEN':'Genesis','DC':'Dreamcast',
    'XB':'Xbox','X360':'360','XOne':'One','PC':'PC','MSD':'MS-DOS','WinP':'Windows',
    'Linux':'Linux','OSX':'macOS','PCFX':'PC-FX','TG16':'TurboGrafx-16','PCE':'PC Engine',
    'C64':'C64','C128':'C128','FDS':'FDS','MSX':'MSX','BBCM':'BBC Micro','AMIG':'Amiga','CD32':'CD32',
    'ACPC':'Amstrad CPC','ApII':'Apple II','Int':'Intellivision','Arc':'Arcadia','CDi':'CD-i'
}
INVERT_MAP = {v:k for k,v in READABLE_NAMES.items()}

# ——— Platform URL lookup ———————————————————————————————————————
PLATFORMS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'platforms.csv')

def load_platform_info():
    """Return mapping of platform code -> {'dir': directory, 'url': url}"""
    if not os.path.exists(PLATFORMS_CSV):
        return {}
    df = pd.read_csv(PLATFORMS_CSV)
    df.fillna('', inplace=True)
    info = {}
    for _, row in df.iterrows():
        info[row['Platform']] = {
            'dir': row['Directory'],
            'url': row['URL']
        }
    return info

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

    # blacklist lives next to the script so it persists across snapshots
    blacklist_path = os.path.join(RESULTS_DIR, 'blacklist.csv')
    if os.path.exists(blacklist_path):
        blacklist_df = pd.read_csv(blacklist_path)
        # ignore legacy columns that may be present
        blacklist_df = blacklist_df[['Search_Term', 'Platform']]
    else:
        blacklist_df = pd.DataFrame(columns=['Search_Term','Platform'])
    blacklisted_pairs = set(zip(blacklist_df.get('Search_Term', []), blacklist_df.get('Platform', [])))

    platform_info = load_platform_info()
    blacklisted = {
        code
        for code, info in platform_info.items()
        if str(info.get('url', '')).strip().upper() == 'BLACKLIST'
    }

    # available platform codes from unmatched CSV (excluding blacklisted)
    all_codes = sorted(
        c for c in unmatched_df['Platform'].unique() if c not in blacklisted
    )
    zero_names = summary_df[summary_df['ROMs'] == 0]['Platform'].tolist()
    zero_codes = sorted(
        INVERT_MAP.get(n, n) for n in zero_names
        if INVERT_MAP.get(n, n) not in blacklisted
    )

    print("Available platforms:")
    print(", ".join(all_codes))
    if zero_codes:
        print("Platforms with zero ROMs:", ", ".join(zero_codes))

    all_map = {c.lower(): c for c in all_codes}
    zero_map = {c.lower(): c for c in zero_codes}

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
            if low not in all_map and low not in zero_map:
                print(f"Unknown platform: {tok}")
                continue
            code = all_map.get(low) or zero_map.get(low)
            if code in active:
                active.remove(code)
                print(f"Removed {code}")
            else:
                active.add(code)
                print(f"Added {code}")
        if active:
            print("Currently selected:", ", ".join(sorted(active)))
        else:
            print("No platforms selected")
    platforms = sorted(active)
    print(f"Running for platforms: {', '.join(platforms)}")

    platform_targets = {}
    total_targets = 0
    if platforms:
        for platform_name in platforms:
            ds_code = INVERT_MAP.get(platform_name, platform_name)
            info = platform_info.get(ds_code, {})
            url = info.get('url', '')
            if not url:
                continue
            subset = unmatched_df[unmatched_df['Platform'] == ds_code]
            subset = subset[~subset['Name'].apply(lambda n: (n, ds_code) in blacklisted_pairs)]
            if ds_code in ('PS3', 'X360'):
                pc_names = set(unmatched_df[unmatched_df['Platform'] == 'PC']['Name'])
                subset = subset[~subset['Name'].isin(pc_names)]
            if ds_code == 'X360':
                ps3_names = set(unmatched_df[unmatched_df['Platform'] == 'PS3']['Name'])
                subset = subset[~subset['Name'].isin(ps3_names)]
            num = len(subset.sort_values('Global_Sales', ascending=False).head(count))
            platform_targets[ds_code] = num
            total_targets += num

    rows = []
    total_done = 0

    for platform_name in platforms:
        ds_code = INVERT_MAP.get(platform_name, platform_name)
        info = platform_info.get(ds_code, {})
        url = info.get('url', '')
        out_dir_name = info.get('dir', ds_code)
        if not url:
            print(f"Skipping {platform_name}: no URL configured.")
            continue
        print(f"Processing {platform_name} ({ds_code})")

        # scrape zips
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        zips = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.zip')]
        file_map = {}
        file_names = {}
        group_map = {}
        for h in zips:
            name = unquote(os.path.splitext(h)[0])
            key = norm(name)
            file_map[key] = h
            file_names[key] = name
            base_key = norm(remove_disc(name))
            group_map.setdefault(base_key, []).append(key)

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

        platform_total = platform_targets.get(ds_code, len(top_df))
        platform_done = 0

        for _, row in top_df.iterrows():
            platform_done += 1
            total_done += 1
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
                search_term = game
                norm_search = norm(search_term)
                best_keys = None
                while True:
                    clear_screen()
                    print(f"Platform {ds_code}: {progress_bar(platform_done, platform_total)} | Total: {progress_bar(total_done, total_targets)}")
                    print(f"{HEADER_COLOR}{BOLD}Options for {game}:{RESET}")
                    if search_term != game:
                        print(f"Searching for '{search_term}'")
                    options = []
                    ts_results = process.extract(norm_search, list(file_map.keys()), scorer=fuzz.token_sort_ratio, limit=None)
                    if not ts_results:
                        print("  No candidates found")
                    else:
                        groups = {}
                        for cand, ts_score, _ in ts_results:
                            base_key = norm(remove_disc(file_names[cand]))
                            g = groups.setdefault(base_key, {'keys': [], 'score': 0})
                            g['keys'].append(cand)
                            g['score'] = max(g['score'], ts_score)
                        sorted_groups = sorted(groups.values(), key=lambda g: g['score'], reverse=True)
                        for idx, g in enumerate(sorted_groups[:3], start=1):
                            main_key = max(g['keys'], key=lambda k: fuzz.partial_ratio(norm_search, k))
                            pr_score = fuzz.partial_ratio(norm_search, main_key)
                            opt_score = int(round(min(g['score'], pr_score)))
                            disp = ' + '.join(file_names[k] for k in sorted(g['keys'], key=lambda k: disc_number(file_names[k])))
                            disp = color_diff(disp, search_term)
                            options.append((g['keys'], opt_score))
                            print(f"  [{idx}] {disp} (score {opt_score})")
                    choice = input("Press 1-3, 'b' to blacklist, or ENTER to skip: ").strip()
                    if choice.lower() == 'b':
                        blacklist_df.loc[len(blacklist_df)] = {
                            'Search_Term': game,
                            'Platform': ds_code
                        }
                        blacklisted_pairs.add((game, ds_code))
                        print(f"  Blacklisted {game}")
                        best_keys = None
                        break
                    if choice in {'1','2','3'}:
                        idx_choice = int(choice) - 1
                        if idx_choice < len(options):
                            best_keys, score = options[idx_choice]
                            if score < threshold:
                                print(f"  Selected option below threshold ({score} < {threshold})")
                        break
                    if choice == '':
                        best_keys = None
                        break
                    # treat input as new search term
                    search_term = choice
                    norm_search = norm(search_term)
                    continue
                if not best_keys:
                    continue
                selected_keys = best_keys
                # If a disc was chosen, gather any additional discs for the same
                # game so they can all be stored as separate rows.
                selected_names = [file_names[k] for k in selected_keys]
                if any(DISC_RE.search(n) for n in selected_names):
                    base = norm(remove_disc(selected_names[0]))
                    all_keys = group_map.get(base, selected_keys)
                    if len(all_keys) > len(selected_keys):
                        # sort by disc number to keep a stable order
                        selected_keys = sorted(all_keys, key=lambda k: disc_number(file_names[k]))
            else:
                max_ts = max(score for _, score, _ in ts_results)
                keys_at_max = [key for key, score, _ in ts_results if score == max_ts]
                best_key = max(keys_at_max, key=lambda k: region_priority(unquote(file_map[k])))
                pr_score = fuzz.partial_ratio(normed, best_key)
                score = int(round(min(max_ts, pr_score)))
                if score < threshold:
                    print(f"  [SKIP] {game} (score below threshold: {score})")
                    continue
                selected_keys = group_map.get(norm(remove_disc(file_names[best_key])), [best_key])

            # remove duplicate disc entries while preserving order
            uniq_keys = []
            seen = set()
            for k in selected_keys:
                if k not in seen:
                    uniq_keys.append(k)
                    seen.add(k)

            for k in uniq_keys:
                href = file_map[k]
                download_url = urljoin(url, href)
                matched_title = unquote(href)

                rows.append({
                    'Search_Term': game,
                    'Platform': ds_code,
                    'Directory': out_dir_name,
                    'Matched_Title': matched_title,
                    'Score': score,
                    'Global_Sales': sales,
                    'URL': download_url
                })
                print(f"  Matched {game} -> {matched_title} (score {score}, sales {sales})")

    # write outputs
    df_out = pd.DataFrame(rows)
    df_out.fillna('', inplace=True)
    csv_path = os.path.join(snapshot_path, 'download_list.csv')
    df_out.to_csv(csv_path, index=False)
    print(f"Wrote download list CSV ({len(rows)} matches) to {csv_path}")

    if not blacklist_df.empty:
        blacklist_df.to_csv(blacklist_path, index=False)
        print(f"Updated blacklist written to {blacklist_path}")

    print(f"Summary: {len(rows)} games matched across {len(platforms)} platforms.")

if __name__ == '__main__':
    main()
