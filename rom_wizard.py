import os
import sys
import csv
import shutil
import datetime
import pandas as pd
import xml.etree.ElementTree as ET
from rapidfuzz import process, fuzz
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import subprocess
from rich.console import Console
from rich.table import Table
from rich.progress import track

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'wizardry')
SALES_CSV = os.path.join(DATA_DIR, 'sales_2019.csv')
PLATFORMS_CSV = os.path.join(DATA_DIR, 'platforms.csv')
ARIA2 = os.path.join(DATA_DIR, 'aria2c.exe')
CHDMAN = os.path.join(DATA_DIR, 'chdman.exe')
# The ROM library lives in the parent directory of this script under "roms".
# Make ROMS_ROOT point there so the wizard can locate existing ROMs when run
# from the scripts_github folder.
ROMS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'roms'))

_ROMAN = {'ix':9,'viii':8,'vii':7,'vi':6,'iv':4,'iii':3,'ii':2,'i':1}
import re
import unicodedata
_ROMAN_RE = re.compile(r'\b(' + '|'.join(sorted(_ROMAN, key=len, reverse=True)) + r')\b')

# Region mapping for summary statistics
REGION_SYNONYMS = {
    'EU': ['eu', 'europe', 'eur', 'pal', 'uk'],
    'US': ['us', 'usa', 'ntsc', 'ntsc-u'],
    'JP': ['jp', 'jpn', 'japan', 'ntsc-j'],
    'World': ['world']
}
REGIONS = list(REGION_SYNONYMS.keys()) + ['Other']

# Extensions considered "ROM" files for summary statistics and duplicate
# detection.  Metadata like videos or gamelist files should be ignored.
ROM_EXTS = {
    'a26','a52','a78','bin','chd','gb','gba','gbc','iso','j64',
    'md','nds','nes','pce','rvz','sfc','sms','xex','z64','zip'
}

# Terminal colors for highlighting differences
RESET = "\033[0m"
BOLD = "\033[1m"
DIFF_COLOR = "\033[93m"
HEADER_COLOR = "\033[95m"

# Track sort cycling state for snapshot summaries
SUMMARY_CYCLE_IDX = 0
SUMMARY_COLS: list[str] = []

# Path to persistent blacklist used for manual matching
BLACKLIST_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'blacklist.csv')

# Directory aliases that map alternate ROM folders to platform codes
DIR_ALIASES = {
    'ds': 'DS',
    'psx': 'PS',
    'dc': 'DC',
    'ns': 'NS',
    'sega_cd': 'SCD',
}


def load_platform_mappings():
    df = pd.read_csv(PLATFORMS_CSV)
    df['ignore'] = df.get('ignore', False).astype(str).str.upper().isin(['TRUE', '1', 'YES'])

    # Ensure every directory under ROMS_ROOT with a gamelist.xml exists in platforms.csv
    if os.path.isdir(ROMS_ROOT):
        rom_dirs = {
            d for d in os.listdir(ROMS_ROOT)
            if os.path.isdir(os.path.join(ROMS_ROOT, d))
            and os.path.isfile(os.path.join(ROMS_ROOT, d, 'gamelist.xml'))
        }
        known_dirs = set(df['Directory'].str.lower())
        missing = sorted(rom_dirs - known_dirs)
        if missing:
            add_rows = pd.DataFrame(
                [
                    {
                        'Platform': m,
                        'Directory': m,
                        'URL': '',
                        'ignore': False,
                    }
                    for m in missing
                ]
            )
            df = pd.concat([df, add_rows], ignore_index=True)
            df.to_csv(PLATFORMS_CSV, index=False)

    ignore_set = {row['Platform'] for _, row in df[df['ignore']].iterrows()}
    active = df[~df['ignore']]
    plat_map = {row['Directory'].lower(): row['Platform'] for _, row in active.iterrows()}
    dataset_to_console = {row['Platform'].lower(): row['Directory'] for _, row in active.iterrows()}
    for alias, code in DIR_ALIASES.items():
        if code.lower() in dataset_to_console:
            plat_map.setdefault(alias, code)
    return plat_map, dataset_to_console, {p.upper() for p in ignore_set}


PLAT_MAP, DATASET_TO_CONSOLE, IGNORED_PLATFORMS = load_platform_mappings()

# Rich console for nicer output
console = Console()


def shorten_path(path: str, depth: int = 2) -> str:
    """Return ``path`` showing only the last ``depth`` directories and filename."""
    s = str(path)
    if "/" not in s and "\\" not in s:
        return s
    s = s.replace("\\", "/")
    parts = [p for p in s.split("/") if p]
    if len(parts) <= depth + 1:
        return "/" + "/".join(parts)
    return "/" + "/".join(parts[-(depth + 1):])


def print_table(df: pd.DataFrame, sorted_col: str | None = None) -> None:
    """Pretty-print a DataFrame using rich.

    A ``Total`` row is appended and certain platforms are filtered out.  If
    ``sorted_col`` is provided, the corresponding header is marked with an
    ``↑`` arrow to indicate the active sort."""
    if df.empty:
        console.print("[bold red]No data available.[/]")
        return

    skip_plats = {p.lower() for p in IGNORED_PLATFORMS}
    if "Platform" in df.columns:
        df = df[~df["Platform"].str.lower().isin(skip_plats)]

    if df.empty:
        console.print("[bold red]No data available.[/]")
        return

    total_row = {}
    for i, col in enumerate(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            if col == "ROMs%" and {"Matched ROMs", "ROMs"}.issubset(df.columns):
                roms_total = df["ROMs"].sum()
                matched_total = df["Matched ROMs"].sum()
                pct = matched_total / roms_total * 100 if roms_total else 0
                total_row[col] = round(pct, 1)
            else:
                total_row[col] = df[col].sum()
        elif i == 0:
            total_row[col] = "Total"
        else:
            total_row[col] = ""

    df_disp = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    table = Table(show_header=True, header_style="bold magenta")
    for col in df_disp.columns:
        header = f"{col}"
        if sorted_col and col == sorted_col:
            header += " \u2191"
        table.add_column(header)
    for _, row in df_disp.iterrows():
        display = []
        for v in row:
            display.append(shorten_path(v))
        table.add_row(*display)
    console.print(table)


def iter_progress(seq, description: str):
    """Iterate with a progress bar if output is a TTY."""
    return track(seq, description=description) if sys.stdout.isatty() else seq


def get_region_key(game) -> str:
    """Return normalized region key for a gamelist entry."""
    region_text = (game.findtext('region') or '').lower().strip()
    if region_text:
        for reg, toks in REGION_SYNONYMS.items():
            if region_text in toks:
                return reg
    text = os.path.basename(game.findtext('path') or game.findtext('name') or '').lower()
    tags = re.findall(r'\(([^()]+)\)', text)
    for tag in tags:
        for token in re.split('[,;/]', tag):
            tok = token.strip().lower()
            for reg, toks in REGION_SYNONYMS.items():
                if tok in toks:
                    return reg
    return 'Other'


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Prompt the user for a yes/no answer."""
    ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not ans:
        return default
    return ans in {'y', 'yes'}

def norm(text):
    s = unicodedata.normalize('NFKD', str(text))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r'[\(\[].*?[\)\]]', '', s)
    s = re.sub(r'^the\s+', '', s)
    s = s.replace('&', ' and ')
    s = _ROMAN_RE.sub(lambda m: str(_ROMAN[m.group(1)]), s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def token_set(text: str) -> set[str]:
    tokens = text.split()
    while tokens and tokens[0].isdigit():
        tokens.pop(0)
    return set(tokens)


def region_priority(filename: str) -> int:
    lf = filename.lower()
    if re.search(r'\b(eu|eur|europe)\b', lf):
        return 3
    if re.search(r'\b(us|usa|ntsc-?u)\b', lf):
        return 2
    if re.search(r'\b(jp|japan|ntsc-?j)\b', lf):
        return 1
    return 0


def filter_best_region(keys, file_names):
    """Return keys belonging to the highest-priority region."""
    if not keys:
        return keys
    max_prio = max(region_priority(file_names[k]) for k in keys)
    return [k for k in keys if region_priority(file_names[k]) == max_prio]


def color_diff(candidate: str, search: str) -> str:
    sm = SequenceMatcher(None, candidate.lower(), search.lower())
    out = []
    for op, a1, a2, _, _ in sm.get_opcodes():
        segment = candidate[a1:a2]
        if op == 'equal':
            out.append(BOLD + segment + RESET)
        else:
            out.append(DIFF_COLOR + segment + RESET)
    return ''.join(out)

def filter_wii_no_controller(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Wii rows without controller support."""
    sales = pd.read_csv(SALES_CSV, usecols=['Name','Platform','controller_support'])
    mask = (sales['Platform'].str.lower() == 'wii') & (
        sales['controller_support'].astype(str).str.lower() == 'false')
    bad_names = set(sales.loc[mask, 'Name'])
    return df[~((df['Platform'].str.upper() == 'WII') & df['Dataset Name'].isin(bad_names))]


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def progress_bar(current, total, width=20):
    if total == 0:
        return '[----------] 0/0'
    filled = int(width * current / total)
    return '[' + '#' * filled + '-' * (width - filled) + f'] {current}/{total}'

# --- Multi-disc helpers -----------------------------------------------------
DISC_RE = re.compile(r'(?:disc|disk|cd)\s*([0-9]+)', re.I)

def remove_disc(text: str) -> str:
    """Return string with disc information removed."""
    return re.sub(r'\s*[\[(]?(?:disc|disk|cd)\s*[0-9]+[^\])]*[\])]?', '', text,
                  flags=re.I).strip()

def disc_number(text: str) -> int:
    match = DISC_RE.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return 0
    return 0


def ask_threshold(default=90):
    try:
        ans = input(f"Fuzzy match threshold [0-100] (default {default}): ").strip()
        if not ans:
            return default
        val = int(ans)
        if 0 <= val <= 100:
            return val
    except ValueError:
        pass
    print('Using default threshold', default)
    return default


def write_rom_filenames(root_dir: str, output_path: str) -> None:
    """Write a sorted list of ROM filenames to ``output_path``.

    The extensions mirror those supported by ``2_relocate_duplicate_ROMs.py``
    and the former ``list_rom_filenames.py`` helper script."""
    allowed = {
        'a26','a52','a78','bin','chd','gb','gba','gbc','iso','j64',
        'md','mp4','nds','nes','pce','rvz','sfc','sms','xex','xml','z64','zip'
    }

    filenames = []
    for dirpath, _, files in os.walk(root_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower().lstrip('.')
            if ext in allowed:
                rel = os.path.relpath(os.path.join(dirpath, fname), root_dir)
                filenames.append(rel)

    filenames.sort(key=str.lower)
    with open(output_path, 'w', encoding='utf-8') as f:
        for name in filenames:
            f.write(name + '\n')



def create_snapshot():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    snap_dir = os.path.join(os.path.dirname(__file__), f'snapshot_{ts}')
    os.makedirs(os.path.join(snap_dir, 'roms'), exist_ok=True)
    os.makedirs(os.path.join(snap_dir, 'duplicate_roms'), exist_ok=True)

    threshold = ask_threshold()
    sales = pd.read_csv(SALES_CSV, low_memory=False)
    sales = sales[~sales['Platform'].isin(IGNORED_PLATFORMS)]
    sales['key'] = sales['Name'].apply(norm)
    max_sales = sales['Global_Sales'].max()

    summary_rows = []
    matched_records = []
    unmatched_keys = set(zip(sales['Platform'], sales['key']))
    found_consoles = set()

    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Scanning ROMs"):
        gl_path = os.path.join(ROMS_ROOT, console_name, 'gamelist.xml')
        if not os.path.isfile(gl_path):
            continue
        ds_plat = PLAT_MAP.get(console_name.lower())
        if not ds_plat or ds_plat in IGNORED_PLATFORMS:
            continue
        found_consoles.add(console_name.lower())
        tree = ET.parse(gl_path)
        games = [g for g in tree.getroot().findall('game')
                 if os.path.splitext(g.findtext('path') or '')[1].lower().lstrip('.') in ROM_EXTS]
        if not games:
            continue
        platform = console_name

        sales_subset = sales[sales['Platform'].str.lower() == ds_plat.lower()]
        match_keys = list(sales_subset['key'])
        sales_map = dict(zip(sales_subset['key'], sales_subset['Global_Sales']))
        name_map = dict(zip(sales_subset['key'], sales_subset['Name']))
        dataset_size = sales_subset['key'].nunique()

        region_counts = {r: 0 for r in REGIONS}
        matched = 0

        for g in games:
            title = g.findtext('name') or ''
            k = norm(title)

            region_key = get_region_key(g)
            region_counts[region_key] += 1

            res = process.extractOne(k, match_keys, scorer=fuzz.token_sort_ratio)
            if res and res[1] >= threshold and token_set(k) == token_set(res[0]):
                matched += 1
                key = res[0]
                matched_records.append({
                    'Dataset Name': name_map[key],
                    'Platform': platform,
                    'ROM': title,
                    'Sales': sales_map[key],
                    'Match Score': res[1]
                })
                unmatched_keys.discard((ds_plat, key))

        summary_rows.append({
            'Platform': platform,
            'ROMs': len(games),
            'Dataset': dataset_size,
            'Matched ROMs': matched,
            **region_counts
        })

    # Add dataset platforms missing from the ROM library
    all_codes = sales['Platform'].dropna().unique()
    for code in all_codes:
        mapped = DATASET_TO_CONSOLE.get(code.lower(), code)
        if mapped.lower() in found_consoles:
            continue
        size = sales[sales['Platform'].str.lower() == code.lower()]['key'].nunique()
        summary_rows.append({'Platform': mapped, 'ROMs': 0, 'Dataset': size,
                             'Matched ROMs': 0, **{r: 0 for r in REGIONS}})

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df['ROMs%'] = (summary_df['Matched ROMs'] / summary_df['ROMs'].replace(0,1) * 100)
        for col in summary_df.select_dtypes(include='float'):
            summary_df[col] = summary_df[col].round(1)
    summary_csv = os.path.join(snap_dir, 'summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    unmatched_list = [{'Platform': p, 'key': k} for p, k in unmatched_keys]
    unmatched_df = pd.DataFrame(unmatched_list)
    unmatched_df = unmatched_df.merge(sales[['Platform','Name','Global_Sales','key']], on=['Platform','key'], how='left')
    unmatched_df = unmatched_df.rename(columns={'Name':'Dataset Name','Global_Sales':'Sales'})
    unmatched_df = unmatched_df[['Dataset Name','Platform','Sales']]
    unmatched_csv = os.path.join(snap_dir, 'unmatched_summary.csv')
    unmatched_df.to_csv(unmatched_csv, index=False)

    match_df = pd.DataFrame(matched_records, columns=['Dataset Name','Platform','ROM','Sales','Match Score'])
    match_df.to_csv(os.path.join(snap_dir, 'match_summary.csv'), index=False)

    # Write list of ROM filenames for analysis
    write_rom_filenames(ROMS_ROOT, os.path.join(snap_dir, 'rom_filenames.txt'))

    console.print(f"[bold green]Snapshot created at {shorten_path(snap_dir)}[/]")
    print_table(summary_df)
    return snap_dir


def detect_duplicates(snapshot_dir):
    from rapidfuzz import fuzz
    threshold = ask_threshold()
    dup_root = os.path.join(snapshot_dir, 'duplicate_roms')
    os.makedirs(dup_root, exist_ok=True)
    report_rows = []
    summary_rows = []

    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Scanning ROMs"):
        console_dir = os.path.join(ROMS_ROOT, console_name)
        if not os.path.isdir(console_dir):
            continue
        code = PLAT_MAP.get(console_name.lower())
        if code in IGNORED_PLATFORMS:
            continue
        rom_count = 0
        dup_count = 0
        for root, _, files in os.walk(console_dir):
            files = [f for f in files if os.path.splitext(f)[1].lower().lstrip('.') in ROM_EXTS and 'disc' not in f.lower()]
            rom_count += len(files)
            norm_map = {f: norm(f) for f in files}
            base_map = {f: norm(remove_disc(f)) for f in files}
            disc_map = {f: disc_number(f) for f in files}
            # Break filenames into tokens and drop any leading numbers which are
            # often used by ROM sets for simple indexing (e.g. ``"18. Super"``).
            #
            # These numeric prefixes shouldn't cause a mismatch when detecting
            # duplicates, but numeric tokens appearing later in the name are
            # significant (e.g. "Super Mario Bros. 2" vs "Super Mario Bros. 3").
            token_map = {}
            token_no_num = {}
            num_map = {}
            for f in files:
                tokens = norm_map[f].split()
                # Drop catalog numbers at the beginning of the filename
                while tokens and tokens[0].isdigit():
                    tokens.pop(0)
                token_map[f] = tokens
                token_no_num[f] = [t for t in tokens if not t.isdigit()]
                num_map[f] = [t for t in tokens if t.isdigit()]
            moved = set()
            for i, f in enumerate(files):
                if f in moved:
                    continue
                for g in files[i+1:]:
                    if g in moved:
                        continue
                    score = fuzz.token_set_ratio(norm_map[f], norm_map[g])
                    if score >= threshold:
                        # Skip sequels with different numbering
                        if num_map[f] != num_map[g] and (num_map[f] or num_map[g]):
                            continue
                        # Skip titles with different word sets (likely different games)
                        if set(token_no_num[f]) != set(token_no_num[g]):
                            continue
                        # Skip multi-disc sets
                        if base_map[f] == base_map[g] and disc_map[f] != disc_map[g] and (disc_map[f] or disc_map[g]):
                            continue
                        src = os.path.join(root, g)
                        rel_dir = os.path.relpath(root, ROMS_ROOT)
                        dst_dir = os.path.join(dup_root, rel_dir)
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.move(src, os.path.join(dst_dir, g))
                        keep_rel = os.path.relpath(os.path.join(root, f), ROMS_ROOT)
                        move_rel = os.path.relpath(src, ROMS_ROOT)
                        report_rows.append([console_name, score, norm_map[f], norm_map[g], keep_rel, move_rel])
                        moved.add(g)
                        dup_count += 1
        if rom_count == 0:
            continue
        pct = dup_count / rom_count * 100
        summary_rows.append({'Platform': console_name, 'ROMs': rom_count, 'Duplicates': dup_count, 'Duplicate %': round(pct, 1)})
    csv_path = os.path.join(snapshot_dir, 'duplicates_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Platform','Match Score','Kept Key','Moved Key','Kept File','Moved File'])
        writer.writerows(report_rows)
    console.print('[bold green]Duplicate scan complete.[/]')
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print_table(df)


def generate_playlists(snapshot_dir):
    """Create m3u playlists in the snapshot's roms folder."""
    pattern = re.compile(r'(?i)^(.+?)\s*\(disc\s*(\d+)\)')
    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Generating playlists"):
        folder = os.path.join(ROMS_ROOT, console_name)
        if not os.path.isdir(folder):
            continue
        code = PLAT_MAP.get(console_name.lower())
        if code in IGNORED_PLATFORMS:
            continue
        rom_files = [f for f in os.listdir(folder)
                     if os.path.splitext(f)[1].lower().lstrip('.') in ROM_EXTS]
        if not rom_files:
            continue
        groups = {}
        for fname in rom_files:
            if 'disc' not in fname.lower():
                continue
            m = pattern.match(fname)
            if not m:
                continue
            base = m.group(1).strip()
            num = int(m.group(2))
            groups.setdefault(base, []).append((num, fname))
        for base, discs in groups.items():
            if len(discs) < 2:
                continue
            discs.sort()
            out_folder = os.path.join(snapshot_dir, 'roms', console_name)
            os.makedirs(out_folder, exist_ok=True)
            path = os.path.join(out_folder, base + '.m3u')
            with open(path, 'w', encoding='utf-8') as f:
                for _,name in discs:
                    f.write(name+'\n')
            print('Playlist created:', shorten_path(path))


def apply_sales(snapshot_dir):
    threshold = ask_threshold()
    sales = pd.read_csv(SALES_CSV, low_memory=False)
    sales = sales[~sales['Platform'].isin(IGNORED_PLATFORMS)]
    sales['key'] = sales['Name'].apply(norm)
    max_sales = sales['Global_Sales'].max()
    unmatched_keys = set(zip(sales['Platform'], sales['key']))
    output_root = os.path.join(snapshot_dir, 'roms')
    os.makedirs(output_root, exist_ok=True)
    match_rows = []
    summary_rows = []
    found_consoles = set()

    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Applying sales data"):
        gl_path = os.path.join(ROMS_ROOT, console_name, 'gamelist.xml')
        if not os.path.isfile(gl_path):
            continue
        ds_plat = PLAT_MAP.get(console_name.lower())
        if not ds_plat or ds_plat in IGNORED_PLATFORMS:
            continue
        found_consoles.add(console_name.lower())
        tree = ET.parse(gl_path)
        root = tree.getroot()
        games = [g for g in root.findall('game')
                 if os.path.splitext(g.findtext('path') or '')[1].lower().lstrip('.') in ROM_EXTS]
        if not games:
            continue

        subset = sales[sales['Platform'].str.lower() == ds_plat.lower()]
        match_keys = list(subset['key'])
        sales_map = dict(zip(subset['key'], subset['Global_Sales']))
        name_map = dict(zip(subset['key'], subset['Name']))
        dataset_size = subset['key'].nunique()

        region_counts = {r: 0 for r in REGIONS}
        matched = 0

        for g in games:
            for tag in g.findall('rating') + g.findall('ratingMax'):
                g.remove(tag)

            title = g.findtext('name') or ''
            k = norm(title)

            region_key = get_region_key(g)
            region_counts[region_key] += 1

            res = process.extractOne(k, match_keys, scorer=fuzz.token_sort_ratio)
            if res and res[1] >= threshold and token_set(k) == token_set(res[0]):
                key = res[0]
                gs = sales_map[key]
                rating = gs / max_sales * 100
                ET.SubElement(g, 'rating').text = f"{rating:.2f}"
                ET.SubElement(g, 'ratingMax').text = '100'
                match_rows.append({
                    'Dataset Name': name_map[key],
                    'Platform': console_name,
                    'ROM': title,
                    'Sales': gs,
                    'Match Score': res[1]
                })
                matched += 1
                unmatched_keys.discard((ds_plat, key))

        out_dir = os.path.join(output_root, console_name)
        os.makedirs(out_dir, exist_ok=True)
        tree.write(os.path.join(out_dir, 'gamelist.xml'), encoding='utf-8', xml_declaration=True)
        summary_rows.append({
            'Platform': console_name,
            'ROMs': len(games),
            'Dataset': dataset_size,
            'Matched ROMs': matched,
            **region_counts
        })

    # Add dataset platforms missing from the ROM library
    all_codes = sales['Platform'].dropna().unique()
    for code in all_codes:
        mapped = DATASET_TO_CONSOLE.get(code.lower(), code)
        if mapped.lower() in found_consoles:
            continue
        size = sales[sales['Platform'].str.lower() == code.lower()]['key'].nunique()
        summary_rows.append({'Platform': mapped, 'ROMs': 0, 'Dataset': size,
                             'Matched ROMs': 0, **{r: 0 for r in REGIONS}})

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df['ROMs%'] = (summary_df['Matched ROMs'] / summary_df['ROMs'].replace(0,1) * 100)
        for col in summary_df.select_dtypes(include='float'):
            summary_df[col] = summary_df[col].round(1)
    summary_df.to_csv(os.path.join(snapshot_dir, 'summary.csv'), index=False)
    pd.DataFrame(match_rows).to_csv(os.path.join(snapshot_dir, 'match_summary.csv'), index=False)

    unmatched_list = [{'Platform': p, 'key': k} for p, k in unmatched_keys]
    unmatched_df = pd.DataFrame(unmatched_list)
    unmatched_df = unmatched_df.merge(sales[['Platform','Name','Global_Sales','key']], on=['Platform','key'], how='left')
    unmatched_df = unmatched_df.rename(columns={'Name':'Dataset Name','Global_Sales':'Sales'})
    unmatched_df = unmatched_df[['Dataset Name','Platform','Sales']]
    unmatched_df.to_csv(os.path.join(snapshot_dir, 'unmatched_summary.csv'), index=False)

    console.print('[bold green]Sales data applied.[/]')
    print_table(summary_df)


def scrape_file_list(code):
    df = pd.read_csv(PLATFORMS_CSV)
    df['ignore'] = df.get('ignore', False).astype(str).str.upper().isin(['TRUE', '1', 'YES'])
    row = df[(df['Platform'] == code) & (~df['ignore'])]
    if row.empty:
        return None, None, []
    url = row['URL'].iloc[0]
    directory = row['Directory'].iloc[0] or code
    if not url:
        return None, directory, []
    r = requests.get(url)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    files = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.zip')]
    names = [requests.utils.unquote(os.path.splitext(f)[0]) for f in files]
    return url, directory, list(zip(names, files))


def manual_add_games(snapshot_dir):
    """Interactive manual mode for adding download entries."""
    unmatched_path = os.path.join(snapshot_dir, 'unmatched_summary.csv')
    summary_path = os.path.join(snapshot_dir, 'summary.csv')
    if not os.path.isfile(unmatched_path):
        print('unmatched_summary.csv not found. Run snapshot first.')
        return

    unmatched_df = pd.read_csv(unmatched_path)
    summary_df = pd.read_csv(summary_path) if os.path.isfile(summary_path) else pd.DataFrame()

    if os.path.exists(BLACKLIST_CSV):
        blacklist_df = pd.read_csv(BLACKLIST_CSV)
        blacklist_df = blacklist_df[['Search_Term', 'Platform']]
    else:
        blacklist_df = pd.DataFrame(columns=['Search_Term', 'Platform'])
    blacklisted_pairs = set(zip(blacklist_df.get('Search_Term', []), blacklist_df.get('Platform', [])))

    def canonical(code: str) -> str:
        """Return canonical, upper-case platform code."""
        if not isinstance(code, str):
            return ''
        return PLAT_MAP.get(code.lower(), code).upper()

    # Normalize platform codes for consistent comparisons
    unmatched_df['Platform'] = unmatched_df['Platform'].apply(canonical)
    if not summary_df.empty and 'Platform' in summary_df.columns:
        summary_df['Platform'] = summary_df['Platform'].apply(canonical)
    if prompt_yes_no('Exclude Wii games without controller support?'):
        unmatched_df = filter_wii_no_controller(unmatched_df)

    plat_df = pd.read_csv(PLATFORMS_CSV)
    plat_df['ignore'] = plat_df.get('ignore', False).astype(str).str.upper().isin(['TRUE', '1', 'YES'])
    info_map = {canonical(row['Platform']): {
                    'dir': row.get('Directory', row['Platform']),
                    'url': row.get('URL', '')
                } for _, row in plat_df.iterrows()}
    blacklisted_plats = {canonical(row['Platform']) for _, row in plat_df[plat_df['ignore']].iterrows()}

    available = {canonical(c) for c in unmatched_df['Platform'].dropna()}
    zero_codes = {canonical(c) for c in summary_df[summary_df.get('ROMs', 1) == 0]['Platform'].dropna()}
    available = sorted(available - blacklisted_plats)
    zero_codes = sorted(zero_codes - blacklisted_plats)
    all_codes = sorted(set(available) | set(zero_codes))
    print(f"All platforms: [{' '.join(all_codes)}]")
    print(f"Platforms with zero ROMs: [{' '.join(zero_codes)}]")
    print("Enter platform codes. Use:\n'empty' to add empty platforms, \n'all' to toggle all platforms, \n'run' to continue:")

    active = set()
    while True:
        ans = input('> ').strip()
        if ans.lower() == 'run':
            break
        for tok in filter(None, re.split(r'[\s,]+', ans)):
            low = tok.lower()
            if low == 'empty':
                active.update(zero_codes)
                continue
            if low == 'all':
                if active >= set(available):
                    active.clear()
                else:
                    active.update(available)
                continue
            code = canonical(tok)
            if code not in all_codes:
                print(f'Unknown platform: {tok}')
                continue
            if code in active:
                active.remove(code)
                print(f'Removed {code}')
            else:
                active.add(code)
                print(f'Added {code}')
        if active:
            print('Currently selected:', ', '.join(sorted(active)))
        else:
            print('No platforms selected')

    platforms = sorted(active)
    if not platforms:
        print('No platforms selected.')
        return

    try:
        count = int(input('Number of top games per platform [10]: ') or '10')
    except ValueError:
        count = 10

    # Use a fixed fuzzy match threshold for manual mode
    threshold = 70

    platform_targets = {}
    total_targets = 0
    for code in platforms:
        subset = unmatched_df[unmatched_df['Platform'] == code]
        subset = subset[~subset['Dataset Name'].apply(lambda n: (n, code) in blacklisted_pairs)]
        num = len(subset.sort_values('Sales', ascending=False).head(count))
        platform_targets[code] = num
        total_targets += num

    total_done = 0

    download_rows = []

    for code in platforms:
        info = info_map.get(code, {})
        url = info.get('url')
        directory = info.get('dir', code) or code
        platform_total = platform_targets.get(code, 0)
        platform_done = 0
        if not url:
            print(f'Skipping {code}: no URL configured.')
            continue
        subset = unmatched_df[unmatched_df['Platform'] == code]
        subset = subset[~subset['Dataset Name'].apply(lambda n: (n, code) in blacklisted_pairs)]
        subset = subset.sort_values('Sales', ascending=False).head(count)
        if subset.empty:
            print(f'No unmatched entries for {code}.')
            continue
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        files = [a['href'] for a in soup.find_all('a', href=True)
                 if a['href'].lower().endswith('.zip')]
        filtered = []
        for href in files:
            name = requests.utils.unquote(os.path.splitext(href)[0])
            low = name.lower()
            if '(demo)' in low or '(virtual console)' in low:
                continue
            filtered.append(href)
        files = filtered

        file_map = {}
        file_names = {}
        group_map = {}
        for href in files:
            name = requests.utils.unquote(os.path.splitext(href)[0])
            key = norm(name)
            file_map[key] = href
            file_names[key] = name
            base_key = norm(remove_disc(name))
            group_map.setdefault(base_key, []).append(key)

        for _, row in subset.iterrows():
            platform_done += 1
            total_done += 1
            search_term = row['Dataset Name']
            norm_search = norm(search_term)
            selected_keys = None
            score = 0
            while True:
                clear_screen()
                print(f"Platform {code}: {progress_bar(platform_done, platform_total)} | Total: {progress_bar(total_done, total_targets)}")
                print(f"   {search_term}")
                ts_results = process.extract(norm_search, list(file_map.keys()), scorer=fuzz.token_sort_ratio, limit=None)
                options = []
                if ts_results:
                    groups = {}
                    for cand, ts_score, _ in ts_results:
                        base = norm(remove_disc(file_names[cand]))
                        g = groups.setdefault(base, {'keys': [], 'score': 0})
                        g['keys'].append(cand)
                        g['score'] = max(g['score'], ts_score)
                    sorted_groups = sorted(groups.values(), key=lambda g: g['score'], reverse=True)
                    for idx, g in enumerate(sorted_groups[:3], start=1):
                        keys = filter_best_region(g['keys'], file_names)
                        main_key = max(keys, key=lambda k: fuzz.partial_ratio(norm_search, k))
                        pr_score = fuzz.partial_ratio(norm_search, main_key)
                        opt_score = int(round(min(g['score'], pr_score)))
                        disp = ' + '.join(file_names[k] for k in sorted(keys, key=lambda k: disc_number(file_names[k])))
                        disp = color_diff(disp, search_term)
                        options.append((keys, opt_score))
                        print(f"{idx}. {disp}")
                else:
                    print('No candidates found')
                sel = input("Press 1-3, 'b' to blacklist, or ENTER to skip: ").strip()
                if sel == '':
                    break
                if sel.lower() == 'b':
                    blacklist_df.loc[len(blacklist_df)] = {'Search_Term': row['Dataset Name'], 'Platform': code}
                    blacklisted_pairs.add((row['Dataset Name'], code))
                    print(f"Blacklisted {row['Dataset Name']}")
                    selected_keys = None
                    break
                if sel.isdigit() and options:
                    j = int(sel) - 1
                    if 0 <= j < len(options):
                        selected_keys, score = options[j]
                        if score < threshold:
                            print(f'Selected option below threshold ({score} < {threshold})')
                        break
                    else:
                        print('Invalid selection.')
                        continue
                else:
                    search_term = sel
                    norm_search = norm(search_term)
                    continue
            if not selected_keys:
                clear_screen()
                continue
            selected_names = [file_names[k] for k in selected_keys]
            if any(DISC_RE.search(n) for n in selected_names):
                base = norm(remove_disc(selected_names[0]))
                all_keys = [k for k in group_map.get(base, selected_keys)
                            if region_priority(file_names[k]) == region_priority(selected_names[0])]
                if len(all_keys) > len(selected_keys):
                    selected_keys = sorted(all_keys, key=lambda k: disc_number(file_names[k]))

            uniq_keys = []
            seen = set()
            for k in selected_keys:
                if k not in seen:
                    uniq_keys.append(k)
                    seen.add(k)
            for k in uniq_keys:
                filehref = file_map[k]
                title = requests.utils.unquote(os.path.basename(filehref))
                download_rows.append({'Search_Term': row['Dataset Name'], 'Platform': code,
                                     'Directory': directory, 'Matched_Title': title,
                                     'Score': score, 'URL': requests.compat.urljoin(url, filehref)})
            clear_screen()

    if not download_rows:
        print('No entries added.')
        return

    csv_path = os.path.join(snapshot_dir, 'download_list.csv')
    df_existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    new_df = pd.DataFrame(download_rows)
    df = pd.concat([df_existing, new_df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    console.print('[bold green]Download list updated.[/]')
    if not blacklist_df.empty:
        blacklist_df.drop_duplicates().to_csv(BLACKLIST_CSV, index=False)
    if prompt_yes_no('Download now?'):
        download_games(snapshot_dir)


def auto_add_games(snapshot_dir):
    """Automatic mode for adding download entries using fuzzy matching."""
    unmatched_path = os.path.join(snapshot_dir, 'unmatched_summary.csv')
    summary_path = os.path.join(snapshot_dir, 'summary.csv')
    if not os.path.isfile(unmatched_path):
        print('unmatched_summary.csv not found. Run snapshot first.')
        return

    unmatched_df = pd.read_csv(unmatched_path)
    summary_df = pd.read_csv(summary_path) if os.path.isfile(summary_path) else pd.DataFrame()

    if os.path.exists(BLACKLIST_CSV):
        blacklist_df = pd.read_csv(BLACKLIST_CSV)
        blacklist_df = blacklist_df[['Search_Term', 'Platform']]
    else:
        blacklist_df = pd.DataFrame(columns=['Search_Term', 'Platform'])
    blacklisted_pairs = set(zip(blacklist_df.get('Search_Term', []), blacklist_df.get('Platform', [])))

    def canonical(code: str) -> str:
        if not isinstance(code, str):
            return ''
        return PLAT_MAP.get(code.lower(), code).upper()


    unmatched_df['Platform'] = unmatched_df['Platform'].apply(canonical)
    if not summary_df.empty and 'Platform' in summary_df.columns:
        summary_df['Platform'] = summary_df['Platform'].apply(canonical)
    if prompt_yes_no('Exclude Wii games without controller support?'):
        unmatched_df = filter_wii_no_controller(unmatched_df)
    plat_df = pd.read_csv(PLATFORMS_CSV)
    plat_df['ignore'] = plat_df.get('ignore', False).astype(str).str.upper().isin(['TRUE', '1', 'YES'])
    info_map = {canonical(row['Platform']): {
                    'dir': row.get('Directory', row['Platform']),
                    'url': row.get('URL', '')
                } for _, row in plat_df.iterrows()}
    blacklisted_plats = {canonical(row['Platform']) for _, row in plat_df[plat_df['ignore']].iterrows()}

    available = {canonical(c) for c in unmatched_df['Platform'].dropna()}
    zero_codes = {canonical(c) for c in summary_df[summary_df.get('ROMs', 1) == 0]['Platform'].dropna()}
    available = sorted(available - blacklisted_plats)
    zero_codes = sorted(zero_codes - blacklisted_plats)
    all_codes = sorted(set(available) | set(zero_codes))
    print(f"All platforms: [{' '.join(all_codes)}]")
    print(f"Platforms with zero ROMs: [{' '.join(zero_codes)}]")
    print("Enter platform codes. Use:\n'empty' to add empty platforms, \n'all' to toggle all platforms, \n'run' to continue:")

    active = set()
    while True:
        ans = input('> ').strip()
        if ans.lower() == 'run':
            break
        for tok in filter(None, re.split(r'[\s,]+', ans)):
            low = tok.lower()
            if low == 'empty':
                active.update(zero_codes)
                continue
            if low == 'all':
                if active >= set(available):
                    active.clear()
                else:
                    active.update(available)
                continue
            code = canonical(tok)
            if code not in all_codes:
                print(f'Unknown platform: {tok}')
                continue
            if code in active:
                active.remove(code)
                print(f'Removed {code}')
            else:
                active.add(code)
                print(f'Added {code}')
        if active:
            print('Currently selected:', ', '.join(sorted(active)))
        else:
            print('No platforms selected')

    platforms = sorted(active)
    if not platforms:
        print('No platforms selected.')
        return

    try:
        count = int(input('Number of top games per platform [10]: ') or '10')
    except ValueError:
        count = 10

    threshold = ask_threshold(default=75)

    download_rows = []
    summary_rows = []

    for code in platforms:
        info = info_map.get(code, {})
        url = info.get('url')
        directory = info.get('dir', code) or code
        if not url:
            print(f'Skipping {code}: no URL configured.')
            continue
        subset = unmatched_df[unmatched_df['Platform'] == code]
        subset = subset[~subset['Dataset Name'].apply(lambda n: (n, code) in blacklisted_pairs)]
        subset = subset.sort_values('Sales', ascending=False).head(count)
        if subset.empty:
            print(f'No unmatched entries for {code}.')
            continue
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        files = [a['href'] for a in soup.find_all('a', href=True)
                 if a['href'].lower().endswith('.zip')]
        filtered = []
        for href in files:
            name = requests.utils.unquote(os.path.splitext(href)[0])
            low = name.lower()
            if '(demo)' in low or '(virtual console)' in low:
                continue
            filtered.append(href)
        files = filtered

        file_map = {}
        file_names = {}
        group_map = {}
        for href in files:
            name = requests.utils.unquote(os.path.splitext(href)[0])
            key = norm(name)
            file_map[key] = href
            file_names[key] = name
            base_key = norm(remove_disc(name))
            group_map.setdefault(base_key, []).append(key)

        match_count = 0
        scores = []
        for _, row in subset.iterrows():
            game = row['Dataset Name']
            normed = norm(game)
            ts_results = process.extract(normed, list(file_map.keys()), scorer=fuzz.token_sort_ratio, limit=None)
            if not ts_results:
                print(f"  [SKIP] {game} (no candidates)")
                continue
            max_ts = max(score for _, score, _ in ts_results)
            keys_at_max = [key for key, score, _ in ts_results if score == max_ts]
            best_key = max(keys_at_max, key=lambda k: region_priority(file_names[k]))
            pr_score = fuzz.partial_ratio(normed, best_key)
            score = int(round(min(max_ts, pr_score)))
            if score < threshold:
                print(f"  [SKIP] {game} (score below threshold: {score})")
                continue
            selected_keys = group_map.get(norm(remove_disc(file_names[best_key])), [best_key])

            uniq_keys = []
            seen = set()
            for k in selected_keys:
                if k not in seen:
                    uniq_keys.append(k)
                    seen.add(k)
            for k in uniq_keys:
                filehref = file_map[k]
                title = requests.utils.unquote(os.path.basename(filehref))
                download_rows.append({'Search_Term': game, 'Platform': code,
                                     'Directory': directory, 'Matched_Title': title,
                                     'Score': score, 'URL': requests.compat.urljoin(url, filehref)})
            match_count += 1
            scores.append(score)

        pct = match_count / len(subset) * 100 if len(subset) else 0
        summary_rows.append({
            'Platform': code,
            'Added': match_count,
            'Match %': round(pct, 1),
            'Lowest Score': min(scores) if scores else 0,
            'Average Score': round(sum(scores) / len(scores), 1) if scores else 0
        })

    if not download_rows:
        print('No entries added.')
        return

    csv_path = os.path.join(snapshot_dir, 'download_list.csv')
    df_existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    new_df = pd.DataFrame(download_rows)
    df = pd.concat([df_existing, new_df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    console.print('[bold green]Download list updated.[/]')
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows,
                                 columns=['Platform','Added','Match %','Lowest Score','Average Score'])
        print_table(summary_df)
    if not blacklist_df.empty:
        blacklist_df.drop_duplicates().to_csv(BLACKLIST_CSV, index=False)
    if prompt_yes_no('Download now?'):
        download_games(snapshot_dir)


def download_games(snapshot_dir):
    csv_path = os.path.join(snapshot_dir, 'download_list.csv')
    if not os.path.isfile(csv_path):
        print('download_list.csv not found.')
        return
    links_file = os.path.join(snapshot_dir, 'aria2_links.txt')
    entries = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row['URL']
            title = requests.utils.unquote(os.path.basename(row['Matched_Title']))
            out_dir = os.path.join(snapshot_dir, 'roms', row['Directory'])
            os.makedirs(out_dir, exist_ok=True)
            entries.append(f"{url}\n  out={title}\n  dir={out_dir}")
    with open(links_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(entries))
    cmd = [ARIA2, '-i', links_file, '-j', '5']
    print('Running:', ' '.join(shorten_path(c) for c in cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print('Error: aria2c not found. Ensure aria2c.exe is present in the wizardry folder or added to PATH.')
    except subprocess.CalledProcessError as e:
        print(f'aria2c exited with code {e.returncode}')


def convert_to_chd():
    TARGET_EXTS = {'.cue','.bin','.gdi','.iso'}
    rows = []
    available = []
    zero_codes = []
    code_map = {}
    for console_name in sorted(os.listdir(ROMS_ROOT)):
        folder = os.path.join(ROMS_ROOT, console_name)
        if not os.path.isdir(folder):
            continue
        code = PLAT_MAP.get(console_name.lower())
        if not code or code in IGNORED_PLATFORMS:
            continue
        count = 0
        for root,_,files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in TARGET_EXTS:
                    count += 1
        code = code.upper()
        code_map[code] = console_name
        rows.append({'Platform': code, 'Convertible Files': count})
        if count:
            available.append(code)
        else:
            zero_codes.append(code)
    df = pd.DataFrame(rows, columns=['Platform','Convertible Files'])
    print_table(df)
    all_codes = sorted(set(available) | set(zero_codes))
    print(f"All platforms: [{' '.join(all_codes)}]")
    print(f"Platforms with zero ROMs: [{' '.join(zero_codes)}]")
    print("Enter platform codes. Use:\n'empty' to add empty platforms, \n'all' to toggle all platforms, \n'run' to continue:")
    active = set()
    while True:
        ans = input('> ').strip()
        if ans.lower() == 'run':
            break
        for tok in filter(None, re.split(r'[\s,]+', ans)):
            low = tok.lower()
            if low == 'empty':
                active.update(zero_codes)
                continue
            if low == 'all':
                if active >= set(all_codes):
                    active.clear()
                else:
                    active.update(all_codes)
                continue
            code = PLAT_MAP.get(low, tok).upper() if low in PLAT_MAP else tok.upper()
            if code not in all_codes:
                print(f'Unknown platform: {tok}')
                continue
            if code in active:
                active.remove(code)
                print(f'Removed {code}')
            else:
                active.add(code)
                print(f'Added {code}')
        if active:
            print('Currently selected:', ', '.join(sorted(active)))
        else:
            print('No platforms selected')
    platforms = sorted(active)
    if not platforms:
        print('No platforms selected.')
        return
    for code in platforms:
        console_name = code_map.get(code, code)
        folder = os.path.join(ROMS_ROOT, console_name)
        for root,_,files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() not in TARGET_EXTS:
                    continue
                in_path = os.path.join(root, f)
                out_path = os.path.splitext(in_path)[0] + '.chd'
                if os.path.exists(out_path):
                    continue
                cmd = [CHDMAN, 'createcd', '-i', in_path, '-o', out_path]
                try:
                    subprocess.run(cmd, check=True)
                except FileNotFoundError:
                    print('Error: chdman.exe not found. Ensure it is in the wizardry folder or in your PATH.')
                    return
                except subprocess.CalledProcessError as e:
                    print(f'chdman exited with code {e.returncode}')
                    continue
                os.remove(in_path)
    print('Conversion complete.')


def get_latest_snapshot():
    base = os.path.dirname(__file__)
    snaps = [d for d in os.listdir(base) if d.startswith('snapshot_') and os.path.isdir(os.path.join(base, d))]
    if not snaps:
        return None
    snaps.sort(reverse=True)
    return os.path.join(base, snaps[0])


def select_snapshot():
    """Prompt the user to load the latest snapshot or create/select one."""
    latest = get_latest_snapshot()
    if latest and prompt_yes_no(f"Load latest snapshot at {shorten_path(latest)}?", True):
        return latest
    path = input('Enter path to existing snapshot or press ENTER to create new: ').strip()
    if path:
        if os.path.isdir(path):
            return path
        print('Invalid path, creating new snapshot.')
    return create_snapshot()


def show_snapshot_summary(snapshot_dir):
    """Print the snapshot summary and cycle through sorted columns."""
    global SUMMARY_CYCLE_IDX, SUMMARY_COLS
    summary_file = os.path.join(snapshot_dir, 'summary.csv')
    if not os.path.isfile(summary_file):
        print('summary.csv not found.')
        return
    df = pd.read_csv(summary_file)
    if not SUMMARY_COLS:
        SUMMARY_COLS = list(df.columns)
        if 'ROMs' in SUMMARY_COLS:
            SUMMARY_CYCLE_IDX = SUMMARY_COLS.index('ROMs')
    if SUMMARY_COLS:
        sort_col = SUMMARY_COLS[SUMMARY_CYCLE_IDX]
        view = df.sort_values(sort_col, ascending=False)
    else:
        sort_col = None
        view = df
    for col in view.select_dtypes(include='float'):
        view[col] = view[col].round(1)
    print_table(view, sorted_col=sort_col)
    if SUMMARY_COLS:
        SUMMARY_CYCLE_IDX = (SUMMARY_CYCLE_IDX + 1) % len(SUMMARY_COLS)


def wizard_menu(snapshot_dir):
    """Main interactive menu for a given snapshot."""
    global SUMMARY_CYCLE_IDX, SUMMARY_COLS
    SUMMARY_COLS = []
    SUMMARY_CYCLE_IDX = 0
    while True:
        sort_col = None
        summary_file = os.path.join(snapshot_dir, 'summary.csv')
        if not SUMMARY_COLS and os.path.isfile(summary_file):
            SUMMARY_COLS = list(pd.read_csv(summary_file, nrows=0).columns)
            if 'ROMs' in SUMMARY_COLS:
                SUMMARY_CYCLE_IDX = SUMMARY_COLS.index('ROMs')
        if SUMMARY_COLS:
            sort_col = SUMMARY_COLS[SUMMARY_CYCLE_IDX]

        dup_exists = os.path.isdir(os.path.join(snapshot_dir, 'duplicate_roms'))
        m3u_exists = False
        gl_exists = False
        roms_root = os.path.join(snapshot_dir, 'roms')
        for root, _, files in os.walk(roms_root):
            for f in files:
                if f.lower().endswith('.m3u'):
                    m3u_exists = True
                if f == 'gamelist.xml':
                    gl_exists = True
            if m3u_exists and gl_exists:
                break

        console.print('\n[bold cyan]ROM Wizard Menu[/]')
        console.print(f"0) Show snapshot summary (sorted by: {sort_col or 'N/A'})")
        console.print(f"1) {'[green]Re-detect duplicates[/]' if dup_exists else 'Detect duplicates'}")
        console.print(f"2) {'[green]Re-generate .m3u playlists[/]' if m3u_exists else 'Generate .m3u playlists'}")
        console.print(f"3) {'[green]Re-match gamelist.xml with sales data[/]' if gl_exists else 'Match gamelist.xml with sales data'}")
        console.print('4) Add unmatched games to download list (manual filtering)')
        console.print('5) Add unmatched games to download list (automatic filtering)')
        console.print('6) Download unmatched games in list')
        console.print('7) Convert downloaded disc images to CHD')
        console.print('8) Quit')
        choice = input('Select option: ').strip()
        if choice == '0':
            show_snapshot_summary(snapshot_dir)
        elif choice == '1':
            detect_duplicates(snapshot_dir)
        elif choice == '2':
            generate_playlists(snapshot_dir)
        elif choice == '3':
            apply_sales(snapshot_dir)
        elif choice == '4':
            manual_add_games(snapshot_dir)
        elif choice == '5':
            auto_add_games(snapshot_dir)
        elif choice == '6':
            download_games(snapshot_dir)
        elif choice == '7':
            convert_to_chd()
        elif choice == '8':
            return prompt_yes_no('Restart wizard?')


def main():
    while True:
        snap_dir = select_snapshot()
        restart = wizard_menu(snap_dir)
        if not restart:
            break

if __name__ == '__main__':
    main()
