import os
import sys
import csv
import shutil
import argparse
import datetime
import pandas as pd
import xml.etree.ElementTree as ET
from rapidfuzz import process, fuzz
import requests
from bs4 import BeautifulSoup

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

# Mapping from ROM directory names to platform codes used in the sales dataset
PLAT_MAP = {
    'atari2600':'2600','atari5200':'5200','atari7800':'7800','atarist':'AST',
    '3do':'3DO','3ds':'3DS','nds':'DS','ds':'DS',
    'gb':'GB','gbc':'GBC','gba':'GBA','nes':'NES','snes':'SNES','n64':'N64',
    'gamecube':'GC','wii':'Wii','wiiu':'WiiU','switch':'NS','ns':'NS','ngage':'NGage',
    'virtualboy':'VB','ps':'PS','psx':'PS','ps2':'PS2','ps3':'PS3','ps4':'PS4','ps5':'PS5',
    'psp':'PSP','psvita':'PSV','saturn':'SAT','segacd':'SCD','sega_cd':'SCD','sega32x':'S32X',
    'gamegear':'GG','megadrive':'GEN','dreamcast':'DC','dc':'DC',
    'xbox':'XB','xbox360':'X360','xboxone':'XOne',
    'pc':'PC','dos':'MSD','windows':'WinP','linux':'Linux','osx':'OSX','pcfx':'PCFX',
    'pcengine':'TG16','pce':'PCE','pcenginecd':'PCE','c64':'C64','c128':'C128','fds':'FDS',
    'msx1':'MSX','msx2':'MSX','msx2+':'MSX','msxturbor':'MSX','bbcmicro':'BBCM',
    'amiga1200':'AMIG','amiga4000':'AMIG','amiga500':'AMIG','amigacd32':'CD32',
    'amigacdtv':'CD32','amstradcpc':'ACPC','apple2':'ApII','apple2gs':'ApII',
    'intellivision':'Int','arcadia':'Arc','astrocade':'Arc','cdi':'CDi'
}

# Reverse lookup for convenience when mapping dataset platforms back to directories
DATASET_TO_CONSOLE = {}
for k, v in PLAT_MAP.items():
    DATASET_TO_CONSOLE.setdefault(v.lower(), k)

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


def create_snapshot():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    snap_dir = os.path.join(os.path.dirname(__file__), f'snapshot_{ts}')
    os.makedirs(os.path.join(snap_dir, 'roms'), exist_ok=True)
    os.makedirs(os.path.join(snap_dir, 'duplicate_roms'), exist_ok=True)

    threshold = ask_threshold()
    sales = pd.read_csv(SALES_CSV, low_memory=False)
    sales['key'] = sales['Name'].apply(norm)
    max_sales = sales['Global_Sales'].max()

    summary_rows = []
    matched_records = []
    unmatched_keys = set(zip(sales['Platform'], sales['key']))
    found_consoles = set()

    for console in os.listdir(ROMS_ROOT):
        gl_path = os.path.join(ROMS_ROOT, console, 'gamelist.xml')
        if not os.path.isfile(gl_path):
            continue
        found_consoles.add(console.lower())
        tree = ET.parse(gl_path)
        games = [g for g in tree.getroot().findall('game')
                 if os.path.splitext(g.findtext('path') or '')[1].lower().lstrip('.') in ROM_EXTS]
        platform = console
        ds_plat = PLAT_MAP.get(console.lower(), console)

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

            region_text = (g.findtext('region') or '').lower().strip()
            region_key = None
            if region_text:
                for reg, toks in REGION_SYNONYMS.items():
                    if region_text in toks:
                        region_key = reg
                        break
            if not region_key:
                text = os.path.basename(g.findtext('path') or title).lower()
                tags = re.findall(r'\(([^()]+)\)', text)
                for tag in tags:
                    for token in re.split('[,;/]', tag):
                        t = token.strip().lower()
                        for reg, toks in REGION_SYNONYMS.items():
                            if t in toks:
                                region_key = reg
                                break
                        if region_key:
                            break
                    if region_key:
                        break
            if not region_key:
                region_key = 'Other'
            region_counts[region_key] += 1

            res = process.extractOne(k, match_keys, scorer=fuzz.token_set_ratio)
            if res and res[1] >= threshold:
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

    print('Snapshot created at', snap_dir)
    print(summary_df.to_string(index=False))
    return snap_dir


def detect_duplicates(snapshot_dir):
    from rapidfuzz import fuzz
    threshold = ask_threshold()
    dup_root = os.path.join(snapshot_dir, 'duplicate_roms')
    os.makedirs(dup_root, exist_ok=True)
    report_rows = []

    for console in os.listdir(ROMS_ROOT):
        console_dir = os.path.join(ROMS_ROOT, console)
        if not os.path.isdir(console_dir):
            continue
        for root, _, files in os.walk(console_dir):
            files = [f for f in files if os.path.splitext(f)[1].lower().lstrip('.') in ROM_EXTS and 'disc' not in f.lower()]
            norm_map = {f: norm(f) for f in files}
            moved = set()
            for i, f in enumerate(files):
                if f in moved:
                    continue
                for g in files[i+1:]:
                    if g in moved:
                        continue
                    score = fuzz.token_set_ratio(norm_map[f], norm_map[g])
                    if score >= threshold:
                        src = os.path.join(root, g)
                        rel_dir = os.path.relpath(root, ROMS_ROOT)
                        dst_dir = os.path.join(dup_root, rel_dir)
                        os.makedirs(dst_dir, exist_ok=True)
                        shutil.move(src, os.path.join(dst_dir, g))
                        keep_rel = os.path.relpath(os.path.join(root, f), ROMS_ROOT)
                        move_rel = os.path.relpath(src, ROMS_ROOT)
                        report_rows.append([console, score, norm_map[f], norm_map[g], keep_rel, move_rel])
                        moved.add(g)
    csv_path = os.path.join(snapshot_dir, 'duplicates_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Platform','Match Score','Kept Key','Moved Key','Kept File','Moved File'])
        writer.writerows(report_rows)
    print('Duplicate scan complete.')


def generate_playlists():
    pattern = re.compile(r'(?i)^(.+?)\s*\(disc\s*(\d+)\)')
    for console in os.listdir(ROMS_ROOT):
        folder = os.path.join(ROMS_ROOT, console)
        if not os.path.isdir(folder):
            continue
        groups = {}
        for fname in os.listdir(folder):
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
            path = os.path.join(folder, base + '.m3u')
            with open(path, 'w', encoding='utf-8') as f:
                for _,name in discs:
                    f.write(name+'\n')
            print('Playlist created:', path)


def apply_sales(snapshot_dir):
    threshold = ask_threshold()
    sales = pd.read_csv(SALES_CSV, low_memory=False)
    sales['key'] = sales['Name'].apply(norm)
    max_sales = sales['Global_Sales'].max()
    unmatched_keys = set(zip(sales['Platform'], sales['key']))
    output_root = os.path.join(snapshot_dir, 'roms')
    os.makedirs(output_root, exist_ok=True)
    match_rows = []
    summary_rows = []
    found_consoles = set()

    for console in os.listdir(ROMS_ROOT):
        gl_path = os.path.join(ROMS_ROOT, console, 'gamelist.xml')
        if not os.path.isfile(gl_path):
            continue
        found_consoles.add(console.lower())
        tree = ET.parse(gl_path)
        root = tree.getroot()
        games = [g for g in root.findall('game')
                 if os.path.splitext(g.findtext('path') or '')[1].lower().lstrip('.') in ROM_EXTS]

        ds_plat = PLAT_MAP.get(console.lower(), console)
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

            region_text = (g.findtext('region') or '').lower().strip()
            region_key = None
            if region_text:
                for reg, toks in REGION_SYNONYMS.items():
                    if region_text in toks:
                        region_key = reg
                        break
            if not region_key:
                text = os.path.basename(g.findtext('path') or title).lower()
                tags = re.findall(r'\(([^()]+)\)', text)
                for tag in tags:
                    for token in re.split('[,;/]', tag):
                        t = token.strip().lower()
                        for reg, toks in REGION_SYNONYMS.items():
                            if t in toks:
                                region_key = reg
                                break
                        if region_key:
                            break
                    if region_key:
                        break
            if not region_key:
                region_key = 'Other'
            region_counts[region_key] += 1

            res = process.extractOne(k, match_keys, scorer=fuzz.token_set_ratio)
            if res and res[1] >= threshold:
                key = res[0]
                gs = sales_map[key]
                rating = gs / max_sales * 100
                ET.SubElement(g, 'rating').text = f"{rating:.2f}"
                ET.SubElement(g, 'ratingMax').text = '100'
                match_rows.append({
                    'Dataset Name': name_map[key],
                    'Platform': console,
                    'ROM': title,
                    'Sales': gs,
                    'Match Score': res[1]
                })
                matched += 1
                unmatched_keys.discard((ds_plat, key))

        out_dir = os.path.join(output_root, console)
        os.makedirs(out_dir, exist_ok=True)
        tree.write(os.path.join(out_dir, 'gamelist.xml'), encoding='utf-8', xml_declaration=True)
        summary_rows.append({
            'Platform': console,
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

    print('Sales data applied.')


def scrape_file_list(code):
    df = pd.read_csv(PLATFORMS_CSV)
    row = df[df['Platform']==code]
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

    available = sorted(unmatched_df['Platform'].unique())
    zero_codes = sorted(summary_df[summary_df.get('ROMs', 1) == 0]['Platform'].unique())

    print('Available platforms:')
    print(' '.join(available))
    if zero_codes:
        print('Platforms with zero ROMs:', ' '.join(zero_codes))
    print("Enter platform codes separated by spaces. Use 'empty' to add empty platforms, 'all' to toggle all platforms, or 'run' to continue:")

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
            code = tok.upper()
            if code not in available and code not in zero_codes:
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

    threshold = ask_threshold()

    plat_df = pd.read_csv(PLATFORMS_CSV)
    info_map = {row['Platform']: {'dir': row.get('Directory', row['Platform']), 'url': row.get('URL', '')} for _, row in plat_df.iterrows()}

    download_rows = []

    for code in platforms:
        info = info_map.get(code, {})
        url = info.get('url')
        directory = info.get('dir', code) or code
        if not url:
            print(f'Skipping {code}: no URL configured.')
            continue
        subset = unmatched_df[unmatched_df['Platform'] == code]
        subset = subset.sort_values('Sales', ascending=False).head(count)
        if subset.empty:
            print(f'No unmatched entries for {code}.')
            continue
        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        files = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.zip')]
        names = [requests.utils.unquote(os.path.splitext(f)[0]) for f in files]

        for _, row in subset.iterrows():
            search_term = row['Dataset Name']
            while True:
                results = process.extract(norm(search_term), names, scorer=fuzz.token_set_ratio, limit=20)
                choices = [(cand, score, idx) for cand, score, idx in results if score >= threshold][:5]
                if not choices:
                    print(f'No matches found for "{search_term}".')
                else:
                    for i, (name, score, idx) in enumerate(choices, 1):
                        print(f"{i}. {name} (score {score})")
                sel = input(f"{row['Dataset Name']}: select 1-{len(choices)} or enter new search term, ENTER to skip: ").strip()
                if not sel:
                    break
                if sel.isdigit():
                    j = int(sel) - 1
                    if 0 <= j < len(choices):
                        name, score, idx = choices[j]
                        filehref = files[idx]
                        download_rows.append({'Search_Term': row['Dataset Name'], 'Platform': code, 'Directory': directory, 'Matched_Title': filehref, 'Score': score, 'URL': requests.compat.urljoin(url, filehref)})
                        break
                    else:
                        print('Invalid selection.')
                else:
                    search_term = sel
                    continue

    if not download_rows:
        print('No entries added.')
        return

    csv_path = os.path.join(snapshot_dir, 'download_list.csv')
    df_existing = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()
    new_df = pd.DataFrame(download_rows)
    df = pd.concat([df_existing, new_df], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print('Download list updated.')
    ans = input('Download now? [y/N]: ').lower()
    if ans == 'y':
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
            title = row['Matched_Title']
            out_dir = os.path.join(snapshot_dir, 'roms', row['Directory'])
            os.makedirs(out_dir, exist_ok=True)
            entries.append(f"{url}\n  out={title}\n  dir={out_dir}")
    with open(links_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(entries))
    cmd = [ARIA2, '-i', links_file, '-j', '5']
    print('Running:', ' '.join(cmd))
    os.system(' '.join(cmd))


def convert_to_chd():
    TARGET_EXTS = {'.cue','.bin','.gdi','.iso'}
    for console in os.listdir(ROMS_ROOT):
        folder = os.path.join(ROMS_ROOT, console)
        if not os.path.isdir(folder):
            continue
        for root,_,files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() not in TARGET_EXTS:
                    continue
                in_path = os.path.join(root, f)
                out_path = os.path.splitext(in_path)[0] + '.chd'
                if os.path.exists(out_path):
                    continue
                os.system(f"{CHDMAN} createcd -i \"{in_path}\" -o \"{out_path}\"")
                os.remove(in_path)
    print('Conversion complete.')


def get_latest_snapshot():
    base = os.path.dirname(__file__)
    snaps = [d for d in os.listdir(base) if d.startswith('snapshot_') and os.path.isdir(os.path.join(base, d))]
    if not snaps:
        return None
    snaps.sort(reverse=True)
    return os.path.join(base, snaps[0])


def main():
    snap_dir = None
    latest = get_latest_snapshot()
    if latest and input(f"Load latest snapshot at {latest}? [Y/n] ").strip().lower() in {'', 'y', 'yes'}:
        snap_dir = latest
    else:
        path = input('Enter path to existing snapshot or press ENTER to create new: ').strip()
        if path:
            if os.path.isdir(path):
                snap_dir = path
            else:
                print('Invalid path, creating new snapshot.')
                snap_dir = create_snapshot()
        else:
            snap_dir = create_snapshot()

    while True:
        print('\nROM Wizard Menu')
        print('1) Detect duplicates')
        print('2) Generate m3u playlists')
        print('3) Apply sales data to gamelists')
        print('4) Add new games (manual mode)')
        print('5) Download games from list')
        print('6) Convert disc images to CHD')
        print('7) Quit')
        choice = input('Select option: ').strip()
        if choice=='1':
            detect_duplicates(snap_dir)
        elif choice=='2':
            generate_playlists()
        elif choice=='3':
            apply_sales(snap_dir)
        elif choice=='4':
            manual_add_games(snap_dir)
        elif choice=='5':
            download_games(snap_dir)
        elif choice=='6':
            convert_to_chd()
        elif choice=='7':
            break

if __name__ == '__main__':
    main()
