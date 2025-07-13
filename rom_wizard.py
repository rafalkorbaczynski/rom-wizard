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
ROMS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'roms'))

_ROMAN = {'ix':9,'viii':8,'vii':7,'vi':6,'iv':4,'iii':3,'ii':2,'i':1}
import re
import unicodedata
_ROMAN_RE = re.compile(r'\b(' + '|'.join(sorted(_ROMAN, key=len, reverse=True)) + r')\b')

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


def ask_threshold(default=85):
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

    for console in os.listdir(ROMS_ROOT):
        gl_path = os.path.join(ROMS_ROOT, console, 'gamelist.xml')
        if not os.path.isfile(gl_path):
            continue
        tree = ET.parse(gl_path)
        games = tree.getroot().findall('game')
        keys = [norm(g.findtext('name') or '') for g in games]
        platform = console
        sales_subset = sales[sales['Platform'].str.lower()==platform.lower()]
        sales_map = dict(zip(sales_subset['key'], sales_subset['Global_Sales']))
        match_keys = list(sales_map.keys())
        matched = 0
        for k in keys:
            res = process.extractOne(k, match_keys, scorer=fuzz.token_set_ratio)
            if res and res[1] >= threshold:
                matched += 1
                matched_records.append((platform, res[0]))
                unmatched_keys.discard((platform, res[0]))
        summary_rows.append({'Platform': platform, 'ROMs': len(games), 'MatchedROMs': matched})

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df['ROMs%'] = summary_df['MatchedROMs']/summary_df['ROMs'].replace(0,1)*100
    summary_csv = os.path.join(snap_dir, 'summary.csv')
    summary_df.to_csv(summary_csv, index=False)

    unmatched_list = [ {'Platform':p, 'key':k} for p,k in unmatched_keys ]
    unmatched_df = pd.DataFrame(unmatched_list)
    unmatched_csv = os.path.join(snap_dir, 'unmatched_summary.csv')
    unmatched_df.to_csv(unmatched_csv, index=False)

    match_df = pd.DataFrame(matched_records, columns=['Platform','key'])
    match_df.to_csv(os.path.join(snap_dir, 'matched_summary.csv'), index=False)

    print('Snapshot created at', snap_dir)
    print(summary_df.to_string(index=False))
    return snap_dir


def detect_duplicates(snapshot_dir):
    from rapidfuzz import fuzz
    threshold = ask_threshold()
    ALLOWED = {'a26','a52','a78','bin','chd','gb','gba','gbc','iso','j64','md','mp4','nds','nes','pce','rvz','sfc','sms','xex','xml','z64','zip'}
    dup_root = os.path.join(snapshot_dir, 'duplicate_roms')
    os.makedirs(dup_root, exist_ok=True)
    report_rows = []
    for console in os.listdir(ROMS_ROOT):
        folder = os.path.join(ROMS_ROOT, console)
        if not os.path.isdir(folder):
            continue
        files = [f for f in os.listdir(folder) if os.path.splitext(f)[1].lower().lstrip('.') in ALLOWED and 'disc' not in f.lower()]
        norm_map = {f:norm(f) for f in files}
        moved = set()
        for i,f in enumerate(files):
            for j,g in enumerate(files[i+1:], i+1):
                if g in moved or f in moved:
                    continue
                if fuzz.token_set_ratio(norm_map[f], norm_map[g]) >= threshold:
                    src = os.path.join(folder, g)
                    dst_dir = os.path.join(dup_root, console)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.move(src, dst_dir)
                    report_rows.append([console, f, g])
                    moved.add(g)
    csv_path = os.path.join(snapshot_dir, 'duplicates_report.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Console','Keep','Duplicate'])
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
    key_to_sales = dict(zip(sales['key'], sales['Global_Sales']))
    key_to_name = dict(zip(sales['key'], sales['Name']))
    output_root = os.path.join(snapshot_dir, 'roms')
    os.makedirs(output_root, exist_ok=True)
    match_rows = []
    unmatched = []
    summary_rows = []

    for console in os.listdir(ROMS_ROOT):
        gl_path = os.path.join(ROMS_ROOT, console, 'gamelist.xml')
        if not os.path.isfile(gl_path):
            continue
        tree = ET.parse(gl_path)
        root = tree.getroot()
        games = root.findall('game')
        matched = 0
        for g in games:
            for tag in g.findall('rating') + g.findall('ratingMax'):
                g.remove(tag)
            title = g.findtext('name') or ''
            k = norm(title)
            res = process.extractOne(k, sales['key'], scorer=fuzz.token_set_ratio)
            if res and res[1] >= threshold:
                s_key = res[0]
                gs = key_to_sales[s_key]
                rating = gs / max_sales * 100
                ET.SubElement(g, 'rating').text = f"{rating:.2f}"
                ET.SubElement(g, 'ratingMax').text = '100'
                match_rows.append({'Dataset_Name': key_to_name[s_key], 'Sales_Platform': sales.loc[sales['key']==s_key,'Platform'].iloc[0], 'ROM_File': title, 'Global_Sales': gs, 'Match_Score': res[1]})
                matched += 1
                unmatched_key = (sales.loc[sales['key']==s_key,'Platform'].iloc[0], s_key)
                if unmatched_key in unmatched:
                    unmatched.remove(unmatched_key)
            else:
                pass
        out_dir = os.path.join(output_root, console)
        os.makedirs(out_dir, exist_ok=True)
        tree.write(os.path.join(out_dir, 'gamelist.xml'), encoding='utf-8', xml_declaration=True)
        summary_rows.append({'Platform': console, 'ROMs': len(games), 'MatchedROMs': matched})

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df['ROMs%'] = summary_df['MatchedROMs']/summary_df['ROMs'].replace(0,1)*100
    summary_df.to_csv(os.path.join(snapshot_dir, 'summary.csv'), index=False)
    pd.DataFrame(match_rows).to_csv(os.path.join(snapshot_dir, 'match_summary.csv'), index=False)
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


def find_new_games(snapshot_dir):
    unmatched_path = os.path.join(snapshot_dir, 'unmatched_summary.csv')
    if not os.path.isfile(unmatched_path):
        print('unmatched_summary.csv not found. Run snapshot first.')
        return
    threshold = ask_threshold()
    unmatched_df = pd.read_csv(unmatched_path)
    codes = sorted(unmatched_df['Platform'].unique())
    code = input(f"Platform code {codes}: ")
    if code not in codes:
        print('Invalid platform.')
        return
    url, directory, files = scrape_file_list(code)
    if not url:
        print('No URL configured for platform.')
        return
    download_rows = []
    while True:
        term = input('Search term (blank to finish): ').strip()
        if not term:
            break
        results = process.extract(norm(term), [n for n,_ in files], scorer=fuzz.token_set_ratio, limit=20)
        choices = [(name,score,idx) for name,score,idx in results if score >= threshold][:5]
        if not choices:
            print('No matches above threshold.')
            continue
        for i,(name,score,idx) in enumerate(choices,1):
            print(f"{i}. {files[idx][0]} (score {score})")
        sel = input('Select number or ENTER to skip: ').strip()
        if not sel:
            continue
        try:
            idx = int(sel)-1
            name,filehref = files[choices[idx][2]]
        except (ValueError,IndexError):
            print('Invalid choice')
            continue
        download_rows.append({'Search_Term': term, 'Platform': code, 'Directory': directory, 'Matched_Title': filehref, 'Score': choices[idx][1], 'URL': requests.compat.urljoin(url, filehref)})
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
    if ans=='y':
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
        print('4) Find new games (manual search)')
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
            find_new_games(snap_dir)
        elif choice=='5':
            download_games(snap_dir)
        elif choice=='6':
            convert_to_chd()
        elif choice=='7':
            break

if __name__ == '__main__':
    main()
