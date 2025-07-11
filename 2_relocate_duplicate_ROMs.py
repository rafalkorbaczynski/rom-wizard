#!/usr/bin/env python3
import os
import argparse
import unicodedata
import re
import csv
import shutil
from rapidfuzz import fuzz

# ——— Determine script and project directories ———————————————————————
# Script is in /Retrobat/scripts, ROMs in ../roms, duplicates moved to ../duplicate_roms
template_dir = os.path.dirname(os.path.abspath(__file__))
roms_root_default = os.path.abspath(os.path.join(template_dir, os.pardir, 'roms'))
duplicate_root = os.path.abspath(os.path.join(template_dir, os.pardir, 'duplicate_roms'))
os.makedirs(duplicate_root, exist_ok=True)

# ——— Allowed ROM formats —————————————————————————————————————
ALLOWED_FORMATS = {
    'a26','a52','a78','bin','chd','gb','gba','gbc','iso','j64',
    'md','mp4','nds','nes','pce','rvz','sfc','sms','xex','xml','z64','zip'
}

# ——— Argument parsing ———————————————————————————————————————
parser = argparse.ArgumentParser(
    description="""Detect duplicate ROM files within each platform folder using fuzzy matching,
    move duplicates to a separate folder structure under duplicate_roms, and output results to CSV."""
)
parser.add_argument(
    "--root-dir",
    default=roms_root_default,
    help="Top-level ROMs directory containing per-console subfolders."
)
parser.add_argument(
    "--threshold",
    type=int,
    default=85,
    help="Minimum fuzzy match score (0-100) to consider two filenames duplicates."
)
parser.add_argument(
    "--csv-output",
    default=os.path.join(template_dir, 'duplicates_report.csv'),
    help="Path to the CSV file for writing duplicate pairs report."
)
args = parser.parse_args()
args.root_dir = os.path.abspath(args.root_dir)

# ——— Helpers for normalization & preferences —————————————————————————
_ROMAN = {'ix':9,'viii':8,'vii':7,'vi':6,'iv':4,'iii':3,'ii':2,'i':1}
# compile regex once with longest tokens first for stable matching
_ROMAN_RE = re.compile(r'\b(' + '|'.join(sorted(_ROMAN, key=len, reverse=True)) + r')\b')
STOP_WORDS = {'version','special','edition','rev','s','and','the'}
SEQUEL_TOKENS = set(['part'] + list(_ROMAN.keys()))

# Normalize filename for matching
def norm(filename):
    name, _ = os.path.splitext(filename)
    # strip leading numeric prefixes like "19. " or "01-"
    name = re.sub(r'^\d+\s*[\.\-]?\s*', '', name)
    s = unicodedata.normalize('NFKD', name)
    s = ''.join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r'[\(\[].*?[\)\]]', '', s)
    s = re.sub(r'^the\s+', '', s)
    s = s.replace('&', ' and ')
    s = _ROMAN_RE.sub(lambda m: str(_ROMAN[m.group(1)]), s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

# Preference: translated versions preferred
def is_translated(fname):
    return 'translated' in fname.lower()

# Preference: region priority EU > USA > JP
def region_priority(fname):
    lf = fname.lower()
    if re.search(r'\b(eu|eur|europe)\b', lf):
        return 3
    if re.search(r'\b(us|usa|ntsc-?u)\b', lf):
        return 2
    if re.search(r'\b(jp|japan|ntsc-?j)\b', lf):
        return 1
    return 0

# ——— Duplicate detection —————————————————————————————————————
dup_stats = {}
total_duplicates = 0
total_files = 0

for console_folder in os.listdir(args.root_dir):
    folder_path = os.path.join(args.root_dir, console_folder)
    if not os.path.isdir(folder_path):
        continue

    # Gather ROM files excluding multi-disc
    rom_files = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower().lstrip('.') in ALLOWED_FORMATS
                 and 'disc' not in f.lower()]
    total_files += len(rom_files)

    keys = [norm(f) for f in rom_files]
    duplicates = []

    for i, fname in enumerate(rom_files):
        key_i = keys[i]
        for j, fname_j in enumerate(rom_files[i+1:], start=i+1):
            if 'disc' in fname_j.lower():
                continue
            key_j = keys[j]
            if key_i.isdigit() or key_j.isdigit():
                continue
            score = fuzz.token_set_ratio(key_i, key_j)
            if score < args.threshold:
                continue
            tokens_i = {t for t in key_i.split() if t not in STOP_WORDS and not t.isdigit()}
            tokens_j = {t for t in key_j.split() if t not in STOP_WORDS and not t.isdigit()}
            if len(tokens_i & tokens_j) < 2:
                continue
            extra_j = tokens_j - tokens_i
            extra_i = tokens_i - tokens_j
            if extra_j and extra_j.issubset(SEQUEL_TOKENS):
                continue
            if extra_i and extra_i.issubset(SEQUEL_TOKENS):
                continue
            nums_i = re.findall(r'\d+', key_i)
            nums_j = re.findall(r'\d+', key_j)
            if (nums_i or nums_j) and set(nums_i) != set(nums_j):
                continue
            if tokens_i < tokens_j or tokens_j < tokens_i:
                continue

            # Determine primary vs duplicate based on translation, region, then file size
            p1_name, p2_name = fname, fname_j
            # Translation preference
            if is_translated(p1_name) != is_translated(p2_name):
                primary_name, dup_name = (p2_name, p1_name) if is_translated(p2_name) else (p1_name, p2_name)
                key_primary, key_dup = (key_j, key_i) if is_translated(p2_name) else (key_i, key_j)
            else:
                # Region preference
                pr_i = region_priority(p1_name)
                pr_j = region_priority(p2_name)
                if pr_i != pr_j:
                    primary_name, dup_name = (p2_name, p1_name) if pr_j > pr_i else (p1_name, p2_name)
                    key_primary, key_dup = (key_j, key_i) if pr_j > pr_i else (key_i, key_j)
                else:
                    # Size preference: keep smaller
                    path1 = os.path.join(folder_path, p1_name)
                    path2 = os.path.join(folder_path, p2_name)
                    size1 = os.path.getsize(path1)
                    size2 = os.path.getsize(path2)
                    if size1 <= size2:
                        primary_name, dup_name = p1_name, p2_name
                        key_primary, key_dup = key_i, key_j
                    else:
                        primary_name, dup_name = p2_name, p1_name
                        key_primary, key_dup = key_j, key_i

            duplicates.append((console_folder, score, key_primary, key_dup, primary_name, dup_name))

    dup_stats[console_folder] = {'pairs': duplicates, 'total_files': len(rom_files)}
    total_duplicates += len(duplicates)

# ——— Move duplicates —————————————————————————————————————
files_to_move = {}
for console, stats in dup_stats.items():
    for rec in stats['pairs']:
        _, _, _, _, _, dup_name = rec
        src_path = os.path.join(args.root_dir, console, dup_name)
        files_to_move.setdefault(console, set()).add(src_path)

for console, paths in files_to_move.items():
    dest_dir = os.path.join(duplicate_root, console)
    os.makedirs(dest_dir, exist_ok=True)
    for src in paths:
        try:
            shutil.move(src, dest_dir)
        except Exception as e:
            print(f"Error moving {src} to {dest_dir}: {e}")

# ——— Write CSV —————————————————————————————————————
with open(args.csv_output, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['console','score','key1','key2','file1','file2'])
    for console, stats in dup_stats.items():
        for rec in stats['pairs']:
            console_name, score, key1, key2, primary_name, dup_name = rec
            writer.writerow([console_name, score, key1, key2, primary_name, dup_name])

# ——— Summary —————————————————————————————————————
print(f"Detected {total_duplicates} duplicate pairs across {total_files} files")
print(f"Duplicate ROMs moved to: {duplicate_root}")
print(f"CSV output saved to {args.csv_output}")
