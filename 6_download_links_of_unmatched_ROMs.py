#!/usr/bin/env python3
"""
Download files listed in a CSV using aria2c with 5 concurrent downloads,
organizing them into platform subfolders under new_roms/ mirroring RetroBat.

When run, this script prompts for the snapshot directory that contains the
generated CSV files from ``5_make_links_for_unmatched_ROMs.py``. The ``-f``
option specifies the CSV filename inside that directory (default:
``download_list.csv``).

This script:
  1. Reads the CSV and extracts columns 'Platform', 'Matched_Title', and 'URL'.
  2. Writes an aria2 input file (aria2_links.txt) where each entry includes:
       URL
         out=<Matched_Title>
         dir=<new_roms>/<Platform>/
  3. Calls aria2c with -i aria2_links.txt and -j 5 to download concurrently.
"""
import os
import sys
import argparse
import csv
import subprocess
import readline
import glob
import re
from urllib.parse import urlsplit, urlunsplit
import requests

# Determine directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = SCRIPT_DIR
# Place downloaded files under a sibling 'new_roms' directory relative to the
# RetroBat folder. Since this script lives in 'RetroBat/scripts', we only need
# to go up one directory level to reach the RetroBat root.
NEW_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, 'new_roms'))

# Multi-disc detection regex and helpers
DISC_RE = re.compile(r"\b(?P<word>disc|disk|cd)[ _.-]*(?P<num>[0-9]+|[ivx]+)\b", re.I)
ROMAN_TO_INT = {'i':1,'ii':2,'iii':3,'iv':4,'v':5,'vi':6,'vii':7,'viii':8}
INT_TO_ROMAN = {v:k.upper() for k,v in ROMAN_TO_INT.items()}

def setup_tab_completion():
    """Enable TAB-completion for directory names in input()."""
    def complete_dir(text, state):
        base = text or '.'
        matches = glob.glob(base + '*')
        matches = [m + ('/' if os.path.isdir(m) else '') for m in matches]
        try:
            return matches[state]
        except IndexError:
            return None

    readline.set_completer(complete_dir)
    readline.parse_and_bind('tab: complete')

# Argument parsing
def parse_args():
    p = argparse.ArgumentParser(description="Bulk download via aria2c from CSV URLs into platform subfolders.")
    p.add_argument('-f', '--file', default='download_list.csv',
                   help='Path to CSV containing Platform, Matched_Title, and URL columns')
    p.add_argument('-j', '--concurrent', type=int, default=5,
                   help='Number of simultaneous downloads (aria2c -j)')
    return p.parse_args()


def find_extra_discs(url: str, filename: str, max_discs: int = 8):
    """Return a list of ``(url, filename)`` tuples for adjacent discs.

    The function looks for a disc number in ``filename`` and then probes both
    earlier and later disc numbers (e.g. if ``Disc 2`` is found it will also
    check ``Disc 1``, ``Disc 3`` ... up to ``max_discs``).  It first attempts a
    ``HEAD`` request to verify the file exists.  If the ``HEAD`` request fails
    due to network issues or an unsupported method, it falls back to a tiny
    ranged ``GET`` request.  Any URL that returns a status code below 400 is
    considered valid.
    """
    m = DISC_RE.search(filename)
    if not m:
        return []

    num_str = m.group('num')
    try:
        disc_num = int(num_str)
    except ValueError:
        disc_num = ROMAN_TO_INT.get(num_str.lower(), 0)
    if disc_num <= 0 or disc_num >= max_discs:
        return []

    prefix = filename[:m.start('num')]
    suffix = filename[m.end('num'):]

    split_url = list(urlsplit(url))
    base_path = os.path.dirname(split_url[2])

    # Build candidate disc numbers, trying earlier discs first for predictable
    # ordering, then later discs up to ``max_discs``.
    candidates = list(range(disc_num - 1, 0, -1))
    candidates.extend(range(disc_num + 1, max_discs + 1))

    results = []
    for n in candidates:
        if num_str.isdigit():
            repl = str(n).zfill(len(num_str))
        else:
            roman = INT_TO_ROMAN.get(n)
            if not roman:
                continue
            repl = roman if num_str.isupper() else roman.lower()

        new_name = prefix + repl + suffix
        split_url[2] = os.path.join(base_path, new_name)
        candidate_url = urlunsplit(split_url)

        try:
            r = requests.head(candidate_url, allow_redirects=True, timeout=5)
            if r.status_code >= 400:
                raise requests.RequestException
        except requests.RequestException:
            try:
                r = requests.get(
                    candidate_url,
                    headers={"Range": "bytes=0-0"},
                    allow_redirects=True,
                    timeout=5,
                )
            except requests.RequestException:
                continue
            if r.status_code >= 400:
                continue

        results.append((candidate_url, new_name))

    return results


def main():
    setup_tab_completion()
    args = parse_args()

    # Prompt for snapshot directory like 5_make_links_for_unmatched_ROMs.py
    while True:
        snap = input("Enter snapshot directory name: ").strip()
        snapshot_path = os.path.join(RESULTS_DIR, snap)
        if os.path.isdir(snapshot_path):
            break
        print(f"Directory '{snap}' not found. Please try again.")

    # Resolve CSV path relative to snapshot directory unless absolute
    if os.path.isabs(args.file):
        csv_path = args.file
    else:
        csv_path = os.path.join(snapshot_path, args.file)
    csv_path = os.path.abspath(csv_path)
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    links_path = os.path.join(snapshot_path, 'aria2_links.txt')

    # Prepare aria2 input file
    entries = []
    seen = set()

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        required = ('Platform', 'Directory', 'Matched_Title', 'URL')
        for col in required:
            if col not in reader.fieldnames:
                print(f"Error: '{col}' column not found in CSV headers.")
                sys.exit(1)

        for row in reader:
            platform = row['Directory'].strip() or row['Platform'].strip()
            for idx in range(1, 100):
                url_col = 'URL' if idx == 1 else f'URL_{idx}'
                title_col = 'Matched_Title' if idx == 1 else f'Matched_Title_{idx}'
                if url_col not in row:
                    break
                url = row[url_col].strip()
                if not url:
                    continue
                title = row.get(title_col, row.get('Matched_Title', '')).strip()
                if not title:
                    title = os.path.basename(url)
                out_dir = os.path.join(NEW_ROOT, platform)
                os.makedirs(out_dir, exist_ok=True)
                key = (url, title, out_dir)
                if key not in seen:
                    entry = url + '\n'
                    entry += f"  out={title}\n"
                    entry += f"  dir={out_dir}\n"
                    entries.append(entry)
                    seen.add(key)

                extras = find_extra_discs(url, title)
                if extras:
                    print(f"Multi-disc detected for '{title}':")
                    for _, etitle in extras:
                        print(f"  -> {etitle}")
                    for extra_url, extra_title in extras:
                        key = (extra_url, extra_title, out_dir)
                        if key in seen:
                            continue
                        extra_entry = extra_url + '\n'
                        extra_entry += f"  out={extra_title}\n"
                        extra_entry += f"  dir={out_dir}\n"
                        entries.append(extra_entry)
                        seen.add(key)

    if not entries:
        print("No valid download entries found in CSV. Exiting.")
        sys.exit(0)

    # Write aria2 input
    with open(links_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(entries))
    print(f"Wrote {len(entries)} entries to {links_path}")

    # Run aria2c
    cmd = ['aria2c', '-i', links_path, '-j', str(args.concurrent)]
    print("Executing:", ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: aria2c not found. Install aria2c and add to your PATH.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"aria2c exited with code {e.returncode}")
        sys.exit(e.returncode)

    print("Downloads complete.")

if __name__ == '__main__':
    main()
