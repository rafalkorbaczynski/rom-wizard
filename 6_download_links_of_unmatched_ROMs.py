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

# Determine directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = SCRIPT_DIR
# Place downloaded files under a sibling 'new_roms' directory relative to the
# RetroBat folder. Since this script lives in 'RetroBat/scripts', we only need
# to go up one directory level to reach the RetroBat root.
NEW_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, 'new_roms'))

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

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        required = ('Platform', 'Directory', 'Matched_Title', 'URL')
        for col in required:
            if col not in reader.fieldnames:
                print(f"Error: '{col}' column not found in CSV headers.")
                sys.exit(1)

        for row in reader:
            url = row['URL'].strip()
            platform = row['Directory'].strip() or row['Platform'].strip()
            title = row['Matched_Title'].strip()
            if not url:
                continue
            # determine output directory
            out_dir = os.path.join(NEW_ROOT, platform)
            os.makedirs(out_dir, exist_ok=True)
            # prepare aria2 entry
            entry = url + '\n'
            entry += f"  out={title}\n"
            entry += f"  dir={out_dir}\n"
            entries.append(entry)

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
