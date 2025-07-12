#!/usr/bin/env python3
"""Convert disc images to CHD format using chdman.

This script scans ROM folders for disc-based consoles and converts
.cue/.bin/.gdi/.iso files to .chd. After a successful conversion the
original files are removed.
"""
import os
import argparse
import subprocess
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, 'roms'))
DEFAULT_CHDMAN = os.path.join(SCRIPT_DIR, 'chdman.exe')

# Consoles that support CHD
CHD_CONSOLES = [
    '3do', 'amigacd32', 'cdi', 'dreamcast', 'naomi',
    'pcenginecd', 'pcfx', 'ps2', 'psx', 'saturn', 'segacd', 'xbox'
]

TARGET_EXTS = {'.cue', '.bin', '.gdi', '.iso'}


def parse_cue_files(cue_path):
    """Return a list of files referenced by the cue sheet."""
    files = []
    try:
        with open(cue_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                m = re.search(r'^\s*FILE\s+"([^"]+)"', line, re.IGNORECASE)
                if m:
                    files.append(m.group(1))
    except OSError as e:
        print(f"Error reading {cue_path}: {e}")
    return files


def convert_image(chdman, input_path, output_path):
    """Run chdman to convert the image and return True on success."""
    cmd = [chdman, 'createcd', '-i', input_path, '-o', output_path]
    print(' '.join(cmd))
    result = subprocess.run(cmd)
    return result.returncode == 0


def remove_files(base_dir, paths):
    for name in paths:
        p = os.path.join(base_dir, name)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                print(f"Error deleting {p}: {e}")


def process_directory(console_dir, chdman):
    for dirpath, _, files in os.walk(console_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in TARGET_EXTS:
                continue
            in_path = os.path.join(dirpath, fname)
            out_path = os.path.splitext(in_path)[0] + '.chd'
            if os.path.exists(out_path):
                continue
            print(f"Converting {in_path} -> {out_path}")
            if convert_image(chdman, in_path, out_path):
                if ext == '.cue':
                    referenced = parse_cue_files(in_path)
                    remove_files(dirpath, referenced)
                # delete the source file itself
                remove_files(dirpath, [fname])
            else:
                print(f"Failed to convert {in_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert disc images to CHD format and remove originals.'
    )
    parser.add_argument('--root-dir', default=DEFAULT_ROOT,
                        help='Root ROM directory (default: ../roms)')
    parser.add_argument('--chdman', default=DEFAULT_CHDMAN,
                        help='Path to chdman executable (default: ./chdman.exe)')
    args = parser.parse_args()

    root = os.path.abspath(args.root_dir)
    chdman = os.path.abspath(args.chdman)

    for folder in CHD_CONSOLES:
        console_path = os.path.join(root, folder)
        if os.path.isdir(console_path):
            process_directory(console_path, chdman)


if __name__ == '__main__':
    main()
