#!/usr/bin/env python3
"""Generate a list of ROM filenames under the given root directory.

Recursively scans each subfolder for files with extensions allowed in
``2_relocate_duplicate_ROMs.py`` and writes the relative paths to a text
file. This helps analyze region naming patterns used across a collection.
"""
import os
import argparse

# Determine script directory for default paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, 'roms'))
DEFAULT_OUT = os.path.join(SCRIPT_DIR, 'rom_filenames.txt')

# Allowed ROM formats (mirrors script 2)
ALLOWED_FORMATS = {
    'a26','a52','a78','bin','chd','gb','gba','gbc','iso','j64',
    'md','mp4','nds','nes','pce','rvz','sfc','sms','xex','xml','z64','zip'
}


def parse_args():
    p = argparse.ArgumentParser(
        description='Collect all ROM filenames under ROOT_DIR.'
    )
    p.add_argument('--root-dir', default=DEFAULT_ROOT,
                   help='Top-level ROMs directory (default: ../roms)')
    p.add_argument('--output', default=DEFAULT_OUT,
                   help='Output text file (default: rom_filenames.txt)')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.root_dir):
        print(f"Error: root directory not found: {args.root_dir}")
        return

    filenames = []
    for dirpath, _, files in os.walk(args.root_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower().lstrip('.')
            if ext in ALLOWED_FORMATS:
                rel = os.path.relpath(os.path.join(dirpath, fname), args.root_dir)
                filenames.append(rel)

    filenames.sort(key=str.lower)
    with open(args.output, 'w', encoding='utf-8') as f:
        for name in filenames:
            f.write(name + '\n')

    print(f"Wrote {len(filenames)} filenames to {args.output}")


if __name__ == '__main__':
    main()
