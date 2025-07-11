#!/usr/bin/env python3
import os
import argparse
import re

"""
Scan each console subfolder under the ROMs root for files with "disc" in their names,
then generate an .m3u playlist per multi-disc title listing all disc files in order.
"""

def find_multi_disc_sets(root_dir):
    """
    Walk the directory tree one level deep under root_dir, and find files
    containing 'disc' in their name. Group them by common base name.
    Returns a dict mapping folder_path -> { base_name: [(disc_number, filename), ...], ... }
    """
    multi_disc = {}
    # pattern: capture base name and disc number
    pattern = re.compile(r'(?i)^(.+?)\s*\(disc\s*(\d+)\)')

    for console in os.listdir(root_dir):
        console_path = os.path.join(root_dir, console)
        if not os.path.isdir(console_path):
            continue

        groups = {}
        for fname in os.listdir(console_path):
            if 'disc' not in fname.lower():
                continue
            m = pattern.match(fname)
            if not m:
                continue
            base = m.group(1).strip()
            disc_num = int(m.group(2))
            groups.setdefault(base, []).append((disc_num, fname))

        # keep only true multi-disc sets
        multi = {base: discs for base, discs in groups.items() if len(discs) > 1}
        if multi:
            multi_disc[console_path] = multi
    return multi_disc


def write_playlists(multi_disc):
    """
    Given the multi_disc mapping, write an .m3u file for each base in its folder.
    """
    for folder_path, sets in multi_disc.items():
        for base, discs in sets.items():
            # sort by disc number
            sorted_discs = sorted(discs, key=lambda x: x[0])
            playlist_name = f"{base}.m3u"
            playlist_path = os.path.join(folder_path, playlist_name)
            with open(playlist_path, 'w', encoding='utf-8') as pf:
                for _, fname in sorted_discs:
                    pf.write(fname + '\n')
            print(f"Created playlist: {playlist_path} ({len(sorted_discs)} tracks)")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_root = os.path.abspath(os.path.join(script_dir, os.pardir, 'roms'))

    parser = argparse.ArgumentParser(
        description="Generate .m3u playlists for multi-disc ROM sets."
    )
    parser.add_argument(
        "--root-dir",
        default=default_root,
        help="Top-level ROMs directory containing per-console subfolders (default: ../roms)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root_dir):
        print(f"Error: root directory not found: {args.root_dir}")
        return

    multi_disc = find_multi_disc_sets(args.root_dir)
    if not multi_disc:
        print("No multi-disc sets found.")
        return

    write_playlists(multi_disc)


if __name__ == '__main__':
    main()
