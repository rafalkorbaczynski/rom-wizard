import os
import sys
import csv
import shutil
import datetime
import math
from dataclasses import dataclass
from typing import Optional, Iterable

import pandas as pd
import xml.etree.ElementTree as ET
from rapidfuzz import process, fuzz
import requests
from bs4 import BeautifulSoup
from difflib import SequenceMatcher
import subprocess
from urllib.parse import urljoin, urlparse
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
SNAPSHOT_BASE_DIR = os.path.normpath('D:/')
ROM_STATE_DIRS = {
    'main': 'roms_main',
    'duplicates': 'roms_duplicates',
    'unrated': 'roms_unrated',
    'unpopular': 'roms_unpopular',
    'new': 'roms_new',
}

_ROMAN = {'ix':9,'viii':8,'vii':7,'vi':6,'iv':4,'iii':3,'ii':2,'i':1}
import re
import unicodedata
_ROMAN_RE = re.compile(r'\b(' + '|'.join(sorted(_ROMAN, key=len, reverse=True)) + r')\b')

BETA_DEMO_TAG_RE = re.compile(r'\((beta|demo)\)', re.IGNORECASE)
BETA_DEMO_WORD_RE = re.compile(r'\b(beta|demo)\b', re.IGNORECASE)
BETA_DEMO_PROTO_TAG_RE = re.compile(r'\((beta|prototype|demo)\)', re.IGNORECASE)
NUMBERED_DUPLICATE_RE = re.compile(r'^(?P<stem>.+?)\.(?P<index>\d+)(?P<ext>\.[^.]+)$', re.IGNORECASE)

# Region mapping for summary statistics
REGION_SYNONYMS = {
    'EU': ['eu', 'europe', 'eur', 'pal', 'uk'],
    'US': ['us', 'usa', 'ntsc', 'ntsc-u'],
    'JP': ['jp', 'jpn', 'japan', 'ntsc-j'],
    'World': ['world']
}
REGIONS = list(REGION_SYNONYMS.keys()) + ['Other']

SUMMARY_ROM_SALES_MATCH_COL = 'ROMs\nSales match'
SUMMARY_DATASET_SALES_COL = 'Dataset\nSales entries'
SUMMARY_SALES_COVERAGE_COL = 'Sales %'
SUMMARY_ROM_RATING_MATCH_COL = 'ROMs\nAge rating'
SUMMARY_DATASET_RATING_COL = 'Dataset\nAge ratings'
SUMMARY_AGE_COVERAGE_COL = 'Age %'
SUMMARY_IGNORE_AGE_COL = 'Ignore age'
SUMMARY_AVG_SIZE_COL = 'Avg (MB)'

# Extensions considered "ROM" files for summary statistics and duplicate
# detection.  Metadata like videos or gamelist files should be ignored.
ROM_EXTS = {
    'a26','a52','a78','bin','chd','gb','gg','gba','gbc','iso','j64','int',
    'md','nds','nes','ngc','pce','psvita','rvz','sfc','sms','sg','xex','z64','zip','vb'
}

# Some platforms (notably PS Vita) use directory-based dumps with a file
# extension suffix on the folder name.  Treat those specially so the wizard
# recognises them as ROM entries when scanning the library.
ROM_DIR_EXTS = {'psvita'}

# Directory prefixes that correspond to firmware/system files that should be
# ignored when scanning the ROM library.  The prefixes are relative to
# ``ROMS_ROOT`` and compared in a case-insensitive manner.
SYSTEM_PATH_PREFIXES = [
    ('psp', 'psp_game'),
    ('psvita', 'os0'),
    ('psvita', 'pd0'),
    ('psvita', 'ux0'),
    ('psvita', 'vs0'),
]


def rom_extension(name: str) -> str:
    """Return the lower-case extension (without leading dot) for ``name``."""

    return os.path.splitext(name)[1].lower().lstrip('.')


def has_rom_extension(name: str) -> bool:
    """Return ``True`` if ``name`` matches a known ROM extension."""

    return rom_extension(name) in ROM_EXTS


def is_rom_directory_name(name: str) -> bool:
    """Return ``True`` when ``name`` represents a directory-based ROM entry."""

    return rom_extension(name) in ROM_DIR_EXTS


def _normalized_parts(rel_path: str) -> list[str]:
    """Return normalised, lower-case path components for ``rel_path``."""

    if not rel_path:
        return []
    normalized = os.path.normpath(rel_path).replace('\\', '/').split('/')
    return [part.lower() for part in normalized if part not in {'', '.'}]


SYSTEM_PATH_PREFIXES = [tuple(part.lower() for part in prefix) for prefix in SYSTEM_PATH_PREFIXES]


def is_system_rom_path(rel_path: str | os.PathLike[str] | None) -> bool:
    """Return ``True`` if ``rel_path`` resides inside a known system directory."""

    if rel_path is None:
        return False
    parts = _normalized_parts(str(rel_path))
    if not parts:
        return False
    for prefix in SYSTEM_PATH_PREFIXES:
        if len(parts) >= len(prefix) and parts[: len(prefix)] == list(prefix):
            return True
    return False


def relative_to_roms_root(path: str) -> str | None:
    """Return ``path`` relative to ``ROMS_ROOT`` or ``None`` if outside."""

    try:
        rel = os.path.relpath(path, ROMS_ROOT)
    except ValueError:
        return None
    if rel.startswith('..'):
        return None
    return os.path.normpath(rel)


def is_system_rom_full_path(path: str) -> bool:
    """Return ``True`` if absolute ``path`` lives under a known system directory."""

    rel = relative_to_roms_root(path)
    if rel is None:
        return False
    return is_system_rom_path(rel)


def directory_size_bytes(path: str, extensions: Optional[set[str]] = None) -> int:
    """Return the cumulative size of files under ``path``.

    If ``extensions`` is provided, only files whose extension matches one of
    the entries (case-insensitive, without the leading dot) are counted.
    ``OSError`` from unreadable files is ignored so a single problematic file
    does not abort the scan.
    """

    total = 0
    for root, dirnames, files in os.walk(path):
        rel_root = relative_to_roms_root(root)
        if is_system_rom_path(rel_root):
            dirnames[:] = []
            continue

        dirnames[:] = [
            d for d in dirnames if not is_system_rom_full_path(os.path.join(root, d))
        ]

        if extensions is not None:
            rel_root_local = os.path.relpath(root, path)
            if rel_root_local == '.':
                special_context = False
            else:
                parts = [p for p in rel_root_local.split(os.sep) if p not in {'.', ''}]
                special_context = any(is_rom_directory_name(p) for p in parts)
        else:
            special_context = False
        for fname in files:
            if is_system_rom_full_path(os.path.join(root, fname)):
                continue
            if extensions is not None:
                ext = rom_extension(fname)
                if ext not in extensions and not special_context:
                    continue
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                continue
    return total


def count_rom_entries(path: str) -> int:
    """Return the number of ROM entries beneath ``path``."""

    if not os.path.isdir(path):
        return 0

    total = 0
    for root, dirnames, files in os.walk(path):
        rel_root = relative_to_roms_root(root)
        if is_system_rom_path(rel_root):
            dirnames[:] = []
            continue

        dirnames[:] = [
            d for d in dirnames if not is_system_rom_full_path(os.path.join(root, d))
        ]
        rom_dirs = [d for d in dirnames if is_rom_directory_name(d)]
        total += len(rom_dirs)
        dirnames[:] = [d for d in dirnames if d not in rom_dirs]
        for fname in files:
            if is_system_rom_full_path(os.path.join(root, fname)):
                continue
            if has_rom_extension(fname):
                total += 1
    return total


def iter_rom_library_entries(path: str):
    """Yield (relative_path, entry_name) pairs for ROM entries under ``path``."""

    if not os.path.isdir(path):
        return

    for root, dirnames, files in os.walk(path):
        rel_root_full = relative_to_roms_root(root)
        if is_system_rom_path(rel_root_full):
            dirnames[:] = []
            continue

        cleaned_dirnames: list[str] = []
        rom_dirs: list[str] = []
        for d in dirnames:
            full_dir = os.path.join(root, d)
            if is_system_rom_full_path(full_dir):
                continue
            if is_rom_directory_name(d):
                rom_dirs.append(d)
            else:
                cleaned_dirnames.append(d)
        dirnames[:] = cleaned_dirnames

        rel_root = os.path.relpath(root, path)
        rel_root = '' if rel_root == '.' else rel_root
        for d in rom_dirs:
            rel = os.path.join(rel_root, d) if rel_root else d
            yield os.path.normpath(rel), d

        for fname in files:
            if is_system_rom_full_path(os.path.join(root, fname)):
                continue
            if has_rom_extension(fname):
                rel = os.path.join(rel_root, fname) if rel_root else fname
                yield os.path.normpath(rel), fname


def get_rom_state_dir(snapshot_dir: str, state: str) -> str:
    """Return the path for the ``state`` ROM directory within ``snapshot_dir``."""

    try:
        subdir = ROM_STATE_DIRS[state]
    except KeyError as exc:
        raise KeyError(f"Unknown ROM state: {state}") from exc
    return os.path.join(snapshot_dir, subdir)


def ensure_rom_state_dir(snapshot_dir: str, state: str) -> str:
    """Ensure the ROM directory for ``state`` exists within ``snapshot_dir``."""

    path = get_rom_state_dir(snapshot_dir, state)
    os.makedirs(path, exist_ok=True)
    return path


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
    if 'ignore_ratings' in df.columns:
        df['ignore_ratings'] = (
            df['ignore_ratings'].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
        )
    else:
        df['ignore_ratings'] = False

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
    ignore_ratings_set = {
        row['Platform']
        for _, row in df.iterrows()
        if bool(row.get('ignore_ratings'))
    }
    active = df[~df['ignore']]
    plat_map = {row['Directory'].lower(): row['Platform'] for _, row in active.iterrows()}
    dataset_to_console = {row['Platform'].lower(): row['Directory'] for _, row in active.iterrows()}
    for alias, code in DIR_ALIASES.items():
        if code.lower() in dataset_to_console:
            plat_map.setdefault(alias, code)
    return (
        plat_map,
        dataset_to_console,
        {p.upper() for p in ignore_set},
        {p.upper() for p in ignore_ratings_set},
    )


PLAT_MAP, DATASET_TO_CONSOLE, IGNORED_PLATFORMS, IGNORE_RATINGS_PLATFORMS = load_platform_mappings()

# Rich console for nicer output
console = Console()


def parse_platform_urls(url_field: str) -> list[str]:
    """Split a platform URL field into individual URLs.

    ``platforms.csv`` entries historically allowed a single URL, but some
    platforms now provide a semi-colon separated list so that multiple archive
    pages can be indexed together.  The data is user-maintained, meaning we may
    encounter a mixture of separators (``;``, newlines, Windows ``\r\n``) or
    accidental whitespace.  Treat any semi-colon immediately followed by a URL
    as a delimiter while leaving literal query-string semi-colons untouched.
    """

    if not isinstance(url_field, str):
        return []

    normalized = url_field.replace('\r', '\n')
    parts = re.split(r';(?=https?://)|\n+', normalized)
    return [p.strip() for p in parts if p and p.strip()]


def filename_from_url(full_url: str) -> str:
    """Return the decoded filename component for ``full_url``."""

    path = urlparse(full_url).path
    return requests.utils.unquote(os.path.basename(path))


def fetch_platform_file_index(url_field: str):
    """Return available download entries for the URLs in ``url_field``."""

    urls = parse_platform_urls(url_field)
    if not urls:
        return [], {}, {}, {}

    file_map = {}
    file_names = {}
    group_map = {}
    seen_urls = set()
    errors: list[tuple[str, Exception]] = []

    session = requests.Session()
    default_headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/124.0 Safari/537.36 ROMWizard/1.0'
    }

    for page_url in urls:
        # Normalize Archive.org item URLs to the canonical download listing with
        # a trailing slash so relative file hrefs resolve correctly.
        try:
            parsed = urlparse(page_url)
            if parsed.netloc.endswith('archive.org'):
                parts = parsed.path.strip('/').split('/')
                if parts and parts[0] in {'details', 'download'} and len(parts) >= 2:
                    item = parts[1]
                    # Rebuild as https://archive.org/download/<item>/
                    page_url = f"https://archive.org/download/{item}/"
        except Exception:
            # If anything goes wrong in normalization, fall back to original URL
            pass
        try:
            resp = session.get(page_url, headers=default_headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            errors.append((page_url, exc))
            if len(urls) == 1:
                raise
            console.print(f"[bold yellow]Warning:[/] Failed to fetch {page_url}: {exc}")
            continue

        soup = BeautifulSoup(resp.text, 'html.parser')
        # Prefer anchors inside Archive.org directory listing table when present
        anchor_scope = soup.select('table.directory-listing-table a[href]') or soup.find_all('a', href=True)
        for anchor in anchor_scope:
            href = anchor['href']
            # Use the (possibly normalized) page_url as base to avoid servers
            # that don't redirect to a trailing slash form.
            full_url = urljoin(page_url, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            path = urlparse(full_url).path
            if not has_rom_extension(path):
                continue
            name = requests.utils.unquote(os.path.splitext(os.path.basename(path))[0])
            if not name:
                continue
            low = name.lower()
            if '(virtual console)' in low:
                continue
            key = norm(name)
            file_map[key] = full_url
            file_names[key] = name
            base_key = norm(remove_disc(name))
            group_map.setdefault(base_key, []).append(key)

        # Fallback: Archive.org exposes a <item>_files.xml index with all files.
        # If we didn't capture anything via anchors and this is an archive.org
        # item page, attempt to parse that XML.
        if not file_map:
            try:
                parsed = urlparse(page_url)
                if parsed.netloc.endswith('archive.org'):
                    parts = parsed.path.strip('/').split('/')
                    if parts and parts[0] in {'details', 'download'} and len(parts) >= 2:
                        item = parts[1]
                        files_xml_url = f"https://archive.org/download/{item}/{item}_files.xml"
                        fx = session.get(files_xml_url, headers=default_headers, timeout=30)
                        if fx.ok and fx.text:
                            try:
                                root = ET.fromstring(fx.text)
                                for f in root.findall('.//file'):
                                    fname = f.get('name') or ''
                                    if not fname:
                                        continue
                                    if not has_rom_extension(fname):
                                        continue
                                    name = requests.utils.unquote(os.path.splitext(os.path.basename(fname))[0])
                                    key = norm(name)
                                    full_url = f"https://archive.org/download/{item}/{requests.utils.quote(fname)}"
                                    if full_url in seen_urls:
                                        continue
                                    seen_urls.add(full_url)
                                    file_map[key] = full_url
                                    file_names[key] = name
                                    base_key = norm(remove_disc(name))
                                    group_map.setdefault(base_key, []).append(key)
                            except ET.ParseError:
                                pass
            except Exception:
                pass

    if not file_map and errors and len(errors) == len(urls):
        raise errors[-1][1]
    return urls, file_map, file_names, group_map


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
            if col in {'Year', 'Generation'}:
                total_row[col] = ''
            elif col in {SUMMARY_SALES_COVERAGE_COL, 'Sales Coverage %'}:
                numerator_candidates = [
                    SUMMARY_ROM_SALES_MATCH_COL,
                    'Sales ROMs',
                    'Matched ROMs',
                ]
                denominator_candidates = [
                    SUMMARY_DATASET_SALES_COL,
                    'Dataset',
                    'ROMs',
                ]
                numerator_col = next((c for c in numerator_candidates if c in df.columns), None)
                denominator_col = next((c for c in denominator_candidates if c in df.columns), None)
                if numerator_col and denominator_col:
                    denom_total = df[denominator_col].sum()
                    num_total = df[numerator_col].sum()
                    pct = num_total / denom_total * 100 if denom_total else 0
                    total_row[col] = round(pct, 1)
                else:
                    total_row[col] = 0
            elif col in {SUMMARY_AGE_COVERAGE_COL, 'Age Coverage %'}:
                numerator_candidates = [SUMMARY_ROM_RATING_MATCH_COL, 'ROMs with age rating']
                denominator_candidates = [SUMMARY_DATASET_RATING_COL, 'Age-eligible ROMs']
                numerator_col = next((c for c in numerator_candidates if c in df.columns), None)
                denominator_col = next((c for c in denominator_candidates if c in df.columns), None)
                if numerator_col and denominator_col:
                    denom_total = df[denominator_col].sum()
                    num_total = df[numerator_col].sum()
                    pct = num_total / denom_total * 100 if denom_total else 0
                    total_row[col] = round(pct, 1)
                else:
                    total_row[col] = 0
            elif col.endswith('%'):
                numeric_values = pd.to_numeric(df[col], errors='coerce').dropna()
                total_row[col] = round(numeric_values.mean(), 1) if not numeric_values.empty else 0
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
        for col_name, value in zip(df_disp.columns, row):
            if pd.isna(value):
                display.append('')
                continue
            if isinstance(value, float):
                if col_name in {'Year', 'Generation'}:
                    formatted = str(int(round(value)))
                elif col_name.endswith('%'):
                    formatted = f"{value:.1f}"
                elif value.is_integer():
                    formatted = str(int(round(value)))
                else:
                    formatted = f"{value:.1f}"
            else:
                formatted = str(value)
            display.append(shorten_path(formatted))
        table.add_row(*display)
    console.print(table)


def prepare_summary_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` formatted for on-screen summary display."""

    display_df = df.copy()
    for col in ['Year', 'Generation']:
        if col in display_df.columns:
            display_df[col] = (
                pd.to_numeric(display_df[col], errors='coerce')
                .round()
                .astype('Int64')
            )
    if 'FullName' in display_df.columns:
        platform_codes = (
            display_df['Platform']
            if 'Platform' in display_df.columns
            else pd.Series(index=display_df.index, dtype='object')
        )
        names = display_df['FullName']
        names = names.astype('object').replace('', pd.NA)
        platform_codes = platform_codes.astype('object').replace('', pd.NA)
        display_df['Platform'] = (
            names.combine_first(platform_codes).fillna('Unknown Platform')
        )
        display_df = display_df.drop(columns=['FullName'])
    return display_df


@dataclass
class RatingCoverageResult:
    coverage_df: pd.DataFrame
    unmatched_entries: list[dict[str, str]]
    platforms_scanned: int
    total_roms_scanned: int
    total_with_rating: int


def compute_rating_coverage(
    sales: pd.DataFrame,
    threshold: int,
    progress_label: str = 'Evaluating rating coverage',
) -> RatingCoverageResult:
    """Return rating coverage metrics and unmatched ROM entries."""

    if 'new_rating' not in sales.columns:
        return RatingCoverageResult(pd.DataFrame(), [], 0, 0, 0)

    working = sales.copy()
    working = working[~working['Platform'].isin(IGNORED_PLATFORMS)].copy()
    working['new_rating'] = pd.to_numeric(working['new_rating'], errors='coerce')
    working['key'] = working['Name'].apply(norm)
    working['plat_lower'] = working['Platform'].str.lower()

    rated_sales = working.dropna(subset=['new_rating'])
    rated_sales = rated_sales[
        (rated_sales['new_rating'] >= 3) & (rated_sales['new_rating'] <= 18)
    ]

    platform_groups = {
        plat: grp.drop_duplicates(subset=['key'])
        for plat, grp in rated_sales.groupby('plat_lower')
    }
    all_platform_groups = {
        plat: grp.drop_duplicates(subset=['key'])
        for plat, grp in working.groupby('plat_lower')
    }

    summary_rows: list[dict[str, object]] = []
    unmatched_entries: list[dict[str, str]] = []
    seen_unmatched_paths: set[str] = set()
    platforms_scanned = 0
    total_roms_scanned = 0
    total_with_rating = 0

    for console_name in iter_progress(os.listdir(ROMS_ROOT), progress_label):
        console_dir = os.path.join(ROMS_ROOT, console_name)
        if not os.path.isdir(console_dir):
            continue

        ds_plat = PLAT_MAP.get(console_name.lower())
        if not ds_plat:
            continue

        plat_upper = ds_plat.upper()
        if plat_upper in IGNORED_PLATFORMS:
            continue

        ignore_ratings_platform = plat_upper in IGNORE_RATINGS_PLATFORMS
        plat_key = ds_plat.lower()
        rated_subset = platform_groups.get(plat_key)
        all_subset = all_platform_groups.get(plat_key)

        rom_entries: list[dict[str, str]] = []
        gamelist_entry_paths: set[str] = set()

        gl_path = os.path.join(console_dir, 'gamelist.xml')
        if os.path.isfile(gl_path):
            tree = ET.parse(gl_path)
            for g in tree.getroot().findall('game'):
                path_text = g.findtext('path') or ''
                if not has_rom_extension(path_text):
                    continue
                full_path, rel_path = normalize_rom_entry_path(console_dir, path_text)
                if (
                    not full_path
                    or not rel_path
                    or is_system_rom_path(rel_path)
                ):
                    continue
                title = g.findtext('name') or ''
                rom_entries.append(
                    {
                        'title': title,
                        'key': norm(title),
                        'full_path': full_path,
                        'rel_path': os.path.normpath(rel_path),
                    }
                )
                norm_rel = os.path.normcase(
                    os.path.normpath(os.path.relpath(full_path, console_dir))
                )
                gamelist_entry_paths.add(norm_rel)

        for rel_path, entry_name in iter_rom_library_entries(console_dir):
            rel_from_root = os.path.normpath(os.path.join(console_name, rel_path))
            if is_system_rom_path(rel_from_root):
                continue
            norm_rel = os.path.normcase(os.path.normpath(rel_path))
            if norm_rel in gamelist_entry_paths:
                continue

            full_path = os.path.normpath(os.path.join(console_dir, rel_path))
            if not os.path.exists(full_path):
                continue

            try:
                rom_rel_path = os.path.relpath(full_path, ROMS_ROOT)
            except ValueError:
                continue

            if rom_rel_path.startswith('..'):
                continue

            candidate_name = os.path.splitext(entry_name)[0]
            rom_entries.append(
                {
                    'title': candidate_name,
                    'key': norm(candidate_name),
                    'full_path': full_path,
                    'rel_path': os.path.normpath(rom_rel_path),
                }
            )

        if not rom_entries:
            continue

        platforms_scanned += 1
        total_roms_scanned += len(rom_entries)

        match_keys = rated_subset['key'].tolist() if rated_subset is not None else []
        all_match_keys = (
            all_subset['key'].tolist() if all_subset is not None else []
        )
        key_to_rating = (
            dict(zip(all_subset['key'], all_subset['new_rating']))
            if all_subset is not None
            else {}
        )

        matched = 0
        for entry in rom_entries:
            key = entry['key']
            matched_entry = False
            if key and match_keys:
                res = process.extractOne(key, match_keys, scorer=fuzz.token_sort_ratio)
                if res and res[1] >= threshold and token_set(key) == token_set(res[0]):
                    matched += 1
                    matched_entry = True

            if matched_entry:
                continue

            match_reason = 'no_dataset_entry'
            if key and all_match_keys:
                res_all = process.extractOne(key, all_match_keys, scorer=fuzz.token_sort_ratio)
                if res_all and res_all[1] >= threshold and token_set(key) == token_set(res_all[0]):
                    rating_val = key_to_rating.get(res_all[0])
                    if rating_val is None or not (3 <= rating_val <= 18) or pd.isna(rating_val):
                        match_reason = 'missing_new_rating'

            full_path = entry['full_path']
            rel_path = entry['rel_path']
            if not full_path or not rel_path:
                continue
            if not os.path.exists(full_path):
                continue
            if full_path in seen_unmatched_paths:
                continue

            unmatched_entries.append(
                {
                    'platform_code': ds_plat,
                    'console_name': console_name,
                    'full_path': full_path,
                    'rel_path': rel_path,
                    'reason': match_reason,
                    'ignore_ratings': ignore_ratings_platform,
                    'title': entry.get('title', ''),
                }
            )
            seen_unmatched_paths.add(full_path)

        rom_total = len(rom_entries)
        total_with_rating += matched

        eligible_roms = 0 if ignore_ratings_platform else rom_total
        eligible_with_rating = matched if not ignore_ratings_platform else 0
        if eligible_roms:
            coverage_pct = round(eligible_with_rating / eligible_roms * 100, 1)
        else:
            coverage_pct = math.nan

        summary_rows.append(
            {
                'Platform': ds_plat,
                'ROMs with age rating': matched,
                'Age-eligible ROMs': eligible_roms,
                'Age Coverage %': coverage_pct,
                'Ignore ratings': 'Yes' if ignore_ratings_platform else 'No',
            }
        )

    coverage_df = pd.DataFrame(summary_rows)
    if not coverage_df.empty:
        coverage_df = coverage_df[
            [
                'Platform',
                'ROMs with age rating',
                'Age-eligible ROMs',
                'Age Coverage %',
                'Ignore ratings',
            ]
        ]

    return RatingCoverageResult(
        coverage_df=coverage_df,
        unmatched_entries=unmatched_entries,
        platforms_scanned=platforms_scanned,
        total_roms_scanned=total_roms_scanned,
        total_with_rating=total_with_rating,
    )


def write_rating_gap_entries(snapshot_dir: str, entries: list[dict[str, str]]) -> None:
    """Persist unmatched rating entries for follow-up moves."""

    path = os.path.join(snapshot_dir, 'rating_gap_entries.csv')
    rows = [
        {
            'Platform': entry.get('platform_code', ''),
            'Console': entry.get('console_name', ''),
            'Title': entry.get('title', ''),
            'Reason': entry.get('reason', ''),
            'IgnoreRatings': 'Yes' if entry.get('ignore_ratings') else 'No',
            'FullPath': entry.get('full_path', ''),
            'RelativePath': entry.get('rel_path', ''),
        }
        for entry in entries
    ]
    gap_df = pd.DataFrame(
        rows,
        columns=[
            'Platform',
            'Console',
            'Title',
            'Reason',
            'IgnoreRatings',
            'FullPath',
            'RelativePath',
        ],
    )
    gap_df.to_csv(path, index=False)


def iter_progress(seq, description: str):
    """Iterate with a progress bar if output is a TTY."""
    return track(seq, description=description) if sys.stdout.isatty() else seq


def infer_region_from_filename(text: str) -> str:
    """Infer region identifier from a ROM filename or path."""

    lowered = os.path.basename(text or '').lower()
    tags = re.findall(r'[\(\[]([^()\[\]]+)[\)\]]', lowered)
    for tag in tags:
        for token in re.split('[,;/]', tag):
            tok = token.strip().lower()
            for reg, toks in REGION_SYNONYMS.items():
                if tok in toks:
                    return reg
    return 'Other'


def get_region_key(game) -> str:
    """Return normalized region key for a gamelist entry."""
    region_text = (game.findtext('region') or '').lower().strip()
    if region_text:
        for reg, toks in REGION_SYNONYMS.items():
            if region_text in toks:
                return reg
    text = game.findtext('path') or game.findtext('name') or ''
    return infer_region_from_filename(text)


def prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Prompt the user for a yes/no answer."""
    ans = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    if not ans:
        return default
    return ans in {'y', 'yes'}


def prompt_age_rating() -> int:
    """Return the PEGI rating ceiling selected by the user."""

    valid_choices = {'3', '7', '12', '16', '18'}
    while True:
        choice = input(
            "Oldest allowed age rating (PEGI 3, 7, 12, 16, 18): "
        ).strip()
        if choice in valid_choices:
            return int(choice)
        console.print("[bold red]Please enter one of: 3, 7, 12, 16, 18.[/]")


def update_kidgame_tag(game: ET.Element, value: bool | None) -> None:
    """Set or clear the ``<kidgame>`` tag for ``game`` based on ``value``."""

    for node in game.findall('kidgame'):
        game.remove(node)
    if value is None:
        return
    ET.SubElement(game, 'kidgame').text = 'true' if value else 'false'


def normalize_rom_entry_path(console_dir: str, path_text: str) -> tuple[str | None, str | None]:
    """Return absolute and ROMS_ROOT-relative paths for a gamelist entry."""

    raw = (path_text or '').strip()
    if not raw:
        return None, None

    candidate = raw.replace('\\', os.sep)
    if os.path.isabs(candidate):
        full_path = os.path.normpath(candidate)
    else:
        while candidate.startswith('./'):
            candidate = candidate[2:]
        while candidate.startswith(f'.{os.sep}'):
            candidate = candidate[2:]
        full_path = os.path.normpath(os.path.join(console_dir, candidate))

    try:
        rel_path = os.path.relpath(full_path, ROMS_ROOT)
    except ValueError:
        return None, None

    if rel_path.startswith('..'):
        return None, None

    return full_path, rel_path

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


def unique_preserve_order(keys: Iterable[str]) -> list[str]:
    """Return ``keys`` without duplicates while preserving order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def entry_allows_beta_demo(entry_name: object) -> bool:
    """Return ``True`` if ``entry_name`` explicitly references beta/demo builds."""

    if not isinstance(entry_name, str):
        return False
    return bool(BETA_DEMO_WORD_RE.search(entry_name))


def candidate_allowed_for_entry(file_title: str, entry_name: object) -> bool:
    """Return ``True`` when ``file_title`` is eligible for ``entry_name``."""

    if not isinstance(file_title, str):
        return False
    if not BETA_DEMO_TAG_RE.search(file_title):
        return True
    return entry_allows_beta_demo(entry_name)


def filter_candidate_keys_for_entry(
    keys: Iterable[str], file_names: dict[str, str], entry_name: object
) -> list[str]:
    """Return keys whose titles are valid matches for ``entry_name``."""

    allowed: list[str] = []
    for key in keys:
        name = file_names.get(key)
        if not name:
            continue
        if candidate_allowed_for_entry(name, entry_name):
            allowed.append(key)
    return allowed


def expand_multidisc_keys(
    selected_keys: Iterable[str],
    group_map: dict[str, list[str]],
    file_names: dict[str, str],
    entry_name: object,
) -> list[str]:
    """Return ``selected_keys`` expanded to include all matching disc entries."""

    ordered = unique_preserve_order(selected_keys)
    if not ordered:
        return []

    names = [file_names.get(key, '') for key in ordered if file_names.get(key)]
    if not names:
        return ordered

    has_disc = any(DISC_RE.search(name) for name in names)
    if not has_disc:
        # Still filter against beta/demo restrictions while preserving order.
        filtered = unique_preserve_order(
            filter_candidate_keys_for_entry(ordered, file_names, entry_name)
        )
        return filtered or ordered

    base_key = norm(remove_disc(names[0]))
    region_score = region_priority(names[0])
    group_candidates = group_map.get(base_key, ordered)
    same_region = [
        key
        for key in group_candidates
        if region_priority(file_names.get(key, '')) == region_score
    ]
    candidate_pool = same_region or group_candidates
    expanded = unique_preserve_order(
        filter_candidate_keys_for_entry(candidate_pool, file_names, entry_name)
    )
    if not expanded:
        expanded = ordered

    return sorted(expanded, key=lambda k: disc_number(file_names.get(k, '')))


def extract_known_urls(df: pd.DataFrame) -> set[str]:
    """Return a set of URL strings already present in ``df``."""

    if df.empty or 'URL' not in df.columns:
        return set()

    urls: set[str] = set()
    for value in df['URL']:
        if pd.isna(value):
            continue
        if isinstance(value, str):
            if value:
                urls.add(value)
        else:
            text = str(value)
            if text and text.lower() != 'nan':
                urls.add(text)
    return urls


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
        'md','mp4','nds','nes','pce','psvita','rvz','sfc','sms','xex','xml','z64','zip'
    }
    dir_allowed = ROM_DIR_EXTS

    filenames = []
    for dirpath, dirnames, files in os.walk(root_dir):
        rel_root = relative_to_roms_root(dirpath)
        if is_system_rom_path(rel_root):
            dirnames[:] = []
            continue

        dirnames[:] = [
            d for d in dirnames if not is_system_rom_full_path(os.path.join(dirpath, d))
        ]

        rom_dirs = [d for d in dirnames if rom_extension(d) in dir_allowed]
        if rom_dirs:
            dirnames[:] = [d for d in dirnames if d not in rom_dirs]
            for d in rom_dirs:
                full_dir = os.path.join(dirpath, d)
                if is_system_rom_full_path(full_dir):
                    continue
                rel = os.path.relpath(full_dir, root_dir)
                filenames.append(rel)
        for fname in files:
            if is_system_rom_full_path(os.path.join(dirpath, fname)):
                continue
            ext = rom_extension(fname)
            if ext in allowed:
                rel = os.path.relpath(os.path.join(dirpath, fname), root_dir)
                filenames.append(rel)

    filenames.sort(key=str.lower)
    with open(output_path, 'w', encoding='utf-8') as f:
        for name in filenames:
            f.write(name + '\n')



def create_snapshot():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    try:
        os.makedirs(SNAPSHOT_BASE_DIR, exist_ok=True)
    except OSError:
        pass
    snap_dir = os.path.join(SNAPSHOT_BASE_DIR, f'snapshot_{ts}')
    os.makedirs(snap_dir, exist_ok=True)
    for state in ROM_STATE_DIRS:
        ensure_rom_state_dir(snap_dir, state)

    threshold = ask_threshold()
    sales = pd.read_csv(SALES_CSV, low_memory=False)
    sales = sales[~sales['Platform'].isin(IGNORED_PLATFORMS)]
    sales['key'] = sales['Name'].apply(norm)
    if 'new_rating' in sales.columns:
        sales['new_rating'] = pd.to_numeric(sales['new_rating'], errors='coerce')
        sales.loc[~sales['new_rating'].between(3, 18, inclusive='both'), 'new_rating'] = math.nan
    else:
        sales['new_rating'] = math.nan
    max_sales = sales['Global_Sales'].max()

    summary_rows = []
    matched_records = []
    unmatched_keys = set(zip(sales['Platform'], sales['key']))
    found_consoles = set()
    gamelist_roms_total = 0
    filesystem_only_total = 0

    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Scanning ROMs"):
        console_dir = os.path.join(ROMS_ROOT, console_name)
        gl_path = os.path.join(console_dir, 'gamelist.xml')
        ds_plat = PLAT_MAP.get(console_name.lower())
        if not ds_plat or ds_plat in IGNORED_PLATFORMS:
            continue
        found_consoles.add(console_name.lower())
        games: list[ET.Element] = []
        gamelist_entry_paths: set[str] = set()
        if os.path.isfile(gl_path):
            tree = ET.parse(gl_path)
            for g in tree.getroot().findall('game'):
                path_text = g.findtext('path') or ''
                if not has_rom_extension(path_text):
                    continue
                full_path, rel_path = normalize_rom_entry_path(console_dir, path_text)
                if (
                    not full_path
                    or not rel_path
                    or is_system_rom_path(rel_path)
                ):
                    continue
                games.append(g)
                rel = os.path.relpath(full_path, console_dir)
                norm_rel = os.path.normcase(os.path.normpath(rel))
                gamelist_entry_paths.add(norm_rel)
        sales_subset = sales[sales['Platform'].str.lower() == ds_plat.lower()]
        match_keys = list(sales_subset['key'])
        sales_map = dict(zip(sales_subset['key'], sales_subset['Global_Sales']))
        name_map = dict(zip(sales_subset['key'], sales_subset['Name']))
        rating_map = {}
        for key, rating in zip(sales_subset['key'], sales_subset['new_rating']):
            if key not in rating_map or pd.isna(rating_map[key]):
                rating_map[key] = rating if pd.notna(rating) else None
        dataset_size = sales_subset['key'].nunique()
        rating_values = pd.to_numeric(sales_subset['new_rating'], errors='coerce')
        dataset_age_total = sales_subset.loc[
            (rating_values >= 3) & (rating_values <= 18),
            'key',
        ].nunique()

        folder_bytes = directory_size_bytes(console_dir, ROM_EXTS)
        fs_rom_total = count_rom_entries(console_dir)
        rom_total = max(len(games), fs_rom_total)
        if rom_total == 0:
            continue
        gamelist_roms_total += len(games)
        fs_only = max(rom_total - len(games), 0)
        filesystem_only_total += fs_only
        avg_size_mb = (folder_bytes / rom_total) / (1024 ** 2) if rom_total else 0

        region_counts = {r: 0 for r in REGIONS}
        matched = 0
        age_rated = 0

        for g in games:
            title = g.findtext('name') or ''
            k = norm(title)

            region_key = get_region_key(g)
            region_counts[region_key] += 1

            res = process.extractOne(k, match_keys, scorer=fuzz.token_sort_ratio)
            if res and res[1] >= threshold and token_set(k) == token_set(res[0]):
                matched += 1
                key = res[0]
                rating_val = rating_map.get(key)
                if pd.notna(rating_val):
                    age_rated += 1
                matched_records.append({
                    'Dataset Name': name_map[key],
                    'Platform': ds_plat,
                    'ROM': title,
                    'Sales': sales_map[key],
                    'Match Score': res[1]
                })
                unmatched_keys.discard((ds_plat, key))

        for rel_path, entry_name in iter_rom_library_entries(console_dir):
            rel_from_root = os.path.normpath(os.path.join(console_name, rel_path))
            if is_system_rom_path(rel_from_root):
                continue
            norm_rel = os.path.normcase(os.path.normpath(rel_path))
            if norm_rel in gamelist_entry_paths:
                continue
            region_key = infer_region_from_filename(entry_name)
            region_counts[region_key] += 1

            if not match_keys:
                continue

            candidate_name = os.path.splitext(entry_name)[0]
            candidate_key = norm(candidate_name)
            if not candidate_key:
                continue

            res = process.extractOne(candidate_key, match_keys, scorer=fuzz.token_sort_ratio)
            if (
                res
                and res[1] >= threshold
                and token_set(candidate_key) == token_set(res[0])
                and (ds_plat, res[0]) in unmatched_keys
            ):
                matched += 1
                key = res[0]
                rating_val = rating_map.get(key)
                if pd.notna(rating_val):
                    age_rated += 1
                matched_records.append({
                    'Dataset Name': name_map[key],
                    'Platform': ds_plat,
                    'ROM': candidate_name,
                    'Sales': sales_map[key],
                    'Match Score': res[1]
                })
                unmatched_keys.discard((ds_plat, key))

        summary_rows.append({
            'Platform': ds_plat,
            'ROMs': rom_total,
            SUMMARY_ROM_SALES_MATCH_COL: matched,
            SUMMARY_DATASET_SALES_COL: dataset_size,
            SUMMARY_DATASET_RATING_COL: dataset_age_total,
            SUMMARY_AVG_SIZE_COL: avg_size_mb,
            **region_counts
        })

    # Add dataset platforms missing from the ROM library
    all_codes = sales['Platform'].dropna().unique()
    for code in all_codes:
        mapped = DATASET_TO_CONSOLE.get(code.lower(), code)
        if mapped.lower() in found_consoles:
            continue
        size = sales[sales['Platform'].str.lower() == code.lower()]['key'].nunique()
        platform_mask = sales['Platform'].str.lower() == code.lower()
        ds_subset = sales.loc[platform_mask]
        rating_values = pd.to_numeric(ds_subset['new_rating'], errors='coerce')
        dataset_age_total = ds_subset.loc[
            (rating_values >= 3) & (rating_values <= 18),
            'key',
        ].nunique()
        summary_rows.append({
            'Platform': code,
            'ROMs': 0,
            SUMMARY_ROM_SALES_MATCH_COL: 0,
            SUMMARY_DATASET_SALES_COL: size,
            SUMMARY_DATASET_RATING_COL: dataset_age_total,
            SUMMARY_AVG_SIZE_COL: 0.0,
            **{r: 0 for r in REGIONS}
        })

    summary_df = pd.DataFrame(summary_rows)

    rating_result = compute_rating_coverage(
        sales,
        threshold,
        progress_label='Evaluating rating coverage',
    )
    if not summary_df.empty and not rating_result.coverage_df.empty:
        summary_df = summary_df.merge(
            rating_result.coverage_df,
            on='Platform',
            how='left',
        )
    write_rating_gap_entries(snap_dir, rating_result.unmatched_entries)

    summary_df = summary_df.rename(
        columns={
            'ROMs with age rating': SUMMARY_ROM_RATING_MATCH_COL,
            'Ignore ratings': SUMMARY_IGNORE_AGE_COL,
        }
    )
    for drop_col in ['Age-eligible ROMs', 'Age Coverage %', 'Age-rated ROMs']:
        if drop_col in summary_df.columns:
            summary_df = summary_df.drop(columns=[drop_col])
    for col in [SUMMARY_DATASET_SALES_COL, SUMMARY_DATASET_RATING_COL]:
        if col not in summary_df.columns:
            summary_df[col] = 0
    if SUMMARY_ROM_RATING_MATCH_COL not in summary_df.columns:
        summary_df[SUMMARY_ROM_RATING_MATCH_COL] = 0
    if SUMMARY_IGNORE_AGE_COL not in summary_df.columns:
        summary_df[SUMMARY_IGNORE_AGE_COL] = 'No'
    if SUMMARY_SALES_COVERAGE_COL not in summary_df.columns:
        summary_df[SUMMARY_SALES_COVERAGE_COL] = pd.Series([math.nan] * len(summary_df))
    if SUMMARY_AGE_COVERAGE_COL not in summary_df.columns:
        summary_df[SUMMARY_AGE_COVERAGE_COL] = pd.Series([math.nan] * len(summary_df))
    if not summary_df.empty:
        summary_df['ROMs'] = (
            pd.to_numeric(summary_df['ROMs'], errors='coerce').fillna(0).astype('Int64')
        )
        summary_df[SUMMARY_ROM_SALES_MATCH_COL] = (
            pd.to_numeric(summary_df[SUMMARY_ROM_SALES_MATCH_COL], errors='coerce')
            .fillna(0)
            .astype('Int64')
        )
        summary_df[SUMMARY_DATASET_SALES_COL] = (
            pd.to_numeric(summary_df[SUMMARY_DATASET_SALES_COL], errors='coerce')
            .fillna(0)
            .astype('Int64')
        )
        summary_df[SUMMARY_DATASET_RATING_COL] = (
            pd.to_numeric(summary_df[SUMMARY_DATASET_RATING_COL], errors='coerce')
            .fillna(0)
            .astype('Int64')
        )
        summary_df[SUMMARY_ROM_RATING_MATCH_COL] = (
            pd.to_numeric(summary_df[SUMMARY_ROM_RATING_MATCH_COL], errors='coerce')
            .fillna(0)
            .astype('Int64')
        )
        summary_df[SUMMARY_AVG_SIZE_COL] = pd.to_numeric(
            summary_df[SUMMARY_AVG_SIZE_COL], errors='coerce'
        ).fillna(0.0)
        for region in REGIONS:
            if region in summary_df.columns:
                summary_df[region] = (
                    pd.to_numeric(summary_df[region], errors='coerce')
                    .fillna(0)
                    .astype('Int64')
                )
        summary_df[SUMMARY_IGNORE_AGE_COL] = summary_df[SUMMARY_IGNORE_AGE_COL].fillna('No')
        with pd.option_context('mode.use_inf_as_na', True):
            rom_den = summary_df['ROMs'].astype(float).replace(0, math.nan)
            summary_df[SUMMARY_SALES_COVERAGE_COL] = (
                summary_df[SUMMARY_ROM_SALES_MATCH_COL]
                / rom_den
                * 100
            )
            summary_df[SUMMARY_AGE_COVERAGE_COL] = (
                summary_df[SUMMARY_ROM_RATING_MATCH_COL]
                / rom_den
                * 100
            )
        ignore_mask = (
            summary_df[SUMMARY_IGNORE_AGE_COL]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(['yes', 'true', '1'])
        )
        summary_df.loc[ignore_mask, SUMMARY_AGE_COVERAGE_COL] = 100.0
        summary_df[SUMMARY_SALES_COVERAGE_COL] = (
            pd.to_numeric(summary_df[SUMMARY_SALES_COVERAGE_COL], errors='coerce')
            .fillna(0)
        )
        summary_df[SUMMARY_AGE_COVERAGE_COL] = (
            pd.to_numeric(summary_df[SUMMARY_AGE_COVERAGE_COL], errors='coerce')
            .fillna(0)
        )
        summary_df[SUMMARY_AVG_SIZE_COL] = summary_df[SUMMARY_AVG_SIZE_COL].round(1)
        summary_df[SUMMARY_SALES_COVERAGE_COL] = summary_df[SUMMARY_SALES_COVERAGE_COL].round(1)
        summary_df[SUMMARY_AGE_COVERAGE_COL] = summary_df[SUMMARY_AGE_COVERAGE_COL].round(1)

        plat_info = pd.read_csv(
            PLATFORMS_CSV, usecols=['Platform', 'FullName', 'ReleaseYear', 'Generation']
        )
        plat_info = plat_info.rename(columns={'ReleaseYear': 'Year'})
        summary_df = summary_df.merge(plat_info, on='Platform', how='left')
        base_cols = ['Platform', 'FullName', 'Year', 'Generation']
        metric_cols = [
            'ROMs',
            SUMMARY_ROM_SALES_MATCH_COL,
            SUMMARY_DATASET_SALES_COL,
            SUMMARY_SALES_COVERAGE_COL,
            SUMMARY_ROM_RATING_MATCH_COL,
            SUMMARY_DATASET_RATING_COL,
            SUMMARY_AGE_COVERAGE_COL,
            SUMMARY_IGNORE_AGE_COL,
            SUMMARY_AVG_SIZE_COL,
        ] + REGIONS
        ordered_metrics = [c for c in metric_cols if c in summary_df.columns]
        remaining = [
            c for c in summary_df.columns
            if c not in set(base_cols + ordered_metrics)
        ]
        summary_df = summary_df[base_cols + ordered_metrics + remaining]
        for col in ['Year', 'Generation']:
            if col in summary_df.columns:
                summary_df[col] = (
                    pd.to_numeric(summary_df[col], errors='coerce')
                    .round()
                    .astype('Int64')
                )

    summary_csv = os.path.join(snap_dir, 'summary.csv')
    export_df = summary_df.drop(columns=['FullName'], errors='ignore')
    export_df.to_csv(summary_csv, index=False)

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
    display_df = prepare_summary_display(summary_df)
    print_table(display_df)

    total_tracked_roms = gamelist_roms_total + filesystem_only_total
    if total_tracked_roms:
        gamelist_pct = gamelist_roms_total / total_tracked_roms * 100
        console.print(
            "[bold cyan]ROM counting sources:[/] "
            f"{gamelist_roms_total} from gamelist.xml ({gamelist_pct:.1f}%), "
            f"{filesystem_only_total} from filesystem scan"
        )
    else:
        console.print(
            "[bold cyan]ROM counting sources:[/] no ROM entries were detected to tally"
        )
    return snap_dir


def remove_duplicates_and_special_roms(snapshot_dir):
    """Move simple duplicate ROMs and optional beta/prototype/demo builds."""

    dup_root = ensure_rom_state_dir(snapshot_dir, 'duplicates')
    numbered_candidates: list[dict[str, str]] = []
    special_candidates: list[dict[str, str]] = []

    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Scanning for simple duplicates"):
        console_dir = os.path.join(ROMS_ROOT, console_name)
        if not os.path.isdir(console_dir):
            continue
        code = PLAT_MAP.get(console_name.lower())
        if code in IGNORED_PLATFORMS:
            continue
        for root, dirnames, files in os.walk(console_dir):
            rom_dirs = [d for d in dirnames if is_rom_directory_name(d)]
            dirnames[:] = [d for d in dirnames if d not in rom_dirs]
            entries = [(d, True) for d in rom_dirs]
            entries.extend((f, False) for f in files)
            if not entries:
                continue
            lookup = {name.lower(): name for name, _ in entries}
            for name, is_dir in entries:
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, ROMS_ROOT)
                if rel_path.startswith('..'):
                    continue
                if is_dir:
                    if not is_rom_directory_name(name):
                        continue
                else:
                    if not has_rom_extension(name):
                        continue
                match = NUMBERED_DUPLICATE_RE.match(name)
                if match:
                    index_val = int(match.group('index'))
                    if index_val >= 1:
                        base_candidate = f"{match.group('stem')}{match.group('ext')}"
                        base_name = lookup.get(base_candidate.lower())
                        if base_name and base_name != name:
                            base_rel = os.path.relpath(os.path.join(root, base_name), ROMS_ROOT)
                            numbered_candidates.append({'path': full_path, 'rel': rel_path, 'base_rel': base_rel})
                            continue
                if BETA_DEMO_PROTO_TAG_RE.search(name):
                    special_candidates.append({'path': full_path, 'rel': rel_path})

    if not numbered_candidates and not special_candidates:
        console.print('[bold cyan]No numbered duplicates or beta/demo/prototype ROMs were found.[/]')
        return

    def preview(entries: list[dict[str, str]], heading: str) -> None:
        console.print(heading)
        sample = entries[:5]
        for item in sample:
            base_rel = item.get('base_rel')
            if base_rel:
                console.print(f"  - {item['rel']} (keeping {base_rel})")
            else:
                console.print(f"  - {item['rel']}")
        if len(entries) > len(sample):
            console.print(f"  ... and {len(entries) - len(sample)} more")

    moved_any = False
    moved_paths: set[str] = set()

    if numbered_candidates:
        preview(numbered_candidates, f"Found {len(numbered_candidates)} numbered duplicate ROM(s) (e.g. name.1.ext).")
        if prompt_yes_no('Move these numbered duplicates to the duplicates folder?', True):
            for entry in numbered_candidates:
                path = entry['path']
                if path in moved_paths or not os.path.exists(path):
                    continue
                rel = entry['rel']
                dst_dir = os.path.join(dup_root, os.path.dirname(rel))
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(path, os.path.join(dst_dir, os.path.basename(path)))
                moved_paths.add(path)
                moved_any = True
            console.print('[bold green]Numbered duplicates moved.[/]')

    remaining_special = [entry for entry in special_candidates if entry['path'] not in moved_paths]
    if remaining_special:
        preview(remaining_special, f"Found {len(remaining_special)} ROM(s) tagged as beta/prototype/demo.")
        if prompt_yes_no('Move these ROMs to the duplicates folder?', False):
            for entry in remaining_special:
                path = entry['path']
                if path in moved_paths or not os.path.exists(path):
                    continue
                rel = entry['rel']
                dst_dir = os.path.join(dup_root, os.path.dirname(rel))
                os.makedirs(dst_dir, exist_ok=True)
                shutil.move(path, os.path.join(dst_dir, os.path.basename(path)))
                moved_paths.add(path)
                moved_any = True
            console.print('[bold green]Selected beta/prototype/demo ROMs moved.[/]')

    if not moved_any:
        console.print('[bold yellow]No files were moved.[/]')


def detect_duplicates(snapshot_dir):
    from rapidfuzz import fuzz
    threshold = ask_threshold()
    dup_root = ensure_rom_state_dir(snapshot_dir, 'duplicates')
    report_rows = []
    summary_rows = []
    moved_any = False

    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Scanning ROMs"):
        console_dir = os.path.join(ROMS_ROOT, console_name)
        if not os.path.isdir(console_dir):
            continue
        code = PLAT_MAP.get(console_name.lower())
        if code in IGNORED_PLATFORMS:
            continue
        rom_count = 0
        dup_count = 0
        for root, dirnames, files in os.walk(console_dir):
            rom_dirs = [d for d in dirnames if is_rom_directory_name(d)]
            if rom_dirs:
                dirnames[:] = [d for d in dirnames if d not in rom_dirs]
            entries = [
                f for f in files
                if has_rom_extension(f) and 'disc' not in f.lower()
            ]
            entries.extend(rom_dirs)
            if not entries:
                continue
            rom_count += len(entries)
            norm_map = {f: norm(f) for f in entries}
            base_map = {f: norm(remove_disc(f)) for f in entries}
            disc_map = {f: disc_number(f) for f in entries}
            # Break filenames into tokens and drop any leading numbers which are
            # often used by ROM sets for simple indexing (e.g. ``"18. Super"``).
            #
            # These numeric prefixes shouldn't cause a mismatch when detecting
            # duplicates, but numeric tokens appearing later in the name are
            # significant (e.g. "Super Mario Bros. 2" vs "Super Mario Bros. 3").
            token_map = {}
            token_no_num = {}
            num_map = {}
            for f in entries:
                tokens = norm_map[f].split()
                # Drop catalog numbers at the beginning of the filename
                while tokens and tokens[0].isdigit():
                    tokens.pop(0)
                token_map[f] = tokens
                token_no_num[f] = [t for t in tokens if not t.isdigit()]
                num_map[f] = [t for t in tokens if t.isdigit()]
            moved = set()
            for i, f in enumerate(entries):
                if f in moved:
                    continue
                for g in entries[i+1:]:
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
                        moved_any = True
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
    if moved_any:
        console.print(
            "[bold yellow]Reminder:[/] clean gamelists & remove unused media in RetroBat "
            "frontend developer options so non-existent ROMs don't count toward the snapshot summary."
        )
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        print_table(df)


def generate_playlists(snapshot_dir):
    """Create m3u playlists in the snapshot's roms folder."""
    pattern = re.compile(r'(?i)^(.+?)\s*\(disc\s*(\d+)\)')
    base_rom_dir = ensure_rom_state_dir(snapshot_dir, 'main')
    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Generating playlists"):
        folder = os.path.join(ROMS_ROOT, console_name)
        if not os.path.isdir(folder):
            continue
        code = PLAT_MAP.get(console_name.lower())
        if code in IGNORED_PLATFORMS:
            continue
        rom_files = [f for f in os.listdir(folder) if has_rom_extension(f)]
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
            out_folder = os.path.join(base_rom_dir, console_name)
            os.makedirs(out_folder, exist_ok=True)
            path = os.path.join(out_folder, base + '.m3u')
            with open(path, 'w', encoding='utf-8') as f:
                for _,name in discs:
                    f.write(name+'\n')
            print('Playlist created:', shorten_path(path))


def enrich_game_lists(snapshot_dir):
    threshold = ask_threshold()
    enable_parental = prompt_yes_no('Turn on parental controls?')
    allowed_rating = prompt_age_rating() if enable_parental else None
    attach_sales = prompt_yes_no('Attach sales data to game lists?', True)

    sales = pd.read_csv(SALES_CSV, low_memory=False)
    sales = sales[~sales['Platform'].isin(IGNORED_PLATFORMS)]
    sales['key'] = sales['Name'].apply(norm)
    if 'new_rating' in sales.columns:
        sales['new_rating'] = pd.to_numeric(sales['new_rating'], errors='coerce')
        sales.loc[~sales['new_rating'].between(3, 18, inclusive='both'), 'new_rating'] = math.nan
    else:
        sales['new_rating'] = math.nan
    max_sales = sales['Global_Sales'].max()
    unmatched_keys = set(zip(sales['Platform'], sales['key']))
    output_root = ensure_rom_state_dir(snapshot_dir, 'main')
    match_rows = []
    summary_rows = []
    found_consoles = set()

    for console_name in iter_progress(os.listdir(ROMS_ROOT), "Enriching game lists"):
        console_dir = os.path.join(ROMS_ROOT, console_name)
        gl_path = os.path.join(console_dir, 'gamelist.xml')
        if not os.path.isfile(gl_path):
            continue
        ds_plat = PLAT_MAP.get(console_name.lower())
        if not ds_plat or ds_plat in IGNORED_PLATFORMS:
            continue
        found_consoles.add(console_name.lower())
        tree = ET.parse(gl_path)
        root = tree.getroot()
        games = [
            g for g in root.findall('game')
            if has_rom_extension(g.findtext('path') or '')
        ]
        if not games:
            continue

        subset = sales[sales['Platform'].str.lower() == ds_plat.lower()].copy()
        match_keys = list(subset['key'])
        sales_map = dict(zip(subset['key'], subset['Global_Sales']))
        name_map = dict(zip(subset['key'], subset['Name']))
        rating_map = {}
        for key, rating in zip(subset['key'], subset['new_rating']):
            if key not in rating_map or pd.isna(rating_map[key]):
                rating_map[key] = rating if pd.notna(rating) else None
        dataset_size = subset['key'].nunique()
        rating_values = pd.to_numeric(subset['new_rating'], errors='coerce')
        dataset_age_total = subset.loc[
            (rating_values >= 3) & (rating_values <= 18),
            'key',
        ].nunique()
        rom_total = len(games)
        folder_bytes = directory_size_bytes(console_dir, ROM_EXTS)
        avg_size_mb = (folder_bytes / rom_total) / (1024 ** 2) if rom_total else 0

        region_counts = {r: 0 for r in REGIONS}
        matched = 0
        age_rated_count = 0
        ignore_ratings_platform = ds_plat.upper() in IGNORE_RATINGS_PLATFORMS

        for g in games:
            if attach_sales:
                for tag in g.findall('rating') + g.findall('ratingMax'):
                    g.remove(tag)

            title = g.findtext('name') or ''
            k = norm(title)

            region_key = get_region_key(g)
            region_counts[region_key] += 1

            res = process.extractOne(k, match_keys, scorer=fuzz.token_sort_ratio)
            game_rating_value = None
            has_rating = False
            if res and res[1] >= threshold and token_set(k) == token_set(res[0]):
                key = res[0]
                gs = sales_map[key]
                game_rating_value = rating_map.get(key)
                if pd.notna(game_rating_value):
                    has_rating = True
                    age_rated_count += 1
                if attach_sales:
                    rating = gs / max_sales * 100 if max_sales else 0
                    ET.SubElement(g, 'rating').text = f"{rating:.2f}"
                    ET.SubElement(g, 'ratingMax').text = '100'
                match_rows.append({
                    'Dataset Name': name_map[key],
                    'Platform': ds_plat,
                    'ROM': title,
                    'Sales': gs,
                    'Match Score': res[1]
                })
                matched += 1
                unmatched_keys.discard((ds_plat, key))
            if enable_parental:
                if has_rating:
                    kidgame_value = float(game_rating_value) <= allowed_rating
                else:
                    kidgame_value = ignore_ratings_platform
                update_kidgame_tag(g, kidgame_value)
            else:
                update_kidgame_tag(g, None)

        out_dir = os.path.join(output_root, console_name)
        os.makedirs(out_dir, exist_ok=True)
        tree.write(os.path.join(out_dir, 'gamelist.xml'), encoding='utf-8', xml_declaration=True)
        summary_rows.append({
            'Platform': ds_plat,
            'ROMs': rom_total,
            SUMMARY_ROM_SALES_MATCH_COL: matched,
            SUMMARY_DATASET_SALES_COL: dataset_size,
            SUMMARY_DATASET_RATING_COL: dataset_age_total,
            SUMMARY_AVG_SIZE_COL: avg_size_mb,
            **region_counts
        })

    # Add dataset platforms missing from the ROM library
    all_codes = sales['Platform'].dropna().unique()
    for code in all_codes:
        mapped = DATASET_TO_CONSOLE.get(code.lower(), code)
        if mapped.lower() in found_consoles:
            continue
        ds_subset = sales[sales['Platform'].str.lower() == code.lower()]
        size = ds_subset['key'].nunique()
        rating_values = pd.to_numeric(ds_subset['new_rating'], errors='coerce')
        dataset_age_total = ds_subset.loc[
            (rating_values >= 3) & (rating_values <= 18),
            'key',
        ].nunique()
        summary_rows.append({
            'Platform': code,
            'ROMs': 0,
            SUMMARY_ROM_SALES_MATCH_COL: 0,
            SUMMARY_DATASET_SALES_COL: size,
            SUMMARY_DATASET_RATING_COL: dataset_age_total,
            SUMMARY_AVG_SIZE_COL: 0.0,
            **{r: 0 for r in REGIONS}
        })

    summary_df = pd.DataFrame(summary_rows)

    rating_result = compute_rating_coverage(
        sales,
        threshold,
        progress_label='Evaluating rating coverage',
    )
    if not summary_df.empty and not rating_result.coverage_df.empty:
        summary_df = summary_df.merge(
            rating_result.coverage_df,
            on='Platform',
            how='left',
        )
    write_rating_gap_entries(snapshot_dir, rating_result.unmatched_entries)

    summary_df = summary_df.rename(
        columns={
            'ROMs with age rating': SUMMARY_ROM_RATING_MATCH_COL,
            'Ignore ratings': SUMMARY_IGNORE_AGE_COL,
        }
    )
    for drop_col in ['Age-eligible ROMs', 'Age Coverage %', 'Age-rated ROMs']:
        if drop_col in summary_df.columns:
            summary_df = summary_df.drop(columns=[drop_col])
    for col in [SUMMARY_DATASET_SALES_COL, SUMMARY_DATASET_RATING_COL]:
        if col not in summary_df.columns:
            summary_df[col] = 0
    if SUMMARY_ROM_RATING_MATCH_COL not in summary_df.columns:
        summary_df[SUMMARY_ROM_RATING_MATCH_COL] = 0
    if SUMMARY_IGNORE_AGE_COL not in summary_df.columns:
        summary_df[SUMMARY_IGNORE_AGE_COL] = 'No'
    if SUMMARY_SALES_COVERAGE_COL not in summary_df.columns:
        summary_df[SUMMARY_SALES_COVERAGE_COL] = pd.Series([math.nan] * len(summary_df))
    if SUMMARY_AGE_COVERAGE_COL not in summary_df.columns:
        summary_df[SUMMARY_AGE_COVERAGE_COL] = pd.Series([math.nan] * len(summary_df))
    if not summary_df.empty:
        summary_df['ROMs'] = (
            pd.to_numeric(summary_df['ROMs'], errors='coerce').fillna(0).astype('Int64')
        )
        summary_df[SUMMARY_ROM_SALES_MATCH_COL] = (
            pd.to_numeric(summary_df[SUMMARY_ROM_SALES_MATCH_COL], errors='coerce')
            .fillna(0)
            .astype('Int64')
        )
        summary_df[SUMMARY_DATASET_SALES_COL] = (
            pd.to_numeric(summary_df[SUMMARY_DATASET_SALES_COL], errors='coerce')
            .fillna(0)
            .astype('Int64')
        )
        summary_df[SUMMARY_DATASET_RATING_COL] = (
            pd.to_numeric(summary_df[SUMMARY_DATASET_RATING_COL], errors='coerce')
            .fillna(0)
            .astype('Int64')
        )
        summary_df[SUMMARY_ROM_RATING_MATCH_COL] = (
            pd.to_numeric(summary_df[SUMMARY_ROM_RATING_MATCH_COL], errors='coerce')
            .fillna(0)
            .astype('Int64')
        )
        summary_df[SUMMARY_AVG_SIZE_COL] = pd.to_numeric(
            summary_df[SUMMARY_AVG_SIZE_COL], errors='coerce'
        ).fillna(0.0)
        for region in REGIONS:
            if region in summary_df.columns:
                summary_df[region] = (
                    pd.to_numeric(summary_df[region], errors='coerce')
                    .fillna(0)
                    .astype('Int64')
                )
        summary_df[SUMMARY_IGNORE_AGE_COL] = summary_df[SUMMARY_IGNORE_AGE_COL].fillna('No')
        with pd.option_context('mode.use_inf_as_na', True):
            rom_den = summary_df['ROMs'].astype(float).replace(0, math.nan)
            summary_df[SUMMARY_SALES_COVERAGE_COL] = (
                summary_df[SUMMARY_ROM_SALES_MATCH_COL]
                / rom_den
                * 100
            )
            summary_df[SUMMARY_AGE_COVERAGE_COL] = (
                summary_df[SUMMARY_ROM_RATING_MATCH_COL]
                / rom_den
                * 100
            )
        ignore_mask = (
            summary_df[SUMMARY_IGNORE_AGE_COL]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(['yes', 'true', '1'])
        )
        summary_df.loc[ignore_mask, SUMMARY_AGE_COVERAGE_COL] = 100.0
        summary_df[SUMMARY_SALES_COVERAGE_COL] = (
            pd.to_numeric(summary_df[SUMMARY_SALES_COVERAGE_COL], errors='coerce')
            .fillna(0)
        )
        summary_df[SUMMARY_AGE_COVERAGE_COL] = (
            pd.to_numeric(summary_df[SUMMARY_AGE_COVERAGE_COL], errors='coerce')
            .fillna(0)
        )
        summary_df[SUMMARY_AVG_SIZE_COL] = summary_df[SUMMARY_AVG_SIZE_COL].round(1)
        summary_df[SUMMARY_SALES_COVERAGE_COL] = summary_df[SUMMARY_SALES_COVERAGE_COL].round(1)
        summary_df[SUMMARY_AGE_COVERAGE_COL] = summary_df[SUMMARY_AGE_COVERAGE_COL].round(1)

        plat_info = pd.read_csv(
            PLATFORMS_CSV, usecols=['Platform', 'FullName', 'ReleaseYear', 'Generation']
        )
        plat_info = plat_info.rename(columns={'ReleaseYear': 'Year'})
        summary_df = summary_df.merge(plat_info, on='Platform', how='left')
        base_cols = ['Platform', 'FullName', 'Year', 'Generation']
        metric_cols = [
            'ROMs',
            SUMMARY_ROM_SALES_MATCH_COL,
            SUMMARY_DATASET_SALES_COL,
            SUMMARY_SALES_COVERAGE_COL,
            SUMMARY_ROM_RATING_MATCH_COL,
            SUMMARY_DATASET_RATING_COL,
            SUMMARY_AGE_COVERAGE_COL,
            SUMMARY_IGNORE_AGE_COL,
            SUMMARY_AVG_SIZE_COL,
        ] + REGIONS
        ordered_metrics = [c for c in metric_cols if c in summary_df.columns]
        remaining = [
            c for c in summary_df.columns
            if c not in set(base_cols + ordered_metrics)
        ]
        summary_df = summary_df[base_cols + ordered_metrics + remaining]
        for col in ['Year', 'Generation']:
            if col in summary_df.columns:
                summary_df[col] = (
                    pd.to_numeric(summary_df[col], errors='coerce')
                    .round()
                    .astype('Int64')
                )

    export_df = summary_df.drop(columns=['FullName'], errors='ignore')
    export_df.to_csv(os.path.join(snapshot_dir, 'summary.csv'), index=False)
    pd.DataFrame(match_rows).to_csv(os.path.join(snapshot_dir, 'match_summary.csv'), index=False)

    unmatched_list = [{'Platform': p, 'key': k} for p, k in unmatched_keys]
    unmatched_df = pd.DataFrame(unmatched_list)
    unmatched_df = unmatched_df.merge(sales[['Platform','Name','Global_Sales','key']], on=['Platform','key'], how='left')
    unmatched_df = unmatched_df.rename(columns={'Name':'Dataset Name','Global_Sales':'Sales'})
    unmatched_df = unmatched_df[['Dataset Name','Platform','Sales']]
    unmatched_df.to_csv(os.path.join(snapshot_dir, 'unmatched_summary.csv'), index=False)

    console.print('[bold green]Game lists enriched.[/]')
    display_df = prepare_summary_display(summary_df)
    print_table(display_df)

def _gap_rows_to_entries(df: pd.DataFrame) -> list[dict[str, str]]:
    """Convert gap rows into move-entry dictionaries."""

    entries: list[dict[str, str]] = []
    if df.empty:
        return entries
    for row in df.itertuples(index=False):
        full_path = getattr(row, 'FullPath', None)
        rel_path = getattr(row, 'RelativePath', None)
        if not isinstance(full_path, str) or not full_path:
            continue
        if not isinstance(rel_path, str) or not rel_path:
            continue
        entry: dict[str, str] = {
            'full_path': os.path.normpath(full_path),
            'rel_path': os.path.normpath(rel_path),
        }
        ignore_val = getattr(row, 'IgnoreRatings', None)
        if isinstance(ignore_val, bool):
            entry['ignore_ratings'] = ignore_val
        entries.append(entry)
    return entries


def _load_rating_summary_totals(snapshot_dir: str) -> tuple[int, int, int]:
    """Return (platforms, total ROMs, ROMs with age-rating coverage)."""

    summary_path = os.path.join(snapshot_dir, 'summary.csv')
    if not os.path.isfile(summary_path):
        return 0, 0, 0
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return 0, 0, 0

    roms_series = pd.to_numeric(
        df.get('ROMs', pd.Series(dtype='float64')),
        errors='coerce',
    ).fillna(0)
    platforms_scanned = int((roms_series > 0).sum())
    total_roms = int(roms_series.sum())

    rating_col = SUMMARY_ROM_RATING_MATCH_COL
    if rating_col not in df.columns:
        rating_col = 'ROMs with age rating'
    with_rating_series = pd.to_numeric(
        df.get(rating_col, pd.Series(dtype='float64')),
        errors='coerce',
    ).fillna(0)
    total_with_rating = int(with_rating_series.sum())
    return platforms_scanned, total_roms, total_with_rating


def count_new_rating_matches(snapshot_dir=None):
    """Prompt to move ROMs missing sales or rating data using stored results."""

    console.print(
        "[bold cyan]Review the snapshot summary for rating gaps and optionally move "
        "ROMs missing sales or age rating data.[/]"
    )
    if not snapshot_dir:
        print('Snapshot path is required to continue.')
        return

    gap_path = os.path.join(snapshot_dir, 'rating_gap_entries.csv')
    if not os.path.isfile(gap_path):
        print(
            'rating_gap_entries.csv not found. Run a snapshot or apply sales data to refresh the summary.'
        )
        return

    gap_df = pd.read_csv(gap_path)
    if gap_df.empty:
        print('All ROMs have the necessary sales and age rating data.')
        platforms_scanned, total_roms_scanned, total_with_rating = _load_rating_summary_totals(snapshot_dir)
        console.print(
            "[bold green]Rating gap review complete:[/] scanned "
            f"{platforms_scanned} platform(s), {total_roms_scanned} ROM(s), "
            f"and found {total_with_rating} with new-rating coverage."
        )
        return

    gap_df['Reason'] = gap_df.get('Reason', '').astype(str)
    gap_df['IgnoreRatings'] = (
        gap_df.get('IgnoreRatings', 'No')
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(['yes', 'true', '1'])
    )

    sales_only_df = gap_df[gap_df['Reason'] == 'no_dataset_entry']
    missing_rating_df = gap_df[gap_df['Reason'] == 'missing_new_rating']
    moveable_missing_rating_df = missing_rating_df[~missing_rating_df['IgnoreRatings']]
    skipped_missing_rating = int(missing_rating_df['IgnoreRatings'].sum())

    sales_entries = _gap_rows_to_entries(sales_only_df)
    unrated_entries = _gap_rows_to_entries(moveable_missing_rating_df)

    total_gap = len(gap_df)
    missing_entry = len(sales_entries)
    missing_rating = len(unrated_entries)

    print(f"{total_gap} ROM(s) are missing sales or age rating data.")
    if missing_entry:
        print(f" - {missing_entry} ROM(s) are missing sales data.")
    if missing_rating:
        print(
            " - "
            f"{missing_rating} ROM(s) are missing age ratings (platforms that ignore ratings are excluded)."
        )
    if skipped_missing_rating:
        print(
            f" - {skipped_missing_rating} ROM(s) missing age ratings were skipped because their platform ignores ratings."
        )

    unpopular_root = get_rom_state_dir(snapshot_dir, 'unpopular')
    unrated_root = get_rom_state_dir(snapshot_dir, 'unrated')

    if missing_entry and prompt_yes_no(
        f"Move {missing_entry} un-popular ROM(s) without sales data to {shorten_path(unpopular_root)}?",
        default=False,
    ):
        move_unpopular_roms(snapshot_dir, sales_entries)

    if missing_rating and prompt_yes_no(
        "Move "
        f"{missing_rating} un-rated ROM(s) without age ratings (excluding ignore_ratings platforms) "
        f"to {shorten_path(unrated_root)}?",
        default=False,
    ):
        move_unmatched_roms(snapshot_dir, unrated_entries)

    platforms_scanned, total_roms_scanned, total_with_rating = _load_rating_summary_totals(snapshot_dir)
    console.print(
        "[bold green]Rating gap review complete:[/] scanned "
        f"{platforms_scanned} platform(s), {total_roms_scanned} ROM(s), "
        f"and found {total_with_rating} with new-rating coverage."
    )


def move_entries_to(entries: list[dict[str, str]], target_root: str, label: str) -> int:
    """Move ROM ``entries`` to ``target_root`` and report the result."""

    if not entries:
        print(f'No {label} ROMs to move.')
        return 0

    resolved_root = target_root.replace('\\', os.sep) if os.name != 'nt' else target_root
    resolved_root = os.path.abspath(resolved_root)
    os.makedirs(resolved_root, exist_ok=True)

    moved = 0
    for entry in entries:
        src = entry['full_path']
        rel_path = entry['rel_path']
        dest_path = os.path.abspath(os.path.join(resolved_root, rel_path))
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)

        if not os.path.exists(src):
            print('Source missing, skipping:', shorten_path(src))
            continue
        if os.path.exists(dest_path):
            print('Destination exists, skipping:', shorten_path(dest_path))
            continue

        shutil.move(src, dest_path)
        moved += 1

    print(f"Moved {moved} {label} ROM(s) to {shorten_path(resolved_root)}.")
    if moved:
        console.print(
            "[bold yellow]Reminder:[/] clean gamelists & remove unused media in RetroBat "
            "frontend developer options so non-existent ROMs don't count toward the snapshot summary."
        )
    return moved


def move_unmatched_roms(snapshot_dir: str, entries: list[dict[str, str]]) -> int:
    """Move un-rated ROMs to the snapshot's unrated ROM directory."""

    target = ensure_rom_state_dir(snapshot_dir, 'unrated')
    return move_entries_to(entries, target, 'un-rated')


def move_unpopular_roms(snapshot_dir: str, entries: list[dict[str, str]]) -> int:
    """Move ROMs without sales data to the snapshot's unpopular ROM directory."""

    target = ensure_rom_state_dir(snapshot_dir, 'unpopular')
    return move_entries_to(entries, target, 'un-popular')


def scrape_file_list(code):
    df = pd.read_csv(PLATFORMS_CSV)
    df['ignore'] = df.get('ignore', False).astype(str).str.upper().isin(['TRUE', '1', 'YES'])
    row = df[(df['Platform'] == code) & (~df['ignore'])]
    if row.empty:
        return None, None, []
    url_field = row['URL'].iloc[0]
    directory = row['Directory'].iloc[0] or code
    urls, file_map, file_names, _ = fetch_platform_file_index(url_field)
    if not urls:
        return None, directory, []
    entries = [(file_names[k], file_map[k]) for k in file_map]
    return urls[0], directory, entries


def manual_add_games(snapshot_dir):
    """Interactive manual mode for adding download entries."""
    unmatched_path = os.path.join(snapshot_dir, 'unmatched_summary.csv')
    summary_path = os.path.join(snapshot_dir, 'summary.csv')
    if not os.path.isfile(unmatched_path):
        print('unmatched_summary.csv not found. Run snapshot first.')
        return

    unmatched_df = pd.read_csv(unmatched_path)
    summary_df = pd.read_csv(summary_path) if os.path.isfile(summary_path) else pd.DataFrame()

    download_list_path = os.path.join(snapshot_dir, 'download_list.csv')
    existing_dl_df = (
        pd.read_csv(download_list_path) if os.path.exists(download_list_path) else pd.DataFrame()
    )
    known_urls = extract_known_urls(existing_dl_df)
    duplicate_skips = 0

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
        url_field = info.get('url')
        directory = info.get('dir', code) or code
        platform_total = platform_targets.get(code, 0)
        platform_done = 0
        try:
            urls, file_map, file_names, group_map = fetch_platform_file_index(url_field)
        except requests.RequestException as exc:
            print(f'Failed to fetch {code}: {exc}')
            continue
        if not urls:
            print(f'Skipping {code}: no URL configured.')
            continue
        if not file_map:
            print(f'No downloadable entries found for {code}.')
            continue
        subset = unmatched_df[unmatched_df['Platform'] == code]
        subset = subset[~subset['Dataset Name'].apply(lambda n: (n, code) in blacklisted_pairs)]
        subset = subset.sort_values('Sales', ascending=False).head(count)
        if subset.empty:
            print(f'No unmatched entries for {code}.')
            continue
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
                candidate_keys = filter_candidate_keys_for_entry(
                    file_map.keys(), file_names, row['Dataset Name']
                )
                ts_results = process.extract(
                    norm_search, candidate_keys, scorer=fuzz.token_sort_ratio, limit=None
                )
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
            selected_keys = expand_multidisc_keys(
                selected_keys, group_map, file_names, row['Dataset Name']
            )
            fresh_keys: list[str] = []
            for k in selected_keys:
                filehref = file_map[k]
                if filehref in known_urls:
                    duplicate_skips += 1
                    continue
                known_urls.add(filehref)
                fresh_keys.append(k)
            if not fresh_keys:
                clear_screen()
                continue
            for k in fresh_keys:
                filehref = file_map[k]
                title = filename_from_url(filehref)
                download_rows.append({'Search_Term': row['Dataset Name'], 'Platform': code,
                                     'Directory': directory, 'Matched_Title': title,
                                     'Score': score, 'URL': filehref})
            clear_screen()

    if not download_rows:
        if duplicate_skips:
            console.print(f"[bold yellow]Skipped {duplicate_skips} duplicate download(s).[/]")
        print('No entries added.')
        return

    new_df = pd.DataFrame(download_rows)
    if not existing_dl_df.empty:
        df = pd.concat([existing_dl_df, new_df], ignore_index=True)
    else:
        df = new_df
    if 'URL' in df.columns:
        df = df.drop_duplicates(subset=['URL'], keep='first')
    df.to_csv(download_list_path, index=False)
    console.print('[bold green]Download list updated.[/]')
    if duplicate_skips:
        console.print(f"[bold yellow]Skipped {duplicate_skips} duplicate download(s).[/]")
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

    download_list_path = os.path.join(snapshot_dir, 'download_list.csv')
    existing_dl_df = (
        pd.read_csv(download_list_path) if os.path.exists(download_list_path) else pd.DataFrame()
    )
    known_urls = extract_known_urls(existing_dl_df)
    duplicate_skips = 0

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
        url_field = info.get('url')
        directory = info.get('dir', code) or code
        try:
            urls, file_map, file_names, group_map = fetch_platform_file_index(url_field)
        except requests.RequestException as exc:
            print(f'Failed to fetch {code}: {exc}')
            continue
        if not urls:
            print(f'Skipping {code}: no URL configured.')
            continue
        if not file_map:
            print(f'No downloadable entries found for {code}.')
            continue
        subset = unmatched_df[unmatched_df['Platform'] == code]
        subset = subset[~subset['Dataset Name'].apply(lambda n: (n, code) in blacklisted_pairs)]
        subset = subset.sort_values('Sales', ascending=False).head(count)
        if subset.empty:
            print(f'No unmatched entries for {code}.')
            continue
        match_count = 0
        scores = []
        for _, row in subset.iterrows():
            game = row['Dataset Name']
            normed = norm(game)
            candidate_keys = filter_candidate_keys_for_entry(file_map.keys(), file_names, game)
            ts_results = process.extract(
                normed, candidate_keys, scorer=fuzz.token_sort_ratio, limit=None
            )
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
            group_key = norm(remove_disc(file_names[best_key]))
            group_candidates = group_map.get(group_key, [best_key])
            selected_keys = filter_candidate_keys_for_entry(
                group_candidates, file_names, game
            )
            if not selected_keys:
                continue

            selected_keys = expand_multidisc_keys(selected_keys, group_map, file_names, game)
            fresh_keys: list[str] = []
            for k in selected_keys:
                filehref = file_map[k]
                if filehref in known_urls:
                    duplicate_skips += 1
                    continue
                known_urls.add(filehref)
                fresh_keys.append(k)
            if not fresh_keys:
                continue
            for k in fresh_keys:
                filehref = file_map[k]
                title = filename_from_url(filehref)
                download_rows.append({'Search_Term': game, 'Platform': code,
                                     'Directory': directory, 'Matched_Title': title,
                                     'Score': score, 'URL': filehref})
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
        if duplicate_skips:
            console.print(f"[bold yellow]Skipped {duplicate_skips} duplicate download(s).[/]")
        print('No entries added.')
        return

    new_df = pd.DataFrame(download_rows)
    if not existing_dl_df.empty:
        df = pd.concat([existing_dl_df, new_df], ignore_index=True)
    else:
        df = new_df
    if 'URL' in df.columns:
        df = df.drop_duplicates(subset=['URL'], keep='first')
    df.to_csv(download_list_path, index=False)
    console.print('[bold green]Download list updated.[/]')
    if duplicate_skips:
        console.print(f"[bold yellow]Skipped {duplicate_skips} duplicate download(s).[/]")
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows,
                                 columns=['Platform','Added','Match %','Lowest Score','Average Score'])
        print_table(summary_df)
    if not blacklist_df.empty:
        blacklist_df.drop_duplicates().to_csv(BLACKLIST_CSV, index=False)
    if prompt_yes_no('Download now?'):
        download_games(snapshot_dir)


def enforce_download_targets(snapshot_dir):
    """Ensure each platform meets its configured download target."""

    console.print(
        "[bold cyan]Review each platform's download target, add missing download links to "
        "download_list.csv, and respect ignore and rating settings.[/]"
    )
    summary_path = os.path.join(snapshot_dir, 'summary.csv')
    match_path = os.path.join(snapshot_dir, 'match_summary.csv')
    if not os.path.isfile(summary_path):
        print('summary.csv not found. Run snapshot first.')
        return
    if not os.path.isfile(match_path):
        print('match_summary.csv not found. Run snapshot first.')
        return

    try:
        plat_df = pd.read_csv(PLATFORMS_CSV)
    except FileNotFoundError:
        print('platforms.csv not found.')
        return

    if 'download_count' not in plat_df.columns:
        print("The platforms.csv file does not contain a 'download_count' column.")
        return

    try:
        sales_df = pd.read_csv(SALES_CSV, low_memory=False)
    except FileNotFoundError:
        print('sales_2019.csv not found.')
        return

    summary_df = pd.read_csv(summary_path)
    match_df = pd.read_csv(match_path)
    download_list_path = os.path.join(snapshot_dir, 'download_list.csv')
    existing_dl_df = pd.read_csv(download_list_path) if os.path.isfile(download_list_path) else pd.DataFrame()
    known_urls = extract_known_urls(existing_dl_df)
    duplicate_skips = 0

    if os.path.exists(BLACKLIST_CSV):
        blacklist_df = pd.read_csv(BLACKLIST_CSV)
        blacklist_df = blacklist_df[['Search_Term', 'Platform']]
    else:
        blacklist_df = pd.DataFrame(columns=['Search_Term', 'Platform'])

    def canonical(code: str) -> str:
        if not isinstance(code, str):
            return ''
        mapped = PLAT_MAP.get(code.lower())
        return (mapped or code).upper()

    plat_df['Platform'] = plat_df['Platform'].apply(canonical)
    summary_df['Platform'] = summary_df['Platform'].apply(canonical)
    if not match_df.empty:
        match_df['Platform'] = match_df['Platform'].apply(canonical)
    if not existing_dl_df.empty and 'Platform' in existing_dl_df.columns:
        existing_dl_df['Platform'] = existing_dl_df['Platform'].apply(canonical)

    blacklisted_pairs = set(
        (row['Search_Term'], canonical(row['Platform']))
        for _, row in blacklist_df.dropna(subset=['Search_Term', 'Platform']).iterrows()
        if isinstance(row.get('Search_Term'), str)
    )

    plat_df['ignore'] = plat_df.get('ignore', False).astype(str).str.upper().isin(['TRUE', '1', 'YES'])
    ignore_ratings_source = None
    if 'ignore_ratings' in plat_df.columns:
        ignore_ratings_source = 'ignore_ratings'
    elif 'ignore_rating' in plat_df.columns:
        ignore_ratings_source = 'ignore_rating'
    if ignore_ratings_source:
        plat_df['ignore_ratings'] = (
            plat_df[ignore_ratings_source].astype(str).str.upper().isin(['TRUE', '1', 'YES'])
        )
    else:
        plat_df['ignore_ratings'] = False
    plat_df['download_count'] = pd.to_numeric(plat_df['download_count'], errors='coerce').fillna(0).astype(int)

    if 'new_rating' not in sales_df.columns:
        sales_df['new_rating'] = ''
    sales_df['Global_Sales'] = pd.to_numeric(sales_df.get('Global_Sales', 0), errors='coerce').fillna(0)
    sales_df['Platform'] = sales_df['Platform'].astype(str)

    platform_matches = {}
    if not match_df.empty:
        for platform, group in match_df.groupby('Platform'):
            rom_map: dict[str, str | None] = {}
            dataset_names = set()
            for _, mrow in group.iterrows():
                rom_name = mrow.get('ROM')
                dataset_name = mrow.get('Dataset Name')
                if isinstance(rom_name, str):
                    rom_map[rom_name] = dataset_name if isinstance(dataset_name, str) else None
                if isinstance(dataset_name, str):
                    dataset_names.add(dataset_name)
            platform_matches[platform] = {
                'rom_to_dataset': rom_map,
                'dataset_names': dataset_names,
            }

    existing_pairs: set[tuple[str, str]] = set()
    if not existing_dl_df.empty and {'Search_Term', 'Platform'}.issubset(existing_dl_df.columns):
        for _, drow in existing_dl_df.dropna(subset=['Search_Term', 'Platform']).iterrows():
            search_term = drow.get('Search_Term')
            platform = drow.get('Platform')
            if isinstance(search_term, str) and isinstance(platform, str):
                existing_pairs.add((search_term, platform))

    threshold = ask_threshold(default=75)
    console.print(
        f"[bold cyan]Fuzzy match threshold set to {threshold}. Scanning platforms for missing downloads...[/]"
    )

    new_download_rows: list[dict[str, str | int]] = []
    results_rows: list[dict[str, int | str]] = []
    total_added = 0
    platforms_updated = 0

    for _, row in plat_df.iterrows():
        code = row['Platform']
        if not code or row.get('ignore') or code in IGNORED_PLATFORMS:
            continue
        target_count = int(row.get('download_count', 0))
        if target_count <= 0:
            continue

        directory_val = row.get('Directory') if isinstance(row.get('Directory'), str) else None
        directory = directory_val or DATASET_TO_CONSOLE.get(code.lower(), code.lower())
        ignore_ratings = bool(row.get('ignore_ratings'))

        sales_subset = sales_df[sales_df['Platform'].str.lower() == code.lower()].copy()
        if sales_subset.empty:
            print(f'No sales data available for {code}.')
            results_rows.append({'Platform': code, 'Added': 0, 'Still Needed': target_count})
            continue
        sales_subset['Name'] = sales_subset['Name'].astype(str)
        if not ignore_ratings:
            rating_mask = ~sales_subset['new_rating'].astype(str).str.strip().eq('')
            sales_subset = sales_subset[rating_mask]
        sales_subset = sales_subset.sort_values('Global_Sales', ascending=False)
        if sales_subset.empty:
            print(f'No eligible sales entries for {code}.')
            results_rows.append({'Platform': code, 'Added': 0, 'Still Needed': target_count})
            continue

        top_sequence = [name for name in sales_subset['Name'] if name]
        if target_count > 0:
            top_target_set = set(top_sequence[:target_count])
        else:
            top_target_set = set()
        match_info = platform_matches.get(code, {'rom_to_dataset': {}, 'dataset_names': set()})
        dataset_names = {
            name for name in match_info.get('dataset_names', set())
            if isinstance(name, str)
        }

        platform_dl_names: set[str] = set()
        if not existing_dl_df.empty and {'Search_Term', 'Platform'}.issubset(existing_dl_df.columns):
            dl_subset = existing_dl_df[existing_dl_df['Platform'] == code]
            platform_dl_names = {
                term
                for term in dl_subset.get('Search_Term', [])
                if isinstance(term, str) and term
            }

        covered_names = set(dataset_names) | platform_dl_names
        if top_target_set:
            covered_top_names = covered_names & top_target_set
        else:
            covered_top_names = set(covered_names)
        if len(covered_top_names) >= target_count:
            results_rows.append({'Platform': code, 'Added': 0, 'Still Needed': 0})
            continue

        needed = max(target_count - len(covered_top_names), 0)

        url_field = row.get('URL', '')
        new_dataset_names: list[str] = []

        if needed > 0:
            try:
                urls, file_map, file_names, group_map = fetch_platform_file_index(url_field)
            except requests.RequestException as exc:
                print(f'Failed to fetch {code}: {exc}')
                urls, file_map, file_names, group_map = [], {}, {}, {}
            if not url_field or not urls:
                print(f'Skipping download lookup for {code}: no URL configured.')
            elif not file_map:
                print(f'No downloadable entries found for {code}.')
            else:
                for name in top_sequence:
                    if len(new_dataset_names) >= needed:
                        break
                    if not isinstance(name, str) or not name:
                        continue
                    if name in covered_names:
                        continue
                    if name in platform_dl_names:
                        continue
                    if (name, code) in blacklisted_pairs:
                        continue
                    if (name, code) in existing_pairs:
                        continue
                    normed = norm(name)
                    candidate_keys = filter_candidate_keys_for_entry(
                        file_map.keys(), file_names, name
                    )
                    ts_results = process.extract(
                        normed, candidate_keys, scorer=fuzz.token_sort_ratio, limit=None
                    )
                    if not ts_results:
                        continue
                    max_ts = max(score for _, score, _ in ts_results)
                    keys_at_max = [key for key, score, _ in ts_results if score == max_ts]
                    best_key = max(keys_at_max, key=lambda k: region_priority(file_names[k]))
                    pr_score = fuzz.partial_ratio(normed, best_key)
                    score = int(round(min(max_ts, pr_score)))
                    if score < threshold:
                        continue
                    group_key = norm(remove_disc(file_names[best_key]))
                    group_candidates = group_map.get(group_key, [best_key])
                    selected_keys = filter_candidate_keys_for_entry(
                        group_candidates, file_names, name
                    )
                    if not selected_keys:
                        continue

                    selected_keys = expand_multidisc_keys(selected_keys, group_map, file_names, name)
                    fresh_keys: list[str] = []
                    for key in selected_keys:
                        filehref = file_map[key]
                        if filehref in known_urls:
                            duplicate_skips += 1
                            continue
                        known_urls.add(filehref)
                        fresh_keys.append(key)
                    if not fresh_keys:
                        continue

                    for key in fresh_keys:
                        filehref = file_map[key]
                        title = filename_from_url(filehref)
                        new_download_rows.append({
                            'Search_Term': name,
                            'Platform': code,
                            'Directory': directory,
                            'Matched_Title': title,
                            'Score': score,
                            'URL': filehref,
                        })
                        existing_pairs.add((name, code))
                    new_dataset_names.append(name)
                    platform_dl_names.add(name)
                    covered_names.add(name)
                    if not top_target_set or name in top_target_set:
                        covered_top_names.add(name)

        still_needed = max(target_count - len(covered_top_names), 0)
        if new_dataset_names:
            total_added += len(new_dataset_names)
            platforms_updated += 1

        results_rows.append({
            'Platform': code,
            'Added': len(new_dataset_names),
            'Still Needed': still_needed,
        })

    if not summary_df.empty and 'ROMs' in summary_df.columns:
        numerator_candidates = [
            SUMMARY_ROM_SALES_MATCH_COL,
            'Sales ROMs',
            'Matched ROMs',
        ]
        denominator_candidates = [
            SUMMARY_DATASET_SALES_COL,
            'Dataset',
            'ROMs',
        ]
        numerator_col = next((c for c in numerator_candidates if c in summary_df.columns), None)
        denominator_col = next((c for c in denominator_candidates if c in summary_df.columns), None)
        coverage_col = (
            SUMMARY_SALES_COVERAGE_COL
            if SUMMARY_SALES_COVERAGE_COL in summary_df.columns
            else 'Sales Coverage %'
        )
        if numerator_col and denominator_col:
            with pd.option_context('mode.use_inf_as_na', True):
                denom_series = summary_df[denominator_col].astype(float).replace(0, math.nan)
                summary_df[coverage_col] = (
                    summary_df[numerator_col]
                    / denom_series
                    * 100
                )
            summary_df[coverage_col] = summary_df[coverage_col].round(1)

    summary_df.to_csv(summary_path, index=False)

    if new_download_rows:
        new_df = pd.DataFrame(new_download_rows)
        if not existing_dl_df.empty:
            combined = pd.concat([existing_dl_df, new_df], ignore_index=True)
        else:
            combined = new_df
        if 'URL' in combined.columns:
            combined = combined.drop_duplicates(subset=['URL'], keep='first')
        combined.to_csv(download_list_path, index=False)
        console.print('[bold green]Download list updated.[/]')
        if duplicate_skips:
            console.print(f"[bold yellow]Skipped {duplicate_skips} duplicate download(s).[/]")
    else:
        print('No new download entries added.')
        if duplicate_skips:
            console.print(f"[bold yellow]Skipped {duplicate_skips} duplicate download(s).[/]")

    if results_rows:
        results_df = pd.DataFrame(results_rows, columns=['Platform', 'Added', 'Still Needed'])
        print_table(results_df)

    console.print(
        "[bold green]Download target review complete:[/] added "
        f"{total_added} download link(s) across {platforms_updated} platform(s)."
    )


def download_games(snapshot_dir):
    csv_path = os.path.join(snapshot_dir, 'download_list.csv')
    if not os.path.isfile(csv_path):
        print('download_list.csv not found.')
        return
    links_file = os.path.join(snapshot_dir, 'aria2_links.txt')
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print('No download entries found in download_list.csv.')
        return

    snapshot_name = os.path.basename(os.path.normpath(snapshot_dir)) or 'snapshot'
    default_base_dir = ensure_rom_state_dir(snapshot_dir, 'main')
    alt_base_dir = ensure_rom_state_dir(snapshot_dir, 'new')

    print('Select download destination:')
    print(f"1) {shorten_path(default_base_dir)} (default)")
    print(f"2) {shorten_path(alt_base_dir)}")

    while True:
        choice = input('Enter choice [1/2]: ').strip()
        if choice in {'', '1'}:
            download_base_dir = default_base_dir
            break
        if choice == '2':
            download_base_dir = alt_base_dir
            break
        print('Invalid selection. Please enter 1 or 2.')

    skip_archive = prompt_yes_no('Skip downloads from archive.org?')

    download_jobs = []
    aria_entries = []
    skipped = 0
    dir_failures = 0
    for row in rows:
        url = row['URL']
        if skip_archive:
            parsed = urlparse(url)
            if parsed.netloc.lower().endswith('archive.org'):
                skipped += 1
                continue
        title = requests.utils.unquote(os.path.basename(row['Matched_Title']))
        out_dir = os.path.join(download_base_dir, row['Directory'])
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError as exc:
            dir_failures += 1
            print(f'Failed to create download directory {out_dir}: {exc}')
            continue
        aria_entries.append(f"{url}\n  out={title}\n  dir={out_dir}")
        download_jobs.append({
            'url': url,
            'title': title,
            'directory': out_dir,
        })

    if skip_archive and skipped:
        print(f'Skipped {skipped} archive.org download(s).')

    if dir_failures:
        print(f'Skipped {dir_failures} download(s) due to directory creation errors.')

    if not download_jobs:
        print('No downloads to process.')
        return

    try:
        os.makedirs(download_base_dir, exist_ok=True)
    except OSError as exc:
        print(f'Unable to prepare download directory {download_base_dir}: {exc}')
        return
    with open(links_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(aria_entries))

    total_jobs = len(download_jobs)
    console.print(f"[bold cyan]Prepared {total_jobs} download(s). Starting sequential aria2c downloads...[/]")
    successes = 0
    failures = 0

    for idx, job in enumerate(download_jobs, start=1):
        remaining_queue = total_jobs - idx
        console.print(
            f"[bold blue]Downloading ({idx}/{total_jobs}):[/] {job['title']}"
        )
        cmd = [
            ARIA2,
            job['url'],
            '-d',
            job['directory'],
            '-o',
            job['title'],
            '--auto-file-renaming=false',
            '--allow-overwrite=true',
        ]
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print('Error: aria2c not found. Ensure aria2c.exe is present in the wizardry folder or added to PATH.')
            return
        except subprocess.CalledProcessError as e:
            failures += 1
            console.print(
                f"[bold red]aria2c exited with code {e.returncode} while downloading {job['title']}.[/]"
            )
        else:
            successes += 1
        console.print(
            f"[bold cyan]Progress:[/] {successes} downloaded, {remaining_queue} remaining (of {total_jobs}); failures so far: {failures}."
        )

    if failures:
        console.print(f"[bold yellow]{failures} download(s) failed. See messages above for details.[/]")
    console.print(
        f"[bold green]Finished downloads: {successes}/{total_jobs} completed successfully.[/]"
    )
    if successes and (
        os.path.normcase(os.path.abspath(download_base_dir))
        == os.path.normcase(os.path.abspath(alt_base_dir))
    ):
        console.print(
            "[bold yellow]Reminder:[/] update gamelists and scrape through new games so "
            "they appear in the snapshot summary."
        )


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
    base = SNAPSHOT_BASE_DIR
    if not os.path.isdir(base):
        return None
    snaps = [
        d
        for d in os.listdir(base)
        if d.startswith('snapshot_') and os.path.isdir(os.path.join(base, d))
    ]
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
    display_df = prepare_summary_display(df)
    if not SUMMARY_COLS:
        SUMMARY_COLS = list(display_df.columns)
        if 'ROMs' in SUMMARY_COLS:
            SUMMARY_CYCLE_IDX = SUMMARY_COLS.index('ROMs')
    if SUMMARY_COLS:
        sort_col = SUMMARY_COLS[SUMMARY_CYCLE_IDX]
        view = display_df.sort_values(sort_col, ascending=False)
    else:
        sort_col = None
        view = display_df
    for col in view.select_dtypes(include='float'):
        view[col] = view[col].round(1)
    print_table(view, sorted_col=sort_col)
    if SUMMARY_COLS:
        SUMMARY_CYCLE_IDX = (SUMMARY_CYCLE_IDX + 1) % len(SUMMARY_COLS)


def manage_duplicates_menu(snapshot_dir: str) -> None:
    """Provide a focused menu for duplicate detection and cleanup."""

    while True:
        dup_exists = os.path.isdir(get_rom_state_dir(snapshot_dir, 'duplicates'))
        console.print("\n[bold cyan]Duplicate management[/]")
        if dup_exists:
            console.print("1) [green]Re-detect duplicates[/]")
        else:
            console.print("1) Detect duplicates")
        console.print("2) Remove duplicates, demos, prototypes and betas")
        console.print("3) Return to main menu")
        choice = input('Select option: ').strip()
        if choice == '1':
            detect_duplicates(snapshot_dir)
        elif choice == '2':
            remove_duplicates_and_special_roms(snapshot_dir)
        elif choice == '3':
            return
        else:
            console.print('[red]Invalid selection. Please try again.[/]')


def manage_download_list_menu(snapshot_dir: str) -> None:
    """Provide a focused menu for download list operations."""

    while True:
        console.print("\n[bold cyan]Download list management[/]")
        console.print("1) Add games to download list (manual filtering)")
        console.print("2) Add games to download list (automatic filtering)")
        console.print("3) Enforce download list targets")
        console.print("4) Return to main menu")
        choice = input('Select option: ').strip()
        if choice == '1':
            manual_add_games(snapshot_dir)
        elif choice == '2':
            auto_add_games(snapshot_dir)
        elif choice == '3':
            enforce_download_targets(snapshot_dir)
        elif choice == '4':
            return
        else:
            console.print('[red]Invalid selection. Please try again.[/]')


def wizard_menu(snapshot_dir):
    """Main interactive menu for a given snapshot."""
    global SUMMARY_CYCLE_IDX, SUMMARY_COLS
    SUMMARY_COLS = []
    SUMMARY_CYCLE_IDX = 0
    while True:
        sort_col = None
        summary_file = os.path.join(snapshot_dir, 'summary.csv')
        if not SUMMARY_COLS and os.path.isfile(summary_file):
            cols_df = pd.read_csv(summary_file, nrows=0)
            display_cols = prepare_summary_display(cols_df).columns
            SUMMARY_COLS = list(display_cols)
            if 'ROMs' in SUMMARY_COLS:
                SUMMARY_CYCLE_IDX = SUMMARY_COLS.index('ROMs')
        if SUMMARY_COLS:
            sort_col = SUMMARY_COLS[SUMMARY_CYCLE_IDX]

        dup_exists = os.path.isdir(get_rom_state_dir(snapshot_dir, 'duplicates'))
        m3u_exists = False
        gl_exists = False
        roms_root = get_rom_state_dir(snapshot_dir, 'main')
        if os.path.isdir(roms_root):
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

        console.print('\n[bold]Manage current ROM library[/]')
        console.print('1) Move ROMs missing sales or ratings')
        if dup_exists:
            console.print('2) [green]Manage duplicates, demos, prototypes and betas[/]')
        else:
            console.print('2) Manage duplicates, demos, prototypes and betas')
        if gl_exists:
            console.print('3) [green]Re-enrich game lists[/]')
        else:
            console.print('3) Enrich game lists')
        if m3u_exists:
            console.print('4) [green]Re-generate .m3u playlists[/]')
        else:
            console.print('4) Generate .m3u playlists')

        console.print('\n[bold]Build download queue[/]')
        console.print('5) Manage download list')

        console.print('\n[bold]Download and post-process[/]')
        console.print('6) Download unmatched games in list')
        console.print('7) Convert downloaded disc images to CHD')

        console.print('\n[bold]Exit[/]')
        console.print('8) Quit')
        choice = input('Select option: ').strip()
        if choice == '0':
            show_snapshot_summary(snapshot_dir)
        elif choice == '1':
            count_new_rating_matches(snapshot_dir)
        elif choice == '2':
            manage_duplicates_menu(snapshot_dir)
        elif choice == '3':
            enrich_game_lists(snapshot_dir)
        elif choice == '4':
            generate_playlists(snapshot_dir)
        elif choice == '5':
            manage_download_list_menu(snapshot_dir)
        elif choice == '6':
            download_games(snapshot_dir)
        elif choice == '7':
            convert_to_chd()
        elif choice == '8':
            return prompt_yes_no('Restart wizard?')
        else:
            console.print('[red]Invalid selection. Please try again.[/]')


def main():
    while True:
        snap_dir = select_snapshot()
        restart = wizard_menu(snap_dir)
        if not restart:
            break

if __name__ == '__main__':
    main()
