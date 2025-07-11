#!/usr/bin/env python3
import os
import sys
import pandas as pd
import xml.etree.ElementTree as ET
import unicodedata
import re
import math
import argparse
from datetime import datetime
from rapidfuzz import process, fuzz

# ——— Title normalization helper —————————————————————————————————
_ROMAN = {'ix':9,'viii':8,'vii':7,'vi':6,'iv':4,'iii':3,'ii':2,'i':1}

def norm(title):
    s = unicodedata.normalize('NFKD', str(title))
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    s = re.sub(r'[\(\[].*?[\)\]]', '', s)
    s = re.sub(r'^the\s+', '', s)
    s = s.replace('&', ' and ')
    s = re.sub(r'\b(' + '|'.join(_ROMAN.keys()) + r')\b',
               lambda m: str(_ROMAN[m.group(1)]), s)
    s = re.sub(r'[^a-z0-9 ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

# ——— Determine script directory and create timestamped snapshot folder —————————————————
template_dir = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(template_dir, f'snapshot_{timestamp}')
os.makedirs(results_dir, exist_ok=True)
print(f"Snapshot directory: {results_dir}")

# ——— Define ROMs root —————————————————————————————————————————————————
roms_root = os.path.abspath(os.path.join(template_dir, os.pardir, 'roms'))

# ——— Argument parsing —————————————————————————————————————————
parser = argparse.ArgumentParser(
    description="Inject global sales into RetroBat gamelist.xml files and output results to a timestamped snapshot folder."
)
parser.add_argument(
    "--scale", choices=["linear","log"], default="linear",
    help="Normalization scale: linear=gs/max*100, log=log(gs+1)/log(max+1)*100"
)
parser.add_argument(
    "--fuzzy-threshold", type=int, default=85,
    help="Minimum fuzzy match score (0–100) to accept a sales match."
)
parser.add_argument(
    "--dataset-csv", default="vgsales.csv",
    help="Name of the CSV dataset file located in the scripts directory."
)
args = parser.parse_args()

use_log = (args.scale == "log")
fuzzy_threshold = args.fuzzy_threshold

# ——— Load & normalize sales data ———————————————————————————
sales_csv = os.path.join(template_dir, args.dataset_csv)
print(f"Loading sales data from {sales_csv}")
sales = pd.read_csv(sales_csv, low_memory=False)
required_cols = {'Name','Platform','Global_Sales'}
missing = required_cols - set(sales.columns)
if missing:
    raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

# Parse release date or Year
if 'release_date' in sales.columns:
    sales['release_date'] = pd.to_datetime(sales['release_date'], errors='coerce')
    sales['Year'] = sales['release_date'].dt.year.fillna(0).astype(int)
elif 'Year' in sales.columns:
    sales['Year'] = pd.to_numeric(sales['Year'], errors='coerce').fillna(0).astype(int)
else:
    sales['Year'] = 0

# Filter out entries with no sales
sales_cols = [c for c in sales.columns if c.endswith('_Sales')]
sales = sales[sales[sales_cols].fillna(0).sum(axis=1) > 0]
print(f"Dataset rows after filtering empty sales: {len(sales)}")

# Precompute normalization
sales['key'] = sales['Name'].apply(norm)
key_to_platform = dict(zip(sales['key'], sales['Platform']))
max_sales = sales['Global_Sales'].max()

# ——— Platform mapping —————————————————————————————————————————————
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
READABLE_NAMES = {
    '2600':'Atari 2600','5200':'Atari 5200','7800':'Atari 7800','AST':'Atari ST',
    '3DO':'3DO','3DS':'Nintendo 3DS','DS':'Nintendo DS','GB':'Game Boy','GBC':'Game Boy Color',
    'GBA':'Game Boy Advance','NES':'NES','SNES':'SNES','N64':'Nintendo 64','GC':'GameCube',
    'Wii':'Wii','WiiU':'Wii U','NS':'Switch','NGage':'N-Gage','VB':'Virtual Boy',
    'PS':'PlayStation','PS2':'PS2','PS3':'PS3','PS4':'PS4','PS5':'PS5','PSP':'PSP','PSV':'Vita',
    'SAT':'Saturn','SCD':'Sega CD','S32X':'32X','GG':'Game Gear','GEN':'Genesis','DC':'Dreamcast',
    'XB':'Xbox','X360':'360','XOne':'One','PC':'PC','MSD':'MS-DOS','WinP':'Windows',
    'Linux':'Linux','OSX':'macOS','PCFX':'PC-FX','TG16':'TurboGrafx-16','PCE':'PC Engine',
    'C64':'C64','C128':'C128','FDS':'FDS','MSX':'MSX','BBCM':'BBC Micro','AMIG':'Amiga','CD32':'CD32',
    'ACPC':'Amstrad CPC','ApII':'Apple II','Int':'Intellivision','Arc':'Arcadia','CDi':'CD-i'
}

matched_records = []
detailed_matches = []
console_stats = {}
total_games = 0

# ——— Matching & XML injection ——————————————————————————————————————
print("Beginning gamelist processing...")
for root_dir, _, files in os.walk(roms_root):
    if 'gamelist.xml' not in files:
        continue
    console = os.path.basename(root_dir).lower()
    ds_plat = PLAT_MAP.get(console)
    if not ds_plat:
        continue

    print(f"Processing {console}...")
    rel = os.path.relpath(root_dir, roms_root)
    out_dir = os.path.join(results_dir, rel)
    os.makedirs(out_dir, exist_ok=True)

    subset = sales[sales['Platform'].str.lower().isin(
        ['gbc','gb'] if ds_plat.lower()=='gbc' else [ds_plat.lower()]
    )]
    sales_keys = subset['key'].tolist()
    key_to_sales = dict(zip(sales_keys, subset['Global_Sales']))
    key_to_name = dict(zip(sales_keys, subset['Name']))

    tree = ET.parse(os.path.join(root_dir, 'gamelist.xml'))
    root = tree.getroot()
    games = root.findall('game')
    total_games += len(games)
    for game in games:
        for old in game.findall('rating') + game.findall('ratingMax'):
            game.remove(old)

        title = game.findtext('name') or os.path.splitext(game.findtext('path') or '')[0]
        if not title:
            continue
        key = norm(title)
        if key in key_to_sales:
            best, score = key, 100
        else:
            res = process.extractOne(key, sales_keys, scorer=fuzz.token_set_ratio)
            if not res or res[1] < fuzzy_threshold:
                continue
            best, score = res[0], res[1]
        gs = key_to_sales[best]
        matched_records.append((ds_plat, best))
        detailed_matches.append({
            'Dataset_Name': key_to_name[best],
            'Sales_Platform': key_to_platform[best],
            'ROM_File': os.path.basename(game.findtext('path') or title),
            'Global_Sales': gs,
            'Match_Score': score
        })
        val = (math.log(gs+1)/math.log(max_sales+1)*100) if use_log else (gs/max_sales*100)
        ET.SubElement(game, 'rating').text = f"{val:.2f}"
        ET.SubElement(game, 'ratingMax').text = '100'

    out_xml = os.path.join(out_dir, 'gamelist.xml')
    tree.write(out_xml, encoding='utf-8', xml_declaration=True)
    console_stats[ds_plat] = {'matched_roms': len([r for r in detailed_matches if r['Sales_Platform']==ds_plat]),
                               'total_roms': len(games)}
    print(f"  Wrote updated gamelist for {READABLE_NAMES.get(ds_plat, ds_plat)}: {out_xml}")

# ——— Write CSV summaries once ——————————————————————————————————————
print("Writing CSV summaries...")
# detailed matches
pd.DataFrame(detailed_matches,columns=['Dataset_Name','Sales_Platform','ROM_File','Global_Sales','Match_Score']).to_csv(os.path.join(results_dir, 'match_summary.csv'), index=False)
# unmatched
matched_df = pd.DataFrame(matched_records, columns=['Platform','key'])
df = sales[['Name','Platform','Global_Sales','key']]
merged = df.merge(matched_df.drop_duplicates(), on=['Platform','key'], how='left', indicator=True)
merged[merged['_merge']=='left_only'].sort_values(['Platform','Global_Sales'], ascending=[True,False]).to_csv(os.path.join(results_dir, 'unmatched_summary.csv'), index=False)
# coverage summary
df1 = sales.groupby('Platform')['key'].nunique().rename('Titles')
df2 = matched_df.groupby('Platform')['key'].nunique().rename('MatchedTitles')
stats = pd.concat([df1, df2], axis=1).fillna(0)
stats['Data%'] = stats['MatchedTitles']/stats['Titles'].replace(0,1)*100
roms = pd.DataFrame.from_dict(console_stats, orient='index')
roms.index.name='Platform'
roms.rename(columns={'total_roms':'ROMs','matched_roms':'MatchedROMs'}, inplace=True)
roms['ROMs%'] = roms['MatchedROMs']/roms['ROMs'].replace(0,1)*100
summary = pd.concat([roms, stats], axis=1).fillna(0)
summary.rename(index=READABLE_NAMES, inplace=True)
summary[['Titles','MatchedTitles','Data%','ROMs','MatchedROMs','ROMs%']].to_csv(os.path.join(results_dir, 'summary.csv'), index_label='Platform')
print("All summaries saved in snapshot folder.")
