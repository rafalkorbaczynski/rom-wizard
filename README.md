# rom-wizard

This repo contains helper scripts for working with ROM collections.
You can generate a plain text list of all ROM filenames (with supported extensions) by running `list_rom_filenames.py`. The script scans the `roms` directory and writes `rom_filenames.txt` in this folder.

Platform download URLs used by `5_make_links_for_unmatched_ROMs.py` are stored
in `platforms.csv`.  Each row now contains three columns:

```
Platform,Directory,URL
```

* **Platform** – name exactly as it appears in the sales dataset
* **Directory** – folder name under `roms/` or `new_roms/` where files are
  stored
* **URL** – base URL to scrape for downloadable archives

Fill in the `URL` column for platforms you wish to scrape.

To populate missing URLs automatically, run `scrape_platform_urls.py`.

Run `7_convert_to_chd.py` to convert disc images to CHD format.
