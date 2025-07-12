# rom-wizard

This repo contains helper scripts for working with ROM collections.

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
