# rom-wizard

This repo contains helper scripts for working with ROM collections.

Platform download URLs used by `5_make_links_for_unmatched_ROMs.py` are stored
in `platforms.csv`. The file now has three columns:

```
Platform,Dir,URL
```

- **Platform** – dataset platform code (e.g. `PS3`, `Wii`)
- **Dir** – name of the output folder for downloaded ROMs
- **URL** – base URL to scrape for ZIP files

Fill in the `URL` column for any platforms you want to scrape.
