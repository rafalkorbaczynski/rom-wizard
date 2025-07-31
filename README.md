# rom-wizard

This repo contains helper scripts for managing ROM collections.

For a streamlined experience, run `rom_wizard.py` which provides an
interactive menu to manage snapshots and ROM files.  On start, the
script asks whether to load the newest existing snapshot or create a
fresh one.  Before fuzzy searches such as duplicate detection or sales
matching, you will be prompted for a match threshold.

When adding games, you may optionally exclude Wii titles that do not list controller support in `sales_2019.csv`.
Helper files (`aria2c.exe`, `chdman.exe`, `sales_2019.csv`,
`platforms.csv`) must reside in the `wizardry/` folder.
When a snapshot is created, a `rom_filenames.txt` file listing all
ROM paths is written to the snapshot folder (replacing the old
`list_rom_filenames.py` helper). Platform download URLs are configured in
`wizardry/platforms.csv` with the columns:

```
Platform,Directory,URL
```

Fill in the `URL` column for platforms you wish to scrape.  To
populate missing URLs automatically, run `scrape_platform_urls.py`.
