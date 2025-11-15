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
Platform,Directory,URL,ignore
```

Fill in the `URL` column for platforms you wish to scrape and set
`ignore` to `TRUE` for systems that should be skipped by
`rom_wizard.py`.  To populate missing URLs automatically, run
`scrape_platform_urls.py`.

## ROM Wizard workflow

The `rom_wizard.py` script is organised into a predictable loop so you can
understand which files and reports are touched in every phase.

### 1. Startup and snapshot selection

1. `main()` repeatedly calls `select_snapshot()` until you choose to exit.
2. `select_snapshot()` offers the newest snapshot if one exists and
   otherwise lets you browse to an existing folder or create a fresh
   snapshot via `create_snapshot()`.

### 2. Snapshot creation and analysis

`create_snapshot()` performs the heavy lifting:

* Scans the ROM library for every platform, merges platform metadata, and
  fuzzy-matches ROM names against `sales_2019.csv` so that sales and age
  ratings can be summarized.
* Writes `summary.csv`, `match_summary.csv`, `unmatched_summary.csv`, and
  `rating_gap_entries.csv` inside the snapshot folder.
* Logs ROM filename inventory to `rom_filenames.txt` and prints an overview
  table in the terminal so you can review coverage immediately.

### 3. Interactive menu (per snapshot)

After a snapshot is selected, `wizard_menu()` exposes task groupings:

* **Show snapshot summary** – cycles the on-screen table through each
  sortable column for quick status checks.
* **Manage current ROM library** –
  * `count_new_rating_matches()` can move ROMs without sales/ratings to
    staging folders.
  * `manage_duplicates_menu()` calls into duplicate detection and optional
    beta/demo/prototype cleanup.
  * `enrich_game_lists()` rewrites `gamelist.xml` entries with sales, new
    ratings, and optional parental-control filtering.
  * `generate_playlists()` writes multi-disc `.m3u` playlists to the
    snapshot copy of the ROM library.
* **Build download queue** – `manual_add_games()`, `auto_add_games()`, and
  `enforce_download_targets()` each append rows to `download_list.csv`
  using the unmatched dataset report.
* **Download and post-process** –
  * `download_games()` walks the download list, prepares aria2c commands,
    and streams progress for each file.
  * `convert_to_chd()` scans the ROM tree for disc images and optionally
    runs `chdman.exe` per platform to convert supported files into CHD
    archives.

### 4. Mermaid overview

```mermaid
flowchart TD
    start([Launch rom_wizard.py]) --> select{Load latest snapshot?}
    select -->|Yes| load[Select latest snapshot]
    select -->|No| create[Create snapshot]
    create --> menu
    load --> menu
    menu["wizard_menu() loop"] --> summary[Show snapshot summary]
    menu --> library{Manage ROM library}
    menu --> queue{Build download queue}
    menu --> download[Download & post-process]
    library --> rating[count_new_rating_matches()]
    library --> duplicates[manage_duplicates_menu()]
    library --> enrich[enrich_game_lists()]
    library --> playlists[generate_playlists()]
    queue --> manual[manual_add_games()]
    queue --> auto[auto_add_games()]
    queue --> enforce[enforce_download_targets()]
    download --> fetch[download_games()]
    download --> chd[convert_to_chd()]
    fetch --> menu
    chd --> menu
```

Use the diagram as a quick reference for how user actions in the menu map
back to the major functions inside `rom_wizard.py`.
