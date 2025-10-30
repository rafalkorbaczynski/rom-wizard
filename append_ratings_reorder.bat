@echo off
REM Get the folder of this batch file
set SCRIPT_DIR=%~dp0

python "%SCRIPT_DIR%append_ratings_pipeline_reorder.py" ^
  --in "%SCRIPT_DIR%wizardry/sales_gpt_pegi_nodupes_nopcps4x360xone.csv" ^
  --out "%SCRIPT_DIR%ratings_output/sales_gpt_pegi_nodupes_allratings_nopcps4x360xone.csv" ^
  --config "%SCRIPT_DIR%wizardry/ratings_config.json" ^
  --platforms-csv "%SCRIPT_DIR%wizardry/platforms.csv" ^
  --title-col "Name" ^
  --platform-col "Platform" ^
  --year-col "Year" ^
  --per-platform-limit 100 ^
  --priority-col "Global_Sales" ^
  --descending ^
  --summary-every 1 ^
  --summary-json "%SCRIPT_DIR%ratings_output/ratings_summary.json" ^
  --summary-platform-csv "%SCRIPT_DIR%ratings_output/ratings_summary_by_platform.csv" ^
  --enable-giantbomb-backfill ^
  --enable-rawg-backfill ^
  --rps-giantbomb 0.05 ^
  --gb-min-spacing 20 ^
  --gb-hourly 190 ^
  --workers 6 ^
  --checkpoint-every 100 ^
  --pegi-input-col PEGI_Rating ^
  --verbose
