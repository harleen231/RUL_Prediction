# AeroGuard: NASA C-MAPSS Turbofan Engine Degradation Dataset Directory

This directory stores the NASA C-MAPSS dataset subsets (FD001, FD002, FD003, FD004).

## Files Description:
- `train_FD00x.txt`: Full run-to-failure trajectory sensor recordings for training engines.
- `test_FD00x.txt`: Sensor recordings of operational engines stopped at some time prior to system failure.
- `RUL_FD00x.txt`: The true Remaining Useful Life values at the final operational cycle recorded in `test_FD00x.txt`.

## Automatic Download:
AeroGuard automatically attempts to download the dataset using `kagglehub.dataset_download("behrad3d/nasa-cmaps")`.

## Manual Setup:
If automated download fails (e.g., due to firewall, proxy, or Kaggle authentication), place the text files directly into this directory:
- `data/train_FD001.txt`
- `data/test_FD001.txt`
- `data/RUL_FD001.txt`
- (Optionally FD002, FD003, FD004)
