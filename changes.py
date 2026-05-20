import pandas as pd
from pathlib import Path

# Folder containing all Excel files
FOLDER_PATH = "./excel_files"   # change this

# Output file name
OUTPUT_FILE = "combined_output.xlsx"

# Get all Excel files
excel_files = list(Path(FOLDER_PATH).glob("*.xlsx"))

if not excel_files:
    print("No Excel files found.")
    exit()

all_dataframes = []

print(f"Found {len(excel_files)} Excel files\n")

for file in excel_files:
    try:
        print(f"Reading: {file.name}")

        # Read Excel
        df = pd.read_excel(file, engine="openpyxl")

        # Optional: remove completely empty rows
        df = df.dropna(how="all")

        all_dataframes.append(df)

    except Exception as e:
        print(f"Error reading {file.name}: {e}")

# Combine all files
combined_df = pd.concat(all_dataframes, ignore_index=True)

# Optional: remove duplicate rows
combined_df = combined_df.drop_duplicates()

# Save combined file
combined_df.to_excel(OUTPUT_FILE, index=False)

print("\n✅ All Excel files combined successfully!")
print(f"Total rows: {len(combined_df)}")
print(f"Saved as: {OUTPUT_FILE}")


project/
│
├── combine_excel.py
├── excel_files/
│   ├── file1.xlsx
│   ├── file2.xlsx
│   ├── file3.xlsx
│   └── ...
Install dependency
pip install pandas openpyxl
Run
python combine_excel.py

import pandas as pd

# ── File paths ─────────────────────────────────────────────
MAIN_FILE = "main_excel.xlsx"
REFERENCE_FILE = "machine_models.xlsx"

OUTPUT_FILE = "updated_main_excel.xlsx"

# ── Read Excels ────────────────────────────────────────────
main_df = pd.read_excel(MAIN_FILE, engine="openpyxl")
ref_df  = pd.read_excel(REFERENCE_FILE, engine="openpyxl")

# ── Clean column names ─────────────────────────────────────
main_df.columns = main_df.columns.str.strip()
ref_df.columns  = ref_df.columns.str.strip()

# ── Optional: clean spaces in values ───────────────────────
main_df["Line No"] = main_df["Line No"].astype(str).str.strip()
main_df["Machine"] = main_df["Machine"].astype(str).str.strip()

ref_df["Line No"] = ref_df["Line No"].astype(str).str.strip()
ref_df["Machine"] = ref_df["Machine"].astype(str).str.strip()

# ── Keep only required columns from reference ──────────────
ref_df = ref_df[["Line No", "Machine", "Model Name"]]

# ── Merge using Line No + Machine ──────────────────────────
merged_df = pd.merge(
    main_df,
    ref_df,
    on=["Line No", "Machine"],
    how="left"
)

# ── Save output ────────────────────────────────────────────
merged_df.to_excel(OUTPUT_FILE, index=False)

print("✅ Model names added successfully!")
print(f"Saved file: {OUTPUT_FILE}")

  
