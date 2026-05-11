Great that head positions are fixed — that's crucial. Now let me answer your core doubt honestly.

Can Pogo Pin Detect Sub-millimeter Upliftment?
Yes — but ONLY if you choose the right spring force.
This is the most critical parameter.

The Physics of It
CORRECT (flush):          UPLIFTED by 0.3mm:

Head stops at H           Head stops at same H
      |                         |
   [Pogo]                    [Pogo]
   compresses 4mm            compresses 3.7mm
      |                         |
   [PCB flush]              [PCB uplifted 0.3mm]

Difference = exactly 0.3mm compression difference
Since your head position is fixed, the pogo pin will always compress by exactly (total travel - upliftment). So even 0.2–0.3mm upliftment = 0.2–0.3mm difference in compression. The pogo pin sees this directly and mechanically.

The Problem with Microswitch for Sub-mm
Here is where you need to be careful:
Microswitch travel tolerance = typically ±0.2 to ±0.5mm

If upliftment = 0.3mm
And switch tolerance = ±0.3mm

→ Switch may or may not trigger = UNRELIABLE
Microswitch alone is NOT reliable for sub-mm detection.

What You Need Instead — Linear Encoder or Thin Film Pressure Sensor
Option A — Thin Film Force Sensor (FSR) ✓ Best for your case
         [Pogo Pin]
              |
         [FSR Sensor]  ← FlexiForce or Interlink FSR
              |
         [Analog Reading]

Flush PCB    → Pogo compresses fully → Higher force → FSR reads ~800/1023
Uplifted PCB → Less compression     → Lower force  → FSR reads ~600/1023

Difference is clearly measurable even at 0.2mm upliftment

FSR sensors detect force difference not just contact
Even 0.2mm less compression = measurably less force
Thin, flexible, can be as small as 5mm diameter
Costs ~$5-10 each
Analog output → goes to Arduino/microcontroller ADC pin



That's a very valid concern. Let me be honest with you.

Honest Assessment of Fiber Optic in Your Case
A few mm space means:

Fiber tip needs to be precisely aimed into a gap that is itself only fraction of a mm tall
Any vibration, misalignment, or variation in PCB position will throw off readings
Mounting a bracket rigidly enough in a few mm space is mechanically very difficult
Teach/calibration will be extremely sensitive and may need frequent recalibration

Verdict: Fiber optic is risky in sub-5mm space. Not impossible but unreliable in practice.

So What Actually Works in This Tiny Space?
Let me re-think this from scratch given the true constraint.

Best Fit Now: Micro Hall Effect Sensor + Small Magnet
Side view:

[ PCB Board  ]  ← small magnet embedded/stuck on PCB underside at corner
                        ↕ gap (uplifted) or no gap (clamped)
[ JIG        ]  ← micro hall effect sensor recessed into corner area
But jig cannot be touched — so this is out too



project/
│
├── raw_csvs/
│   ├── jan_feb.csv
│   ├── mar_apr.csv
│   ├── may_jun.csv
│
├── merged/
│   └── full_year.csv
│
├── app.py
├── clean_excel.py
├── merge_csv.py
├── index.html

merge_csv.py

import pandas as pd
import glob
import os

# Folder containing all CSV files
INPUT_FOLDER = "raw_csvs"

# Output merged file
OUTPUT_FILE = "merged/full_year.csv"

# Find all CSV files
csv_files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))

if not csv_files:
    print("No CSV files found.")
    exit()

all_data = []

for file in csv_files:
    print(f"Reading: {file}")

    try:
        df = pd.read_csv(file)

        # Optional: remove completely empty rows
        df = df.dropna(how='all')

        all_data.append(df)

    except Exception as e:
        print(f"Error reading {file}: {e}")

# Combine all CSVs
merged_df = pd.concat(all_data, ignore_index=True)

# Optional: remove duplicate rows
merged_df = merged_df.drop_duplicates()

# Create output folder if not exists
os.makedirs("merged", exist_ok=True)

# Save final merged CSV
merged_df.to_csv(OUTPUT_FILE, index=False)

print(f"\nMerged {len(csv_files)} files successfully.")
print(f"Final rows: {len(merged_df)}")
print(f"Saved to: {OUTPUT_FILE}")


After putting all CSV files in raw_csvs/:

python merge_csv.py

It creates:

merged/full_year.csv
Then Run Your Existing Cleaner

Your current command remains same:

python clean_excel.py --file merged/full_year.csv
