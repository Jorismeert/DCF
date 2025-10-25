import pandas as pd
import json
import requests
from io import StringIO

# Fetch Wikipedia page with a User-Agent
url = "https://en.wikipedia.org/wiki/Nasdaq-100"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
html = requests.get(url, headers=headers).text

# Wrap the HTML in StringIO to avoid the FutureWarning
tables = pd.read_html(StringIO(html))

# Inspect tables to find the Nasdaq-100 components table
# Usually it has "Company" and "Ticker" columns
nasdaq_df = None
for table in tables:
    if "Company" in table.columns and "Ticker" in table.columns:
        nasdaq_df = table
        break

if nasdaq_df is None:
    raise ValueError("Could not find Nasdaq-100 table on the page.")

# Keep only the needed columns
nasdaq_df = nasdaq_df[["Company", "Ticker"]].dropna().reset_index(drop=True)

# Convert to JSON
nasdaq_list = [{"company": row["Company"], "ticker": row["Ticker"]} 
               for _, row in nasdaq_df.iterrows()]
nasdaq100 = {"Nasdaq100": nasdaq_list}

# Save to file
output_path = "/Users/jorismeert/Desktop/Python/ProjectDCF/data/nasdaq100.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(nasdaq100, f, indent=2, ensure_ascii=False)

print(f"✅ Nasdaq-100 JSON saved successfully to: {output_path}")
