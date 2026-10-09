# Automated Insight Generation: District-Level Healthcare Analytics

`data/healthcare.csv` is a **synthetic dataset for software testing and education**. It is not sourced from health departments and must not be interpreted or cited as actual healthcare statistics for Gujarat.

Regenerate the 2025 district-month dataset with a fixed random seed:

```sh
python3 -m pip install -r requirements.txt
python3 scripts/generate_dataset.py
```

The generator validates the schema, coverage ranges, row uniqueness, missing values, monthly completeness, and the designed trend, outlier, and correlation patterns before writing the CSV.
