from pathlib import Path

import pandas as pd
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent / "data" / "healthcare.csv"

st.set_page_config(page_title="District Healthcare Analytics", layout="wide")
st.title("District-Level Healthcare Analytics")
st.warning(
    "This dashboard uses synthetic data for testing and education. "
    "It does not represent actual healthcare statistics."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.stop()

data = pd.read_csv(DATA_PATH)
district = st.selectbox("District", sorted(data["district"].unique()))
district_data = data.loc[data["district"] == district].sort_values("month")

st.line_chart(
    district_data.set_index("month")[
        ["anc_coverage", "institutional_delivery", "immunization"]
    ]
)
st.dataframe(district_data, use_container_width=True, hide_index=True)
