import streamlit as st
import pandas as pd
import json
import plotly.express as px
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="Dissipation Analysis", layout="wide")
st.title("🔁 Non‑Equilibrium Statistical Mechanics for Markets")
st.caption("Entropy production rate (KL divergence) | Fluctuation theorem | Jarzynski proxy")

# Custom CSS
st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; padding: 0.5rem; border-radius: 0.5rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

OUTPUT_REPO = config.OUTPUT_REPO
HF_TOKEN = config.HF_TOKEN

@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        files = [f['name'] for f in fs.ls(f"datasets/{OUTPUT_REPO}", detail=True, recursive=True) if f['type'] == 'file']
        return files
    except Exception as e:
        return [f"Error: {e}"]

def find_latest_json(files):
    json_files = [f for f in files if f.endswith('.json') and 'dissipation' in f]
    if not json_files:
        return None
    json_files.sort(reverse=True)
    return json_files[0]

@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

files = list_repo_files()
latest = find_latest_json(files)
if not latest:
    st.error("No dissipation results found. Run trainer first.")
    st.stop()

data = load_json(latest)
if "error" in data:
    st.error(f"Error loading JSON: {data['error']}")
    st.stop()

st.sidebar.header("ℹ️ Info")
st.sidebar.write(f"**Run date:** {data['run_date']}")
st.sidebar.write(f"**Next trading day:** {next_trading_day()}")
st.sidebar.write("**Method:** KL divergence (forward vs reverse) as entropy production rate. High = far from equilibrium → unstable.")

universes = data["universes"]
if not universes:
    st.warning("No universe data.")
    st.stop()

# ========== TOP UNSTABLE ETFS PER UNIVERSE ==========
st.header("⚠️ Highest Entropy Production (Most Unstable)")

for universe_name, uni_data in universes.items():
    rankings = uni_data.get("rankings", [])
    if not rankings:
        continue
    selected_win = uni_data.get("selected_window", "?")
    st.subheader(f"📌 {universe_name} (window = {selected_win} days)")
    cols = st.columns(min(len(rankings), config.TOP_N))
    for idx, rec in enumerate(rankings[:config.TOP_N]):
        with cols[idx]:
            st.metric(rec["ticker"], f"{rec['entropy_production']:.3f}", "entropy rate")
            st.caption(f"Jarzynski: {rec['jarzynski_exponential']:.2f}")
            st.caption(f"Neg fraction: {rec['fraction_negative_entropy']:.2f}")
    st.divider()

# ========== DETAILED VIEW ==========
universe_names = list(universes.keys())
selected = st.selectbox("Select Universe for detailed view", universe_names)

if selected:
    uni_data = universes[selected]
    all_tickers = uni_data.get("all_tickers", {})
    if not all_tickers:
        st.info("No data")
    else:
        # Build DataFrame for all tickers
        rows = []
        for ticker, metrics in all_tickers.items():
            rows.append({
                "ETF": ticker,
                "Entropy Production": metrics["entropy_production"],
                "Jarzynski Exp": metrics["jarzynski_exponential"],
                "Fraction Negative Entropy": metrics["fraction_negative_entropy"],
                "Mean Window Entropy": metrics["mean_window_entropy"]
            })
        df = pd.DataFrame(rows).sort_values("Entropy Production", ascending=False)
        st.subheader("📊 Full Rankings")
        st.dataframe(df, use_container_width=True, hide_index=True)
        # Bar chart
        fig = px.bar(df, x="ETF", y="Entropy Production", title="Entropy Production Rate (Higher = more unstable)",
                     color="Entropy Production", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

st.caption("Interpretation: High entropy production = time‑asymmetry, far from equilibrium, likely regime instability.")
