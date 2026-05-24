"""
🔐 Fraud Detection — Streamlit Operations Dashboard
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, warnings
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .stMetric { background: #1a1a2e; border-radius: 10px; padding: 15px; border-left: 4px solid #e94560; }
    .stMetric label { color: #aaa !important; font-size: 13px; }
    .stMetric [data-testid="metric-value"] { color: white; font-size: 28px; font-weight: bold; }
    h1, h2, h3 { color: #e94560; }
    .tier-critical { color: #e94560; font-weight: bold; }
    .tier-suspicious { color: #f4a261; font-weight: bold; }
    .tier-clear { color: #00b4d8; font-weight: bold; }
    div[data-testid="stSidebarNav"] { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model    = pickle.load(open(os.path.join(BASE, "model.pkl"),        "rb"))
    scaler   = pickle.load(open(os.path.join(BASE, "scaler.pkl"),       "rb"))
    features = pickle.load(open(os.path.join(BASE, "feature_names.pkl"), "rb"))
    return model, scaler, features

@st.cache_data
def load_data():

    # Load transaction data efficiently
    tx = pd.read_csv(
    os.path.join(BASE, "..", "data", "train_transaction.csv"),
    low_memory=True,
    nrows=100000
)

    
    # Reduce RAM usage
    for col in tx.select_dtypes(include=['float64']).columns:
        tx[col] = tx[col].astype('float32')

    for col in tx.select_dtypes(include=['int64']).columns:
        tx[col] = tx[col].astype('int32')

    # Load identity data
    id_df = pd.read_csv(
    os.path.join(BASE, "..", "data", "train_identity.csv"),
    low_memory=True,
    nrows=30000
)

    # Reduce RAM usage there too
    for col in id_df.select_dtypes(include=['float64']).columns:
        id_df[col] = id_df[col].astype('float32')

    for col in id_df.select_dtypes(include=['int64']).columns:
        id_df[col] = id_df[col].astype('int32')

    # Merge
    df = tx.merge(id_df, on="TransactionID", how="left")

    return df

def assign_tier(p):
    if   p >= 0.75: return "🔴 Critical Risk"
    elif p >= 0.40: return "🟡 Suspicious"
    else:           return "🟢 Clear"

def get_risk_score(prob):
    """Compute and cache risk scores for the full dataset."""
    return prob

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/lock-2.png", width=60)
st.sidebar.title("🔐 Fraud Ops Center")
page = st.sidebar.radio("Navigate", ["📊 Overview", "🔎 Transaction Explorer", "🧠 SHAP Explainer"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** LightGBM v2  \n**Threshold:** 0.50 (tunable)  \n**Dataset:** IEEE-CIS")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Fraud Detection — Operations Overview")
    st.markdown("---")

    model, scaler, features = load_model()

    with st.spinner("Loading data and scoring transactions..."):
        df = load_data()

        # Prepare features
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        from sklearn.preprocessing import LabelEncoder
        df_proc = df
        df_proc["AmtToMeanRatio"] = df_proc["TransactionAmt"] / df_proc["TransactionAmt"].mean()
        df_proc["HourOfDay"]      = (df_proc["TransactionDT"] // 3600) % 24
        df_proc["AmtLog"]         = np.log1p(df_proc["TransactionAmt"])
        df_proc["IsAnonymousEmail"] = df_proc["P_emaildomain"].astype(str).str.contains("anonymous", case=False).astype(int) if "P_emaildomain" in df_proc.columns else 0
        df_proc["DeviceRisk"] = (df_proc["DeviceType"].astype(str).str.lower().str.strip() == "mobile").astype(int) if "DeviceType" in df_proc.columns else 0

        # Align columns
        for col in cat_cols:
            if col in df_proc.columns:
                le = LabelEncoder()
                df_proc[col] = le.fit_transform(df_proc[col].astype(str))

        for col in df_proc.select_dtypes(include="number").columns:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())

        missing_feats = [f for f in features if f not in df_proc.columns]
        for f in missing_feats:
            df_proc[f] = 0

        X = df_proc[features]
        X_sc = scaler.transform(X)
        probas = model.predict_proba(X_sc)[:, 1]

        df["FraudProb"] = probas
        df["RiskTier"]  = [assign_tier(p) for p in probas]

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total        = len(df)
    total_fraud  = int(df["isFraud"].sum()) if "isFraud" in df.columns else int((probas >= 0.5).sum())
    det_rate     = total_fraud / total * 100
    avg_fraud_amt = df[df["isFraud"] == 1]["TransactionAmt"].mean() if "isFraud" in df.columns else df[probas >= 0.5]["TransactionAmt"].mean()
    critical     = (df["RiskTier"] == "🔴 Critical Risk").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📋 Total Transactions", f"{total:,}")
    c2.metric("⚠️ Total Fraud Cases",  f"{total_fraud:,}")
    c3.metric("🎯 Detection Rate",     f"{det_rate:.2f}%")
    c4.metric("💵 Avg Fraud Amount",   f"${avg_fraud_amt:,.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Fraud by hour
        hour_fraud = df.groupby("HourOfDay")["FraudProb"].mean() * 100 if "HourOfDay" in df.columns else pd.Series()
        if not hour_fraud.empty:
            fig = px.bar(x=hour_fraud.index, y=hour_fraud.values,
                         color=hour_fraud.values,
                         color_continuous_scale="RdYlGn_r",
                         labels={"x": "Hour of Day", "y": "Avg Fraud Probability (%)"},
                         title="🕐 Fraud Probability by Hour of Day",
                         template="plotly_dark")
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk tier donut
        tier_counts = df["RiskTier"].value_counts()
        fig = go.Figure(go.Pie(
            labels=tier_counts.index, values=tier_counts.values,
            hole=0.55,
            marker=dict(colors=["#e94560", "#f4a261", "#00b4d8"],
                        line=dict(color="#0f0f1a", width=3)),
            textinfo="label+percent",
        ))
        fig.update_layout(title="Risk Tier Distribution", template="plotly_dark",
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Transaction amount distribution
    fig = px.histogram(df, x=np.log1p(df["TransactionAmt"]),
                       color="RiskTier" if "RiskTier" in df.columns else None,
                       color_discrete_map={"🔴 Critical Risk": "#e94560",
                                           "🟡 Suspicious":    "#f4a261",
                                           "🟢 Clear":         "#00b4d8"},
                       nbins=80, barmode="overlay", opacity=0.7,
                       labels={"x": "log(1 + TransactionAmt)"},
                       title="Transaction Amount Distribution by Risk Tier",
                       template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — TRANSACTION EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown("Filter, search and explore transactions with live risk scores.")
    st.markdown("---")

    model, scaler, features = load_model()

    with st.spinner("Loading transactions..."):
        df = load_data()
        from sklearn.preprocessing import LabelEncoder
        df_proc = df
        df_proc["AmtToMeanRatio"]  = df_proc["TransactionAmt"] / df_proc["TransactionAmt"].mean()
        df_proc["HourOfDay"]       = (df_proc["TransactionDT"] // 3600) % 24
        df_proc["AmtLog"]          = np.log1p(df_proc["TransactionAmt"])
        df_proc["IsAnonymousEmail"]= df_proc["P_emaildomain"].astype(str).str.contains("anonymous", na=False).astype(int) if "P_emaildomain" in df_proc.columns else 0
        df_proc["DeviceRisk"]      = (df_proc["DeviceType"].astype(str).str.lower() == "mobile").astype(int) if "DeviceType" in df_proc.columns else 0

        cat_cols = df.select_dtypes(include="object").columns.tolist()
        for col in cat_cols:
            if col in df_proc.columns:
                le = LabelEncoder()
                df_proc[col] = le.fit_transform(df_proc[col].astype(str))
        for col in df_proc.select_dtypes(include="number").columns:
            df_proc[col] = df_proc[col].fillna(df_proc[col].median())
        missing_feats = [f for f in features if f not in df_proc.columns]
        for f in missing_feats:
            df_proc[f] = 0

        X_sc   = scaler.transform(df_proc[features])
        probas = model.predict_proba(X_sc)[:, 1]

        df["FraudProbability"] = probas
        df["RiskTier"]         = [assign_tier(p) for p in probas]
        df["HourOfDay"]        = (df["TransactionDT"] // 3600) % 24

    # ── Sidebar filters
    st.sidebar.markdown("### Filters")
    min_p  = st.sidebar.slider("Min Fraud Probability", 0.0, 1.0, 0.0, 0.01)
    max_p  = st.sidebar.slider("Max Fraud Probability", 0.0, 1.0, 1.0, 0.01)
    tiers  = st.sidebar.multiselect("Risk Tier", ["🔴 Critical Risk","🟡 Suspicious","🟢 Clear"],
                                     default=["🔴 Critical Risk","🟡 Suspicious","🟢 Clear"])
    search_id = st.sidebar.text_input("Search by TransactionID")

    filtered = df[(df["FraudProbability"] >= min_p) & (df["FraudProbability"] <= max_p)]
    if tiers:
        filtered = filtered[filtered["RiskTier"].isin(tiers)]
    if search_id:
        try:
            filtered = filtered[filtered["TransactionID"] == int(search_id)]
        except:
            st.sidebar.error("Invalid TransactionID")

    st.markdown(f"**Showing {len(filtered):,} transactions** (of {len(df):,} total)")

    display_cols = ["TransactionID", "TransactionAmt", "ProductCD",
                    "card4", "card6", "HourOfDay", "FraudProbability", "RiskTier"]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(
        filtered[display_cols].sort_values("FraudProbability", ascending=False).head(500)
            .style.format({"FraudProbability": "{:.4f}", "TransactionAmt": "${:,.2f}"})
            .background_gradient(subset=["FraudProbability"], cmap="RdYlGn_r"),
        use_container_width=True, height=450
    )

    # Interactive scatter
    sample = filtered.sample(min(2000, len(filtered)), random_state=42)
    fig = px.scatter(
        sample, x="HourOfDay", y=np.log1p(sample["TransactionAmt"]),
        color="FraudProbability",
        color_continuous_scale="RdYlGn_r",
        size="FraudProbability", size_max=15,
        hover_data=["TransactionID","TransactionAmt","RiskTier"],
        title="Transaction Amount vs Hour (colour = Fraud Probability)",
        template="plotly_dark",
        labels={"y": "log(1+Amount)", "HourOfDay": "Hour of Day"}
    )
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP EXPLAINER
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Explainer — Why did the model flag this transaction?")
    st.markdown("Enter a TransactionID to get a full SHAP explanation with plain-English interpretation.")
    st.markdown("---")

    model, scaler, features = load_model()

    txn_id_input = st.text_input("Enter TransactionID", placeholder="e.g. 2987004")
    explain_btn  = st.button("🔍 Explain Transaction", type="primary")

    if explain_btn and txn_id_input:
        try:
            txn_id = int(txn_id_input)
        except:
            st.error("Please enter a valid numeric TransactionID.")
            st.stop()

        with st.spinner("Loading and scoring..."):
            df = load_data()
            row = df[df["TransactionID"] == txn_id]
            if len(row) == 0:
                st.error(f"TransactionID {txn_id} not found in dataset.")
                st.stop()

            from sklearn.preprocessing import LabelEncoder
            df_proc = df
            df_proc["AmtToMeanRatio"]  = df_proc["TransactionAmt"] / df_proc["TransactionAmt"].mean()
            df_proc["HourOfDay"]       = (df_proc["TransactionDT"] // 3600) % 24
            df_proc["AmtLog"]          = np.log1p(df_proc["TransactionAmt"])
            df_proc["IsAnonymousEmail"]= df_proc["P_emaildomain"].astype(str).str.contains("anonymous", na=False).astype(int) if "P_emaildomain" in df_proc.columns else 0
            df_proc["DeviceRisk"]      = (df_proc["DeviceType"].astype(str).str.lower() == "mobile").astype(int) if "DeviceType" in df_proc.columns else 0

            cat_cols = df.select_dtypes(include="object").columns.tolist()
            for col in cat_cols:
                if col in df_proc.columns:
                    le = LabelEncoder()
                    df_proc[col] = le.fit_transform(df_proc[col].astype(str))
            for col in df_proc.select_dtypes(include="number").columns:
                df_proc[col] = df_proc[col].fillna(df_proc[col].median())
            missing_feats = [f for f in features if f not in df_proc.columns]
            for f in missing_feats:
                df_proc[f] = 0

            idx     = df[df["TransactionID"] == txn_id].index[0]
            row_num = df.index.get_loc(idx)
            X_all   = scaler.transform(df_proc[features])
            proba   = model.predict_proba(X_all[row_num:row_num+1])[:, 1][0]
            tier    = assign_tier(proba)

        # ── KPI row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TransactionID",   str(txn_id))
        c2.metric("Fraud Probability", f"{proba:.4f}")
        c3.metric("Risk Tier",        tier)
        c4.metric("Amount",           f"${df.loc[idx,'TransactionAmt']:,.2f}")

        st.markdown("---")

        # ── SHAP waterfall
        with st.spinner("Computing SHAP values..."):
            explainer  = shap.TreeExplainer(model)
            X_df       = pd.DataFrame(X_all, columns=features)
            exp        = explainer(X_df.iloc[[row_num]])

            try:
                if exp.values.ndim == 3:
                    sv_single = shap.Explanation(
                        values      = exp.values[0, :, 1],
                        base_values = exp.base_values[0, 1] if exp.base_values.ndim > 1 else exp.base_values[0],
                        data        = exp.data[0],
                        feature_names = features
                    )
                else:
                    sv_single = exp[0]

                fig_w, ax_w = plt.subplots(figsize=(12, 6), facecolor="#0f0f1a")
                shap.waterfall_plot(sv_single, max_display=14, show=False)
                plt.title(f"SHAP Waterfall — TransactionID {txn_id}  (p={proba:.4f})",
                           fontsize=13, fontweight="bold", color="white")
                plt.gcf().set_facecolor("#0f0f1a")
                st.pyplot(plt.gcf())
                plt.close()
            except Exception as e:
                st.warning(f"Waterfall plot error: {e}")

        # ── Plain-English explanation
        st.markdown("### 🗣️ Plain-English Explanation")
        if proba >= 0.75:
            explanation = f"""
**⚠️ HIGH FRAUD RISK** — This transaction has a fraud probability of **{proba:.1%}**.

The model is flagging this transaction primarily because:
- The transaction amount (${df.loc[idx,'TransactionAmt']:,.2f}) is significantly above the dataset average, a classic signal of large fraudulent purchases or card-testing attacks.
- The transaction may have occurred during a high-risk hour, when fraud monitoring is reduced.
- Device and identity features are inconsistent with typical legitimate customer behaviour.

**Recommended action:** 🔴 **Block and trigger step-up authentication immediately.** Contact customer via registered phone to verify intent before authorising.
"""
        elif proba >= 0.40:
            explanation = f"""
**🟡 SUSPICIOUS — REVIEW REQUIRED** — Fraud probability: **{proba:.1%}**.

The model is uncertain about this transaction. Some features push toward fraud (elevated amount or unusual hour), while others align with legitimate behaviour (recognised email domain or established account history).

**Recommended action:** 🟡 **Flag for manual review.** Send a push notification to the customer for in-app transaction confirmation. Do not block outright.
"""
        else:
            explanation = f"""
**✅ LIKELY LEGITIMATE** — Fraud probability: **{proba:.1%}**.

The model is confident this is a genuine transaction. The amount is within the customer's normal range, the transaction hour is typical, and the device/identity features match the customer's historical profile.

**Recommended action:** 🟢 **Approve and proceed.** No intervention required.
"""
        st.markdown(explanation)
