"""
app/streamlit_app.py

Drift — behavioral intelligence platform.
Premium glassmorphism UI — soft glows, floating cards, Framer aesthetic.
"""

import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_URL = "http://127.0.0.1:8000"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

st.set_page_config(page_title="Drift", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');
:root {
    --bg:#04040e; --bg2:#07071a; --glass:rgba(255,255,255,0.035);
    --glass-b:rgba(255,255,255,0.08); --glow-v:#7b5ea7; --glow-p:#a855f7;
    --glow-r:#ec4899; --glow-b:#3b82f6; --glow-g:#10b981;
    --text:#e8e4f0; --text-muted:#7a7390; --border:rgba(255,255,255,0.07);
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text);background:var(--bg)!important;}
.main,.block-container{background:var(--bg)!important;padding-top:1.5rem!important;}
.main::before{content:'';position:fixed;inset:0;background:
    radial-gradient(ellipse 80% 50% at 20% 10%,rgba(123,94,167,0.12) 0%,transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 90%,rgba(236,72,153,0.08) 0%,transparent 60%),
    radial-gradient(ellipse 50% 60% at 50% 50%,rgba(59,130,246,0.05) 0%,transparent 70%);
    pointer-events:none;z-index:0;}
[data-testid="stSidebar"]{background:rgba(7,7,26,0.97)!important;border-right:1px solid var(--border)!important;backdrop-filter:blur(20px);}
[data-testid="stSidebar"]::before{content:'';position:absolute;top:0;left:0;right:0;height:280px;
    background:radial-gradient(ellipse at 50% 0%,rgba(123,94,167,0.18) 0%,transparent 70%);pointer-events:none;}
.drift-logo{font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;letter-spacing:-0.03em;
    background:linear-gradient(135deg,#c084fc 0%,#ec4899 60%,#f97316 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1;margin-bottom:2px;}
.drift-tag{font-family:'DM Sans',sans-serif;font-size:0.63rem;font-weight:300;letter-spacing:0.22em;
    text-transform:uppercase;color:var(--text-muted);}
.page-title{font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;letter-spacing:-0.04em;
    background:linear-gradient(135deg,#e8e4f0 0%,#9d8ec4 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1.1;margin-bottom:0.2rem;}
.page-sub{font-size:0.75rem;font-weight:300;letter-spacing:0.18em;text-transform:uppercase;
    color:var(--text-muted);margin-bottom:1.5rem;}
.g-card{background:var(--glass);border:1px solid var(--border);border-radius:16px;padding:1.4rem 1.6rem;
    margin-bottom:1rem;backdrop-filter:blur(12px);position:relative;overflow:hidden;}
.g-card::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
    background:linear-gradient(90deg,transparent,rgba(255,255,255,0.12),transparent);}
.g-card-v{box-shadow:0 0 40px rgba(168,85,247,0.08),inset 0 1px 0 rgba(168,85,247,0.08);}
.g-card-r{box-shadow:0 0 40px rgba(236,72,153,0.08),inset 0 1px 0 rgba(236,72,153,0.08);}
.metric-pill{background:var(--glass);border:1px solid var(--border);border-radius:12px;
    padding:1rem 1.2rem;text-align:center;backdrop-filter:blur(8px);}
.metric-pill .val{font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:700;letter-spacing:-0.03em;
    background:linear-gradient(135deg,#e8e4f0,#c084fc);-webkit-background-clip:text;
    -webkit-text-fill-color:transparent;background-clip:text;}
.metric-pill .lbl{font-size:0.68rem;font-weight:300;letter-spacing:0.15em;text-transform:uppercase;
    color:var(--text-muted);margin-top:2px;}
.risk-badge{display:inline-flex;align-items:center;gap:6px;padding:4px 14px;border-radius:999px;
    font-family:'Syne',sans-serif;font-size:0.75rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;}
.risk-critical{background:rgba(236,72,153,0.12);border:1px solid rgba(236,72,153,0.3);color:#f472b6;}
.risk-high{background:rgba(249,115,22,0.12);border:1px solid rgba(249,115,22,0.3);color:#fb923c;}
.risk-medium{background:rgba(234,179,8,0.12);border:1px solid rgba(234,179,8,0.3);color:#fbbf24;}
.risk-low{background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.3);color:#34d399;}
.arch-badge{display:inline-flex;align-items:center;gap:8px;
    background:linear-gradient(135deg,rgba(168,85,247,0.12),rgba(236,72,153,0.08));
    border:1px solid rgba(168,85,247,0.22);border-radius:999px;padding:6px 16px;
    font-family:'Syne',sans-serif;font-size:0.85rem;font-weight:600;color:#d8b4fe;margin-bottom:1.2rem;}
.persona-box{background:linear-gradient(135deg,rgba(123,94,167,0.07),rgba(59,130,246,0.04));
    border:1px solid rgba(168,85,247,0.13);border-radius:14px;padding:1.4rem 1.6rem 1.4rem 2rem;
    font-size:0.94rem;font-weight:300;line-height:1.78;color:#c4bedd;position:relative;overflow:hidden;}
.persona-box::before{content:'';position:absolute;top:0;left:0;width:3px;height:100%;
    background:linear-gradient(180deg,#a855f7,#ec4899);border-radius:3px 0 0 3px;}
.rec-box{background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.15);border-radius:10px;
    padding:0.85rem 1.1rem;color:#6ee7b7;font-size:0.87rem;font-weight:400;}
.rf-row{display:flex;align-items:center;gap:10px;padding:0.55rem 0;
    border-bottom:1px solid var(--border);font-size:0.87rem;}
.rf-dot-high{width:7px;height:7px;border-radius:50%;background:#f472b6;flex-shrink:0;box-shadow:0 0 7px #f472b6;}
.rf-dot-medium{width:7px;height:7px;border-radius:50%;background:#fbbf24;flex-shrink:0;box-shadow:0 0 7px #fbbf24;}
.rf-dot-low{width:7px;height:7px;border-radius:50%;background:#34d399;flex-shrink:0;box-shadow:0 0 7px #34d399;}
.api-ok{display:inline-flex;align-items:center;gap:6px;font-size:0.76rem;color:#34d399;
    background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.18);border-radius:999px;padding:4px 12px;}
.api-err{display:inline-flex;align-items:center;gap:6px;font-size:0.76rem;color:#f472b6;
    background:rgba(236,72,153,0.08);border:1px solid rgba(236,72,153,0.18);border-radius:999px;padding:4px 12px;}
.api-dot{width:6px;height:6px;border-radius:50%;}
.api-dot-ok{background:#34d399;box-shadow:0 0 5px #34d399;}
.api-dot-err{background:#f472b6;box-shadow:0 0 5px #f472b6;}
.sec-label{font-size:0.67rem;font-weight:500;letter-spacing:0.18em;text-transform:uppercase;
    color:var(--text-muted);margin-bottom:0.5rem;margin-top:0.1rem;}
[data-testid="stSidebar"] .stRadio label{font-family:'DM Sans',sans-serif!important;
    font-size:0.87rem!important;color:var(--text-muted)!important;padding:0.3rem 0!important;}
.stButton>button{background:linear-gradient(135deg,#7b5ea7,#a855f7)!important;border:none!important;
    border-radius:10px!important;color:white!important;font-family:'Syne',sans-serif!important;
    font-weight:600!important;font-size:0.87rem!important;letter-spacing:0.04em!important;
    padding:0.6rem 1.4rem!important;box-shadow:0 4px 20px rgba(168,85,247,0.28)!important;transition:all 0.2s!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 6px 28px rgba(168,85,247,0.42)!important;}
.stTabs [data-baseweb="tab-list"]{background:var(--glass)!important;border-radius:10px!important;
    padding:3px!important;gap:2px!important;border:1px solid var(--border)!important;}
.stTabs [data-baseweb="tab"]{border-radius:8px!important;color:var(--text-muted)!important;
    font-family:'DM Sans',sans-serif!important;font-size:0.82rem!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,rgba(168,85,247,0.18),rgba(236,72,153,0.12))!important;color:#d8b4fe!important;}
hr{border-color:var(--border)!important;margin:1.5rem 0!important;}
::-webkit-scrollbar{width:4px;}::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(168,85,247,0.25);border-radius:2px;}
</style>
""", unsafe_allow_html=True)


def check_api():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).status_code == 200
    except:
        return False

def predict_single(f):
    try:
        r = requests.post(f"{API_URL}/predict", json=f, timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"API error: {e}"); return None

def predict_batch(users):
    try:
        r = requests.post(f"{API_URL}/predict/batch", json={"users": users}, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"API error: {e}"); return None

def get_persona(prediction, features):
    try:
        from drift.serving.persona import generate_persona_report
        return generate_persona_report(
            archetype=prediction["archetype"], churn_probability=prediction["churn_probability"],
            risk_level=prediction["risk_level"], risk_factors=prediction["top_risk_factors"],
            recommendation=prediction["recommendation"],
            user_stats={k: features.get(k) for k in ["total_sessions","sessions_per_week","total_purchases","cart_abandonment_rate","avg_price_point","weekend_activity_ratio","recency_days"]},
        )
    except Exception as e:
        return f"Persona unavailable: {e}"

def risk_color(r):
    return {"critical":"#f472b6","high":"#fb923c","medium":"#fbbf24","low":"#34d399"}.get(r,"#ccc")

def churn_gauge(prob, risk):
    c = risk_color(risk)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(prob*100,1),
        number={"suffix":"%","font":{"size":42,"color":c,"family":"Syne"}},
        gauge={"axis":{"range":[0,100],"tickcolor":"rgba(255,255,255,0.08)","tickfont":{"color":"rgba(255,255,255,0.25)","size":9}},
               "bar":{"color":c,"thickness":0.18},"bgcolor":"rgba(255,255,255,0.02)","bordercolor":"rgba(255,255,255,0.05)",
               "steps":[{"range":[0,40],"color":"rgba(16,185,129,0.05)"},{"range":[40,65],"color":"rgba(234,179,8,0.05)"},
                        {"range":[65,85],"color":"rgba(249,115,22,0.05)"},{"range":[85,100],"color":"rgba(236,72,153,0.07)"}],
               "threshold":{"line":{"color":c,"width":2},"thickness":0.75,"value":prob*100}},
        title={"text":"CHURN RISK","font":{"color":"rgba(255,255,255,0.25)","size":10,"family":"DM Sans"}},
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e8e4f0",height=210,margin=dict(t=40,b=0,l=20,r=20))
    return fig

def pdark(fig, h=320):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(255,255,255,0.015)",
                      font_color="#9d8ec4",font_family="DM Sans",height=h,
                      margin=dict(t=24,b=24,l=8,r=8),coloraxis_showscale=False,
                      xaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="rgba(255,255,255,0.05)"),
                      yaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="rgba(255,255,255,0.05)"))
    return fig


# Sidebar
with st.sidebar:
    st.markdown('<div class="drift-logo">drift</div>', unsafe_allow_html=True)
    st.markdown('<div class="drift-tag">behavioral intelligence</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    api_ok = check_api()
    if api_ok:
        st.markdown('<span class="api-ok"><span class="api-dot api-dot-ok"></span>API connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="api-err"><span class="api-dot api-dot-err"></span>API offline</span>', unsafe_allow_html=True)
        st.caption("`uvicorn drift.serving.api:app --port 8000`")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Navigate</div>', unsafe_allow_html=True)
    page = st.radio("", ["Single User","Batch Analysis","Model Performance","Archetypes"], label_visibility="collapsed")
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.73rem;color:#3d3858;line-height:1.8;">109M events<br>3M users<br>Oct – Nov 2019<br>REES46</div>', unsafe_allow_html=True)


# Single User
if page == "Single User":
    st.markdown('<div class="page-title">user analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">predict churn · assign archetype · generate persona</div>', unsafe_allow_html=True)

    with st.expander("⊕  Feature input", expanded=True):
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown('<div class="sec-label">Session behavior</div>', unsafe_allow_html=True)
            recency_days           = st.number_input("Recency (days)",0.0,365.0,12.0,1.0)
            total_sessions         = st.number_input("Total sessions",1,10000,8,1)
            sessions_per_week      = st.number_input("Sessions / week",0.0,50.0,1.5,0.1)
            avg_events_per_session = st.number_input("Avg events / session",0.0,500.0,6.0,0.5)
            avg_duration_min       = st.number_input("Avg duration (min)",0.0,300.0,8.0,0.5)
        with c2:
            st.markdown('<div class="sec-label">Purchase behavior</div>', unsafe_allow_html=True)
            total_revenue            = st.number_input("Total revenue ($)",0.0,100000.0,120.0,10.0)
            avg_session_revenue      = st.number_input("Avg session revenue ($)",0.0,10000.0,15.0,1.0)
            total_views              = st.number_input("Total views",0,100000,60,5)
            total_cart_adds          = st.number_input("Cart adds",0,10000,8,1)
            total_purchases          = st.number_input("Purchases",0,10000,2,1)
            cart_abandonment_rate    = st.slider("Cart abandonment rate",0.0,1.0,0.65,0.01)
            purchase_conversion_rate = st.slider("Purchase conversion rate",0.0,1.0,0.03,0.01)
        with c3:
            st.markdown('<div class="sec-label">Temporal & category</div>', unsafe_allow_html=True)
            weekend_activity_ratio = st.slider("Weekend ratio",0.0,1.0,0.4,0.01)
            night_owl_score        = st.slider("Night owl score",0.0,1.0,0.15,0.01)
            payday_activity_ratio  = st.slider("Payday ratio",0.0,1.0,0.2,0.01)
            activity_trend         = st.slider("Activity trend",-1.0,1.0,-0.05,0.01)
            category_diversity     = st.number_input("Category diversity",0.0,10.0,1.2,0.1)
            brand_loyalty_score    = st.slider("Brand loyalty",0.0,1.0,0.4,0.01)
            avg_price_point        = st.number_input("Avg price point ($)",0.0,10000.0,75.0,5.0)
            price_sensitivity      = st.number_input("Price sensitivity",0.0,5000.0,30.0,5.0)

    if st.button("Analyze User →", type="primary", use_container_width=True):
        if not api_ok:
            st.error("API offline.")
        else:
            features = {
                "recency_days":recency_days,"total_sessions":int(total_sessions),
                "sessions_per_week":sessions_per_week,"total_revenue":total_revenue,
                "avg_session_revenue":avg_session_revenue,"avg_events_per_session":avg_events_per_session,
                "avg_duration_min":avg_duration_min,"total_views":int(total_views),
                "total_cart_adds":int(total_cart_adds),"total_purchases":int(total_purchases),
                "cart_abandonment_rate":cart_abandonment_rate,"purchase_conversion_rate":purchase_conversion_rate,
                "weekend_activity_ratio":weekend_activity_ratio,"night_owl_score":night_owl_score,
                "payday_activity_ratio":payday_activity_ratio,"activity_trend":activity_trend,
                "category_diversity":category_diversity,"brand_loyalty_score":brand_loyalty_score,
                "avg_price_point":avg_price_point,"price_sensitivity":price_sensitivity,
            }
            with st.spinner("Analyzing behavioral signals..."):
                result = predict_single(features)
            if result:
                st.markdown("---")
                c1,c2 = st.columns([1,1.6])
                with c1:
                    st.markdown('<div class="g-card g-card-v">', unsafe_allow_html=True)
                    st.plotly_chart(churn_gauge(result["churn_probability"],result["risk_level"]), use_container_width=True)
                    risk = result["risk_level"]
                    st.markdown(f'<div style="text-align:center;margin-top:-8px;"><span class="risk-badge risk-{risk}">◉ {risk.upper()} RISK</span></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown('<div class="g-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="arch-badge">◈ {result["archetype"]}</div>', unsafe_allow_html=True)
                    st.markdown('<div class="sec-label">Risk factors</div>', unsafe_allow_html=True)
                    for rf in result["top_risk_factors"]:
                        st.markdown(f'<div class="rf-row"><span class="rf-dot-{rf["impact"]}"></span><span style="color:#e8e4f0;font-weight:500;">{rf["factor"]}</span><span style="color:#7a7390;margin-left:4px;">— {rf["detail"]}</span></div>', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown('<div class="sec-label">Recommended intervention</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="rec-box">→ {result["recommendation"]}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")
                st.markdown('<div class="sec-label">AI Persona Report</div>', unsafe_allow_html=True)
                with st.spinner("Generating behavioral narrative..."):
                    persona = get_persona(result, features)
                st.markdown(f'<div class="persona-box">{persona}</div>', unsafe_allow_html=True)


# Batch Analysis
elif page == "Batch Analysis":
    st.markdown('<div class="page-title">batch analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">upload a CSV of users — predictions at scale</div>', unsafe_allow_html=True)
    st.markdown('<div class="g-card">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">Required columns</div>', unsafe_allow_html=True)
    st.code("recency_days, total_sessions, sessions_per_week, total_revenue, avg_session_revenue, avg_events_per_session, avg_duration_min, total_views, total_cart_adds, total_purchases, cart_abandonment_rate, purchase_conversion_rate, weekend_activity_ratio, night_owl_score, payday_activity_ratio, activity_trend, category_diversity, brand_loyalty_score, avg_price_point, price_sensitivity", language=None)
    st.markdown('</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload user CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df):,} users")
        st.dataframe(df.head(5), use_container_width=True)
        if st.button(f"Run predictions →", type="primary"):
            if not api_ok:
                st.error("API offline.")
            else:
                with st.spinner(f"Predicting for {len(df):,} users..."):
                    result = predict_batch(df.to_dict(orient="records"))
                if result:
                    s = result["summary"]; preds = result["predictions"]
                    st.markdown("---")
                    c1,c2,c3,c4 = st.columns(4)
                    for col,(lbl,val) in zip([c1,c2,c3,c4],[("Total users",f"{s['total_users']:,}"),("Predicted churners",f"{s['predicted_churners']:,}"),("Avg churn prob",f"{s['avg_churn_probability']:.1%}"),("High risk",f"{s['high_risk_users']:,}")]):
                        col.markdown(f'<div class="metric-pill"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
                    st.markdown("---")
                    ad = s["archetype_distribution"]
                    fig = px.bar(x=list(ad.keys()),y=list(ad.values()),color=list(ad.values()),color_continuous_scale=["#7b5ea7","#a855f7","#ec4899"],title="Archetype Distribution")
                    st.plotly_chart(pdark(fig), use_container_width=True)
                    rdf = pd.DataFrame([{"churn_probability":p["churn_probability"],"risk_level":p["risk_level"],"archetype":p["archetype"],"recommendation":p["recommendation"]} for p in preds])
                    st.dataframe(rdf, use_container_width=True)
                    st.download_button("Download results CSV", rdf.to_csv(index=False), "drift_predictions.csv","text/csv")


# Model Performance
elif page == "Model Performance":
    st.markdown('<div class="page-title">model performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">evaluation metrics · explainability · SHAP</div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val) in zip([c1,c2,c3,c4],[("ROC-AUC","0.9987"),("Avg Precision","0.9999"),("F1 Score","0.9914"),("Recall","0.9830")]):
        col.markdown(f'<div class="metric-pill"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    plots_dir = DATA_DIR/"plots"
    plot_files = {"ROC Curve":plots_dir/"roc_curve.png","Precision-Recall":plots_dir/"pr_curve.png","SHAP Importance":plots_dir/"shap_importance.png","Score Distribution":plots_dir/"score_distribution.png","Confusion Matrix":plots_dir/"confusion_matrix.png"}
    available = {k:v for k,v in plot_files.items() if v.exists()}
    if available:
        tabs = st.tabs(list(available.keys()))
        for tab,(name,path) in zip(tabs,available.items()):
            with tab:
                st.image(str(path), use_container_width=True)
    if api_ok:
        st.markdown("---")
        st.markdown('<div class="sec-label">Top features by SHAP importance</div>', unsafe_allow_html=True)
        try:
            info = requests.get(f"{API_URL}/model/info",timeout=5).json()
            si = info.get("top_features_shap",{})
            if si:
                sdf = pd.DataFrame(list(si.items()),columns=["Feature","Mean |SHAP|"]).sort_values("Mean |SHAP|")
                fig = px.bar(sdf,x="Mean |SHAP|",y="Feature",orientation="h",color="Mean |SHAP|",color_continuous_scale=["#7b5ea7","#a855f7","#ec4899"])
                fig.update_layout(yaxis={"categoryorder":"total ascending"})
                st.plotly_chart(pdark(fig), use_container_width=True)
        except:
            pass


# Archetypes
elif page == "Archetypes":
    st.markdown('<div class="page-title">archetypes</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">behavioral clusters discovered via HDBSCAN</div>', unsafe_allow_html=True)
    profiles_path = DATA_DIR/"cluster_profiles.json"
    if not profiles_path.exists():
        st.warning("Run `python drift/models/train_cluster.py` first.")
    else:
        with open(profiles_path) as f:
            profiles = json.load(f)
        valid = {k:v for k,v in profiles.items() if int(k)!=-1}
        for cid,profile in valid.items():
            st.markdown('<div class="g-card g-card-v">', unsafe_allow_html=True)
            c1,c2 = st.columns([1,2.2])
            with c1:
                st.markdown(f'<div class="arch-badge">◈ {profile["archetype"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-pill"><div class="val">{profile["user_count"]:,}</div><div class="lbl">users in cluster</div></div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                if profile.get("churn_rate") is not None:
                    cr_color = "#f472b6" if profile["churn_rate"]>0.8 else "#fbbf24"
                    st.markdown(f'<div style="font-family:Syne;font-size:1.6rem;font-weight:700;color:{cr_color};">{profile["churn_rate"]:.1%}</div><div class="sec-label">churn rate</div>', unsafe_allow_html=True)
            with c2:
                feats = profile.get("features",{})
                if feats:
                    fdf = pd.DataFrame(list(feats.items()),columns=["Feature","Mean value"]).head(8)
                    fig = px.bar(fdf,x="Mean value",y="Feature",orientation="h",color="Mean value",color_continuous_scale=["#7b5ea7","#a855f7","#ec4899"])
                    fig.update_layout(yaxis={"categoryorder":"total ascending"},margin=dict(l=0,r=0,t=8,b=0))
                    st.plotly_chart(pdark(fig,260), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")
        st.markdown('<div class="sec-label">AI Executive Summary</div>', unsafe_allow_html=True)
        if st.button("Generate cluster summary →"):
            with st.spinner("Generating..."):
                try:
                    from drift.serving.persona import generate_cluster_summary
                    summary = generate_cluster_summary(profiles)
                    st.markdown(f'<div class="persona-box">{summary}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Could not generate summary: {e}")
