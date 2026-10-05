"""
Flipkart Reviews Sentiment Analysis - Streamlit dashboard (project edition)

Every number on every page comes from this project's own artifacts:
  artifacts/   best_ml_pipeline.joblib (tuned Logistic Regression + TF-IDF), best_dl_model.keras (LSTM), dl_tokenizer.json, preprocessing.json
  results.json, app_data.json, model_comparison.csv   (written by the notebook / evaluation scripts)
  Flipkart_Reviews_Cleaned.csv                         (cleaned dataset from Q8)
Run:  streamlit run app.py     (from the project folder)
"""
import os, io, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from datetime import datetime
import joblib, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
import nlp_utils as nu

ORDER = ["Negative", "Neutral", "Positive"]
COL = {"Positive": "#00f59b", "Neutral": "#ffb703", "Negative": "#ff0055"}
EMO = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
LR_NAME, DL_NAME = "Logistic Regression (TF-IDF)", "LSTM (Keras)"
DATA_PATHS = ["Flipkart_Reviews_Cleaned.csv", "data/Flipkart_Reviews_Cleaned.csv", "Flipkart_Reviews_Sentiment_Analysis.csv"]

st.set_page_config(page_title="Flipkart Sentiment AI", page_icon="🛍️", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
.stApp { background-color:#060813; background-image: radial-gradient(at 0% 0%, rgba(121,40,202,.12) 0, transparent 50%), radial-gradient(at 100% 100%, rgba(0,242,254,.08) 0, transparent 50%); background-attachment: fixed; }
.glass { background: rgba(13,19,38,.75); backdrop-filter: blur(16px); border:1px solid rgba(0,242,254,.18); border-radius:16px; padding:22px; margin-bottom:18px; box-shadow:0 10px 30px -10px rgba(0,0,0,.6); }
.kpi { background: rgba(18,25,48,.65); border:1px solid rgba(255,255,255,.07); border-left:4px solid #00f2fe; border-radius:14px; padding:16px 18px; }
.kpi .t { font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:#94a3b8; font-weight:600; }
.kpi .v { font-size:1.75rem; font-weight:800; margin:4px 0 2px; color:#f0f4fc; }
.kpi .s { font-size:.74rem; color:#94a3b8; }
.gtitle { background: linear-gradient(135deg,#00f2fe 0%,#4facfe 50%,#7928ca 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800; letter-spacing:-.02em; }
.sub { color:#94a3b8; margin-bottom:14px; }
.banner { border-radius:18px; padding:22px; text-align:center; margin:8px 0 14px; }
.b-Positive { background:rgba(0,245,155,.10); border:2px solid #00f59b; box-shadow:0 0 28px rgba(0,245,155,.22); }
.b-Neutral  { background:rgba(255,183,3,.10); border:2px solid #ffb703; box-shadow:0 0 28px rgba(255,183,3,.22); }
.b-Negative { background:rgba(255,0,85,.10); border:2px solid #ff0055; box-shadow:0 0 28px rgba(255,0,85,.22); }
.tok { font-family:'JetBrains Mono',monospace; background:rgba(0,242,254,.08); border:1px solid rgba(0,242,254,.2); border-radius:6px; padding:2px 8px; color:#00f2fe; font-size:.82rem; display:inline-block; margin:2px; }
.warn { border-left:4px solid #ffb703; background:rgba(255,183,3,.08); border-radius:10px; padding:12px 16px; color:#e2e8f0; font-size:.9rem; margin:10px 0; }
.bad  { border-left:4px solid #ff0055; background:rgba(255,0,85,.08); border-radius:10px; padding:12px 16px; color:#e2e8f0; font-size:.9rem; margin:10px 0; }
div.stButton > button:first-child { background: linear-gradient(135deg,#00f2fe,#4facfe) !important; color:#030712 !important; font-weight:700 !important; border:none !important; border-radius:10px !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)
if "history" not in st.session_state: st.session_state.history = []

# ---------------------------------------------------------------- helpers
def html(s): st.markdown(s, unsafe_allow_html=True)
def head(title, sub): html(f"<h1 class='gtitle' style='font-size:2.1rem;margin-bottom:2px'>{title}</h1><p class='sub'>{sub}</p>")
def kpi(title, value, sub, color="#00f2fe"): return f"<div class='kpi' style='border-left-color:{color}'><div class='t'>{title}</div><div class='v'>{value}</div><div class='s'>{sub}</div></div>"
def note(text, kind="warn"): html(f"<div class='{kind}'>{text}</div>")
def style(fig, h=320, legend_bottom=False):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1"), height=h, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(gridcolor="#1e293b"); fig.update_yaxes(gridcolor="#1e293b")
    if legend_bottom: fig.update_layout(legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"))
    return fig
def show(fig, h=320, legend_bottom=False): st.plotly_chart(style(fig, h, legend_bottom), width="stretch")

def find(paths):
    return next((p for p in paths if os.path.exists(p)), None)

# ---------------------------------------------------------------- loaders
@st.cache_resource(show_spinner=False)
def load_ml():
    try:
        pipe = joblib.load("artifacts/best_ml_pipeline.joblib")
        text_col = json.load(open("artifacts/preprocessing.json"))["text_col"]
        return pipe, text_col, None
    except Exception as e:
        return None, "text_lemma", str(e)

@st.cache_resource(show_spinner="Loading LSTM...")
def load_dl():
    try:
        import tensorflow as tf
        from tensorflow import keras
        from keras import layers
        tok = json.load(open("artifacts/dl_tokenizer.json"))
        vec = layers.TextVectorization(max_tokens=len(tok["vocabulary"]), output_sequence_length=tok["maxlen"], standardize=None, split="whitespace")
        vec.set_vocabulary(tok["vocabulary"])
        return keras.models.load_model("artifacts/best_dl_model.keras"), vec, tok.get("best_dl", "LSTM"), None
    except Exception as e:
        return None, None, None, str(e)

@st.cache_data(show_spinner=False)
def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else {}

@st.cache_data(show_spinner=False)
def load_data():
    p = find(DATA_PATHS)
    if not p: return None
    df = pd.read_csv(p)
    if "Review_Head" not in df.columns:
        df["Review_Head"] = df.Review_Text.str.rsplit(", ", n=1).str[0]; df["Review_Tail"] = df.Review_Text.str.rsplit(", ", n=1).str[1]
    return df

PIPE, TEXT_COL, ML_ERR = load_ml()
R, APP = load_json("results.json"), load_json("app_data.json")
TFIDF = PIPE.named_steps["tfidf"] if PIPE is not None else None
CLF = PIPE.named_steps["clf"] if PIPE is not None else None

# ---------------------------------------------------------------- inference
def clean_for_model(summary, text): return nu.build_variant(nu.combine(summary, text), TEXT_COL)

def predict_proba(cleaned_list, engine):
    if engine == LR_NAME: return PIPE.predict_proba(cleaned_list)
    import tensorflow as tf
    model, vec, _, err = load_dl()
    if model is None: raise RuntimeError(f"LSTM could not be loaded: {err}")
    return model.predict(vec(tf.constant(cleaned_list)).numpy(), verbose=0)

def coverage(cleaned):
    toks = cleaned.split(); vocab = TFIDF.vocabulary_
    return (sum(t in vocab for t in toks) / len(toks)) if toks else 0.0, [t for t in toks if t not in vocab]

def explain_lr(cleaned, k):
    X = TFIDF.transform([cleaned]); idx = X.nonzero()[1]
    contrib = np.asarray(X[0, idx].todense()).ravel() * CLF.coef_[k][idx]
    names = np.array(TFIDF.get_feature_names_out())[idx]
    return pd.DataFrame({"term": names, "contribution": contrib}).sort_values("contribution")

# ---------------------------------------------------------------- sidebar
def sidebar():
    with st.sidebar:
        html("<div style='text-align:center;padding:6px 0 14px'><span style='font-size:2.2rem'>🛍️</span><h2 style='margin:0;font-size:1.3rem;font-weight:800'>Flipkart <span style='color:#00f2fe'>Sentiment</span> AI</h2><p style='font-size:.75rem;color:#94a3b8;margin:4px 0 0'>NLP · Machine Learning · Deep Learning</p></div>")
        page = st.radio("Navigation", ["🏠 Home", "🔍 Sentiment Analyzer", "📄 Batch Scoring", "📊 Data Analytics", "🤖 Model Comparison", "🧪 Errors & Features", "ℹ️ About Project"], label_visibility="collapsed")
        st.markdown("---")
        if PIPE is not None:
            nfeat = len(TFIDF.get_feature_names_out())
            html(f"<div style='background:rgba(0,245,155,.08);border:1px solid rgba(0,245,155,.3);padding:12px;border-radius:10px;font-size:.78rem;color:#94a3b8'><b style='color:#00f59b'>● MODELS LOADED</b><br>• <b>ML:</b> tuned Logistic Regression<br>• <b>Features:</b> TF-IDF uni+bigrams, {nfeat:,} terms<br>• <b>Text:</b> {TEXT_COL.replace('text_', '')} pipeline<br>• <b>DL:</b> LSTM (Keras)</div>")
        else:
            html(f"<div class='bad'><b>MODELS NOT FOUND</b><br>Run the notebook first, or place <code>artifacts/</code> beside app.py.<br><small>{ML_ERR}</small></div>")
        st.markdown("---")
        html("<div style='font-size:.74rem;color:#64748b;text-align:center;line-height:1.5'>Student project · 30,000 reviews<br>Macro-F1 is the primary metric</div>")
    return page

# ---------------------------------------------------------------- pages
def page_home():
    uf = R.get("ml_tuned_holdout", {}).get("macro_f1", 0.93); lat = R.get("final_compare", [{}])[0].get("ms_per_review", 0.01)
    dl_f = next((r["holdout_macro_f1"] for r in R.get("dl", []) if r["model"] == "LSTM"), 0.959)
    html("<div class='glass' style='text-align:center;padding:40px 20px'><span style='font-size:3rem'>🛒</span><h1 class='gtitle' style='font-size:2.6rem;margin:8px 0'>FLIPKART SENTIMENT AI</h1><p style='color:#cbd5e1;max-width:760px;margin:0 auto;font-size:1.02rem;line-height:1.6'>An end-to-end sentiment pipeline for 30,000 Flipkart-style reviews: EDA, NLP preprocessing, six models (three classical, three deep learning), an honest evaluation design, and this live dashboard.</p></div>")
    c = st.columns(5)
    vals = [("Reviews", "30,000", "25 columns, 3 classes", "#00f2fe"), ("Text features", f"{len(TFIDF.get_feature_names_out()):,}" if TFIDF else "1,004", "TF-IDF unigrams + bigrams", "#7928ca"),
            ("Unseen-phrase F1", f"{uf:.3f}", "Logistic Regression (honest test)", "#00f59b"), ("Random-split F1", "1.000", "inflated by template overlap", "#ff0055"), ("Inference", f"{lat:.2f} ms", "per review, Logistic Regression", "#ffb703")]
    for col, v in zip(c, vals): col.markdown(kpi(*v[:3], v[3]), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    a, b = st.columns([1.2, 1])
    a.markdown(f"""<div class='glass'><h3 style='margin-top:0;color:#00f2fe'>🔬 What this project found</h3><ul style='color:#cbd5e1;line-height:1.8;padding-left:20px'>
<li><b>Label leakage:</b> <code>Rating</code> fixes <code>Sentiment</code> exactly (1-2 Negative, 3 Neutral, 4-5 Positive), so it is never a model input.</li>
<li><b>Template text:</b> only 960 unique review texts, built from 30 sentiment phrases and 32 aspect phrases. A random split scores 100% because 95% of test reviews have a twin in training.</li>
<li><b>Honest test:</b> 9 of 30 phrases are held out of training. Logistic Regression scores <b>{uf:.3f}</b> and the LSTM <b>{dl_f:.3f}</b> (a statistical tie).</li>
<li><b>Recommendation:</b> Logistic Regression: 52 KB, trains in 0.1 s, readable word weights.</li></ul></div>""", unsafe_allow_html=True)
    b.markdown("""<div class='glass'><h3 style='margin-top:0;color:#4facfe'>🧭 Where to go</h3>
<p style='color:#94a3b8;font-size:.9rem;line-height:1.9'>🔍 <b>Sentiment Analyzer</b>: score one review with either model<br>📄 <b>Batch Scoring</b>: score a CSV and download results<br>📊 <b>Data Analytics</b>: EDA, leakage and template evidence<br>🤖 <b>Model Comparison</b>: six models, two tests, confusion matrices<br>🧪 <b>Errors & Features</b>: top words, failing phrases, stress tests</p></div>""", unsafe_allow_html=True)

EXAMPLES = {
    "🌟 Positive (dataset style)": ("Loved it", "Perfect purchase, loved it, battery life is decent."),
    "⚖️ Neutral (dataset style)": ("Average", "Decent product but nothing special, size was accurate."),
    "🚨 Negative (dataset style)": ("Waste of money", "Product stopped working after a few days."),
    "✍️ Free-text positive": ("", "The phone is amazing, the camera is outstanding and the battery easily lasts a full day."),
    "🔀 Mixed review": ("Great product", "Very good quality but delivery was late and packaging was damaged."),
}
def set_example(name): st.session_state.in_summary, st.session_state.in_text = EXAMPLES[name]
st.session_state.setdefault("in_summary", ""); st.session_state.setdefault("in_text", "")

def page_analyzer():
    head("🔍 Live Sentiment Analyzer", "Score a review with the tuned Logistic Regression, the LSTM, or both side by side.")
    if PIPE is None: return note("Model artifacts not found. Run the notebook to create <code>artifacts/</code>.", "bad")
    cols = st.columns(len(EXAMPLES))
    for c, name in zip(cols, EXAMPLES): c.button(name, key="ex_" + name, on_click=set_example, args=(name,), width="stretch")
    c1, c2 = st.columns([1, 2.4])
    summary = c1.text_input("Review summary (optional)", key="in_summary")
    text = c2.text_area("Review text", key="in_text", height=110, placeholder="Type or paste a customer review...")
    engine = st.radio("Model", [LR_NAME, DL_NAME, "Compare both"], horizontal=True)
    if st.button("⚡ ANALYZE SENTIMENT", disabled=not text.strip()):
        cleaned = clean_for_model(summary, text)
        if not cleaned: return note("Nothing left after cleaning (only punctuation or numbers).", "bad")
        engines = [LR_NAME, DL_NAME] if engine == "Compare both" else [engine]
        try: probs = {e: predict_proba([cleaned], e)[0] for e in engines}
        except Exception as e: return note(str(e), "bad")
        labels = {e: ORDER[int(np.argmax(p))] for e, p in probs.items()}
        for col, e in zip(st.columns(len(engines)), engines):
            p, lab = probs[e], labels[e]
            col.markdown(f"<div class='banner b-{lab}'><div style='font-size:2.4rem'>{EMO[lab]}</div><div style='font-size:.8rem;letter-spacing:.12em;color:#94a3b8;font-weight:700'>{e.upper()}</div><h2 style='font-size:2.3rem;font-weight:800;color:{COL[lab]};margin:4px 0'>{lab.upper()}</h2><div style='font-size:1.05rem'>Confidence <b style='color:{COL[lab]}'>{p.max():.1%}</b></div></div>", unsafe_allow_html=True)
            fig = px.bar(pd.DataFrame({"Class": ORDER, "p": p}), x="p", y="Class", orientation="h", color="Class", color_discrete_map=COL, text=[f"{v:.1%}" for v in p])
            fig.update_layout(showlegend=False, xaxis=dict(range=[0, 1.15], showticklabels=False, showgrid=False)); fig.update_traces(textposition="outside"); col.plotly_chart(style(fig, 200), width="stretch")
        if len(set(labels.values())) > 1: note("⚠️ <b>The two models disagree.</b> The review is probably ambiguous or uses wording the models have not seen. Treat both labels with caution.")
        cov, oov = coverage(cleaned)
        if cov < 0.8: note(f"⚠️ <b>Low vocabulary coverage ({cov:.0%}).</b> These words were never seen in training: {', '.join(oov[:8])}. Models trained on this dataset are far less reliable on wording outside its 30 sentiment phrases.")
        elif cov < 1: note(f"ℹ️ Vocabulary coverage {cov:.0%}. Unknown words: {', '.join(oov[:8])}.", "warn")
        a, b = st.columns([1.2, 1])
        with a:
            k = ORDER.index(labels[LR_NAME]) if LR_NAME in labels else ORDER.index(list(labels.values())[0])
            ex = explain_lr(cleaned, k)
            if len(ex):
                st.markdown(f"**Why Logistic Regression says {ORDER[k]}** (TF-IDF × word weight)")
                fig = px.bar(ex.tail(10), x="contribution", y="term", orientation="h", color="contribution", color_continuous_scale=["#ff0055", "#334155", "#00f59b"], color_continuous_midpoint=0)
                fig.update_layout(coloraxis_showscale=False); show(fig, 280)
        with b:
            st.markdown("**Pipeline details**")
            html(f"<div class='glass' style='padding:14px;font-size:.85rem'><b>Raw length:</b> {len(summary) + len(text)} chars<br><b>Cleaned tokens:</b> {len(cleaned.split())}<br><b>Vocabulary coverage:</b> {cov:.0%}<br><b>Text pipeline:</b> {TEXT_COL}</div>")
            with st.expander("🔎 Normalized text the models see"): html("".join(f"<span class='tok'>{t}</span>" for t in cleaned.split()))
        st.session_state.history.insert(0, {"time": datetime.now().strftime("%H:%M:%S"), "review": (text[:80] + ("..." if len(text) > 80 else "")), **{e.split(" ")[0]: f"{labels[e]} ({probs[e].max():.0%})" for e in engines}})
        st.session_state.history = st.session_state.history[:25]
    if st.session_state.history:
        st.markdown("---"); h1, h2 = st.columns([4, 1]); h1.markdown("### 📜 Session history")
        if h2.button("Clear history", width="stretch"): st.session_state.history = []; st.rerun()
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch", hide_index=True)

def page_batch():
    head("📄 Batch Scoring", "Upload a CSV with a <b>Review_Text</b> column (optional <b>Review_Summary</b>). Up to 5,000 rows are scored.")
    if PIPE is None: return note("Model artifacts not found.", "bad")
    eng = st.radio("Model", [LR_NAME, DL_NAME], horizontal=True)
    up = st.file_uploader("CSV file", type="csv")
    if up is None: return st.caption("Tip: upload Flipkart_Reviews_Cleaned.csv to see agreement with the Sentiment column. It will read ~100% because of template overlap, which is not real accuracy.")
    df = pd.read_csv(up)
    if "Review_Text" not in df.columns: return note("The file needs a Review_Text column.", "bad")
    sub = df.head(5000).copy()
    if len(df) > 5000: st.info(f"Scoring the first 5,000 of {len(df):,} rows.")
    summ = sub["Review_Summary"].fillna("") if "Review_Summary" in sub else pd.Series([""] * len(sub), index=sub.index)
    cleaned = [clean_for_model(s, t) for s, t in zip(summ, sub.Review_Text.fillna("").astype(str))]
    try: P = predict_proba(cleaned, eng)
    except Exception as e: return note(str(e), "bad")
    sub["Predicted_Sentiment"] = [ORDER[i] for i in P.argmax(1)]; sub["Confidence"] = P.max(1).round(3)
    sub["Vocabulary_Coverage"] = [round(coverage(c)[0], 2) for c in cleaned]
    m = st.columns(4)
    for c, s in zip(m[:3], ORDER): c.markdown(kpi(s, f"{(sub.Predicted_Sentiment == s).mean():.1%}", f"{(sub.Predicted_Sentiment == s).sum():,} reviews", COL[s]), unsafe_allow_html=True)
    m[3].markdown(kpi("Low coverage", f"{(sub.Vocabulary_Coverage < 0.8).mean():.1%}", "reviews with unfamiliar wording", "#7928ca"), unsafe_allow_html=True)
    if "Sentiment" in sub: st.success(f"Agreement with the Sentiment column: {(sub.Predicted_Sentiment == sub.Sentiment).mean():.1%}")
    st.dataframe(sub[[c for c in ["Review_Summary", "Review_Text", "Predicted_Sentiment", "Confidence", "Vocabulary_Coverage"] if c in sub]], width="stretch", height=320, hide_index=True)
    st.download_button("⬇️ Download predictions (CSV)", sub.to_csv(index=False).encode(), "flipkart_predictions.csv", "text/csv")

def page_analytics():
    head("📊 Data Analytics", "Exploratory analysis of the 30,000-review corpus, including the two data findings that shape every model result.")
    df = load_data()
    if df is None: return note("Dataset not found. Place <code>Flipkart_Reviews_Cleaned.csv</code> beside app.py.", "bad")
    cats = st.multiselect("Filter by category", sorted(df.Category.unique())); v = df[df.Category.isin(cats)] if cats else df
    k = st.columns(6)
    for c, (t, val, s) in zip(k, [("Reviews", f"{len(v):,}", "in view"), ("Positive", f"{(v.Sentiment == 'Positive').mean():.1%}", ""), ("Neutral", f"{(v.Sentiment == 'Neutral').mean():.1%}", ""), ("Negative", f"{(v.Sentiment == 'Negative').mean():.1%}", ""), ("Avg rating", f"{v.Rating.mean():.2f} ★", ""), ("Avg delivery", f"{v.Delivery_Days.mean():.1f} d", "")]):
        c.markdown(kpi(t, val, s), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["Overview", "⚠️ Leakage & template text", "Operational drivers"])
    with tabs[0]:
        a, b = st.columns(2); sc = v.Sentiment.value_counts().reindex(ORDER).reset_index(); sc.columns = ["Sentiment", "Count"]
        with a: st.markdown("#### Sentiment distribution"); show(px.pie(sc, values="Count", names="Sentiment", hole=0.55, color="Sentiment", color_discrete_map=COL), 300, True)
        with b:
            st.markdown("#### Average rating by category"); cr = v.groupby("Category").Rating.mean().sort_values().reset_index()
            fig = px.bar(cr, x="Rating", y="Category", orientation="h", color_discrete_sequence=["#4facfe"], text=cr.Rating.round(2)); fig.update_xaxes(range=[3.5, 4.1]); show(fig, 300)
        a, b = st.columns(2)
        with a: st.markdown("#### Review volume by category"); cc = v.Category.value_counts().reset_index(); cc.columns = ["Category", "Reviews"]; show(px.bar(cc, x="Reviews", y="Category", orientation="h", color_discrete_sequence=["#00f2fe"]).update_yaxes(autorange="reversed"), 300)
        with b: st.markdown("#### Top 10 brands"); bc = v.Brand.value_counts().head(10).reset_index(); bc.columns = ["Brand", "Reviews"]; show(px.bar(bc, x="Reviews", y="Brand", orientation="h", color_discrete_sequence=["#7928ca"]).update_yaxes(autorange="reversed"), 300)
        st.markdown("#### Average selling price by category (₹)"); pr = v.groupby("Category").Selling_Price.mean().sort_values().reset_index(); show(px.bar(pr, x="Selling_Price", y="Category", orientation="h", color_discrete_sequence=["#ff0080"]), 300)
    with tabs[1]:
        note("<b>Finding 1: label leakage.</b> Every rating maps to exactly one sentiment. <code>Rating</code> and <code>Sentiment_Score</code> are therefore excluded from all models.", "bad")
        ct = pd.crosstab(df.Rating, df.Sentiment)[ORDER]; fig = px.imshow(ct, text_auto=",", color_continuous_scale="Tealgrn", labels=dict(x="Sentiment", y="Rating", color="Reviews")); show(fig, 300)
        heads, tails = df.Review_Head.nunique(), df.Review_Tail.nunique(); uniq = df.Review_Text.nunique()
        note(f"<b>Finding 2: template-built text.</b> Only <b>{uniq:,}</b> unique review texts in {len(df):,} rows, made from <b>{heads}</b> sentiment phrases plus <b>{tails}</b> neutral aspect phrases. A random split therefore leaks near-identical text into the test set (95.4% of test reviews have a twin in training), which is why every model scores 1.000 on it.")
        hp = df.groupby(["Sentiment", "Review_Head"]).size().reset_index(name="Reviews").sort_values(["Sentiment", "Reviews"], ascending=[True, False])
        st.markdown("#### The 30 sentiment phrases (10 per class)"); st.dataframe(hp, width="stretch", hide_index=True, height=300)
    with tabs[2]:
        from scipy.stats import chi2_contingency
        a, b = st.columns(2)
        with a:
            st.markdown("#### Discount band vs sentiment"); d = v.assign(Band=pd.cut(v.Discount_Percentage, [-0.1, 0.01, 10, 20, 30, 60], labels=["0%", "0-10%", "10-20%", "20-30%", "30%+"]))
            dc = pd.crosstab(d.Band, d.Sentiment, normalize="index")[ORDER].reset_index().melt(id_vars="Band", var_name="Sentiment", value_name="Share")
            show(px.bar(dc, x="Band", y="Share", color="Sentiment", color_discrete_map=COL, barmode="stack"), 300, True)
            p = chi2_contingency(pd.crosstab(d.Band, d.Sentiment))[1]; st.caption(f"Chi-square p = {p:.2f}: {'no significant' if p > .05 else 'a significant'} relationship.")
        with b:
            st.markdown("#### Verified purchase vs sentiment"); vc = pd.crosstab(v.Verified_Purchase, v.Sentiment, normalize="index")[ORDER].reset_index().melt(id_vars="Verified_Purchase", var_name="Sentiment", value_name="Share")
            show(px.bar(vc, x="Verified_Purchase", y="Share", color="Sentiment", color_discrete_map=COL, barmode="group"), 300, True)
            p = chi2_contingency(pd.crosstab(v.Verified_Purchase, v.Sentiment))[1]; st.caption(f"Chi-square p = {p:.2f}: {'no significant' if p > .05 else 'a significant'} difference.")
        st.markdown("#### Delivery days vs customer experience score"); g = v.groupby("Delivery_Days").Customer_Experience_Score.agg(["mean", "count"]).reset_index()
        fig = px.scatter(g, x="Delivery_Days", y="mean", size="count", color_discrete_sequence=["#4facfe"], labels={"mean": "Mean experience score", "count": "Reviews"}, hover_data=["count"]); fig.update_yaxes(range=[1, 5]); show(fig, 300)
        st.caption("Marker size = number of reviews. The spikes at 11 and 12 days rest on only 14 and 4 reviews; the rest of the line is flat at about 3.0.")

def cm_fig(cm, scale):
    return px.imshow(np.array(cm), x=ORDER, y=ORDER, text_auto=",", color_continuous_scale=scale, labels=dict(x="Predicted", y="Actual", color="Count"))

def page_models():
    head("🤖 Model Comparison", "Six models, two tests. The random split is saturated, so the unseen-phrase hold-out is what ranks them.")
    com = pd.DataFrame(R.get("common", []))
    if com.empty:
        if os.path.exists("model_comparison.csv"): com = pd.read_csv("model_comparison.csv")
        else: return note("results.json / model_comparison.csv not found.", "bad")
    lat = {r["model"]: r for r in R.get("ml", []) + R.get("dl", [])}
    com["Latency (ms/review)"] = com.model.map(lambda m: lat.get(m, {}).get("ms_per_review", np.nan)); com["Train (s)"] = com.model.map(lambda m: lat.get(m, {}).get("train_sec", np.nan))
    com = com.rename(columns={"family": "Paradigm", "model": "Model", "accuracy": "Accuracy (random)", "precision": "Precision", "recall": "Recall", "macro_f1": "Macro-F1 (random)", "holdout_accuracy": "Accuracy (unseen)", "holdout_macro_f1": "Macro-F1 (unseen)"})
    com = com.sort_values("Macro-F1 (unseen)", ascending=False)
    st.dataframe(com.style.format({c: "{:.3f}" for c in com.columns if c not in ("Paradigm", "Model")}, na_rep="").highlight_max(subset=["Macro-F1 (unseen)"], color="#034d36").highlight_min(subset=["Latency (ms/review)"], color="#034d36"), width="stretch", hide_index=True)
    note("Random split: stratified 80/20 (6,000 test reviews), with 95% exact text overlap with training. Unseen-phrase: 9 of 30 sentiment phrases removed from training, mean of 5 random draws.", "warn")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Macro-F1: random split vs unseen phrases"); m = com.melt(id_vars="Model", value_vars=["Macro-F1 (random)", "Macro-F1 (unseen)"], var_name="Test", value_name="Macro-F1")
        fig = px.bar(m, x="Model", y="Macro-F1", color="Test", barmode="group", color_discrete_sequence=["#64748b", "#00f2fe"], text_auto=".2f"); fig.update_yaxes(range=[0, 1.1]); show(fig, 360, True)
    with b:
        st.markdown("#### Speed vs unseen-phrase generalization"); s = com.dropna(subset=["Latency (ms/review)"])
        fig = px.scatter(s, x="Latency (ms/review)", y="Macro-F1 (unseen)", color="Paradigm", text="Model", log_x=True, color_discrete_map={"Traditional ML": "#00f2fe", "Deep Learning": "#ff0080"}); fig.update_traces(textposition="top center", marker_size=14); fig.update_yaxes(range=[0.6, 1.0]); show(fig, 360, True)
    st.markdown("---"); st.markdown("### 🔲 Confusion matrices"); which = st.radio("Test set", ["Unseen-phrase hold-out (honest)", "Random 80/20 split (saturated)"], horizontal=True)
    key = "unseen" if which.startswith("Unseen") else "random_split"; blk = APP.get(key, {})
    if blk:
        st.caption(f"{blk['n_test']:,} test reviews" + (" · primary hold-out draw (seed 42)" if key == "unseen" else ""))
        a, b = st.columns(2)
        with a: st.markdown("#### Tuned Logistic Regression"); show(cm_fig(blk["lr"], "Blues"), 320)
        with b: st.markdown("#### LSTM"); show(cm_fig(blk["lstm"], "Purples"), 320)
    tuned = R.get("ml_tuned_holdout", {}); dl = next((r for r in R.get("dl", []) if r["model"] == "LSTM"), {}); fc = {r["model"].split(" ")[0]: r for r in R.get("final_compare", [])}
    html(f"""<div class='glass'><h4 style='margin-top:0;color:#00f2fe'>🎯 Selection rationale</h4><p style='color:#cbd5e1;font-size:.92rem;line-height:1.7'>On unseen phrases the LSTM scores <b>{dl.get('holdout_macro_f1', 0):.3f}</b> (±{dl.get('holdout_f1_std', 0):.3f}) and Logistic Regression <b>{tuned.get('macro_f1', 0):.3f}</b> (±{tuned.get('macro_f1_std', 0):.3f}). The gap is smaller than the run-to-run spread, so the two are <b>statistically tied</b>. Logistic Regression is the recommended model: it trains in {fc.get('Tuned', {}).get('train_sec', 0.1):.1f} s, saves at {fc.get('Tuned', {}).get('size_kb', 52):.0f} KB (LSTM: {dl.get('train_sec', 12):.0f} s, {fc.get('LSTM', {}).get('size_kb', 175):.0f} KB), and its word weights can be read directly. The LSTM's probabilities need calibration: it was 100% confident on an ambiguous mixed review.</p></div>""")

PROBES = [("Template positive", "Loved it", "Perfect purchase, loved it, battery life is decent.", "Positive"), ("Plain negation", "", "Not good at all, would not buy again.", "Negative"),
          ("New wording, positive", "", "Absolutely love this phone, the screen is gorgeous and it works flawlessly.", "Positive"), ("New wording, negative", "", "Terrible. It stopped charging within a week and support was useless.", "Negative"),
          ("Hedged neutral", "", "It's okay I guess, does the job but nothing to write home about.", "Neutral"), ("Mixed opinion", "Great product", "Very good quality but delivery was late and packaging was damaged.", "Mixed"),
          ("Sarcasm", "", "Brilliant phone if your goal is to own a paperweight.", "Negative"), ("Ultra-short", "", "fine", "Neutral")]

def page_errors():
    head("🧪 Errors & Features", "What the models learned, where they fail on unseen phrases, and how they behave on genuinely new wording.")
    if PIPE is None: return note("Model artifacts not found.", "bad")
    t1, t2, t3 = st.tabs(["⚡ Influential words", "🔎 Failing phrases (hold-out)", "🧨 Stress test"])
    with t1:
        st.caption("Largest tuned Logistic Regression weights per class (TF-IDF unigrams and bigrams).")
        names = np.array(TFIDF.get_feature_names_out()); cols = st.columns(3)
        for k, (c, s) in enumerate(zip(cols, ORDER)):
            top = np.argsort(CLF.coef_[k])[-10:]; d = pd.DataFrame({"term": names[top], "weight": CLF.coef_[k][top]})
            c.markdown(f"<p style='color:{COL[s]};font-weight:700'>Top → {s}</p>", unsafe_allow_html=True); c.plotly_chart(style(px.bar(d, x="weight", y="term", orientation="h", color_discrete_sequence=[COL[s]]), 340), width="stretch")
        st.caption("Words such as 'disappoint', 'poor' and 'excellent' carry the signal; neutral aspect words appear in every class and carry little.")
    with t2:
        ph = pd.DataFrame(APP.get("unseen", {}).get("phrases", []))
        if ph.empty: return st.info("app_data.json not found.")
        st.caption("Error rate on each of the 9 held-out sentiment phrases (primary draw). Neither model saw these phrases in training.")
        m = ph.melt(id_vars=["phrase", "true"], value_vars=["lr_error", "lstm_error"], var_name="Model", value_name="Error rate"); m["Model"] = m.Model.map({"lr_error": "Logistic Regression", "lstm_error": "LSTM"})
        fig = px.bar(m, x="Error rate", y="phrase", color="Model", barmode="group", orientation="h", color_discrete_sequence=["#00f2fe", "#ff0080"]); fig.update_xaxes(tickformat=".0%", range=[0, 1]); show(fig, 420, True)
        tbl = ph.assign(lr_error=(ph.lr_error * 100).round(1), lstm_error=(ph.lstm_error * 100).round(1)).rename(columns={"phrase": "Held-out phrase", "true": "True class", "n": "Reviews", "lr_error": "LR error %", "lstm_error": "LSTM error %", "lr_pred": "LR mostly predicts", "lstm_pred": "LSTM mostly predicts"})
        st.dataframe(tbl, width="stretch", hide_index=True)
        note("Errors concentrate on a few phrases, not on all nine. Logistic Regression mostly fails on the Neutral phrase 'quality is neither very good nor very bad', where the words 'very good' and 'very bad' pull it to Negative. The LSTM handles that phrase but misses some reviews of two Positive phrases. A random split hides all of this because those phrases would also appear in training.")
    with t3:
        st.caption("Hand-written reviews outside the dataset's templates. 'Expected' is our own judgement, not a dataset label.")
        rows = []
        for name, s, t, exp in PROBES:
            cl = clean_for_model(s, t); pl = PIPE.predict_proba([cl])[0]; row = {"Probe": name, "Review": (s + " | " if s else "") + t, "Expected": exp, "Coverage": f"{coverage(cl)[0]:.0%}", "LR": f"{ORDER[pl.argmax()]} ({pl.max():.0%})"}
            try: pd_ = predict_proba([cl], DL_NAME)[0]; row["LSTM"] = f"{ORDER[pd_.argmax()]} ({pd_.max():.0%})"
            except Exception: row["LSTM"] = "n/a"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        note("Read this table critically: confident answers on unfamiliar wording are not evidence of accuracy. Disagreement between the two models is a useful warning sign, and real Flipkart text would need retraining.")

def page_about():
    head("ℹ️ About This Project", "Scope, pipeline, deliverables and limits.")
    a, b = st.columns(2)
    a.markdown("""<div class='glass'><h3 style='margin-top:0;color:#00f2fe'>📋 What was built</h3><ul style='color:#cbd5e1;font-size:.92rem;line-height:1.8;padding-left:20px'>
<li><b>Phase 1:</b> data understanding and cleaning (nothing dropped, blanks filled)</li><li><b>Phase 2:</b> EDA: 11 analyses and a one-page dashboard</li><li><b>Phase 3:</b> NLP: combine, clean, tokenize, stop words (negations kept), stem vs lemmatize</li>
<li><b>Phase 4:</b> Naive Bayes, Logistic Regression, Linear SVM, tuned with phrase-grouped GridSearchCV</li><li><b>Phase 5:</b> Embedding+Dense, LSTM, BiLSTM, experiments, comparison and prediction pipeline</li></ul>
<h4 style='color:#4facfe'>Stack</h4>""" + "".join(f"<span class='tok'>{t}</span>" for t in ["Python", "pandas", "scikit-learn", "NLTK", "TensorFlow / Keras", "Plotly", "Streamlit"]) + "</div>", unsafe_allow_html=True)
    b.markdown("""<div class='glass'><h3 style='margin-top:0;color:#ff0055'>⚠️ Limits to keep in mind</h3><ul style='color:#cbd5e1;font-size:.92rem;line-height:1.8;padding-left:20px'>
<li>The data is <b>synthetic and template-built</b>; random-split scores of 100% reflect overlap, not skill.</li><li><code>Rating</code> and <code>Sentiment_Score</code> leak the label and are not inputs.</li>
<li>Brand and product pairings are inconsistent (an HP laptop sold by Wakefit), so metadata is for EDA only.</li><li>Sarcasm, mixed opinions and new wording are outside the training data.</li></ul>
<h3 style='color:#7928ca'>🔮 Next steps</h3><ul style='color:#cbd5e1;font-size:.92rem;line-height:1.8;padding-left:20px'><li>Train on real Flipkart reviews and repeat the unseen-phrase test</li><li>Pretrained embeddings or fine-tuned BERT</li><li>Aspect-level sentiment, SHAP explanations, confidence calibration</li></ul></div>""", unsafe_allow_html=True)
    files = ["Flipkart_Sentiment_Analysis.ipynb", "nlp_utils.py", "predict_sentiment.py", "app.py", "model_comparison.csv", "results.json", "app_data.json", "artifacts/", "figures/"]
    html("<div class='glass'><h4 style='margin-top:0;color:#00f2fe'>📦 Project files</h4>" + "".join(f"<span class='tok'>{'✔ ' if os.path.exists(f) else '✘ '}{f}</span>" for f in files) + "</div>")

PAGES = {"🏠 Home": page_home, "🔍 Sentiment Analyzer": page_analyzer, "📄 Batch Scoring": page_batch, "📊 Data Analytics": page_analytics, "🤖 Model Comparison": page_models, "🧪 Errors & Features": page_errors, "ℹ️ About Project": page_about}
PAGES[sidebar()]()
