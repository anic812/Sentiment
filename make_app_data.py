"""Rebuilds app_data.json (confusion matrices + per-phrase errors for the Streamlit app).
Uses the same splits as the notebook: random split seed 42, and unseen-phrase hold-out draw 42 (3 phrases per class).
Usage: python make_app_data.py [output.json]     (LSTM retraining makes the LSTM numbers vary slightly run to run)"""
import os, sys, json, copy, numpy as np, pandas as pd, joblib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import nlp_utils as nu, tensorflow as tf
from tensorflow import keras
from keras import layers
O = ["Negative", "Neutral", "Positive"]; L = {l: i for i, l in enumerate(O)}
df = pd.read_csv("Flipkart_Reviews_Cleaned.csv"); y = df.Sentiment.map(L).values
tc = json.load(open("artifacts/preprocessing.json"))["text_col"]
comb = df.Review_Summary.fillna("").astype(str) + "|" + df.Review_Text
cache = {}
def prep(c, s, t):
    if c not in cache: cache[c] = nu.build_variant(nu.combine(s, t), tc)
    return cache[c]
X = pd.Series([prep(c, s, t) for c, s, t in zip(comb, df.Review_Summary.fillna(""), df.Review_Text)])
tr, te = train_test_split(np.arange(len(df)), test_size=.2, stratify=y, random_state=42)
ml = joblib.load("artifacts/best_ml_pipeline.joblib")
cm_lr_rand = confusion_matrix(y[te], ml.predict(X.iloc[te])).tolist()
head_lab = df.groupby("Review_Head").Sentiment.first(); rng = np.random.RandomState(42); held = []
for s in O: held += list(rng.choice(sorted(head_lab[head_lab == s].index), 3, replace=False))
m = df.Review_Head.isin(held).values; a, b = np.where(~m)[0], np.where(m)[0]
p_lr = copy.deepcopy(ml).fit(X.iloc[a], y[a]).predict(X.iloc[b])
tf.keras.utils.set_random_seed(42); VOC, ML = 118, 10
vec = layers.TextVectorization(max_tokens=VOC, output_sequence_length=ML, standardize=None, split="whitespace"); vec.adapt(list(X.iloc[a]))
sq = lambda idx: vec(tf.constant(list(X.iloc[idx]))).numpy()
i_tr, i_va = train_test_split(np.arange(len(a)), test_size=.2, stratify=y[a], random_state=42)
inp = keras.Input(shape=(ML,), dtype="int32"); x = layers.Embedding(VOC, 32, mask_zero=True)(inp); x = layers.LSTM(32)(x); x = layers.Dropout(.3)(x)
mdl = keras.Model(inp, layers.Dense(3, activation="softmax")(x)); mdl.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
mdl.fit(sq(a)[i_tr], y[a][i_tr], validation_data=(sq(a)[i_va], y[a][i_va]), epochs=6, batch_size=128, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)], verbose=0)
p_ls = mdl.predict(sq(b), verbose=0).argmax(1)
rows = []
for h in held:
    k = df.Review_Head.values[b] == h
    rows.append(dict(phrase=h, true=head_lab[h], n=int(k.sum()), lr_error=float((p_lr[k] != y[b][k]).mean()), lstm_error=float((p_ls[k] != y[b][k]).mean()),
                     lr_pred=O[int(np.bincount(p_lr[k], minlength=3).argmax())], lstm_pred=O[int(np.bincount(p_ls[k], minlength=3).argmax())]))
out = dict(order=O, random_split=dict(n_test=len(te), lr=cm_lr_rand, lstm=np.diag(np.bincount(y[te], minlength=3)).tolist()),
           unseen=dict(n_test=len(b), lr=confusion_matrix(y[b], p_lr).tolist(), lstm=confusion_matrix(y[b], p_ls).tolist(), draw=42, phrases=rows))
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "app_data.json", "w"), indent=1); print("written")
