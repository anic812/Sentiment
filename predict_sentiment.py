"""Standalone Flipkart review sentiment predictor.
Usage:  python predict_sentiment.py "Loved it" "Perfect purchase, works perfectly."   (summary, text)
        python predict_sentiment.py                                                     (interactive)
Requires: nlp_utils.py, artifacts/best_ml_pipeline.joblib (+ artifacts/best_dl_model.keras & dl_tokenizer.json for --dl)
"""
import sys, json, joblib, numpy as np
import nlp_utils as nu
ORDER = ["Negative", "Neutral", "Positive"]
TEXT_COL = json.load(open("artifacts/preprocessing.json"))["text_col"]
_ml = joblib.load("artifacts/best_ml_pipeline.joblib")
_dl = _vec = None

def _load_dl():
    global _dl, _vec
    import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from tensorflow import keras
    from keras import layers
    tok = json.load(open("artifacts/dl_tokenizer.json"))
    _vec = layers.TextVectorization(max_tokens=len(tok["vocabulary"]), output_sequence_length=tok["maxlen"], standardize=None, split="whitespace")
    _vec.set_vocabulary(tok["vocabulary"])
    _dl = keras.models.load_model("artifacts/best_dl_model.keras")

def predict(review_text, review_summary="", engine="ml"):
    txt = nu.build_variant(nu.combine(review_summary, review_text), TEXT_COL)
    if engine == "ml":
        p = _ml.predict_proba([txt])[0]
    else:
        if _dl is None: _load_dl()
        import tensorflow as tf
        p = _dl.predict(_vec(tf.constant([txt])).numpy(), verbose=0)[0]
    k = int(np.argmax(p))
    return ORDER[k], float(p[k]), {ORDER[i]: round(float(p[i]), 3) for i in range(3)}

if __name__ == "__main__":
    engine = "dl" if "--dl" in sys.argv else "ml"; args = [a for a in sys.argv[1:] if a != "--dl"]
    if len(args) == 2: print(predict(args[1], args[0], engine))
    elif len(args) == 1: print(predict(args[0], "", engine))
    else:
        while True:
            t = input("Enter a Flipkart review (blank to quit): ").strip()
            if not t: break
            print(predict(t, "", engine))
