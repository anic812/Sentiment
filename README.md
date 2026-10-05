# Flipkart Reviews Sentiment Analysis - project package

| File / folder | What it is |
|---|---|
| Flipkart_Sentiment_Analysis.ipynb | Executed notebook, all 50 questions (outputs included) |
| app.py | Streamlit dashboard (7 pages): `streamlit run app.py` |
| .streamlit/config.toml | Dark theme for the dashboard |
| make_app_data.py | Rebuilds app_data.json (confusion matrices, per-phrase errors) |
| app_data.json | Data the dashboard reads for confusion matrices and failing phrases |
| requirements.txt | Python packages for the notebook and the app |
| Flipkart_Sentiment_Presentation.pptx | 14-slide PowerPoint deck |
| nlp_utils.py | Shared text-cleaning functions (notebook and predictor both use them) |
| predict_sentiment.py | Predict a new review: `python predict_sentiment.py "Loved it" "Perfect purchase."` (add `--dl` for the LSTM) |
| figures/ | 18 PNG charts, including eda_dashboard.png |
| artifacts/ | best_ml_pipeline.joblib, tfidf_vectorizer.joblib, best_ml_classifier.joblib, best_dl_model.keras, dl_tokenizer.json, preprocessing.json |
| model_comparison.csv | Common comparison table (random split and unseen-phrase hold-out) |
| results.json | Every metric used in the report and slides |
| Flipkart_Reviews_Cleaned.csv | Cleaned dataset (Q8) |

Setup: pip install -r requirements.txt   (also: pip install scipy)
App:   streamlit run app.py   (run from this folder)
Run order: put Flipkart_Reviews_Sentiment_Analysis.csv beside the notebook, then Run All.
