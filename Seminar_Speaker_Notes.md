# Seminar speaker notes: Flipkart Reviews Sentiment Analysis

Total speaking time about 13 minutes across 14 slides, including a 90-second live demo. Timings are per slide.

## Slide 1: Cover  (0:30)

Good morning, everyone. My project is a sentiment analysis system for Flipkart-style product reviews. It takes a review written by a customer and decides whether it is Positive, Neutral or Negative. Over the next ten minutes I will show you the data, how I cleaned and prepared the text, the six models I compared, and a live app. One thing to listen for: my most important result is not a high score. It is why a perfect score here would be misleading.

## Slide 2: Business problem and objective  (0:45)

Why does this matter? A big platform receives far more reviews than any team can read. If a model can label each review automatically, the company can track product perception, delivery experience and service quality at scale. The dataset has 30,000 reviews and 25 columns, and the target has three classes. The mix is uneven: about 62 percent Positive, 20 percent Neutral and 18 percent Negative. Because the classes are uneven, plain accuracy can flatter a model, so my main metric is macro-F1, which gives each class equal weight.

## Slide 3: Dataset: 30,000 reviews, 25 columns  (0:45)

This chart shows the class balance: 18,671 Positive, 6,019 Neutral and 5,310 Negative reviews. On data quality, only three columns had missing values: review summary, customer city and payment method, with 240, 300 and 150 blanks. I filled them with an empty string or 'Unknown' and dropped nothing. There were no duplicate review IDs, no true duplicate rows, and no invalid prices, ratings or dates. So the table itself is clean. The interesting problems are in what the columns mean, and that is the next two slides.

## Slide 4: Finding 1: Rating gives away the label  (1:00)

This is the first key finding. Look at the heatmap on the left: every rating maps to exactly one sentiment. Ratings 1 and 2 are always Negative, 3 is always Neutral, 4 and 5 are always Positive. The sentiment score on the right separates the classes with no overlap at all. That is called target leakage. If I fed Rating into a model, it would score 100 percent without reading a single word. So I removed Rating and Sentiment_Score from every model. All models use the review text only. A real project should always check for this before training.

## Slide 5: Finding 2: the text is template-built  (1:00)

The second finding is about the text. Out of 30,000 reviews there are only 960 unique review texts. Each one is a sentiment phrase, like 'perfect purchase, loved it', plus a neutral aspect phrase, like 'size was accurate'. There are just 30 sentiment phrases, ten per class, and 32 aspect phrases. The consequence is big: if I split the data randomly, 95.4 percent of test reviews have an exact twin in the training set. The model can memorize instead of learn. That is why every model reaches 100 percent on a random split, and why I built a harder test, which I will show on slide 9.

## Slide 6: Exploratory analysis at a glance  (0:45)

This dashboard summarizes the exploration. Categories are balanced, between about 3,600 and 3,800 reviews each. Televisions and laptops are the most expensive, books the cheapest. I also tested whether business factors affect sentiment. Discount has no significant effect, with p equal to 0.10. Verified purchase makes no difference either, p equal to 0.72. Delivery days show no link to the experience score; the spikes at 11 and 12 days come from only 14 and 4 reviews. So in this dataset, the text is the only real signal, which justifies a text-based model.

## Slide 7: NLP pipeline  (0:45)

Here is how raw text becomes model input. Step one: combine the review summary and review text. Step two: lowercase, strip symbols and tokenize with NLTK. Step three: remove stop words, but I deliberately keep 'no', 'nor' and 'not', because dropping them would turn 'not good' into 'good'. Step four: normalize words, comparing stemming and lemmatization. Step five: convert to numbers, using TF-IDF for the classical models and integer sequences for the neural networks. The same functions run in the notebook and in the live app, so training and prediction never drift apart.

## Slide 8: Choosing the text representation  (0:45)

I did not pick the preprocessing by habit; I tested four versions with Logistic Regression. Cleaned only scored 0.846 macro-F1, stop-word removal 0.825, stemming 0.921 and lemmatization 0.921. Notice that stop-word removal alone was worst, because it can delete negations. Stemming and lemmatization tied, so I chose lemmatization, since its words stay readable and are easier to explain. One caution: on a random split all four scored 1.000, so only the harder test could separate them.

## Slide 9: Evaluation design  (1:00)

This slide explains how I measured the models fairly. The first test is the standard stratified 80/20 split, 24,000 reviews to train and 6,000 to test, as the assignment requires. But we saw it is saturated. So I added a second test, the unseen-phrase hold-out. I take 3 of the 10 sentiment phrases in each class, 9 of 30 in total, remove them from training completely, and test on them. Now the model must understand wording, not recognize a template. I repeat this with 5 random choices of held-out phrases and report the average and spread.

## Slide 10: Six models, one fair test  (1:15)

Here are all six models. The grey bars are the random split: every model gets 1.00. The blue bars are the honest test, and now they separate. Naive Bayes is weakest at 0.70, then Linear SVM at 0.81, Embedding plus Dense at 0.83, BiLSTM at 0.88, Logistic Regression at 0.93 and LSTM at 0.96. The LSTM looks best, but the run-to-run spread is about 0.037, which is as large as its lead over Logistic Regression. So they are statistically tied. Logistic Regression trains in 0.1 seconds and saves at 52 kilobytes, against 12 seconds and 175 kilobytes for the LSTM. That is why I recommend Logistic Regression.

## Slide 11: Tuned Logistic Regression  (0:50)

For the recommended model I tuned settings with GridSearchCV, using folds that keep whole phrases together so tuning is also honest. The best setting was C equal to 1 with unigrams plus bigrams. The chart shows the words with the biggest weights: 'disappoint', 'not', 'poor' and 'defective' push toward Negative; 'average', 'okay' and 'decent' toward Neutral; 'excellent', 'love' and 'superb' toward Positive. I also tested imbalance fixes. Class weighting scored 0.853 and oversampling 0.815, both worse than doing nothing at 0.930, so I did not use them.

## Slide 12: Deep learning: LSTM and BiLSTM  (0:50)

On the deep learning side I built an Embedding plus Dense network, an LSTM and a BiLSTM in Keras, with a vocabulary of 118 tokens and a maximum length of 10. The training curves on the left converge within a few epochs, with no gap between training and validation. That looks perfect, but it is actually a warning: the validation set shares templates with the training set, so it cannot reveal overfitting. On unseen phrases the scores were 0.834, 0.959 and 0.881. The simplest recurrent model did best, which is reasonable for such short reviews.

## Slide 13: Live demo: Streamlit app  (1:30)

I built a Streamlit dashboard with seven pages. In the analyzer you type a review, choose Logistic Regression, the LSTM or both, and get a label, confidence and the words that drove the decision. Now the table: clear reviews get correct answers from both models. The highlighted mixed review, 'great quality but late and damaged', is the interesting one. Logistic Regression says Negative at 63 percent; the LSTM says Positive at 100 percent. Both cannot be right, and there may be no single right label. So confidence is not accuracy, and the app warns when the two models disagree. [DEMO: open the app, click the Mixed review example, choose Compare both, press Analyze.]

## Slide 14: Conclusions and next steps  (1:00)

To conclude. One: Rating and Sentiment_Score leak the label and must stay out of the features. Two: a random-split score of 100 percent reflects template overlap, not skill; the honest range is 0.70 to 0.96. Three: Logistic Regression and the LSTM are tied on unseen phrases, and I recommend Logistic Regression for speed, size and interpretability. The limits are clear: the data is synthetic and template-built, so real reviews with sarcasm or mixed opinions will be harder. Next steps are to train on real Flipkart reviews, try pretrained embeddings or BERT, add aspect-level sentiment and calibrate the confidence. Thank you; I am happy to take questions.

## Likely questions and short answers

**Q: Why is accuracy 100% and why do you not trust it?**

Because the dataset has only 960 unique texts and 95.4% of test reviews have an exact twin in training. The model can memorize. I added a test with 9 of 30 sentiment phrases removed from training; there scores range from 0.70 to 0.96.

**Q: Why did you drop Rating? It is the strongest feature.**

Exactly because it is too strong. Sentiment is a direct function of Rating (1-2 Negative, 3 Neutral, 4-5 Positive), so using it is target leakage. A model using it would not learn language and would be useless for reviews that have no rating.

**Q: Why macro-F1 and not accuracy?**

The classes are uneven (62/20/18). Accuracy can look good by favouring Positive. Macro-F1 averages the F1 of each class equally, so Neutral and Negative count as much as Positive.

**Q: Which model is best and why do you recommend Logistic Regression?**

On unseen phrases the LSTM scores 0.959 and Logistic Regression 0.930, but the run-to-run spread is about 0.037, so they are tied. Logistic Regression trains in 0.1 s, is 52 KB, and its word weights can be read directly. The LSTM was also 100% confident on an ambiguous review.

**Q: Why lemmatization instead of stemming?**

They tied at 0.921 on the unseen-phrase test. Lemmas are real words, so explanations are readable.

**Q: Why keep 'not' when removing stop words?**

Standard stop-word lists delete negations, which can flip meaning ('not good' becomes 'good'). Stop-word removal alone scored lowest (0.825).

**Q: Did you handle class imbalance?**

I tested class weighting (0.853) and random oversampling (0.815) against no correction (0.930) on unseen phrases. Both were worse, so I used none.

**Q: Will this work on real Flipkart reviews?**

Not as is. The data is synthetic and template-built, so wording outside the 30 sentiment phrases, sarcasm and mixed opinions are much harder. The next step is retraining on real reviews and repeating the unseen-phrase test.

**Q: What was the hardest case for the models?**

Mixed and unfamiliar wording. On the unseen-phrase test, Logistic Regression failed on about 61% of reviews of the Neutral phrase 'quality is neither very good nor very bad'. The sarcastic review 'Brilliant phone if your goal is to own a paperweight' was labelled Positive by both models.

**Q: How would you improve it?**

Real data, pretrained embeddings or a fine-tuned BERT, aspect-level sentiment so mixed reviews can be split by feature, confidence calibration, and SHAP or LIME explanations.

## One-minute version (if you are short on time)

I built a sentiment classifier for 30,000 Flipkart-style reviews. Two data problems shaped everything: Rating gives away the label, and the text is built from only 30 sentiment phrases, so a random split scores 100%. I removed Rating, and I tested models on phrases they never saw. There, Logistic Regression scored 0.93 and an LSTM 0.96, a statistical tie, so I recommend Logistic Regression because it is small, fast and explainable. I also built a Streamlit dashboard. The main limit is that the data is synthetic, so real reviews would be harder.