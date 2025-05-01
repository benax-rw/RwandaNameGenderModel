# RwandaNameGenderModel

**RwandaNameGenderModel** is a machine learning model that predicts gender based solely on Rwandan first names. It uses a character-level n-gram approach with a logistic regression classifier to provide fast, interpretable, and highly accurate predictions — achieving **96%+ accuracy** on both validation and test sets.

---

## 🧠 Model Overview

- **Type:** Classic ML (Logistic Regression)
- **Vectorization:** Character-level n-grams (2–3 chars)
- **Framework:** scikit-learn
- **Training Set:** 66,735 names (out of 83,419)
- **Validation/Test Accuracy:** ~96.6%

---

## 📁 Project Structure

```
RwandaNameGenderModel/
├── dataset/
│   └── rwandan_names.csv
├── model/
│   ├── logistic_model.joblib
│   └── vectorizer.joblib
├── logs/
│   └── metrics_log.txt
├── train.py
├── inference.py
├── README.md
└── requirements.txt
```

---

## 🚀 Quickstart

### 1. Install requirements
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py
```

### 3. Predict gender from a name
```python
from joblib import load

model = load("model/logistic_model.joblib")
vectorizer = load("model/vectorizer.joblib")

def predict_gender(name):
    X = vectorizer.transform([name])
    return model.predict(X)[0]

predict_gender("Mugisha")  # Output: "male" or "female"
```

---

## 📈 Performance

| Dataset    | Accuracy | Precision | Recall | F1-Score |
|------------|----------|-----------|--------|----------|
| Validation | 96.72%   | 96.90%    | 96.53% | 96.72%   |
| Test       | 96.64%   | 96.94%    | 96.34% | 96.64%   |

Metrics are logged in both `logs/metrics_log.txt` and TensorBoard format.

---

## 🌍 Use Cases

- Demographic analysis
- Smart form processing
- Voice assistant personalization
- NLP preprocessing for Rwandan corpora

---

## 🛡️ Ethical Note

This model predicts binary gender based on patterns in names and may not reflect self-identified gender. It should not be used in sensitive contexts without consent.

---

## 📄 License

This project is maintained by [Gabriel Baziramwabo](https://benax.rw) and is open for research and educational use. For commercial use, please contact the author.

---

## 🤝 Contributing

We welcome improvements and multilingual extensions. Fork this repo, improve, and submit a PR!

---

## 🔗 Links

- [Benax Technologies](https://benax.rw)

