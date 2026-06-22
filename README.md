# Dynamic ML Classification Service (Bloomberg Terminal Edition)

This workspace contains a plug-and-play classification platform. The FastAPI backend and React frontend are completely adaptive: they read the active model pipeline (`model.joblib`) and its schema descriptor (`metadata.json`) on startup. The UI dynamically builds forms, sliders, dropdowns, evaluation charts, and image overlays based on whatever dataset and model you train.

---

## 📁 Workspace Architecture

```text
├── backend/
│   ├── main.py                 # FastAPI Application (Dynamic Inference & Upload API)
│   ├── requirements.txt        # Backend dependencies (fastapi, scikit-learn, etc.)
│   └── artifacts/              # Model assets folder (Do not delete)
│       ├── model.joblib        # Active trained model pipeline
│       ├── metadata.json       # Active model schema descriptor
│       ├── train_chart.png     # Matplotlib training graph (Feature Importance)
│       └── test_chart.png      # Matplotlib testing graph (Confusion Matrix)
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Adaptive Bloomberg Terminal UI React Code
│   │   └── index.css           # Terminal color styles (Amber/Green/Black)
│   └── public/
│       └── images/             # Static prediction images (setosa.png, etc.)
├── model_development.ipynb     # Jupyter Notebook template for training models
└── README.md                   # This instruction manual
```

---

## 🚀 How to Change the Dataset & Features

To train a model on a new dataset (e.g. Titanic Survival, Heart Disease, Text Sentiment, Wine Quality):

1. **Open the Jupyter Notebook**:
   - Launch [model_development.ipynb](file:///e:/capstone%202/model_development.ipynb) inside your Jupyter server.
2. **Load your Data**:
   - In Step 1, replace the Iris dataset loading code with your dataset reader (e.g., `pd.read_csv("my_dataset.csv")`).
3. **Define Features**:
   - Select your feature columns ($X$) and your target classification column ($y$).
4. **Update Feature Schema**:
   - In Step 3, modify the `features` list in the `metadata` dictionary to declare your columns. Each feature needs:
     - `name`: Exact column header string.
     - `type`: Either `"numerical"`, `"categorical"`, or `"text"`.
     - `min` / `max` / `step` / `default` (For numericals): Restricts slider bounds in the web form.
     - `options` (For categoricals): Array of strings/numbers representing dropdown values.
     - `description`: Text helper shown below inputs.

---

## ⚙️ How to Change the Machine Learning Model

Because the backend passes features directly to `model.predict(X)`, you can use **any classifier** (Random Forest, SVM, XGBoost, Naive Bayes, Logistic Regression, etc.) as long as you wrap the preprocessors inside a **Scikit-Learn Pipeline**.

### 1. Wrapping Preprocessing in a Pipeline
Your model pipeline must handle scaling, encoding, or text vectorization internally. This ensures the backend can feed raw input forms directly into `pipeline.predict()` without crashing.

#### Tabular Pipeline (Standard Scaler + SVM Classifier)
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(probability=True, random_state=42))
])
pipeline.fit(X_train, y_train)
```

#### Text Pipeline (TF-IDF Vectorizer + Logistic Regression)
*If you are doing text classification, write your metadata feature list with exactly one feature of type `"text"`.*
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

### 2. Exporting Model & Schema
At the end of your notebook, serialize your trained pipeline and json schema:
```python
import joblib
import json

# Export pipeline binary
joblib.dump(pipeline, "backend/artifacts/model.joblib")

# Save schema configuration
with open("backend/artifacts/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## 📈 Generating Custom Training/Testing Graphs

The **Performance Charts** tab in the Bloomberg UI loads two static files served by the API:
- `train_chart.png` (Training performance)
- `test_chart.png` (Testing evaluation)

You can customize these plots in python before exporting. Keep a dark background style to maintain the Bloomberg aesthetic:

```python
import matplotlib.pyplot as plt

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(6, 4))

# Plot your logic (e.g. learning curve, ROC, confusion matrix, tree graph)
# Use Bloomberg colors: Amber (#ff9900) and Neon Green (#00ff00)
ax.plot(epochs, loss, color='#ff9900', label='Train Loss')
ax.tick_params(colors='#00ff00')

plt.tight_layout()
plt.savefig("backend/artifacts/train_chart.png", facecolor='#000000', dpi=100)
plt.close()
```

---

## 🖼️ Displaying Custom Prediction Images

When the model outputs a prediction (e.g. `"survived"`, `"spam"`, `"setosa"`), the frontend UI looks for a matching image file inside the public assets folder.

To assign custom photos to your predicted classes:
1. Generate or download a square PNG/JPG image representing each class label.
2. Convert the label string to lowercase (e.g., `"versicolor"` or `"survived"`).
3. Save the image inside:
   `frontend/public/images/{class_name}.png`
4. The web dashboard will automatically animate and pop up this image upon successful prediction!
