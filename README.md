# Placement Predictor using Logistic Regression

A Machine Learning project that predicts whether a student will be placed based on their **CGPA** and **IQ** using **Logistic Regression**.

## Project Overview

This project demonstrates a basic Machine Learning workflow, starting from data preprocessing and exploratory data analysis to model training, evaluation, and saving the trained model.

The dataset contains **100 student records** with the following features:

* **CGPA** – Student's academic performance
* **IQ** – Student's IQ score
* **Placement** – Target variable indicating whether the student was placed

  * `1` → Placed
  * `0` → Not Placed

## Machine Learning Workflow

The project follows these steps:

1. Data Preprocessing
2. Exploratory Data Analysis
3. Feature Selection
4. Input and Output Separation
5. Feature Scaling
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Decision Boundary Visualization
10. Model Serialization

## Model Used

### Logistic Regression

Logistic Regression is used because the target variable is binary:

```text
0 → Not Placed
1 → Placed
```

The model uses:

```text
Input Features:
- CGPA
- IQ

Output:
- Placement
```

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* mlxtend
* Pickle
* Jupyter Notebook

## Project Structure

```text
Placement-Predictor/
│
├── placementPredictor.ipynb
├── placement.csv
├── model.pkl
└── README.md
```

## Dataset

The dataset contains **100 records** and initially includes an unnamed index column, which is removed during preprocessing.

### Features

| Feature     | Description      |
| ----------- | ---------------- |
| `cgpa`      | Student CGPA     |
| `iq`        | Student IQ       |
| `placement` | Placement status |

## Data Preprocessing

The unnamed index column is removed:

```python
df = df.iloc[:, 1:]
```

The input and output variables are then separated:

```python
X = df.iloc[:, 0:2]
y = df.iloc[:, -1]
```

Where:

* `X` contains CGPA and IQ
* `y` contains the placement result

## Feature Scaling

`StandardScaler` is used to standardize the input features.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Train-Test Split

The dataset is divided into training and testing data using `train_test_split`.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1
)
```

This uses **90% of the data for training** and **10% for testing**.

## Model Training

A Logistic Regression classifier is trained on the scaled training data:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()

clf.fit(X_train, y_train)
```

Predictions are generated using:

```python
y_pred = clf.predict(X_test)
```

## Model Evaluation

The model is evaluated using the accuracy score:

```python
from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)
```

The notebook also visualizes the model's decision regions using `mlxtend`.

## Model Serialization

The trained model is saved using Python's `pickle` module:

```python
import pickle

pickle.dump(clf, open('model.pkl', 'wb'))
```

This allows the trained model to be saved and reused later.

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Install Required Libraries

```bash
pip install numpy pandas matplotlib scikit-learn mlxtend jupyter
```

### 3. Open the Notebook

```bash
jupyter notebook placementPredictor.ipynb
```

### 4. Run the Notebook

Run the notebook cells in order to preprocess the data, train the model, evaluate it, and generate the trained `model.pkl` file.

## Future Improvements

* Add more student features such as internships, projects, skills, and academic performance.
* Compare Logistic Regression with other classification algorithms.
* Improve model evaluation using precision, recall, F1-score, and confusion matrix.
* Build a web interface for making placement predictions.
* Deploy the trained model as a web application or API.
