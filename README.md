# 🚢 Titanic Survival Analysis

Exploratory data analysis on the classic Titanic dataset using Python, paired with an interactive Power BI dashboard to visualize survival patterns across passenger demographics.

---

## 📌 Objective

To explore what factors most influenced survival during the Titanic disaster, using statistical analysis and clear data visualizations to communicate findings effectively.

---

## ✨ Features

- 🐍 **Python EDA** — Data cleaning, missing value handling, feature analysis
- 📊 **Survival Breakdown** — By gender, passenger class, age group, and embarkation point
- 📈 **Power BI Dashboard** — Interactive filters to explore survival rates across dimensions
- 🔍 **Correlation Analysis** — Identifying strongest predictors of survival

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Analysis | Python (Pandas, Matplotlib, Seaborn) |
| Visualization | Power BI |
| Dataset | [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data) |

---

## 📂 Project Structure

```
Titanic-Survival-Analysis/
│
├── analysis/
│   └── titanic_eda.ipynb        # Python EDA notebook
├── dashboard/
│   └── TitanicDashboard.pbix    # Power BI dashboard
├── data/
│   └── titanic.csv              # Source dataset
└── README.md
```

---

## 🔑 Key Insights

- **Women survived at ~74%** vs ~19% for men — gender was the strongest survival factor
- **1st class passengers** had a 63% survival rate vs 24% in 3rd class
- **Children under 10** had significantly higher survival rates across all classes
- ~19% of the dataset had missing age values — handled via median imputation by class

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Sanjay1318/Titanic-Survival-Analysis.git

# Install dependencies
pip install pandas matplotlib seaborn jupyter

# Launch the notebook
jupyter notebook analysis/titanic_eda.ipynb
```

---

## 👤 Author

**Vadla Sanjay Kumar**  
[LinkedIn](https://www.linkedin.com/in/sanjaychari007/) · [GitHub](https://github.com/Sanjay1318)
