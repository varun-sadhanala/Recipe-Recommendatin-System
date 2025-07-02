# 🍽️ Recipe Recommendation System

This **Recipe Recommendation System** is a Streamlit-based web application that helps users discover recipes based on the ingredients they have or by searching with tags like "vegetarian", "dinner", etc. The system uses NLP techniques and cosine similarity to find and recommend recipes from a cleaned dataset.

---

## 🔍 Features

- 🔎 **Search by Ingredients** – Enter a list of ingredients you have to get recipe recommendations.
- 🏷️ **Search by Tags** – Find recipes based on tags such as "quick", "breakfast", or "low-carb".
- 📊 **Nutritional Information** – View detailed nutrition facts (calories, fat, protein, sugar, etc.).
- 🍽️ **Step-by-Step Instructions** – Get a complete breakdown of recipe preparation steps.
- ⚡ **Efficient Matching** – Uses `CountVectorizer` and `cosine similarity` for fast and accurate suggestions.

---

## 📁 Dataset

The project uses a preprocessed recipe dataset named `Cleaned_Recipes.csv` which includes:

- `ingredients`
- `tags`
- `steps`
- `nutrition` (calories, fat, sugar, etc.)
- Recipe `name`, `id`, and `cooking time`

Make sure to place the dataset in the correct path or use the upload option in the app.

---

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy** – Data manipulation
- **Scikit-learn** – Vectorization and similarity computation
- **Streamlit** – Web application framework
- **AST** – Safe evaluation of ingredient and nutrition lists

---

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/recipe-recommendation-app.git
   cd recipe-recommendation-app
