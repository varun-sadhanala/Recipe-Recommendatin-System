import streamlit as st
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(layout="wide", page_title="Recipe Recommendation System")

# Title centered on the page
st.markdown("<h1 style='text-align: center;'>Recipe Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Find recipes based on ingredients you have!</h3>", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    file_path = r'C:\Users\varun\Downloads\recipe\Cleaned_Recipes.csv'
    df = pd.read_csv(file_path)
    # Include the nutrition column along with other columns``
    df = df[['ingredients', 'name', 'id', 'minutes', 'tags', 'steps', 'nutrition']]
    df.dropna(inplace=True)
    
    # Parse and process ingredients and tags
    df['ingredients_list'] = df['ingredients'].apply(lambda x: ast.literal_eval(x))
    df['ingredients_list'] = df['ingredients_list'].apply(lambda x: [ingredient.replace(' ', '') for ingredient in x])
    df['ingredients_str'] = df['ingredients_list'].apply(lambda x: ' '.join(x))
    
    # Process tags for search
    df['tags_list'] = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Parse nutrition data
    df['nutrition_parsed'] = df['nutrition'].apply(parse_nutrition)
    
    return df

# Function to parse nutrition data
def parse_nutrition(nutrition_str):
    try:
        # Parse the string representation of list to actual list
        nutrition_list = ast.literal_eval(nutrition_str)
        
        # Create a dictionary with labeled nutrition values
        nutrition_dict = {
            'Calories': round(nutrition_list[0], 1),
            'Total Fat': round(nutrition_list[1], 1),
            'Sugar': round(nutrition_list[2], 1),
            'Sodium': round(nutrition_list[3], 1),
            'Protein': round(nutrition_list[4], 1),
            'Saturated Fat': round(nutrition_list[5], 1),
            'Carbohydrates': round(nutrition_list[6], 1)
        }
        return nutrition_dict
    except:
        # Return empty dict if parsing fails
        return {}

# Create vectorizer
@st.cache_resource
def create_vectorizer(df):
    cv = CountVectorizer(max_features=20000, stop_words='english')
    vectors = cv.fit_transform(df['ingredients_str'])
    return cv, vectors

# Function to suggest recipes based on ingredients
def suggest_recipes(user_ingredients, cv, vectors, df):
    input_ingredients_str = ' '.join([ingredient.replace(' ', '') for ingredient in user_ingredients])
    input_vector = cv.transform([input_ingredients_str])
    # Compute cosine similarity between the input vector and recipe vectors
    similarity_scores = cosine_similarity(input_vector, vectors).flatten()
    df_temp = df.copy()
    df_temp['similarity'] = similarity_scores
    top_recipes = df_temp.nlargest(10, 'similarity')[['name', 'minutes', 'tags', 'steps', 'ingredients_list', 'nutrition_parsed', 'similarity']]
    return top_recipes

# Function to search recipes by tags
def search_recipes_by_tags(tag_query, df):
    # Convert tag query to lowercase for case-insensitive matching
    tag_query_lower = tag_query.lower()
    
    # Filter recipes containing the tag
    matching_recipes = []
    
    for _, recipe in df.iterrows():
        tags = recipe['tags_list']
        # Check if any tag contains the query string
        if any(tag_query_lower in tag.lower() for tag in tags):
            matching_recipes.append(recipe)
        
        # Limit to top 10 results
        if len(matching_recipes) >= 10:
            break
    
    # Convert to DataFrame
    if matching_recipes:
        return pd.DataFrame(matching_recipes)[['name', 'minutes', 'tags', 'steps', 'ingredients_list', 'nutrition_parsed']]
    else:
        return pd.DataFrame()

# Display recipe card
def display_recipe(recipe, col):
    with col:
        st.markdown(f"### {recipe['name']}")
        st.markdown(f"**Cooking Time:** {recipe['minutes']} minutes")
        
        # Format tags
        tags = ast.literal_eval(recipe['tags']) if isinstance(recipe['tags'], str) else recipe['tags']
        st.markdown("**Tags:** " + ", ".join(tags[:5]) + ("..." if len(tags) > 5 else ""))
        
        # Format ingredients
        st.markdown("**Ingredients:**")
        ingredients = recipe['ingredients_list']
        for i, ingredient in enumerate(ingredients[:5]):
            st.markdown(f"- {ingredient}")
        if len(ingredients) > 5:
            st.markdown("- ...")
        
        # Display nutrition information
        if 'nutrition_parsed' in recipe and recipe['nutrition_parsed']:
            with st.expander("Nutrition Information"):
                nutrition = recipe['nutrition_parsed']
                
                # Create two columns for nutrition display
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**Calories:** {nutrition.get('Calories', 'N/A')}")
                    st.markdown(f"**Total Fat:** {nutrition.get('Total Fat', 'N/A')}g")
                    st.markdown(f"**Saturated Fat:** {nutrition.get('Saturated Fat', 'N/A')}g")
                    st.markdown(f"**Carbohydrates:** {nutrition.get('Carbohydrates', 'N/A')}g")
                
                with col2:
                    st.markdown(f"**Protein:** {nutrition.get('Protein', 'N/A')}g")
                    st.markdown(f"**Sugar:** {nutrition.get('Sugar', 'N/A')}g")
                    st.markdown(f"**Sodium:** {nutrition.get('Sodium', 'N/A')}mg")
        
        # Format steps - showing ALL steps
        st.markdown("**Steps:**")
        steps = ast.literal_eval(recipe['steps']) if isinstance(recipe['steps'], str) else recipe['steps']
        for i, step in enumerate(steps):
            st.markdown(f"{i+1}. {step}")
        
        # Show similarity score if present
        if 'similarity' in recipe:
            st.progress(min(recipe['similarity'], 1.0))
            st.markdown(f"Match score: {recipe['similarity']:.2f}")

# Main function
def main():
    try:
        # Load data
        with st.spinner("Loading recipe database..."):
            df = load_data()
            cv, vectors = create_vectorizer(df)
        
        # Create tabs for different search methods
        tab1, tab2 = st.tabs(["Search by Ingredients", "Search by Tags"])
        
        with tab1:
            # User input section for ingredients
            st.markdown("## Enter Your Ingredients")
            st.markdown("Enter ingredients separated by commas (e.g., potato, butter, garlic)")
            
            user_input = st.text_input("", placeholder="Enter ingredients here...", key="ingredients_input")
            
            if st.button("Find Recipes", key="ingredients_button"):
                if user_input:
                    # Process user input
                    user_ingredients = [ingredient.strip() for ingredient in user_input.split(',')]
                    
                    with st.spinner("Finding the best recipes for you..."):
                        # Get recipe recommendations
                        suggested_recipes = suggest_recipes(user_ingredients, cv, vectors, df)
                    
                    # Display results
                    st.markdown("## Recommended Recipes")
                    st.markdown(f"Based on your ingredients: **{', '.join(user_ingredients)}**")
                    
                    # Check if we found any recipes
                    if suggested_recipes.empty:
                        st.error("No recipes found with these ingredients. Try adding more common ingredients.")
                    else:
                        # Display recipes in a grid (2 columns)
                        for i in range(0, len(suggested_recipes), 2):
                            cols = st.columns(2)
                            # First recipe in the row
                            display_recipe(suggested_recipes.iloc[i], cols[0])
                            
                            # Second recipe in the row (if it exists)
                            if i + 1 < len(suggested_recipes):
                                display_recipe(suggested_recipes.iloc[i+1], cols[1])
                else:
                    st.warning("Please enter at least one ingredient")
        
        with tab2:
            # User input section for tags
            st.markdown("## Search by Tags")
            st.markdown("Enter a tag to find matching recipes (e.g., dinner, vegetarian, quick)")
            
            tag_input = st.text_input("", placeholder="Enter tag here...", key="tag_input")
            
            if st.button("Search Tags", key="tag_button"):
                if tag_input:
                    with st.spinner("Searching for recipes with matching tags..."):
                        # Get recipes with matching tags
                        tag_recipes = search_recipes_by_tags(tag_input, df)
                    
                    # Display results
                    st.markdown("## Recipes with Matching Tags")
                    st.markdown(f"Tag searched: **{tag_input}**")
                    
                    # Check if we found any recipes
                    if tag_recipes.empty:
                        st.error(f"No recipes found with tag '{tag_input}'. Try a different tag.")
                    else:
                        # Display recipes in a grid (2 columns)
                        for i in range(0, len(tag_recipes), 2):
                            cols = st.columns(2)
                            # First recipe in the row
                            display_recipe(tag_recipes.iloc[i], cols[0])
                            
                            # Second recipe in the row (if it exists)
                            if i + 1 < len(tag_recipes):
                                display_recipe(tag_recipes.iloc[i+1], cols[1])
                else:
                    st.warning("Please enter a tag to search")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please check if the recipe dataset is accessible at the specified path.")

if __name__ == "__main__":
    main()