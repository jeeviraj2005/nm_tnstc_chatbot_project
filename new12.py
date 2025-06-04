from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
from fuzzywuzzy import process
import nltk
from nltk.tokenize import word_tokenize
import re

# Initialize Flask app
app = Flask(__name__, template_folder='template')

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained SVC model and TF-IDF vectorizer
model_path = os.path.join('models', 'svc_model.pkl')
vectorizer_path = os.path.join('models', 'vectorizer.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)

with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Load and preprocess the route data
data = pd.read_csv("cleaned_data_with_extra_columns.csv")
data['From'] = data['From'].str.lower().str.strip()
data['To'] = data['To'].str.lower().str.strip()
if 'Type' in data.columns:
    data['Type'] = data['Type'].str.lower().str.strip()
if 'Route No' in data.columns:
    data['Route No'] = data['Route No'].astype(str).str.strip()

# Prepare location list for fuzzy matching
unique_locations = pd.concat([data['From'], data['To']]).unique()
unique_locations_df = pd.DataFrame(unique_locations, columns=['Location'])

# Helper to clean user query
def clean_query(query):
    query = re.sub(r'[^\w\s]', '', query)
    return query.lower().strip()

# Smart keyword-based query type detection
def detect_query_type_by_keywords(query):
    query = clean_query(query)

    if 'route no' in query or 'routeno' in query or 'route number' in query:
        return 'route_number_query'
    elif 'fare' in query or 'cost' in query or 'price' in query:
        return 'fare_query'
    elif 'time' in query or 'timing' in query or 'departure' in query:
        return 'timing_query'
    elif 'night' in query or 'sleeper' in query:
        return 'night_bus_query'
    elif 'length' in query or 'distance' in query or 'km' in query:
        return 'travel_time_query'
    elif 'bus' in query or 'available' in query or 'type' in query:
        return 'general_bus_query'
    else:
        return None  # Fallback to ML model

# Extracts route info from user query
def find_location_info(query, query_type):
    query = clean_query(query)
    words = query.split()
    location_matches = {
        word: process.extractOne(word, unique_locations_df['Location'].values, score_cutoff=80)
        for word in words
    }

    to_index = words.index('to') if 'to' in words else -1
    from_location, to_location = None, None

    for word, match in location_matches.items():
        if match:
            matched_location = match[0].lower()
            if to_index != -1 and words.index(word) > to_index:
                to_location = matched_location
            elif to_index == -1 or words.index(word) < to_index:
                from_location = matched_location

    if not from_location or not to_location:
        return "Couldn't find complete route information in your query."

    result = data[(data['From'] == from_location) & (data['To'] == to_location)]

    if result.empty:
        return f"No route from {from_location.title()} to {to_location.title()} found."

    return format_result(result, query_type, from_location, to_location)

# Format response based on query type
def format_result(result, query_type, from_location=None, to_location=None):
    if query_type == 'fare_query':
        fare_info = result[['Type', 'Fare']].dropna()
        if fare_info.empty:
            return "Fare information is not available for this route."

        fare_pairs = fare_info.drop_duplicates().values.tolist()
        fare_strings = [f"{bus_type.title()}: {fare} Rs" for bus_type, fare in fare_pairs]
        return f"<strong>Fare Details:</strong> " + ', '.join(fare_strings)

    elif query_type == 'timing_query':
        timings = ', '.join(map(str, result['Departure Timings'].unique()))
        return f"<strong>Departure Timings:</strong> {timings}"

    elif query_type == 'night_bus_query':
        night_buses = result[result['Type'].str.contains('night|sleeper', case=False, na=False)]
        if not night_buses.empty:
            types = ', '.join(map(str, night_buses['Type'].unique()))
            return f"<strong>Night Bus Types Available:</strong> {types}"
        else:
            return "No night buses available for this route."

    elif query_type == 'travel_time_query':
        lengths = ', '.join(map(str, result['Route Length'].unique()))
        return f"<strong>Route Length:</strong> {lengths} km"

    elif query_type == 'general_bus_query':
        types = ', '.join(map(str, result['Type'].unique()))
        return f"<strong>Available Bus Types:</strong> {types}"

    elif query_type == 'route_number_query':
        if 'Route No' in result.columns:
            route_numbers = result['Route No'].dropna()
            route_numbers = route_numbers[route_numbers != '']
            if not route_numbers.empty:
                numbers_str = ', '.join(route_numbers.unique())
                return f"<strong>Route Numbers:</strong> {numbers_str}"
            else:
                return f"Route number is not listed for the route from {from_location.title()} to {to_location.title()}."
        else:
            return "Route number column is missing in the dataset."

    else:
        return "Couldn't process the query properly."

# Main route to handle user form
@app.route('/', methods=['GET', 'POST'])
def query_form():
    response = None
    user_query = ''
    if request.method == 'POST':
        user_query = request.form['query']

        # Try keyword-based detection first
        detected_label = detect_query_type_by_keywords(user_query)

        if detected_label:
            predicted_label = detected_label
        else:
            # Fallback to ML model
            transformed_query = vectorizer.transform([user_query])
            predicted_label = model.predict(transformed_query)[0]

        print(f"Predicted label: {predicted_label}")
        response = find_location_info(user_query, predicted_label)

    return render_template('form1.html', response=response, user_query=user_query)

if __name__ == "__main__":
    app.run(debug=True)
