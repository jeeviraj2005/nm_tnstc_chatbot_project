import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pickle
import os

# Step 1: Prepare training data
training_data = {
    'query': [
        # Fare related queries
        'fare from madurai to chennai',
        'how much is the ticket from chennai to bangalore',
        'cost from trichy to salem',
        'ticket price from trichy to salem',

        # Timing related queries
        'first bus between coimbatore and salem',
        'last bus from madurai to chennai',
        'what time does the bus leave from chennai to salem',
        'when is the next bus from bangalore to hyderabad',

        # Night bus queries
        'are there night buses from bangalore to hyderabad',
        'is there a sleeper bus from madurai to chennai',
        'late night buses from trichy to chennai',
        'overnight buses from chennai to salem',

        # Travel time queries
        'travel time from chennai to trichy',
        'how long it takes from madurai to salem',
        'duration from coimbatore to trichy',
        'how many hours from bangalore to hyderabad',

        # General bus availability queries
        'show all buses from madurai to tirunelveli',
        'bus options from chennai to coimbatore',
        'available buses between salem and erode',
        'list buses from chennai to mumbai',

        # Route number queries (new)
        'what is the route number from chennai to mumbai',
        'tell me the bus route number between bangalore and hyderabad',
        'route number for bus from coimbatore to salem',
        'what is the routeno from chennai to salem',
    ],
    'label': [
        'fare_query', 'fare_query', 'fare_query', 'fare_query',
        'timing_query', 'timing_query', 'timing_query', 'timing_query',
        'night_bus_query', 'night_bus_query', 'night_bus_query', 'night_bus_query',
        'travel_time_query', 'travel_time_query', 'travel_time_query', 'travel_time_query',
        'general_bus_query', 'general_bus_query', 'general_bus_query', 'general_bus_query',
        'route_number_query', 'route_number_query', 'route_number_query','route_number_query'
    ]
}

# Step 2: Load into a dataframe
df = pd.DataFrame(training_data)

# Step 3: Feature Extraction (TF-IDF Vectorization)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['query'])
y = df['label']

# Step 4: Train the Model
model = SVC(kernel='linear', probability=True)
model.fit(X, y)

# Step 5: Save the model and vectorizer
os.makedirs('models', exist_ok=True)  # Create a folder called 'models' if it doesn't exist

with open('models/svc_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('models/vectorizer.pkl', 'wb') as vec_file:
    pickle.dump(vectorizer, vec_file)

print("✅ Model and Vectorizer saved inside 'models' folder successfully!")
