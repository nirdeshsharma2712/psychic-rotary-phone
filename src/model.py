import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from constants import entity_unit_map, ALLOWED_UNITS

# Define conversion factors (from constants.py)
conversion_factors = {
    'centimetre': 0.01, 'metre': 1, 'foot': 0.3048, 'inch': 0.0254, 'millimetre': 0.001, 'yard': 0.9144,
    'gram': 0.001, 'kilogram': 1, 'microgram': 1e-9, 'milligram': 1e-6, 'ounce': 0.0283495, 'pound': 0.453592, 'ton': 1000,
    'kilovolt': 1000, 'millivolt': 0.001, 'volt': 1,
    'kilowatt': 1000, 'watt': 1,
    'centilitre': 0.01, 'litre': 1, 'millilitre': 0.001, 'microlitre': 1e-6, 'gallon': 3.78541,
    'pint': 0.473176, 'quart': 0.946353, 'cup': 0.236588, 'cubic foot': 0.0283168, 'cubic inch': 1.63871e-5, 'decilitre': 0.1,
    'fluid ounce': 0.0295735, 'imperial gallon': 4.54609
}

# Extract value and unit from a given string
def extract_value_and_unit(entity_value):
    pattern = r'([0-9]+\.?[0-9]*)\s*([a-zA-Z ]+)'  # Regex to extract number and unit
    match = re.match(pattern, entity_value)
    if match:
        value = float(match.group(1))
        unit = match.group(2).strip().lower()
        return value, unit
    return None, None

# Convert units to the most appropriate standard unit
def convert_to_standard_unit(entity, value, unit):
    if unit in conversion_factors:
        return value * conversion_factors[unit]
    return value

# Convert the standardized value to the most appropriate unit
def convert_to_preferred_unit(entity, value):
    units = entity_unit_map.get(entity, [])
    # Define thresholds for unit conversion (adjust based on domain knowledge)
    unit_thresholds = {
        'width': {'metre': 1, 'centimetre': 100, 'foot': 3.281, 'inch': 39.37},
        'depth': {'metre': 1, 'centimetre': 100, 'foot': 3.281, 'inch': 39.37},
        'height': {'metre': 1, 'centimetre': 100, 'foot': 3.281, 'inch': 39.37},
        'item_weight': {'kilogram': 1, 'gram': 1000, 'pound': 2.205, 'ounce': 35.274, 'ton': 0.001},
        'maximum_weight_recommendation': {'kilogram': 1, 'gram': 1000, 'pound': 2.205, 'ounce': 35.274, 'ton': 0.001},
        'voltage': {'volt': 1, 'kilovolt': 0.001, 'millivolt': 1000},
        'wattage': {'watt': 1, 'kilowatt': 0.001},
        'item_volume': {'litre': 1, 'centilitre': 100, 'millilitre': 1000, 'cup': 4.227, 'gallon': 0.264}
    }
    
    if entity in unit_thresholds:
        thresholds = unit_thresholds[entity]
        for unit, threshold in thresholds.items():
            if value >= threshold:
                return value / threshold, unit
    return value, next(iter(units))  # Fallback to the first unit if no appropriate unit found

# Preprocess the training data
def preprocess_train_data(df):
    df['standard_value'] = None  # Create new column for standardized values
    
    for index, row in df.iterrows():
        entity = row['entity_name']
        entity_value = row['entity_value']
        
        # Extract value and unit
        value, unit = extract_value_and_unit(entity_value)
        
        # Convert to standard unit if applicable
        if unit in ALLOWED_UNITS:
            standardized_value = convert_to_standard_unit(entity, value, unit)
            df.at[index, 'standard_value'] = standardized_value
    
    return df

# Load train.csv
train_df = pd.read_csv('dataset/train.csv')
train_df = preprocess_train_data(train_df)

# Handle missing values, drop NA rows if any
train_df = train_df.dropna(subset=['standard_value'])

# Feature selection - use 'entity_name' as a categorical feature
X = pd.get_dummies(train_df[['entity_name']])
y = train_df['standard_value']

# Split the data for training and validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Evaluate on validation set
val_predictions = model.predict(X_val)

# Print validation accuracy (just for testing purposes)
print("Validation set predictions:", val_predictions)

#### Step 2: Predicting on test.csv

# Load test.csv
test_df = pd.read_csv('dataset/test.csv')

# Prepare test data (similar to train data preprocessing)
X_test = pd.get_dummies(test_df[['entity_name']])  # Use entity_name for prediction

# Align columns between train and test
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Predict on test.csv
test_df['predicted_entity_value'] = model.predict(X_test)

# Map each entity to its unit and format predictions
def format_prediction(row):
    entity = row['entity_name']
    value = row['predicted_entity_value']
    # Convert to the most appropriate unit
    standardized_value, unit = convert_to_preferred_unit(entity, value)
    if unit:
        return f"{standardized_value:.2f} {unit}"
    return ""

# Apply the formatting function
test_df['formatted_predicted_entity_value'] = test_df.apply(format_prediction, axis=1)

# Create final output DataFrame with only index and formatted predictions
output_df = test_df[['index', 'formatted_predicted_entity_value']]
output_df.columns = ['index', 'prediction']  # Rename columns to match the required format

# Save the output to a CSV file
output_df.to_csv('dataset/sample_test_out.csv', index=False)
