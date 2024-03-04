# Dylan Porter, Dante Osbourne
# AI 570 Class Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.regularizers import l1, l2
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read in both datasets
climate_change_df = pd.read_csv('./data/climate_change_data.csv')
yield_df = pd.read_csv('./data/yield_df.csv')

# Convert the Date in the climate_change_data.csv to just the year for data merging
climate_change_df['Date'] = pd.to_datetime(climate_change_df['Date'], errors='coerce')
climate_change_df['Year'] = climate_change_df['Date'].dt.year

# Perform the data merge
# Need to average the climate change dataset because it is more granular by location when compared to the other datset.
climate_change_df_average = climate_change_df.groupby(['Country', 'Year']).mean(numeric_only=True).reset_index()
merged_df = pd.merge(climate_change_df_average, yield_df, left_on=['Country', 'Year'], right_on=['Area', 'Year'])

# Remove duplicate rows
merged_df = merged_df.drop_duplicates()

# Remove unnecessary columns
merged_df = merged_df.drop(['Unnamed: 0', 'Area'], axis=1)

# Remove outliers (3 standard deviations) (May want to consider keeping outliers later)
# Find all columns that are numeric
numeric_cols = merged_df.select_dtypes(include=[np.number])

# Find all rows that don't have outliers
df_no_outliers = numeric_cols[(np.abs(stats.zscore(numeric_cols)) < 3).all(axis=1)]

# Remove all rows that contained outliers
cleaned_df = merged_df.loc[df_no_outliers.index]

# This will agreggate the data to find the average hg/ha_yield off of country and year. We essentially remove the granularity of item yield
average_yield_df = cleaned_df.groupby(['Country', 'Year']).agg({
    'hg/ha_yield': 'mean',
    'Temperature': 'mean',
    'CO2 Emissions': 'mean',
    'Sea Level Rise': 'mean',
    'Precipitation': 'mean',
    'Humidity': 'mean',
    'Wind Speed': 'mean',
    'average_rain_fall_mm_per_year': 'mean',
    'pesticides_tonnes': 'mean',
    'avg_temp': 'mean'
}).reset_index()

# One-hot encode country
X = pd.get_dummies(average_yield_df, columns=['Country'])

# X is all the fields that predict hg/ha_yield
X = X.drop('hg/ha_yield', axis=1)
Y = average_yield_df['hg/ha_yield']

# Standardize the dataset to feed into the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2)

# Build a basic regression model (adjust hyperparams later)
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))

model.add(Dense(32, activation='relu', kernel_regularizer=l1(0.1)))
model.add(Dropout(0.5))

model.add(Dense(32, activation='relu', kernel_regularizer=l1(0.1)))
model.add(Dropout(0.5))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=1500, batch_size=16)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progression')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()