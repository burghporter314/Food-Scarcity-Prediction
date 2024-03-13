# Dylan Porter, Dante Osbourne
# AI 570 Class Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, InputLayer
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l1, l2
from keras.callbacks import EarlyStopping
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from kerastuner import RandomSearch

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

# One-hot encode country
X = pd.get_dummies(cleaned_df, columns=['Country'])

X = pd.get_dummies(X, columns=['Item'])

# X is all the fields that predict hg/ha_yield
X = X.drop('hg/ha_yield', axis=1)
Y = cleaned_df['hg/ha_yield']

# Standardize the dataset to feed into the neural network
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2)

# Find the optimal hyperparameters for the problem
# Found via deepnotes and chatgpt for supplementary usage
def build_model(hp):
    model = Sequential()
    model.add(InputLayer(input_shape=(X_train.shape[1],)))

    # Create a variable number of layers between 2 and 5 along with an associated Dropout layer
    for i in range(hp.Int('num_layers', 2, 5)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                        activation='relu'))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0, max_value=0.5, step=0.1)))

    # Regression so just one unit needed
    model.add(Dense(1))

    # Adjust the learning rate dynamically, go off of val_loss for mean_squared_error
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                  loss='mean_squared_error')
    return model

# Find the optimal hyperparameters via keras tuner
tuner = RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=160,
    executions_per_trial=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

tuner.search(X_train, Y_train, validation_split=0.2, epochs=20, batch_size=32)
best_model = tuner.get_best_models(num_models=1)[0]

# Best model based on keras tuner
model = Sequential()

model.add(Dense(384, activation='relu', input_shape=(X_train.shape[1],)))

model.add(Dense(448, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu', kernel_regularizer=l1(0.1)))
model.add(Dropout(0.1))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=100, batch_size=32)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progression')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()