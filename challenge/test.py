import pandas as pd
from model import DelayModel
from sklearn.metrics import confusion_matrix, classification_report

# Initialize the DelayModel class
model = DelayModel()
model._model_path = './../data/xgboost_model.pkl'

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('./../data/data.csv')  # Replace with the actual CSV file path
# Step 1: Preprocess the data
# If you're training the model, specify the target column ('delay' in this case)
# For training
features, target = model.preprocess(data, target_column='delay')

xt, xv, yt, yv = model.split_data(features, target)

# Step 2: Fit the model
model.fit(xt, yt)

# Step 3: Load the pre-trained model
model.load_model(model._model_path)

# Step 4: Make predictions on the new data
predictions = model.predict(xv)

# Display the predictions
print(confusion_matrix(yv, predictions))