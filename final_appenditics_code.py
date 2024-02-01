#Importing necessary libraries
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.pyplot import figure
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
warnings.filterwarnings("ignore")

#Reading the data from a CSV file and loading it into the dataframe
df = pd.read_csv('appendicitis.csv',encoding='windows-1254')
df.head(10)

#Setting the maximum number of rows to display in the output based on the length of the dataframe
pd.set_option('display.max_rows', len(df))

#Checking the size of the dataset
df.shape

#Summary of the dataset
df.info()

#Descriptive statistics of a dataset
df.describe()

#Checking for null values and retrieving only the columns that have null values
null_columns=df.columns[df.isnull().any()]
missing_values = df[null_columns].isnull().sum()
missing_values

#Horizontal Bar plot for missing values
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
plot = sns.barplot(x=missing_values.values, y=missing_values.index, palette='viridis')
plt.title('Missing Values in Each Column')
plt.xlabel('Number of Missing Values')
plt.ylabel('Columns')

# Display the values on the right of the bars with adjusted positions
for i, p in enumerate(plot.patches):
    plot.annotate(f'{p.get_width()}', (p.get_width(), p.get_y() + p.get_height() / 2.), ha='left', va='center', xytext=(5, 0), textcoords='offset points')

plt.tight_layout()  # Adjust layout to prevent clipping of labels
plt.show()

# Fill null values:
# - Numerical columns: Fill with the median
# - Categorical columns: Fill with the mode
df.replace('', np.nan, inplace=True)

for column in df.columns:
    if df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)
    else:
        df[column].fillna(df[column].median(), inplace=True)

#Checking for null values after filling the dataset
relation=df.copy()
null_columns=df.columns[df.isnull().any()]
missing_values = df[null_columns].isnull().sum()
len(missing_values)

#Appying data pre-processing to the dataset
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Applying fit transform to each column in DataFrame
df = df.apply(lambda col: label_encoder.fit_transform(col))
df.head(10)

# Calculate the correlation matrix
corr_matrix = df.corr()

# Print the list of features correlated to Appendicitis in descending order
appendicitis_correlation = corr_matrix["Appendicitis"].sort_values(ascending=False)
print(appendicitis_correlation.head(25))

# Creating a DataFrame with top 10 highly correlated features along with the target
selected_features = ['Appendicitis', 'Pain', 'RECOVD', 'Nausea', 'NUMDAYS', 'OTHER_MEDS', 'STATE', 'Lymphadenopathy', 'Vomiting', 'Pyrexia']
dataframe = df[selected_features]

# Building the model
# Extracting features (X) and target variable (y)
X = dataframe.drop('Appendicitis', axis=1)  # Exclude the target variable from features
y = dataframe['Appendicitis']  # Target variable

from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Displaying the shapes of the resulting sets
print(f'Shape of X_train: {X_train.shape}')
print(f'Shape of y_train: {y_train.shape}')
print(f'Shape of X_test: {X_test.shape}')
print(f'Shape of y_test: {y_test.shape}')

# Training the logistic regression model
from sklearn.linear_model import LogisticRegression

# Create a logistic regression model
lr = LogisticRegression()

# Fit the model on the training data
lr.fit(X_train, y_train)

# Training the Support Vector Machine (SVM) classifier
from sklearn.svm import SVC

# Create an SVM classifier
svm = SVC()

# Fit the model on the training data
svm.fit(X_train, y_train)

# Training the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Fit the model on the training data
rf.fit(X_train, y_train)

# Training the Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree classifier
dt = DecisionTreeClassifier()

# Fit the model on the training data
dt.fit(X_train, y_train)

# Training the Gaussian Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB

# Create a Gaussian Naive Bayes classifier
nb = GaussianNB()

# Fit the model on the training data
nb.fit(X_train, y_train)

# Plotting the Accuracy of the Models
# Set up a list of model names and instances
models = [
    ('Logistic Regression', lr),
    ('SVM', svm),
    ('Random Forest', rf),
    ('Decision Tree', dt),
    ('Naive Bayes', nb)
]

# Set up a list of colors for the plots
colors = ['b', 'g', 'r', 'c', 'm']

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(10, 5))

# Loop through each model and plot the accuracy
for i, (name, model) in enumerate(models):
    # Predict the target variable for the testing data
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = model.score(X_test, y_test) * 100

    # Plot the accuracy as a horizontal bar
    ax.barh(i, accuracy, color=colors[i], label=f'{name}: {accuracy:.2f}%')

# Set the y-axis labels and ticks
ax.set_yticks(np.arange(len(models)))
ax.set_yticklabels([name for name, _ in models])

# Set the x-axis label
ax.set_xlabel('Accuracy %')

# Set the title of the plot
ax.set_title('Model Accuracy')

# Move the legend outside the plot to avoid overlap
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Display the plot
plt.show()

# Get the predicted labels for each model
lr_predicted = lr.predict(X_test)
svm_predicted = svm.predict(X_test)
rf_predicted = rf.predict(X_test)
dt_predicted = dt.predict(X_test)
nb_predicted = nb.predict(X_test)

# Calculate false negatives for each model
false_negatives = [
    np.sum((lr_predicted == 0) & (y_test == 1)),
    np.sum((svm_predicted == 0) & (y_test == 1)),
    np.sum((rf_predicted == 0) & (y_test == 1)),
    np.sum((dt_predicted == 0) & (y_test == 1)),
    np.sum((nb_predicted == 0) & (y_test == 1))
]

# Create a bar chart of false negatives for each model
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(['Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree', 'Naive Bayes'], false_negatives)
ax.set_xlabel('Model')
ax.set_ylabel('False Negatives')
ax.set_title('Count of False Negatives for Each Model')
plt.show()

from sklearn.preprocessing import LabelEncoder

# Create a LabelEncoder object
le = LabelEncoder()

# Iterate through each column in the DataFrame
for column in df.columns:
    # Check if the column data type is object (categorical)
    if df[column].dtype == 'object':
        # Apply LabelEncoder to the column
        df[column] = le.fit_transform(df[column])

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Split the dataset into features and target
X = df.drop(columns=['Appendicitis'])
y = df['Appendicitis']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data for the CNN
X_train = X_train.values.reshape(-1, X_train.shape[1], 1)
X_test = X_test.values.reshape(-1, X_test.shape[1], 1)

# Define the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train the model and store training history
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions on the validation set
y_pred = model.predict(X_test)

# Plot accuracy and loss curves
plt.figure(figsize=(10, 5))

# Subplot for accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Subplot for loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Show the plot
plt.show()

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy*100:.2f}%')

cnn_score = test_accuracy*100

from sklearn.metrics import confusion_matrix

# Calculate the predicted values
y_pred = model.predict(X_test)

# Convert the predicted probabilities to classes
y_pred_classes = np.round(y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Calculate the false negative values
CNN_fn_values = cm[1][0]

# Plot the false negative values
plt.figure(figsize=(5,3))
plt.bar(['False Negatives'], [CNN_fn_values])
plt.title('False Negatives for CNN')
plt.show()

# Import necessary libraries for LSTM
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(units=64))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions on the validation set
y_pred = model.predict(X_test)

# Plot accuracy and loss curves
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy*100:.2f}%')

lstm_score = test_accuracy * 100

from sklearn.metrics import confusion_matrix

# Calculate the predicted values
y_pred = model.predict(X_test)

# Convert the predicted probabilities to classes
y_pred_classes = np.round(y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Calculate the false negative values
LTSM_fn_values = cm[1][0]

# Plot the false negative values
plt.figure(figsize=(5,3))
plt.bar(['False Negatives'], [LTSM_fn_values])
plt.title('False Negatives')
plt.show()

#importing necessary libraries for GRU
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.callbacks import EarlyStopping

# Define the GRU model
model = Sequential()
model.add(GRU(64, input_shape=(X_train.shape[1], 1), activation='tanh', dropout=0.5, recurrent_dropout=0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=3)

model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stop])

# Make predictions on the validation set
y_pred = model.predict(X_test)

# Plot accuracy and loss curves
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy*100:.2f}%')

gru_score = test_accuracy*100

from sklearn.metrics import confusion_matrix

# Calculate the predicted values
y_pred = model.predict(X_test)

# Convert the predicted probabilities to classes
y_pred_classes = np.round(y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Calculate the false negative values
GRU_fn_values = cm[1][0]

# Plot the false negative values
plt.figure(figsize=(5,3))
plt.bar(['False Negatives'], [GRU_fn_values])
plt.title('False Negatives')
plt.show()

# Create a figure with a size of 10 inches wide and 8 inches tall
fig, ax = plt.subplots(figsize=(10, 8))
models = ['CNN','LSTM','GRU']
scores = [cnn_score,lstm_score,gru_score]
plt.bar(models, scores)
plt.ylim(98, 100)
plt.ylabel('Validation Score (%)')
plt.title('Deep Learning Model Comparison')
plt.show()

# Investigating the relationship between COVID-19 and Appendicitis
unique_covid_values = relation['COVID.19'].unique()

# Plotting a scatter plot
plt.scatter(relation['COVID.19'], relation['Appendicitis'])
plt.xlabel('COVID-19')
plt.ylabel('Appendicitis')
plt.title('Correlation between COVID-19 and Appendicitis')
plt.show()

# Count the number of patients with Appendicitis and COVID-19
count = relation.groupby(['COVID.19', 'Appendicitis']).size().reset_index(name='Count')

# Set the figure size
fig, ax = plt.subplots(figsize=(6, 4))

# Plot the bar chart
ax.bar(x=count['COVID.19'].astype(str) + ', ' + count['Appendicitis'].astype(str),
        height=count['Count'])
ax.set_xlabel('COVID-19, Appendicitis', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Number of patients with COVID-19 and Appendicitis', fontsize=12)
for i, v in enumerate(count['Count']):
    ax.text(i, v, str(v), ha='center', va='bottom', fontsize=12)

plt.show()

from scipy.stats import chi2_contingency

# Create a contingency table
cont_table = pd.crosstab(relation['COVID.19'], relation['Appendicitis'])

# Perform chi-square test
stat, p, dof, expected = chi2_contingency(cont_table)
print(f"Chi-square statistic: {stat}")
print(f"P-value: {p}")

significant_columns = []
norelation_columns = []

# Iterate through columns
for col in df.columns:
    # Create a contingency table
    cont_table = pd.crosstab(relation[col], relation['Appendicitis'])

    # Perform chi-square test
    stat, p, dof, expected = chi2_contingency(cont_table)
    print(f"Chi-square statistic: {stat}")
    print(f"P-value: {p}")

    if p <= 0.05:
        print("Null Hypothesis is Rejected \nThere is a significant relationship between Appendicitis and " + str(col) + '\n')
        significant_columns.append(col)
    else:
        print("Alternative Hypothesis is Rejected \nThere is no relationship between Appendicitis and " + str(col) + '\n')
        norelation_columns.append(col)

    print("\n**********************************\n")

norelation_columns

significant_columns

from sklearn.metrics import confusion_matrix

# Create a bar chart of false negatives for each model
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(['Logistic Regression', 'SVM', 'Random Forest', 'Decision Tree', 'Naive Bayes', 'CNN', 'LSTM', 'GRU'],
       [false_negatives[0], false_negatives[1], false_negatives[2], false_negatives[3], false_negatives[4], CNN_fn_values, LTSM_fn_values, GRU_fn_values])
ax.set_xlabel('Models')
ax.set_ylabel('False Negatives')
ax.set_title('Count of False Negatives for Each Model')
plt.show()
