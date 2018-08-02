# Multilayer Perceptron to Predict Time series data  (t+1, given t), Jorge Richard/2018.
import numpy
# To plot the charts
import matplotlib.pyplot as plt
# Pandas for the data sets
import pandas
import math
# Keras module for deep learning
from keras.models import Sequential
from keras.layers import Dense

# Fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset. No matter the values on column one, usecols=[1] to get values from second column.
dataframe = pandas.read_csv('library.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values

# Chage data type for training
dataset = dataset.astype('float32')

# Define size train and test sets; 2/3 for training, 1/3 for test/validation
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size

# Create the data sets
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

# Convert an array of values into a dataset matrix. Look back considers "n" periods before to predict next cycle
def create_dataset(dataset, look_back=1):
    # Define variables
	dataX, dataY = [], []
    # Check for all values in dataset
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0] # Get each a_ieth elemnt so it can be appended on each iteration
		dataX.append(a) # Add to data on time "t"
		dataY.append(dataset[i + look_back, 0]) # Add to data on time "t+1"
	return numpy.array(dataX), numpy.array(dataY) # Get the final array

# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Create and fit a sequential multilayer perceptron model
model = Sequential()

# Add 8 layers and the activation function as relu
model.add(Dense(8, input_dim=look_back, activation='relu'))
model.add(Dense(1))

# Evaluate the MSE
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the ntwork. Consider minimum 200 epochs.
model.fit(trainX, trainY, epochs=200, batch_size=2, verbose=2)

# Evaluate the model by using the train set
trainScore = model.evaluate(trainX, trainY, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore, math.sqrt(trainScore)))
testScore = model.evaluate(testX, testY, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore, math.sqrt(testScore)))

# Generate predictions for training
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot = trainPredict

# Shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2):len(dataset)-2, :] = testPredict

# Plot baseline and predictions
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
