LSTM Stock Price Prediction Model 🚀
====================================

A deep learning model using LSTM (Long Short-Term Memory) networks to predict stock prices with historical data. Currently configured for AAPL (Apple Inc.) stock data.

![image](https://github.com/user-attachments/assets/c8114ca6-1636-4a3f-8f8c-93bc5edde410)


Table of Contents
-----------------

*   [Project Overview](#project-overview)
    
*   [Project Timeline](#project-timeline)
    
*   [Installation](#installation)
    
*   [Dataset](#dataset)
    
*   [Model Architecture](#model-architecture)
    
*   [Mathematical Background](#mathematical-background)
    
*   [Implementation Details](#implementation-details)
    
*   [Usage Guide](#usage-guide)
    
*   [Results](#results)
    
*   [Future Improvements](#future-improvements)
    

Project Overview
----------------

This project of mine implements a deep learning approach to stock price prediction using LSTM networks. It analyzes 10 years of historical stock data to predict future stock prices. The implementation includes data preprocessing, model training, and evaluation phases.

Key Features:

*   LSTM-based neural network architecture
    
*   Rolling window prediction approach
    
*   Multiple technical indicators integration
    
*   Comparative analysis with moving averages
    
*   Interactive visualization of predictions

Installation
------------

```   
# Clone the repository
git clone https://github.com/yourusername/stock-prediction-lstm.git

# Create virtual environment
python -m venv venv  source venv/bin/activate

# On Windows: venv\Scripts\activate

# Install required packages  pip install -r requirements.txt   
```
### Requirements

```
pandas==1.3.0
numpy==1.21.0
tensorflow==2.6.0
keras==2.6.0
matplotlib==3.4.2
scikit-learn==0.24.2
statsmodels==0.12.2
```
Dataset
-------

The model uses historical AAPL stock data from 2012 to 2022, including:

*   Opening price
    
*   Closing price
    
*   High price
    
*   Low price
    
*   Volume
    
*   Adjusted closing price
    

### Data Preprocessing Steps

1.  Data cleaning and null value handling
    

```
df.replace("null", np.nan, inplace=True)
df.dropna(inplace=True)
```

2.  Feature scaling using MinMaxScaler
    

```
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
```

3.  Sequence creation with 60-day windows
    

```
x_train = []
y_train = []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
```

1.  Train-test split (80-20)
    

```
training_data_len = math.ceil(len(dataset) * 0.8)
```

Model Architecture
------------------

```
Model: Sequential
├── LSTM Layer 1
│   ├── Units: 50
│   ├── Return Sequences: True
│   └── Input Shape: (60, 1)
├── LSTM Layer 2
│   ├── Units: 50
│   └── Return Sequences: False
├── Dense Layer 1
│   └── Units: 25
└── Dense Layer 2
    └── Units: 1 (Output)
```

### Implementation Code

```
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
```

Mathematical Background
-----------------------

### LSTM Cell Components

1.  **Forget Gate**
    

```
ft = σ(Wf · [ht-1, xt] + bf)
```

Controls what information to discard from the cell state

1.  **Input Gate**
    

```
it = σ(Wi · [ht-1, xt] + bi)
ct = tanh(Wc · [ht-1, xt] + bc)
```

Decides what new information to store in the cell state

1.  **Cell State Update**
    

```
Ct = ft * Ct-1 + it * ct
```

Updates the long-term memory of the system

1.  **Output Gate**
    

```
ot = σ(Wo · [ht-1, xt] + bo)
ht = ot * tanh(Ct)
```

Controls what parts of the cell state are output

Implementation Details
----------------------

### Model Training

```
# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(x_train, y_train, batch_size=1, epochs=5)
```

### Moving Averages Implementation

```
df_10 = pd.DataFrame()
df_10['Close'] = df['Close'].rolling(window=10).mean()
df_20 = pd.DataFrame()
df_20['Close'] = df['Close'].rolling(window=20).mean()
df_30 = pd.DataFrame()
df_30['Close'] = df['Close'].rolling(window=30).mean()
df_40 = pd.DataFrame()
df_40['Close'] = df['Close'].rolling(window=40).mean()
```

### Prediction Generation

```
# Create the testing data set
test_data = scaled_data[training_data_len-60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
```

Usage Guide
-----------

1.  Prepare your data:
    

```
df = pd.read_csv('your_stock_data.csv')
df.set_index('Date', inplace=True)
```

1.  Run predictions:
    

```
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
```

1.  Visualize results:
    

```
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'])
plt.show()
```

Results
-------

### Performance Metrics

*   Mean Squared Error (MSE): 3.403
    
*   Root Mean Squared Error (RMSE): 1.844
    

### Visualization

The model's predictions closely track actual stock prices, with particularly strong performance during periods of low volatility.

Future Improvements
-------------------

1.  Technical Enhancement
    
    *   Implementation of additional technical indicators
        
    *   Integration of sentiment analysis
        
    *   Enhanced feature engineering
        
2.  Model Architecture
    
    *   Exploration of bidirectional LSTM
        
    *   Implementation of attention mechanisms
        
    *   Hybrid model architecture
        
3.  Practical Features
    
    *   Real-time data integration
        
    *   Multiple stock symbol support
        
    *   Web interface development
        
4.  Performance Optimization
    
    *   Hyperparameter optimization
        
    *   Advanced regularization techniques
        
    *   Ensemble methods implementation
        

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# Install required packages
pip install -r requirements.txt

# THANKS FOR READING 😄👍
