# Fault Injector

Based on DeepCrime we develop Fault Injector to systematically explore bug-space of Sequence-Based Models.

We modify originally collected models to work with Fault Injector by adding placeholder variables for each operator.

Models are trained multiple times to check the statistical significance of the results and metrics.

For classification based models we use accuracy for thresholding while for others we use loss.

# Installation and Setup

To make fault injector work following steps are followed:

1) Create a `model_name.py` file with all preprocessing and training code.
e.g. `LSTM1.py`
2) Wrap all code with a universal function as in LSTM1.py. The file should contain all operators as function arguments. You need to add all the operator parameters in model and data where relevant.

```python
def LSTM1(depth=5, bidirectional=False,
                       mutate_shape=False,
                        with_sequences=True,
                       initialization=False,
            backbone_activation="tanh",
             recurrent_activation="sigmoid",
             kernel_initializer='glorot_uniform',
              recurrent_initializer="orthogonal",
             recurrent_dropout=0.,
             dropout=0.,
             go_backwards=False,
             stateful=False,
             unroll=False,
             num_units=50,
             masking=False,
             kernel_regularizer=None,
    recurrent_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,):
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib.pyplot as plt

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM, Bidirectional
    from keras.layers import Dropout
    import tensorflow as tf

    #--------------------- Data Preprocessing --------------------#
    # Importing and scaling the data
    dataset_train = pd.read_csv("Bitcoin_Stock_Price_Trainset.csv")
    #selecting the right column (we need all rows and column 1) : numpy array
    training_set = dataset_train.iloc[:,1:2].values
    print(training_set)

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    #print(training_set_scaled[0,:])

    #creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []

    for i in range(90,training_set_scaled.size):
        # appending the 60 previous stock prices to the list for i
        # we need to specify the rows and simply pick the first and only column
        X_train.append(training_set_scaled[i-90:i, 0])
        # appending the 60th stock price to the list for i
        y_train.append(training_set_scaled[i, 0])
    # transforming pandas lists to numpy arrays required for the RNN
    X_train, y_train = np.array(X_train), np.array(y_train)
    #print(X_train)


    # Shaping/adding new dimensions to allow adding more indicators: from 2D to 3D
    # 3 input arguments
    # batch_size: number of observations
    # timesteps: number of columns
    # input_dim: number of predictors
    X_train = np.reshape(X_train, [X_train.shape[0], X_train.shape[1], 1])
    #print(X_train)


    #--------------------- Building RNN/LSTM model --------------------#
    #Initializing the RNN

    # independent variable is a sequence of layers: regression and no classification given the continuous output value
    regressor = Sequential()
    #------------------------------------------------------------------#
    # Adding the first LSTM layer
    #------------------------------------------------------------------#

    # 3 inputs
    # number of memory/LSTM units or neurons in each LSTM
    # binary vb to indicate whether there will be further layers of LSTM added to teh model
    # input shape (automatically takes teh first dimension so the reamining only needs to be specified)
    print(X_train.shape[1])
    regressor.add(LSTM(units = num_units, return_sequences=True, activation=backbone_activation,
                       recurrent_activation=recurrent_activation,
             kernel_initializer=kernel_initializer,
              recurrent_initializer=recurrent_initializer,
             recurrent_dropout=recurrent_dropout,
                       kernel_regularizer=kernel_regularizer,
    recurrent_regularizer=recurrent_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,
                       input_shape = (X_train.shape[1], 1)))

    # adding Dropout regularization layers
    # 1 input: amount of neurons to ignore in the layers
    regressor.add(Dropout(dropout))

    #------------------------------------------------------------------#
    # Adding the second LSTM layer
    #------------------------------------------------------------------#
    # no input shape needed given that that is specified in the previous layer
    for _ in range(depth-2):  # -2 for first and last layer
        if not bidirectional:
            regressor.add(LSTM(units = num_units, return_sequences=True, activation=backbone_activation,
                       recurrent_activation=recurrent_activation,
             kernel_initializer=kernel_initializer,
              recurrent_initializer=recurrent_initializer,
             recurrent_dropout=recurrent_dropout,
                       kernel_regularizer=kernel_regularizer,
    recurrent_regularizer=recurrent_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer,))
        else:
            regressor.add(Bidirectional(LSTM(units = num_units, return_sequences=True, activation=backbone_activation,
                       recurrent_activation=recurrent_activation,
             kernel_initializer=kernel_initializer,
              recurrent_initializer=recurrent_initializer,
             recurrent_dropout=recurrent_dropout,
                       kernel_regularizer=kernel_regularizer,
    recurrent_regularizer=recurrent_regularizer,
    bias_regularizer=bias_regularizer,
    activity_regularizer=activity_regularizer)))
        regressor.add(Dropout(dropout))

    # #------------------------------------------------------------------#
    # # Adding the third LSTM layer
    # #------------------------------------------------------------------#
    # regressor.add(LSTM(units = 50, return_sequences=True))
    # regressor.add(Dropout(0.2))

    # #------------------------------------------------------------------#
    # # Adding the forth LSTM layer
    # #------------------------------------------------------------------#
    # regressor.add(LSTM(units = 50, return_sequences=True))
    # regressor.add(Dropout(0.2))

    # #------------------------------------------------------------------#
    # # Adding the fifth LSTM layer
    # #------------------------------------------------------------------#
    # note that this is the final LSTM layer, hence we change the binary argument to False
    regressor.add(LSTM(units = num_units, return_sequences=False, activation=backbone_activation,
                       recurrent_activation=recurrent_activation,
             kernel_initializer=kernel_initializer,
              recurrent_initializer=recurrent_initializer,
             recurrent_dropout=recurrent_dropout,))
    regressor.add(Dropout(dropout))

    #------------------------------------------------------------------#
    # Adding output layer to the RNN to make a fully connected NN
    #------------------------------------------------------------------#
    # one dimensional real output
    regressor.add(Dense(units = 1))


    #--------------------- Compiling the RNN model --------------------#
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    #--------------------- Training RNN model --------------------#
    #connecting the built regressor to the training model
    history = regressor.fit(X_train, y_train, epochs = 5, batch_size = 32, validation_split=0.1)

    return regressor, history
```
3) Add imports for the universal closed function such as LSTM1 in `model.py` such as follows. You also need to update `MODEL_TYPES` dictionary

```
from LSTM1 import LSTM1

MODEL_TYPES = {
                "LSTM1": LSTM1}
```
Viola! the model will work with all operators 
4) Run `script.py` with the operator name and and values.
`python script.py --model 'LSTM1' --operator 'recurrent_activation' --logs_dir './'`

To run the operator multiple times we use this shell script:
```sh
for i in {1..3}; do
    echo "Running iteration $i..."
    python script.py --model 'GRU7' --operator 'recurrent_activation' --logs_dir '/content/drive/MyDrive/models_faults/'
    echo "------------------------------"  # Visual separator for clarity
done
```
5) To perform statistical analysis and killablity check use code in `thresholding.py`. In `thresholding.py`, you can change the criteria. E.g. for regression models use `loss` instead of accuracy. Compare with `val_accuracy` and find killability score.
