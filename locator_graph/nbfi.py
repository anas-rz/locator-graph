import numpy as np
from sklearn.model_selection import train_test_split
from keras import layers, Sequential, losses, metrics
features = np.load('features_inspector_features.npy')
targets = np.load('features_inspector_targets.npy')
features_train, features_test, targets_train, targets_test = train_test_split(features,
                                                                              targets, test_size=0.2, random_state=42)
features_train, features_val, targets_train, targets_val = train_test_split(features_train, targets_train, test_size=0.2, random_state=42)


model = Sequential()
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(11))

model.compile(optimizer='adam',
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy', 'f1_score', metrics.AUC(from_logits=True)])

model.fit(features_train, targets_train, epochs=5, validation_data=(features_val, targets_val))
model.evaluate(features_test, targets_test)
