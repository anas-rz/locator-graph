import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

def main(args):
    # Load the dataset
    df = pd.read_csv(args.input_file)

    # Convert categorical features to numerical values using Label Encoding
    label_encoders = {}
    for column in ['node_type', 'activation', 'recurrent_activation',
                   'kernel_regularizer', 'recurrent_regularizer',
                   'bias_regularizer', 'activity_regularizer', 'buggy_feature']:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Separate features and target
    X = df.drop(columns=['buggy_feature'])
    y = df['buggy_feature']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the MLP model
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    # Train the model
    mlp.fit(X_train, y_train)

    # Predict the buggy features on the test set
    y_pred = mlp.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoders['buggy_feature'].classes_)

    print(f'Accuracy: {accuracy:.2f}')
    print('Classification Report:')
    print(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train an MLP classifier on a dataset.')
    parser.add_argument('input_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()
    main(args)
