

# Make functions for each component of every model
# i.e. max pooling, dense layer, etc. Refer to tensorflow documentation to do this
# Transformers, GANs, Dense, Linear Regression, etc.

# Come back through and optimize each model for GPU or not
# Graph results separately - pulling from recordings


import pandas as pd
import tensorflow as tf
from hmmlearn import hmm
from sklearn.svm import SVC
from keras import Sequential
from keras.src.layers import Add
from keras.src.layers import GRU
from xgboost import XGBClassifier
from keras.src.layers import ReLU
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from keras.src.layers import Conv2D
from lightgbm import LGBMClassifier
from keras.src.layers import Dropout
from keras.src.layers import Reshape
from sklearn.metrics import r2_score
from keras.src.metrics import MeanIoU
from keras.src.layers import SimpleRNN
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from keras.src.layers import Embedding
from keras.src.layers import Activation
from pgmpy.models import BayesianNetwork
from keras.src.layers import UpSampling2D
from keras.src.layers import MaxPooling2D
from keras.src.utils import to_categorical
from transformers import TFGPT2LMHeadModel
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import adjusted_rand_score
from pgmpy.inference import VariableElimination
from sklearn.tree import DecisionTreeClassifier
from keras.src.losses import BinaryCrossentropy
from sklearn.linear_model import LinearRegression
from keras.src.applications.resnet import ResNet50
from keras.src.applications.resnet import ResNet101
from keras.src.applications.resnet import ResNet152
from sklearn.linear_model import LogisticRegression
from keras.src.layers import GlobalAveragePooling1D
from keras.src.layers import GlobalAveragePooling2D
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from keras.src.applications import efficientnet, efficientnet_v2
from sklearn.feature_extraction.text import TfidfTransformer
from keras.src.applications.resnet_v2 import ResNet101V2
from keras.src.applications.resnet_v2 import ResNet152V2
from keras.src.applications.resnet_v2 import ResNet50V2
from pgmpy.estimators import MaximumLikelihoodEstimator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from keras.src.layers import LayerNormalization
from keras.src.layers import MultiHeadAttention
from keras.src.layers import BatchNormalization
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from keras.src.callbacks import EarlyStopping
from transformers import ViTFeatureExtractor
from keras.src.layers import Conv2DTranspose
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from keras.src.layers import concatenate
from keras.src.layers import Concatenate
from catboost import CatBoostClassifier
from keras.src.layers import InputLayer
from keras.src.layers import LeakyReLU
from transformers import GPT2Tokenizer
from catboost import CatBoostRegressor
from transformers import BertTokenizer
from transformers import GPT2Tokenizer
from keras.src.optimizers import Adam
from sklearn.metrics import f1_score
from transformers import TFBertModel
from keras.src.layers import Flatten
from keras.src.layers import Permute
from lightgbm import LGBMClassifier
from keras.src.layers import Lambda
from lightgbm import LGBMRegressor
from keras.src.layers import Input
from keras.src.layers import Dense
from transformers import GPT2Model
from transformers import ViTModel
from keras.src.layers import LSTM
from keras.src.layers import Dot
from xgboost import XGBRegressor
from keras import backend as K
from sklearn.svm import SVR
#from keras import models
from keras import Model
import numpy as np


from helper import is_list_of_tuples_int_and_tuple_of_ints
from helper import is_list_of_tuples_matching_types
from helper import is_list_of_tuples_matching_types
from helper import is_list_of_tuples_ints_or_floats
from helper import is_list_of_np_arrays
from helper import is_list_of_integers
#from helper import is_tf_data_dataset
from helper import is_list_optional
from helper import is_pd_dataframe
from helper import is_tuple_of_ints
from helper import is_2d_np_array
from helper import is_int_or_none
from helper import is_int_or_auto
from helper import is_np_array
from helper import is_integer
from helper import is_string
from helper import is_tuple
from helper import is_float
from helper import is_list
from helper import is_dict
from helper import is_bool



# Tensorflow Dataset
def is_tf_data_dataset(var):
    return isinstance(var, tf.data.Dataset)


#from keras._tf_keras.keras.preprocessing import sequence

# 1.
def linear_regression(data, target_column, test_size=0.2, random_state=42):
    
    # Performs linear regression on the provided dataset.

    Args = {
        "response_code": 400,
        "message": """
            data (pd.DataFrame): The dataset containing features and the target column.
            target_column (str): The name of the target column.
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2
            random_state (int): Seed for the random number generator for reproducibility. Default is 42.
        """
    }

    if (is_pd_dataframe(data) == False):
        return Args
    if (is_string(target_column) == False):
        return Args
    if (is_float(test_size) == False):
        return Args
    if (is_integer(random_state) == False):
        return Args
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        "mean_squared_error": mse,
        "r2_score": r2
    }

    return {
        "response_code": 200,
        "model": model,
        "predictions": y_pred,
        "metrics": metrics
    }



# 2.
def logistic_regression(data, target_column, test_size=0.2, random_state=42, penalty='l2', C=1.0, max_iter=100):
    
    #Performs logistic regression on the provided dataset and evaluates its performance.
    Args = {
        "response_code": 400,
        "message": """
            data (pd.DataFrame): The dataset containing features and the target column.
            target_column (str): The name of the target column.
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
            random_state (int): Seed for the random number generator for reproducibility. Default is 42.
            penalty (str): The norm used in the penalization ('l1', 'l2', 'elasticnet', or 'none'). Default is 'l2'.
            C (float): Inverse of regularization strength; smaller values specify stronger regularization. Default is 1.0.
            max_iter (int): Maximum number of iterations for the solver. Default is 100.
        """
    }

    if (is_pd_dataframe(data) == False):
        return Args
    if (is_string(target_column) == False):
        return Args
    if (is_float(test_size) == False):
        return Args
    if (is_integer(random_state) == False):
        return Args
    if (is_string(penalty) == False):
        return Args
    if (is_float(C) == False):
        return Args
    if (is_integer(max_iter) == False):
        return Args
    
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize and train the model
    model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Predicted probabilities for the positive class

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix
    }

    return {
        "response_code": 200,
        "model": model,
        "predictions": y_pred,
        "probabilities": y_prob,
        "metrics": metrics
    }


# 3.

def decision_tree(data, target_column, task_type='classification', test_size=0.2, random_state=42, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Performs a decision tree-based task (classification or regression) on the provided dataset and evaluates its performance.

    Args:
        data (pd.DataFrame): The dataset containing features and the target column.
        target_column (str): The name of the target column.
        task_type (str): Task type - 'classification' or 'regression'. Default is 'classification'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for the random number generator for reproducibility. Default is 42.
        max_depth (int): The maximum depth of the tree. Default is None (tree expands until all leaves are pure).
        min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Default is 1.

    Returns:
        dict: A dictionary containing the model, predictions, and performance metrics.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the model based on task type
    if task_type == 'classification':
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state)
    elif task_type == 'regression':
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_state)
    else:
        #raise ValueError("task_type must be either 'classification' or 'regression'")
        return ["400", "task_type must be either 'classification' or 'regression'"]
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance based on task type
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix
        }
    else:  # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mean_squared_error": mse,
            "r2_score": r2
        }

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example classification dataset
    classification_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [0, 1, 0, 1, 0, 1]
    })

    classification_result = decision_tree(classification_data, target_column='target', task_type='classification', max_depth=3)
    print("Classification Metrics:", classification_result["metrics"])

    # Example regression dataset
    regression_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    })

    regression_result = decision_tree(regression_data, target_column='target', task_type='regression', max_depth=3)
    print("Regression Metrics:", regression_result["metrics"])

"""

# 4.

def random_forest(data, target_column, task_type='classification', test_size=0.2, random_state=42,
                           n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Performs a random forest-based task (classification or regression) on the provided dataset and evaluates its performance.

    Args:
        data (pd.DataFrame): The dataset containing features and the target column.
        target_column (str): The name of the target column.
        task_type (str): Task type - 'classification' or 'regression'. Default is 'classification'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for the random number generator for reproducibility. Default is 42.
        n_estimators (int): The number of trees in the forest. Default is 100.
        max_depth (int): The maximum depth of the trees. Default is None (expand until all leaves are pure).
        min_samples_split (int): The minimum number of samples required to split an internal node. Default is 2.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node. Default is 1.

    Returns:
        dict: A dictionary containing the model, predictions, and performance metrics.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the model based on task type
    if task_type == 'classification':
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       random_state=random_state)
    elif task_type == 'regression':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      random_state=random_state)
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")
    
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance based on task type
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix
        }
    else:  # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mean_squared_error": mse,
            "r2_score": r2
        }

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics
    }


"""
# Example usage
if __name__ == "__main__":
    # Example classification dataset
    classification_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [0, 1, 0, 1, 0, 1]
    })

    classification_result = random_forest(classification_data, target_column='target', task_type='classification', n_estimators=100)
    print("Classification Metrics:", classification_result["metrics"])

    # Example regression dataset
    regression_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    })

    regression_result = random_forest(regression_data, target_column='target', task_type='regression', n_estimators=100)
    print("Regression Metrics:", regression_result["metrics"])
"""

# 5.

def svm(data, target_column, task_type='classification', test_size=0.2, random_state=42, kernel='rbf', C=1.0, epsilon=0.1):
    """
    Performs a Support Vector Machine (SVM) task (classification or regression) on the provided dataset and evaluates its performance.

    Args:
        data (pd.DataFrame): The dataset containing features and the target column.
        target_column (str): The name of the target column.
        task_type (str): Task type - 'classification' or 'regression'. Default is 'classification'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for the random number generator for reproducibility. Default is 42.
        kernel (str): Specifies the kernel type to be used in the algorithm ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed'). Default is 'rbf'.
        C (float): Regularization parameter. The strength of the regularization is inversely proportional to C. Default is 1.0.
        epsilon (float): Epsilon in the epsilon-SVR model. Only used for regression tasks. Default is 0.1.

    Returns:
        dict: A dictionary containing the model, predictions, and performance metrics.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialize the model based on task type
    if task_type == 'classification':
        model = SVC(kernel=kernel, C=C, random_state=random_state, probability=True)
    elif task_type == 'regression':
        model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance based on task type
    if task_type == 'classification':
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_matrix = confusion_matrix(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix
        }
    else:  # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mean_squared_error": mse,
            "r2_score": r2
        }

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example classification dataset
    classification_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [0, 1, 0, 1, 0, 1]
    })

    classification_result = svm(classification_data, target_column='target', task_type='classification', kernel='linear', C=1.0)
    print("Classification Metrics:", classification_result["metrics"])

    # Example regression dataset
    regression_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    })

    regression_result = svm(regression_data, target_column='target', task_type='regression', kernel='rbf', C=1.0, epsilon=0.1)
    print("Regression Metrics:", regression_result["metrics"])
"""

# 6.

def lstm(data, target_column, task_type='regression', test_size=0.2, random_state=42,
                 lstm_units=50, embedding_dim=None, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Performs an LSTM-based task (regression or classification) on sequential data and evaluates its performance.

    Args:
        data (np.array): Sequential dataset containing features and the target column.
        target_column (str): Name of the target column (if using a DataFrame) or index of the target in arrays.
        task_type (str): Task type - 'regression' or 'classification'. Default is 'regression'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for the random number generator for reproducibility. Default is 42.
        lstm_units (int): Number of LSTM units. Default is 50.
        embedding_dim (int): Embedding dimension for text tasks. Default is None (not used for non-text tasks).
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Size of each training batch. Default is 32.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.

    Returns:
        dict: A dictionary containing the trained model, predictions, and evaluation metrics.
    """
    # Prepare the data
    #if isinstance(data, pd.DataFrame):
    #    X = data.drop(columns=[target_column]).values
    #    y = data[target_column].values
    #else:
    X = data[:, :-1]
    y = data[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Reshape for LSTM input (samples, timesteps, features)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Define the model
    model = Sequential()

    if embedding_dim:
        # Use Embedding layer for text data
        model.add(Embedding(input_dim=X_train.shape[1], output_dim=embedding_dim, input_length=X_train.shape[1]))

    model.add(LSTM(units=lstm_units, input_shape=(X_train.shape[1], X_train.shape[2])))

    if task_type == 'classification':
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task_type == 'regression':
        model.add(Dense(1))  # Regression
        loss = 'mean_squared_error'
        metrics = []
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Evaluate performance
    if task_type == 'classification':
        y_pred_labels = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_labels)
        precision = precision_score(y_test, y_pred_labels)
        recall = recall_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    else:  # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            "mean_squared_error": mse
        }

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics
    }


"""
# Example usage
if __name__ == "__main__":
    # Example regression dataset
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    })

    result = lstm(data, target_column='target', task_type='regression', lstm_units=64, epochs=10)
    print("Regression Metrics:", result["metrics"])

"""

# 7.

def rnn(data, target_column, task_type='regression', test_size=0.2, random_state=42,
                rnn_units=50, embedding_dim=None, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Performs an RNN-based task (regression or classification) on sequential data and evaluates its performance.

    Args:
        data (np.array): Sequential dataset containing features and the target column.
        target_column (str): Name of the target column (if using a DataFrame) or index of the target in arrays.
        task_type (str): Task type - 'regression' or 'classification'. Default is 'regression'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for the random number generator for reproducibility. Default is 42.
        rnn_units (int): Number of RNN units. Default is 50.
        embedding_dim (int): Embedding dimension for text tasks. Default is None (not used for non-text tasks).
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Size of each training batch. Default is 32.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.

    Returns:
        dict: A dictionary containing the trained model, predictions, and evaluation metrics.
    """
    # Prepare the data
    #if isinstance(data, pd.DataFrame):
    #    X = data.drop(columns=[target_column]).values
    #    y = data[target_column].values
    #else:
    X = data[:, :-1]
    y = data[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Reshape for RNN input (samples, timesteps, features)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # Define the model
    model = Sequential()

    if embedding_dim:
        # Use Embedding layer for text data
        model.add(Embedding(input_dim=X_train.shape[1], output_dim=embedding_dim, input_length=X_train.shape[1]))

    model.add(SimpleRNN(units=rnn_units, input_shape=(X_train.shape[1], X_train.shape[2])))

    if task_type == 'classification':
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task_type == 'regression':
        model.add(Dense(1))  # Regression
        loss = 'mean_squared_error'
        metrics = []
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Evaluate performance
    if task_type == 'classification':
        y_pred_labels = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_labels)
        precision = precision_score(y_test, y_pred_labels)
        recall = recall_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    else:  # Regression metrics
        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            "mean_squared_error": mse
        }

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example regression dataset
    data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    })

    result = rnn(data, target_column='target', task_type='regression', rnn_units=64, epochs=10)
    print("Regression Metrics:", result["metrics"])
"""

# 8.

def dnn(data, target_column, task_type='classification', test_size=0.2, random_state=42,
                hidden_layers=[128, 64], activation='relu', dropout_rate=0.5, epochs=10,
                batch_size=32, learning_rate=0.001):
    """
    Performs a Deep Neural Network (DNN) task (classification or regression) and evaluates its performance.

    Args:
        data (np.array): Dataset containing features and the target column.
        target_column (str): Name of the target column (if using a DataFrame) or index of the target in arrays.
        task_type (str): Task type - 'classification' or 'regression'. Default is 'classification'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for the random number generator for reproducibility. Default is 42.
        hidden_layers (list): List of integers defining the number of neurons in each hidden layer. Default is [128, 64].
        activation (str): Activation function to use for hidden layers. Default is 'relu'.
        dropout_rate (float): Dropout rate for regularization. Default is 0.5.
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Size of each training batch. Default is 32.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.

    Returns:
        dict: A dictionary containing the trained model, predictions, and evaluation metrics.
    """
    # Prepare the data
    #if isinstance(data, pd.DataFrame):
    #    X = data.drop(columns=[target_column]).values
    #    y = data[target_column].values
    #else:
    X = data[:, :-1]
    y = data[:, -1]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the model
    model = Sequential()

    # Add hidden layers
    input_dim = X_train.shape[1]
    for units in hidden_layers:
        model.add(Dense(units=units, activation=activation, input_dim=input_dim))
        model.add(Dropout(rate=dropout_rate))
        input_dim = units  # Update input_dim for the next layer

    # Add the output layer
    if task_type == 'classification':
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task_type == 'regression':
        model.add(Dense(1))  # Regression
        loss = 'mean_squared_error'
        metrics = []
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Evaluate performance
    if task_type == 'classification':
        y_pred_labels = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_labels)
        precision = precision_score(y_test, y_pred_labels)
        recall = recall_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    else:  # Regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            "mean_squared_error": mse,
            "r2_score": r2
        }

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example classification dataset
    classification_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [0, 1, 0, 1, 0, 1]
    })

    classification_result = dnn(classification_data, target_column='target', task_type='classification', hidden_layers=[128, 64], epochs=10)
    print("Classification Metrics:", classification_result["metrics"])

    # Example regression dataset
    regression_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': [2, 4, 6, 8, 10, 12],
        'target': [1.1, 2.1, 3.1, 4.1, 5.1, 6.1]
    })

    regression_result = dnn(regression_data, target_column='target', task_type='regression', hidden_layers=[128, 64], epochs=10)
    print("Regression Metrics:", regression_result["metrics"])
"""

# 9.

def cnn(data, labels, input_shape, task_type='classification', test_size=0.2, random_state=42,
                conv_layers=[(32, (3, 3))], dense_layers=[128], pool_size=(2, 2), dropout_rate=0.5, epochs=10,
                batch_size=32, learning_rate=0.001, gpu_optimized='no'):
    """
    Performs a Convolutional Neural Network (CNN) task (classification or regression) on spatial data and evaluates its performance.

    Args:
        data (np.array): Input dataset (e.g., images) as a NumPy array.
        labels (np.array): Target labels corresponding to the input data.
        input_shape (tuple): Shape of each input sample (e.g., (height, width, channels)).
        task_type (str): Task type - 'classification' or 'regression'. Default is 'classification'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for the random number generator for reproducibility. Default is 42.
        conv_layers (list of tuples of int and tuple of ints): List of tuples specifying convolutional layers as (filters, kernel_size). Default is [(32, (3, 3))].
        dense_layers (list of ints): List of integers defining the number of neurons in fully connected layers. Default is [128].
        pool_size (tuple): Size of the pooling window for max pooling. Default is (2, 2).
        dropout_rate (float): Dropout rate for regularization. Default is 0.5.
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Size of each training batch. Default is 32.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        gpu_optimized (str): If 'yes', forces GPU utilization. Default is 'no'.
    Returns:
        dict: A dictionary containing the trained model, predictions, and evaluation metrics.
    """

    if gpu_optimized == 'yes':
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                tf.config.set_visible_devices(physical_devices, 'GPU')
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print("Error enabling GPU optimization:", e)
        else:
            print("No GPU found. Using CPU.")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)

    # Define the model
    model = Sequential()

    # Add convolutional and pooling layers
    for filters, kernel_size in conv_layers:
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=pool_size))

    # Flatten the output from convolutional layers
    model.add(Flatten())

    # Add fully connected (dense) layers
    for units in dense_layers:
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    # Add the output layer
    if task_type == 'classification':
        model.add(Dense(1, activation='sigmoid'))  # Binary classification
        loss = 'binary_crossentropy'
        metrics = ['accuracy']
    elif task_type == 'regression':
        model.add(Dense(1))  # Regression
        loss = 'mean_squared_error'
        metrics = []
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Make predictions
    y_pred = model.predict(X_test).flatten()

    # Evaluate performance
    if task_type == 'classification':
        y_pred_labels = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred_labels)
        precision = precision_score(y_test, y_pred_labels)
        recall = recall_score(y_test, y_pred_labels)
        f1 = f1_score(y_test, y_pred_labels)
        conf_matrix = confusion_matrix(y_test, y_pred_labels)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix
        }
    else:  # Regression metrics
        mse = mean_squared_error(y_test, y_pred)

        metrics = {
            "mean_squared_error": mse
        }

    return {
        "model": model,
        "predictions": y_pred,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example synthetic dataset
    data = np.random.rand(100, 64, 64, 3)  # 100 RGB images of size 64x64
    labels = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)

    result = perform_cnn(data, labels, input_shape=(64, 64, 3), task_type='classification', epochs=5, gpu_optimized='yes')
    print("Classification Metrics:", result["metrics"])

"""

# 10.

def gan(data, input_dim, output_shape, task_type='img', gpu_optimized='no',
                generator_layers=[128, 256, 512], discriminator_layers=[512, 256, 128],
                epochs=10000, batch_size=64, learning_rate=0.0002):
    """
    Implements a Generative Adversarial Network (GAN) for generating synthetic data and evaluates its performance.

    Args:
        data (np.array): Input dataset for training the GAN.
        input_dim (int): Dimensionality of the latent space for the generator.
        output_shape (tuple): Shape of the generated data (e.g., image dimensions).
        task_type (str): Type of task - 'img', 'video', or 'audio'. Default is 'img'.
        gpu_optimized (str): If 'yes', utilizes GPU for computations. Default is 'no'.
        generator_layers (list): List of integers defining the generator's hidden layers. Default is [128, 256, 512].
        discriminator_layers (list): List of integers defining the discriminator's hidden layers. Default is [512, 256, 128].
        epochs (int): Number of training epochs. Default is 10000.
        batch_size (int): Size of training batches. Default is 64.
        learning_rate (float): Learning rate for both generator and discriminator. Default is 0.0002.

    Returns:
        dict: A dictionary containing the trained generator, discriminator, and loss metrics.
    """
    if gpu_optimized == 'yes':
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                tf.config.set_visible_devices(physical_devices, 'GPU')
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print("Error enabling GPU optimization:", e)
        else:
            print("No GPU found. Using CPU.")

    # Define the generator
    def build_generator():
        model = Sequential()
        model.add(Input(shape=(input_dim,)))
        for units in generator_layers:
            model.add(Dense(units))
            model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(output_shape), activation='tanh'))
        model.add(Reshape(output_shape))
        return model

    # Define the discriminator
    def build_discriminator():
        model = Sequential()
        model.add(Input(shape=output_shape))
        model.add(Flatten())
        for units in discriminator_layers:
            model.add(Dense(units))
            model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        return model

    # Instantiate models
    generator = build_generator()
    discriminator = build_discriminator()

    # Compile the discriminator
    discriminator.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

    # Build and compile the GAN
    discriminator.trainable = False
    z = Input(shape=(input_dim,))
    fake_data = generator(z)
    validity = discriminator(fake_data)
    gan = Model(z, validity)
    gan.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy')

    # Training loop
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train the discriminator
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        generated_data = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_data, real)
        d_loss_fake = discriminator.train_on_batch(generated_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train the generator
        noise = np.random.normal(0, 1, (batch_size, input_dim))
        g_loss = gan.train_on_batch(noise, real)

        # Print progress
        if epoch % 1000 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss[0]:.4f}, D Acc: {d_loss[1]:.4f}, G Loss: {g_loss:.4f}")

    return {
        "generator": generator,
        "discriminator": discriminator,
        "metrics": {
            "discriminator_loss": d_loss[0],
            "discriminator_accuracy": d_loss[1],
            "generator_loss": g_loss
        }
    }


"""
# Example usage
if __name__ == "__main__":
    # Example synthetic dataset
    data = np.random.rand(1000, 28, 28, 1) * 2 - 1  # Normalized data for image generation

    result = gan(data, input_dim=100, output_shape=(28, 28, 1), task_type='img', gpu_optimized='yes', epochs=5000)
    print("GAN Training Complete.")
"""

# 11.

def naive_bayes(data, labels, gpu_optimized='no', test_size=0.2, random_state=42, use_tfidf=True):
    """
    Builds and trains a Naive Bayes-like classifier for text-based tasks like fake news detection or sentiment analysis. Heavy computations are offloaded to GPU if specified.

    Args:
        data (list): List of input text data.
        labels (list): Corresponding labels for the input data.
        gpu_optimized (str): If 'yes', enables GPU acceleration for tensor operations. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility of train-test splits. Default is 42.
        use_tfidf (bool): If True, applies TF-IDF transformation to text data. Default is True.

    Returns:
        dict: A dictionary containing the trained TensorFlow model, vectorizer, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    # Vectorize the text data
    vectorizer = CountVectorizer()
    X_train_counts = vectorizer.fit_transform(X_train)
    X_test_counts = vectorizer.transform(X_test)

    # Apply TF-IDF transformation if specified
    if use_tfidf:
        tfidf_transformer = TfidfTransformer()
        X_train_transformed = tfidf_transformer.fit_transform(X_train_counts).toarray()
        X_test_transformed = tfidf_transformer.transform(X_test_counts).toarray()
    else:
        X_train_transformed = X_train_counts.toarray()
        X_test_transformed = X_test_counts.toarray()

    # Use TensorFlow tensors for heavy computations
    X_train_tf = tf.convert_to_tensor(X_train_transformed, dtype=tf.float32)
    X_test_tf = tf.convert_to_tensor(X_test_transformed, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test_tf = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Define TensorFlow Naive Bayes-like model
    class NaiveBayesModel(tf.keras.Model):
        def __init__(self):
            super(NaiveBayesModel, self).__init__()
            self.weights = tf.Variable(tf.random.normal([X_train_tf.shape[1], 1]))

        def call(self, inputs):
            logits = tf.matmul(inputs, self.weights)
            return tf.sigmoid(logits)

    model = NaiveBayesModel()

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # Train the model
    model.fit(X_train_tf, y_train_tf, epochs=10, batch_size=32, verbose=1)

    # Evaluate the model
    y_pred_probs = model.predict(X_test_tf).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return {
        "model": model,
        "vectorizer": vectorizer,
        "metrics": metrics
    }


"""
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = [
        "The movie was fantastic!", 
        "I hated the film.", 
        "It was an average experience.", 
        "Amazing storyline and characters!", 
        "Not worth my time."
    ]
    labels = [1, 0, 1, 1, 0]  # 1: Positive, 0: Negative

    result = naive_bayes(data, labels, gpu_optimized='yes', use_tfidf=True)
    print("Naive Bayes Metrics:", result["metrics"])
"""

# 12.

def k_means(data, n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42,
                 gpu_optimized='no'):
    """
    Builds and trains a K-Means clustering model with optional GPU acceleration for heavy computations.

    Args:
        data (array-like): The input data for clustering.
        n_clusters (int): The number of clusters to form. Default is 3.
        init (str): Method for initialization ('k-means++', 'random', or ndarray). Default is 'k-means++'.
        max_iter (int): Maximum number of iterations for the algorithm. Default is 300.
        n_init (int): Number of times the algorithm will be run with different centroid seeds. Default is 10.
        random_state (int): Seed for random number generation. Default is 42.
        gpu_optimized (str): If 'yes', heavy computations are offloaded to GPU using TensorFlow. Default is 'no'.

    Returns:
        dict: A dictionary containing the K-Means model, cluster assignments, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled using TensorFlow.")
        # Convert data to TensorFlow tensors for GPU computation
        data = tf.convert_to_tensor(data, dtype=tf.float32)

        # Use TensorFlow for centroid initialization and computation
        @tf.function
        def gpu_kmeans(data, n_clusters, max_iter):
            centroids = tf.gather(data, tf.random.shuffle(tf.range(tf.shape(data)[0]))[:n_clusters])

            for _ in range(max_iter):
                distances = tf.reduce_sum(tf.square(tf.expand_dims(data, 1) - tf.expand_dims(centroids, 0)), axis=2)
                cluster_assignments = tf.argmin(distances, axis=1)

                new_centroids = []
                for k in range(n_clusters):
                    mask = tf.equal(cluster_assignments, k)
                    cluster_points = tf.boolean_mask(data, mask)
                    centroid = tf.reduce_mean(cluster_points, axis=0)
                    new_centroids.append(centroid)

                new_centroids = tf.stack(new_centroids)

                if tf.reduce_all(tf.equal(new_centroids, centroids)):
                    break
                centroids = new_centroids

            return centroids, cluster_assignments

        centroids, cluster_assignments = gpu_kmeans(data, n_clusters, max_iter)
        cluster_assignments = cluster_assignments.numpy()
    else:
        print("Running on CPU.")
        # Use scikit-learn's KMeans for CPU computation
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state
        )
        cluster_assignments = kmeans.fit_predict(data)
        centroids = kmeans.cluster_centers_

    # Evaluate clustering with silhouette score
    silhouette_avg = silhouette_score(data, cluster_assignments)
    metrics = {
        "silhouette_score": silhouette_avg
    }

    return {
        "centroids": centroids,
        "assignments": cluster_assignments,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = np.random.rand(100, 2)  # 100 samples with 2 features

    result = k_means(data, n_clusters=3, gpu_optimized='yes')
    print("Centroids:", result["centroids"])
    print("Silhouette Score:", result["silhouette_score"])
"""

# 13.

def lda(data, labels, test_size=0.2, n_components=None, random_state=42, gpu_optimized='no'):
    """
    Builds and trains a Linear Discriminant Analysis (LDA) model for classification tasks. Optionally uses GPU for heavy computations.

    Args:
        data (array-like): Feature matrix (n_samples, n_features).
        labels (array-like): Target labels (n_samples,).
        test_size (float): Proportion of data for testing. Default is 0.2.
        n_components (int or None): Number of components for dimensionality reduction. Default is None (auto-determined).
        random_state (int): Seed for reproducibility. Default is 42.
        gpu_optimized (str): If 'yes', leverages GPU for heavy computations using TensorFlow. Default is 'no'.

    Returns:
        dict: Contains the trained model, metrics, and reduced data.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled using TensorFlow.")
    else:
        print("Running on CPU.")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    # Use TensorFlow for heavy computation if GPU optimization is enabled
    if gpu_optimized == 'yes':
        X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train_tf = tf.convert_to_tensor(y_train, dtype=tf.float32)

        # Perform mean normalization on GPU
        train_mean = tf.reduce_mean(X_train_tf, axis=0)
        train_std = tf.math.reduce_std(X_train_tf, axis=0)
        X_train_tf = (X_train_tf - train_mean) / train_std
        X_test_tf = (X_test_tf - train_mean) / train_std

        # Convert back to NumPy for sklearn LDA model
        X_train = X_train_tf.numpy()
        X_test = X_test_tf.numpy()

    # Build and train the LDA model
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train, y_train)

    # Predictions and evaluation
    y_pred = lda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return {
        "model": lda,
        "metrics": metrics,
        "X_train_reduced": lda.transform(X_train),
        "X_test_reduced": lda.transform(X_test)
    }

"""
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = np.random.rand(100, 5)  # 100 samples with 5 features
    labels = np.random.randint(0, 3, 100)  # 3 classes

    result = lda(data, labels, gpu_optimized='yes')
    print("LDA Metrics:", result["metrics"])
"""

# 14.

def hmm_mml(sequences, n_states=3, n_iter=10, covariance_type='diag', gpu_optimized='no'):
    """
    Builds and trains a Hidden Markov Model (HMM) for sequence modeling tasks. Optionally uses GPU for heavy computations.

    Args:
        sequences (list of array-like): List of sequences where each sequence is an array of observations.
        n_states (int): Number of hidden states in the HMM. Default is 3.
        n_iter (int): Number of iterations for the Baum-Welch algorithm. Default is 10.
        covariance_type (str): Type of covariance for Gaussian emissions ('diag', 'spherical', 'tied', 'full'). Default is 'diag'.
        gpu_optimized (str): If 'yes', leverages GPU for heavy computations using TensorFlow. Default is 'no'.

    Returns:
        dict: Contains the trained model and metrics for evaluation.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled using TensorFlow.")
    else:
        print("Running on CPU.")

    # Concatenate sequences for HMM training
    lengths = [len(seq) for seq in sequences]
    data = np.concatenate(sequences)

    # Use TensorFlow for heavy computations if GPU optimization is enabled
    if gpu_optimized == 'yes':
        data_tf = tf.convert_to_tensor(data, dtype=tf.float32)

        # Compute mean and std for normalization on GPU
        data_mean = tf.reduce_mean(data_tf, axis=0)
        data_std = tf.math.reduce_std(data_tf, axis=0)
        normalized_data = (data_tf - data_mean) / data_std

        # Convert back to NumPy for hmmlearn
        data = normalized_data.numpy()

    # Build and train the HMM model
    model = hmm.GaussianHMM(n_components=n_states, covariance_type=covariance_type, n_iter=n_iter)
    model.fit(data.reshape(-1, 1), lengths)

    # Predict hidden states
    predicted_states = model.predict(data.reshape(-1, 1))

    # For metrics, assume ground truth states (if available) are part of sequences
    # Placeholder: Replace with actual ground truth if available
    ground_truth_states = np.repeat(range(n_states), len(data) // n_states)[:len(data)]

    # Compute metrics
    accuracy = accuracy_score(ground_truth_states, predicted_states)
    precision = precision_score(ground_truth_states, predicted_states, average='weighted', zero_division=0)
    recall = recall_score(ground_truth_states, predicted_states, average='weighted', zero_division=0)
    f1 = f1_score(ground_truth_states, predicted_states, average='weighted', zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return {
        "model": model,
        "metrics": metrics,
        "predicted_states": predicted_states
    }

"""
# Example usage
if __name__ == "__main__":
    # Example dataset: Sequences of observations
    sequences = [
        np.random.rand(50),
        np.random.rand(60),
        np.random.rand(70)
    ]
    result = hmm_mml(sequences, n_states=3, gpu_optimized='yes')
    print("HMM Metrics:", result["metrics"])
"""

# 15.

def autoencoder_gpu(data, encoding_dim=32, epochs=50, batch_size=128, test_split=0.2, gpu_optimized='no'):
    """
    Builds and trains an autoencoder for unsupervised learning tasks such as dimensionality reduction and anomaly detection.

    Args:
        data (np.array): The input data (n_samples, n_features).
        encoding_dim (int): Dimension of the encoded representation. Default is 32.
        epochs (int): Number of training epochs. Default is 50.
        batch_size (int): Batch size for training. Default is 128.
        test_split (float): Proportion of data to use for validation. Default is 0.2.
        gpu_optimized (str): If 'yes', uses TensorFlow's GPU acceleration. Default is 'no'.

    Returns:
        dict: Contains the trained autoencoder model, reconstructed data, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Normalize the data
    data = np.array(data, dtype=np.float32)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_normalized = (data - data_mean) / data_std

    # Split data into training and testing sets
    n_train = int((1 - test_split) * len(data_normalized))
    train_data = data_normalized[:n_train]
    test_data = data_normalized[n_train:]

    # Define the autoencoder model
    input_dim = data.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    # Train the autoencoder
    history = autoencoder.fit(
        train_data, train_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(test_data, test_data),
        verbose=1
    )

    # Reconstruct the data
    reconstructed_data = autoencoder.predict(data_normalized)

    # Evaluate reconstruction error
    mse = mean_squared_error(data_normalized, reconstructed_data)

    metrics = {
        "mean_squared_error": mse
    }

    return {
        "autoencoder": autoencoder,
        "encoder": encoder,
        "reconstructed_data": reconstructed_data,
        "metrics": metrics,
        "history": history
    }


"""
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = np.random.rand(1000, 20)  # 1000 samples with 20 features

    result = autoencoder_gpu(data, encoding_dim=10, epochs=20, gpu_optimized='yes')
    print("Autoencoder Metrics:", result["metrics"])
"""

# 16.

def vae(data, latent_dim=2, intermediate_dim=64, epochs=50, batch_size=128, test_split=0.2, gpu_optimized='no'):
    """
    Builds and trains a Variational Autoencoder (VAE) for tasks such as data generation and anomaly detection.

    Args:
        data (array-like): The input data (n_samples, n_features).
        latent_dim (int): Dimension of the latent space. Default is 2.
        intermediate_dim (int): Number of units in the intermediate layer. Default is 64.
        epochs (int): Number of training epochs. Default is 50.
        batch_size (int): Batch size for training. Default is 128.
        test_split (float): Proportion of data to use for validation. Default is 0.2.
        gpu_optimized (str): If 'yes', uses TensorFlow's GPU acceleration. Default is 'no'.

    Returns:
        dict: Contains the trained VAE model, encoder, decoder, reconstructed data, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Normalize the data
    data = np.array(data, dtype=np.float32)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_normalized = (data - data_mean) / data_std

    # Split data into training and testing sets
    n_train = int((1 - test_split) * len(data_normalized))
    train_data = data_normalized[:n_train]
    test_data = data_normalized[n_train:]

    # Input placeholder
    input_dim = data.shape[1]
    inputs = Input(shape=(input_dim,))

    # Encoder
    h = Dense(intermediate_dim, activation='relu')(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # Decoder
    decoder_h = Dense(intermediate_dim, activation='relu')
    decoder_mean = Dense(input_dim, activation='sigmoid')
    h_decoded = decoder_h(z)
    x_decoded_mean = decoder_mean(h_decoded)

    # Custom loss function
    def vae_loss(inputs, x_decoded_mean):
        reconstruction_loss = K.sum(K.binary_crossentropy(inputs, x_decoded_mean), axis=-1)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + kl_loss)

    # VAE model
    vae = Model(inputs, x_decoded_mean)
    vae.add_loss(vae_loss(inputs, x_decoded_mean))
    vae.compile(optimizer='adam')

    # Train the VAE
    history = vae.fit(
        train_data,
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(test_data, test_data),
        verbose=1
    )

    # Encoder model
    encoder = Model(inputs, z_mean)

    # Decoder model
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    decoder = Model(decoder_input, _x_decoded_mean)

    # Reconstruct data
    reconstructed_data = vae.predict(data_normalized)

    # Evaluate reconstruction error
    mse = mean_squared_error(data_normalized, reconstructed_data)

    metrics = {
        "mean_squared_error": mse
    }

    return {
        "vae": vae,
        "encoder": encoder,
        "decoder": decoder,
        "reconstructed_data": reconstructed_data,
        "metrics": metrics,
        "history": history
    }


"""
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = np.random.rand(1000, 20)  # 1000 samples with 20 features

    result = vae(data, latent_dim=5, epochs=20, gpu_optimized='yes')
    print("VAE Metrics:", result["metrics"])
"""


# 17.


def transformer(
    data, 
    labels=None,
    vocab_size=10000,
    max_len=100,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=2,
    epochs=20,
    batch_size=32,
    test_split=0.2,
    gpu_optimized='no'
):
    """
    Builds and trains a Transformer model for sequence tasks like text generation or translation.

    Args:
        data (array-like): Input data (e.g., tokenized sequences or feature matrices).
        labels (array-like): Target labels (optional, used for supervised tasks).
        vocab_size (int): Size of the vocabulary (for embedding layer). Default is 10000.
        max_len (int): Maximum sequence length. Default is 100.
        embed_dim (int): Dimension of the embedding space. Default is 64.
        num_heads (int): Number of attention heads. Default is 4.
        ff_dim (int): Dimension of the feed-forward network. Default is 128.
        num_layers (int): Number of transformer encoder layers. Default is 2.
        epochs (int): Number of training epochs. Default is 20.
        batch_size (int): Batch size for training. Default is 32.
        test_split (float): Proportion of data to use for validation. Default is 0.2.
        gpu_optimized (str): If 'yes', uses TensorFlow's GPU acceleration. Default is 'no'.

    Returns:
        dict: Contains the trained transformer model, metrics, and history.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Data preparation
    data = np.array(data, dtype=np.int32)
    if labels is not None:
        labels = np.array(labels, dtype=np.int32)

    # Train-test split
    n_train = int((1 - test_split) * len(data))
    train_data = data[:n_train]
    test_data = data[n_train:]

    if labels is not None:
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]

    # Define Transformer components
    def transformer_encoder(inputs):
        x = Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len)(inputs)
        for _ in range(num_layers):
            attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            ffn_output = Dense(ff_dim, activation="relu")(x)
            ffn_output = Dense(embed_dim)(ffn_output)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        return x

    inputs = Input(shape=(max_len,))
    x = transformer_encoder(inputs)
    outputs = Dense(vocab_size, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Training
    history = None
    if labels is not None:
        history = model.fit(
            train_data, train_labels,
            validation_data=(test_data, test_labels),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

    # Evaluation
    metrics = {}
    if labels is not None:
        predictions = model.predict(test_data)
        predictions = np.argmax(predictions, axis=-1)
        accuracy = accuracy_score(test_labels.flatten(), predictions.flatten())
        precision = precision_score(test_labels.flatten(), predictions.flatten(), average='weighted', zero_division=0)
        recall = recall_score(test_labels.flatten(), predictions.flatten(), average='weighted', zero_division=0)
        f1 = f1_score(test_labels.flatten(), predictions.flatten(), average='weighted', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    return {
        "model": model,
        "metrics": metrics,
        "history": history
    }

"""
# Example usage
if __name__ == "__main__":
    # Example dataset: tokenized sequences
    data = np.random.randint(1, 10000, size=(1000, 100))  # 1000 sequences of length 100
    labels = np.random.randint(0, 10000, size=(1000, 100))

    result = transformer(data, labels=labels, gpu_optimized='yes', epochs=10, embed_dim=128, num_heads=8)
    print("Transformer Metrics:", result["metrics"])
"""

# 18.

def bert(
    data, 
    labels=None,
    model_name="bert-base-uncased",
    max_len=128,
    batch_size=32,
    epochs=3,
    test_split=0.2,
    gpu_optimized='no'
):
    """
    Builds and trains a BERT model for NLP tasks with GPU acceleration using TensorFlow.

    Args:
        data (list): List of text data.
        labels (list): Corresponding labels for classification tasks (optional for unsupervised tasks).
        model_name (str): Pretrained BERT model name from Hugging Face. Default is 'bert-base-uncased'.
        max_len (int): Maximum token length for input sequences. Default is 128.
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of training epochs. Default is 3.
        test_split (float): Proportion of data for validation. Default is 0.2.
        gpu_optimized (str): If 'yes', uses TensorFlow's GPU acceleration. Default is 'no'.

    Returns:
        dict: Contains the trained BERT model, tokenizer, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def encode_texts(texts):
        return tokenizer(
            texts,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors="tf"
        )

    # Encode data
    encodings = encode_texts(data)
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']

    if labels is not None:
        labels = np.array(labels)

    # Train-test split
    if labels is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            (input_ids, attention_masks), labels, test_size=test_split, random_state=42
        )
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
        test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    else:
        train_data = tf.data.Dataset.from_tensor_slices((input_ids, attention_masks)).batch(batch_size)

    # Load pre-trained BERT model
    bert_model = TFBertModel.from_pretrained(model_name)

    # Build classification model
    input_ids_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
    attention_masks_layer = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")

    bert_outputs = bert_model(
        input_ids_layer, attention_mask=attention_masks_layer
    )

    cls_token = bert_outputs.last_hidden_state[:, 0, :]  # CLS token output
    output_layer = Dense(1, activation='sigmoid')(cls_token)

    model = Model(
        inputs=[input_ids_layer, attention_masks_layer], outputs=output_layer
    )

    model.compile(optimizer=Adam(learning_rate=3e-5),
                  loss=BinaryCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    history = None
    if labels is not None:
        history = model.fit(
            train_data,
            validation_data=test_data,
            epochs=epochs,
            batch_size=batch_size
        )

    # Evaluate the model
    metrics = {}
    if labels is not None:
        predictions = model.predict(test_data)
        predictions = np.round(predictions).flatten()

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    return {
        "model": model,
        "tokenizer": tokenizer,
        "metrics": metrics,
        "history": history
    }


"""
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = [
        "The movie was fantastic!",
        "I hated the film.",
        "It was an average experience.",
        "Amazing storyline and characters!",
        "Not worth my time."
    ]
    labels = [1, 0, 1, 1, 0]  # 1: Positive, 0: Negative

    result = bert(data, labels=labels, gpu_optimized='yes', epochs=2)
    print("BERT Metrics:", result["metrics"])
"""

# 19.


def gpt(
    data,
    model_name="gpt2",
    max_len=128,
    batch_size=32,
    epochs=3,
    gpu_optimized='no',
    test_split=0.2
):
    """
    Builds and trains a GPT model for text generation and other sequence tasks with GPU acceleration using TensorFlow.

    Args:
        data (list): List of input text data for training.
        model_name (str): Pretrained GPT model name from Hugging Face. Default is 'gpt2'.
        max_len (int): Maximum sequence length. Default is 128.
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of training epochs. Default is 3.
        gpu_optimized (str): If 'yes', enables GPU optimization using TensorFlow. Default is 'no'.
        test_split (float): Proportion of data for validation. Default is 0.2.

    Returns:
        dict: Contains the trained GPT model, tokenizer, history, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def encode_texts(texts):
        return tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )

    # Encode data
    encodings = encode_texts(data)
    input_ids = encodings['input_ids']
    attention_masks = encodings['attention_mask']

    # Train-test split
    n_train = int((1 - test_split) * len(input_ids))
    train_data = input_ids[:n_train]
    test_data = input_ids[n_train:]

    # Load pre-trained GPT model
    gpt_model = TFGPT2LMHeadModel.from_pretrained(model_name)

    # Compile the model
    gpt_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Training
    history = gpt_model.fit(
        train_data, train_data,
        validation_data=(test_data, test_data),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluation
    predictions = gpt_model.predict(test_data)
    predictions = np.argmax(predictions.logits, axis=-1)
    accuracy = accuracy_score(test_data.numpy().flatten(), predictions.flatten())
    precision = precision_score(test_data.numpy().flatten(), predictions.flatten(), average='weighted', zero_division=0)
    recall = recall_score(test_data.numpy().flatten(), predictions.flatten(), average='weighted', zero_division=0)
    f1 = f1_score(test_data.numpy().flatten(), predictions.flatten(), average='weighted', zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return {
        "model": gpt_model,
        "tokenizer": tokenizer,
        "history": history,
        "metrics": metrics
    }


"""
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = [
        "Once upon a time, there was a brave knight.",
        "The wizard cast a powerful spell.",
        "In a distant land, a hero was born.",
        "The dragon guarded the treasure for centuries."
    ]

    result = gpt(data, gpu_optimized='yes', epochs=2)
    print("GPT Metrics:", result["metrics"])
"""

# 20.

def gru(
    data,
    labels,
    vocab_size=10000,
    embedding_dim=128,
    max_len=100,
    gru_units=64,
    dropout_rate=0.2,
    batch_size=32,
    epochs=10,
    test_split=0.2,
    gpu_optimized='no'
):
    """
    Builds and trains a GRU model for sequential data tasks with optional GPU optimization.

    Args:
        data (array-like): Input data (e.g., tokenized sequences or feature matrices).
        labels (array-like): Target labels for classification tasks.
        vocab_size (int): Size of the vocabulary for embedding. Default is 10000.
        embedding_dim (int): Dimension of the embedding space. Default is 128.
        max_len (int): Maximum sequence length. Default is 100.
        gru_units (int): Number of GRU units. Default is 64.
        dropout_rate (float): Dropout rate for regularization. Default is 0.2.
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of training epochs. Default is 10.
        test_split (float): Proportion of data for validation. Default is 0.2.
        gpu_optimized (str): If 'yes', utilizes GPU for training. Default is 'no'.

    Returns:
        dict: Contains the trained GRU model, metrics, and training history.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Train-test split
    data = np.array(data, dtype=np.int32)
    labels = np.array(labels, dtype=np.int32)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=test_split, random_state=42)

    # Build GRU model
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        GRU(gru_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Evaluate the model
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return {
        "model": model,
        "metrics": metrics,
        "history": history
    }

"""
# Example usage
if __name__ == "__main__":
    # Example dataset
    data = np.random.randint(1, 10000, size=(1000, 100))  # 1000 samples of sequence length 100
    labels = np.random.randint(0, 2, size=(1000,))  # Binary labels

    result = gru(data, labels, gpu_optimized='yes', epochs=5)
    print("GRU Metrics:", result["metrics"])
"""

# 21.

def bayesian_network(
    data,
    structure,
    labels=None,
    test_split=0.2,
    gpu_optimized='no'
):
    """
    Builds and trains a Bayesian Network for tasks such as causal inference or relationship extraction.

    Args:
        data (array-like): Input data (e.g., pandas DataFrame).
        structure (list of tuples): The structure of the Bayesian Network (edges between variables).
        labels (array-like): Target labels for evaluation, optional for unsupervised tasks.
        test_split (float): Proportion of data for validation. Default is 0.2.
        gpu_optimized (str): If 'yes', utilizes GPU for heavy computations where applicable. Default is 'no'.

    Returns:
        dict: Contains the trained Bayesian Network, inference engine, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Split data into training and testing sets
    n_train = int((1 - test_split) * len(data))
    train_data = data[:n_train]
    test_data = data[n_train:]

    if labels is not None:
        train_labels = labels[:n_train]
        test_labels = labels[n_train:]

    # Build the Bayesian Network
    model = BayesianNetwork(structure)

    # Train the model using Maximum Likelihood Estimation
    model.fit(train_data, estimator=MaximumLikelihoodEstimator)

    # Perform inference
    inference = VariableElimination(model)

    metrics = {}
    if labels is not None:
        # Evaluate the model using predictions
        predictions = []
        for i in range(len(test_data)):
            query_result = inference.map_query(variables=["Label"], evidence=dict(test_data.iloc[i]))
            predictions.append(query_result["Label"])

        # Compute evaluation metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    return {
        "model": model,
        "inference": inference,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Example dataset
    data = pd.DataFrame({
        'A': [0, 0, 1, 1],
        'B': [0, 1, 0, 1],
        'C': [0, 1, 1, 0],
        'Label': [0, 1, 1, 0]
    })

    structure = [('A', 'C'), ('B', 'C'), ('C', 'Label')]

    result = bayesian_network(data, structure, labels=data['Label'], gpu_optimized='yes')
    print("Bayesian Network Metrics:", result["metrics"])
"""

# 22.


def markov_chain(
    transition_matrix,
    states,
    initial_state,
    sequence_length=10,
    gpu_optimized='no',
    true_sequence=None
):
    """
    Builds and simulates a Markov Chain for modeling sequential data and calculates evaluation metrics if a true sequence is provided.

    Args:
        transition_matrix (2D array-like): Transition probabilities between states.
        states (list): List of states in the Markov Chain.
        initial_state (str): Starting state for the Markov Chain.
        sequence_length (int): Number of steps to simulate. Default is 10.
        gpu_optimized (str): If 'yes', utilizes TensorFlow for GPU-accelerated computations. Default is 'no'.
        true_sequence (list, optional): The true sequence of states for calculating metrics. Default is None.

    Returns:
        dict: Contains the simulated sequence and metrics if a true sequence is provided.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Convert transition matrix to TensorFlow tensor if GPU optimization is enabled
    if gpu_optimized == 'yes':
        transition_matrix = tf.convert_to_tensor(transition_matrix, dtype=tf.float32)

    # Mapping of states to indices
    state_to_index = {state: i for i, state in enumerate(states)}
    index_to_state = {i: state for i, state in enumerate(states)}

    # Initialize the state sequence
    current_state = state_to_index[initial_state]
    sequence = [initial_state]

    for _ in range(sequence_length - 1):
        if gpu_optimized == 'yes':
            probabilities = transition_matrix[current_state].numpy()
        else:
            probabilities = transition_matrix[current_state]

        next_state = np.random.choice(len(states), p=probabilities)
        sequence.append(index_to_state[next_state])
        current_state = next_state

    metrics = {}
    if true_sequence:
        # Convert true sequence to indices for comparison
        true_sequence_indices = [state_to_index[state] for state in true_sequence]
        simulated_sequence_indices = [state_to_index[state] for state in sequence]

        # Compute evaluation metrics
        accuracy = accuracy_score(true_sequence_indices, simulated_sequence_indices)
        precision = precision_score(true_sequence_indices, simulated_sequence_indices, average='weighted', zero_division=0)
        recall = recall_score(true_sequence_indices, simulated_sequence_indices, average='weighted', zero_division=0)
        f1 = f1_score(true_sequence_indices, simulated_sequence_indices, average='weighted', zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

    return {
        "sequence": sequence,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example transition matrix and states
    transition_matrix = [
        [0.1, 0.6, 0.3],
        [0.4, 0.4, 0.2],
        [0.2, 0.3, 0.5]
    ]

    states = ["A", "B", "C"]
    initial_state = "A"
    true_sequence = ["A", "B", "C", "B", "A", "C", "C", "B", "A", "B", "C", "A", "C", "B", "A"]

    result = markov_chain(
        transition_matrix,
        states,
        initial_state,
        sequence_length=15,
        gpu_optimized='yes',
        true_sequence=true_sequence
    )

    print("Simulated Markov Chain Sequence:", result["sequence"])
    print("Metrics:", result["metrics"])
"""

# 23.

def isolation_forest(
    data,
    contamination=0.1,
    n_estimators=100,
    max_samples='auto',
    random_state=42,
    gpu_optimized='no'
):
    """
    Builds and trains an Isolation Forest model for anomaly detection tasks, with GPU optimization for heavy computations.

    Args:
        data (array-like): Input data for anomaly detection.
        contamination (float): Proportion of anomalies in the dataset. Default is 0.1.
        n_estimators (int): Number of base estimators in the ensemble. Default is 100.
        max_samples (int or 'auto'): Number of samples to draw from the dataset for each estimator. Default is 'auto'.
        random_state (int): Random seed for reproducibility. Default is 42.
        gpu_optimized (str): If 'yes', utilizes TensorFlow for GPU acceleration. Default is 'no'.

    Returns:
        dict: Contains the trained Isolation Forest model and evaluation metrics if labels are provided.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Ensure data is in numpy format
    data = np.array(data)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1  # Use all available CPU cores
    )

    model.fit(data)

    # Predict anomalies (-1 for outliers, 1 for inliers)
    predictions = model.predict(data)

    # Metrics evaluation (if true labels are available in a semi-supervised setup)
    metrics = {}
    if 'labels' in dir():
        labels = np.array(labels)
        binary_predictions = (predictions == -1).astype(int)  # Convert -1 to 1 for anomalies
        metrics = {
            "precision": precision_score(labels, binary_predictions, zero_division=0),
            "recall": recall_score(labels, binary_predictions, zero_division=0),
            "f1_score": f1_score(labels, binary_predictions, zero_division=0),
            "roc_auc": roc_auc_score(labels, binary_predictions)
        }

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data (normal and anomalous points)
    data = np.random.rand(100, 2)
    anomalies = np.random.rand(10, 2) + 5  # Add some obvious anomalies
    full_data = np.vstack((data, anomalies))

    result = isolation_forest(
        full_data,
        contamination=0.1,
        n_estimators=100,
        gpu_optimized='no'
    )

    print("Predictions:", result["predictions"])
    print("Metrics:", result["metrics"])
"""

# 24.

def dbscan(
    data,
    eps=0.5,
    min_samples=5,
    metric='euclidean',
    gpu_optimized='no'
):
    """
    Builds and trains a DBSCAN model for clustering tasks with optional GPU optimization.

    Args:
        data (np.array): Input data for clustering.
        eps (float): Maximum distance between two samples for them to be considered as in the same neighborhood. Default is 0.5.
        min_samples (int): Number of samples in a neighborhood for a point to be considered a core point. Default is 5.
        metric (str): The metric to use when calculating distance between instances in a feature array. Default is 'euclidean'.
        gpu_optimized (str): If 'yes', uses TensorFlow for GPU-accelerated distance computations. Default is 'no'.

    Returns:
        dict: Contains the DBSCAN model, cluster labels, and evaluation metrics if true labels are provided.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Ensure data is in numpy format
    data = np.array(data)

    # DBSCAN clustering
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    cluster_labels = dbscan_model.fit_predict(data)

    # Metrics evaluation (if ground truth labels are available in a semi-supervised setup)
    metrics = {}
    if 'labels' in dir():  # Assuming labels are defined globally or passed in your environment
        labels = np.array(labels)
        metrics = {
            "precision": precision_score(labels, cluster_labels, average='weighted', zero_division=0),
            "recall": recall_score(labels, cluster_labels, average='weighted', zero_division=0),
            "f1_score": f1_score(labels, cluster_labels, average='weighted', zero_division=0),
            "adjusted_rand_index": adjusted_rand_score(labels, cluster_labels)
        }

    return {
        "model": dbscan_model,
        "cluster_labels": cluster_labels,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data (2D points)
    data = np.random.rand(100, 2)

    result = dbscan(
        data,
        eps=0.3,
        min_samples=5,
        gpu_optimized='yes'
    )

    print("Cluster Labels:", result["cluster_labels"])
    print("Metrics:", result["metrics"])
"""

# 25.

def xgboost_mml(
    data,
    labels,
    task_type="classification",
    gpu_optimized="no",
    test_size=0.2,
    random_state=42,
    **xgb_params
):
    """
    Builds and trains an XGBoost model for classification or regression tasks with optional GPU optimization.

    Args:
        data (np.array): Input feature data.
        labels (np.array): Target labels.
        task_type (str): "classification" or "regression". Default is "classification".
        gpu_optimized (str): If 'yes', uses GPU for training. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.
        **xgb_params: Additional parameters for the XGBoost model.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    # Determine the model type
    if task_type == "classification":
        model = XGBClassifier(
            tree_method="gpu_hist" if gpu_optimized == "yes" else "auto",
            use_label_encoder=False,
            eval_metric="logloss",
            **xgb_params
        )
    elif task_type == "regression":
        model = XGBRegressor(
            tree_method="gpu_hist" if gpu_optimized == "yes" else "auto",
            **xgb_params
        )
    else:
        raise ValueError("Invalid task_type. Use 'classification' or 'regression'.")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Metrics evaluation
    metrics = {}
    if task_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
    elif task_type == "regression":
        metrics = {
            "mean_squared_error": np.mean((y_test - predictions) ** 2),
            "mean_absolute_error": np.mean(np.abs(y_test - predictions)),
        }

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data (classification task)
    data = np.random.rand(1000, 10)
    labels = np.random.randint(0, 2, size=1000)  # Binary classification

    result = xgboost_mml(
        data,
        labels,
        task_type="classification",
        gpu_optimized="yes",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    print("Model Metrics:", result["metrics"])
"""

# 26.

def lightgbm_mml(
    data,
    labels,
    task_type="classification",
    gpu_optimized="no",
    test_size=0.2,
    random_state=42,
    **lgbm_params
):
    """
    Builds and trains a LightGBM model for classification or regression tasks with optional GPU optimization.

    Args:
        data (np.array): Input feature data.
        labels (np.array): Target labels.
        task_type (str): "classification" or "regression". Default is "classification".
        gpu_optimized (str): If 'yes', uses GPU for training. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.
        **lgbm_params: Additional parameters for the LightGBM model.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    # Determine the model type
    if task_type == "classification":
        model = LGBMClassifier(
            device="gpu" if gpu_optimized == "yes" else "cpu",
            **lgbm_params
        )
    elif task_type == "regression":
        model = LGBMRegressor(
            device="gpu" if gpu_optimized == "yes" else "cpu",
            **lgbm_params
        )
    else:
        raise ValueError("Invalid task_type. Use 'classification' or 'regression'.")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Metrics evaluation
    metrics = {}
    if task_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
    elif task_type == "regression":
        metrics = {
            "mean_squared_error": mean_squared_error(y_test, predictions),
            "mean_absolute_error": mean_absolute_error(y_test, predictions),
        }

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
    }


"""
# Example usage
if __name__ == "__main__":
    # Example data (classification task)
    data = np.random.rand(1000, 10)
    labels = np.random.randint(0, 2, size=1000)  # Binary classification

    result = lightgbm_mml(
        data,
        labels,
        task_type="classification",
        gpu_optimized="yes",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )

    print("Model Metrics:", result["metrics"])
"""

# 27.

def catboost_mml(
    data,
    labels,
    task_type="classification",
    gpu_optimized="no",
    test_size=0.2,
    random_state=42,
    **catboost_params
):
    """
    Builds and trains a CatBoost model for classification or regression tasks with optional GPU optimization.

    Args:
        data (np.array): Input feature data.
        labels (np.array): Target labels.
        task_type (str): "classification" or "regression". Default is "classification".
        gpu_optimized (str): If 'yes', uses GPU for training. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Random seed for reproducibility. Default is 42.
        **catboost_params: Additional parameters for the CatBoost model.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=test_size, random_state=random_state
    )

    # Determine the model type
    if task_type == "classification":
        model = CatBoostClassifier(
            task_type="GPU" if gpu_optimized == "yes" else "CPU",
            verbose=0,  # Suppress training output
            **catboost_params
        )
    elif task_type == "regression":
        model = CatBoostRegressor(
            task_type="GPU" if gpu_optimized == "yes" else "CPU",
            verbose=0,  # Suppress training output
            **catboost_params
        )
    else:
        raise ValueError("Invalid task_type. Use 'classification' or 'regression'.")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Metrics evaluation
    metrics = {}
    if task_type == "classification":
        predictions = np.round(predictions)  # Ensure binary predictions for classification
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
    elif task_type == "regression":
        metrics = {
            "mean_squared_error": mean_squared_error(y_test, predictions),
            "mean_absolute_error": mean_absolute_error(y_test, predictions),
        }

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data (classification task)
    data = np.random.rand(1000, 10)
    labels = np.random.randint(0, 2, size=1000)  # Binary classification

    result = catboost_mml(
        data,
        labels,
        task_type="classification",
        gpu_optimized="yes",
        iterations=100,
        depth=6,
        learning_rate=0.1
    )

    print("Model Metrics:", result["metrics"])
"""

# 28.


def capsule_network(
    input_shape,
    n_classes,
    gpu_optimized='no',
    num_primary_capsules=32,
    dim_primary_capsules=8,
    routing_iterations=3,
    learning_rate=0.001,
    epochs=10,
    batch_size=32,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None
):
    """
    Builds and trains a Capsule Network for tasks like pose estimation, gesture recognition, and classification.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (height, width, channels) for images).
        n_classes (int): Number of output classes.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        num_primary_capsules (int): Number of primary capsules. Default is 32.
        dim_primary_capsules (int): Dimensionality of primary capsules. Default is 8.
        routing_iterations (int): Number of routing iterations in the Capsule Network. Default is 3.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Batch size for training. Default is 32.
        X_train (np.array): Training data. Default is None.
        y_train (np.array): Training labels. Default is None.
        X_test (np.array): Test data. Default is None.
        y_test (np.array): Test labels. Default is None.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Define the Capsule Network architecture
    input_layer = Input(shape=input_shape)

    # Convolutional layer
    conv1 = Conv2D(256, (9, 9), strides=1, padding='valid', activation='relu')(input_layer)

    # Primary Capsules
    primary_caps = Conv2D(
        num_primary_capsules * dim_primary_capsules, (9, 9), strides=2, padding='valid', activation='relu'
    )(conv1)
    primary_caps = Reshape((num_primary_capsules, dim_primary_capsules))(primary_caps)

    # Digit Capsules
    digit_caps = Dense(n_classes * dim_primary_capsules, activation='sigmoid')(primary_caps)
    digit_caps = Reshape((n_classes, dim_primary_capsules))(digit_caps)

    # Output layer
    output_layer = Lambda(lambda z: tf.norm(z, axis=-1))(digit_caps)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluation and metrics
    metrics = {}
    if X_test is not None and y_test is not None:
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        metrics = {
            "accuracy": accuracy_score(y_test, predicted_classes),
            "precision": precision_score(y_test, predicted_classes, average='weighted', zero_division=0),
            "recall": recall_score(y_test, predicted_classes, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, predicted_classes, average='weighted', zero_division=0)
        }

    return {
        "model": model,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    input_shape = (28, 28, 1)  # Shape for grayscale images (e.g., MNIST)
    n_classes = 10  # Number of output classes (e.g., digits 0-9)

    # Simulated dataset
    X_train = np.random.rand(1000, 28, 28, 1)
    y_train = np.random.randint(0, 10, size=(1000,))
    X_test = np.random.rand(200, 28, 28, 1)
    y_test = np.random.randint(0, 10, size=(200,))

    # Build and train the Capsule Network
    result = capsule_network(
        input_shape=input_shape,
        n_classes=n_classes,
        gpu_optimized='yes',
        epochs=5,
        batch_size=64,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    print("Metrics:", result["metrics"])
"""

# 29.

def spiking_neural_network(
    input_shape,
    n_classes,
    gpu_optimized='no',
    spike_threshold=0.5,
    learning_rate=0.001,
    epochs=10,
    batch_size=32,
    num_conv_layers=2,
    num_dense_layers=1,
    pool_size=(2, 2),
    conv_filters=[32, 64],  # List of filters in each conv layer
    dense_units=[128],  # List of units in each dense layer
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None
):
    """
    Builds and trains a customizable Spiking Neural Network (SNN) for tasks like gesture recognition and object tracking.

    Args:
        input_shape (tuple): Shape of the input data (e.g., (height, width, channels) for images).
        n_classes (int): Number of output classes.
        gpu_optimized (str): Utilizes GPU for computation if 'yes'. Default is 'no'.
        spike_threshold (float): Threshold for neuron activation. Default is 0.5.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Batch size for training. Default is 32.
        num_conv_layers (int): Number of convolutional layers. Default is 2.
        num_dense_layers (int): Number of dense layers. Default is 1.
        pool_size (tuple): Size of the pooling window. Default is (2, 2).
        conv_filters (list): Number of filters in each convolutional layer.
        dense_units (list): Number of units in each dense layer.
        X_train (np.array): Training data. Default is None.
        y_train (np.array): Training labels. Default is None.
        X_test (np.array): Test data. Default is None.
        y_test (np.array): Test labels. Default is None.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Define the Spiking Neural Network architecture dynamically
    input_layer = Input(shape=input_shape)
    x = input_layer

    # Add convolutional layers dynamically based on input parameters
    for i in range(num_conv_layers):
        x = Conv2D(conv_filters[i], (3, 3), activation='relu', padding='same')(x)
        if i == num_conv_layers - 1:  # Add pooling only after the last conv layer
            x = MaxPooling2D(pool_size=pool_size)(x)

    # Flatten for dense layers
    x = Flatten()(x)

    # Simulating spike behavior with a custom activation function
    x = Lambda(lambda x: tf.cast(x > spike_threshold, tf.float32))(x)

    # Add dense layers dynamically
    for units in dense_units:
        x = Dense(units, activation='relu')(x)

    # Output layer
    output_layer = Dense(n_classes, activation='softmax')(x)

    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Training
    if X_train is not None and y_train is not None:
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluation and metrics
    metrics = {}
    if X_test is not None and y_test is not None:
        predictions = model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        metrics = {
            "accuracy": accuracy_score(y_test, predicted_classes),
            "precision": precision_score(y_test, predicted_classes, average='weighted', zero_division=0),
            "recall": recall_score(y_test, predicted_classes, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, predicted_classes, average='weighted', zero_division=0)
        }

    return {
        "model": model,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    input_shape = (64, 64, 1)  # Shape for grayscale images (e.g., small gesture recognition dataset)
    n_classes = 10  # Number of output classes

    # Simulated dataset
    X_train = np.random.rand(1000, 64, 64, 1)
    y_train = np.random.randint(0, 10, size=(1000,))
    X_test = np.random.rand(200, 64, 64, 1)
    y_test = np.random.randint(0, 10, size=(200,))

    # Build and train the Spiking Neural Network
    result = spiking_neural_network(
        input_shape=input_shape,
        n_classes=n_classes,
        gpu_optimized='yes',
        epochs=5,
        batch_size=64,
        spike_threshold=0.6,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

    print("Metrics:", result["metrics"])
"""

# 30.

def hopfield_network(
    data,
    patterns,
    gpu_optimized='no',
    max_iterations=10,
    threshold=0.5
):
    """
    Builds and evaluates a Hopfield Network for content-addressable memory tasks.

    Args:
        data (np.array): The input data to be recalled.
        patterns (np.array): The stored patterns to train the Hopfield Network.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        max_iterations (int): Maximum number of iterations for convergence. Default is 10.
        threshold (float): Threshold for neuron activation. Default is 0.5.

    Returns:
        dict: Contains the energy state, recalled pattern, and metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")
    else:
        print("Running on CPU.")

    # Initialize weight matrix
    num_neurons = patterns.shape[1]
    weight_matrix = np.zeros((num_neurons, num_neurons))

    # Train the Hopfield Network by storing patterns
    for pattern in patterns:
        weight_matrix += np.outer(pattern, pattern)
    np.fill_diagonal(weight_matrix, 0)  # No self-connections

    # Normalize weight matrix
    weight_matrix /= patterns.shape[0]

    if gpu_optimized == 'yes':
        weight_matrix = tf.convert_to_tensor(weight_matrix, dtype=tf.float32)
        data = tf.convert_to_tensor(data, dtype=tf.float32)

    # Recall process
    def recall(data_point):
        state = data_point.copy()
        for _ in range(max_iterations):
            new_state = np.sign(np.dot(weight_matrix, state))
            new_state[new_state >= threshold] = 1
            new_state[new_state < threshold] = -1
            if np.array_equal(new_state, state):
                break
            state = new_state
        return state

    recalled_patterns = []
    for data_point in data:
        if gpu_optimized == 'yes':
            data_point = tf.convert_to_tensor(data_point, dtype=tf.float32)
            data_point = tf.sign(tf.linalg.matmul(weight_matrix, data_point))
        recalled_patterns.append(recall(data_point.numpy() if gpu_optimized == 'yes' else data_point))

    # Compute metrics
    accuracy = accuracy_score(data, recalled_patterns)
    metrics = {
        'accuracy': accuracy
    }


    return {
        "recalled_patterns": recalled_patterns,
        "metrics": metrics,
        "weight_matrix": weight_matrix.numpy() if gpu_optimized == 'yes' else weight_matrix
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    patterns = np.array([
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
        [1, 1, -1, -1]
    ])
    data = np.array([
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
        [1, 1, -1, -1]
    ])

    # Build and evaluate the Hopfield Network
    result = hopfield_network(
        data=data,
        patterns=patterns,
        gpu_optimized='yes',
        max_iterations=10,
        threshold=0.5
    )

    print("Recalled Patterns:", result["recalled_patterns"])
    print("Accuracy:", result["accuracy"])
"""

# 31.

def gradient_boosting(
    X,
    y,
    gpu_optimized='no',
    test_size=0.2,
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
):
    """
    Builds and evaluates a Gradient Boosting model for tasks like anomaly detection and feature selection.

    Args:
        X (np.array): Input features.
        y (np.array): Target labels.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility. Default is 42.
        n_estimators (int): Number of boosting stages. Default is 100.
        learning_rate (float): Learning rate for updates. Default is 0.1.
        max_depth (int): Maximum depth of the individual estimators. Default is 3.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled.")
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('GPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    else:
        print("Running on CPU.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create Gradient Boosting model
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return {
        "model": model,
        "metrics": metrics,
        "predictions": y_pred
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification

    # Build and evaluate Gradient Boosting model
    result = gradient_boosting(
        X=X,
        y=y,
        gpu_optimized='yes',
        test_size=0.2,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4
    )

    print("Metrics:", result["metrics"])
"""

# 32.

def build_elastic_net(
    X,
    y,
    gpu_optimized='no',
    test_size=0.2,
    random_state=42,
    alpha=1.0,
    l1_ratio=0.5,
    max_iter=1000
):
    """
    Builds and evaluates an Elastic Net model for regression tasks.

    Args:
        X (np.array): Input features.
        y (np.array): Target labels.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility. Default is 42.
        alpha (float): Regularization strength. Default is 1.0.
        l1_ratio (float): The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. Default is 0.5.
        max_iter (int): Maximum number of iterations. Default is 1000.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled.")
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('GPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    else:
        print("Running on CPU.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create ElasticNet model
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
    }

    return {
        "model": model,
        "metrics": metrics,
        "predictions": y_pred
    }


"""
# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.rand(1000)  # Regression target

    # Build and evaluate Elastic Net model
    result = elastic_net(
        X=X,
        y=y,
        gpu_optimized='yes',
        test_size=0.2,
        alpha=0.1,
        l1_ratio=0.7,
        max_iter=500
    )

    print("Metrics:", result["metrics"])
"""

# 33.

def ridge_regression(
    X,
    y,
    gpu_optimized='no',
    test_size=0.2,
    random_state=42,
    alpha=1.0
):
    """
    Builds and evaluates a Ridge Regression model for regression tasks.

    Args:
        X (np.array): Input features.
        y (np.array): Target labels.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility. Default is 42.
        alpha (float): Regularization strength. Default is 1.0.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled.")
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('GPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    else:
        print("Running on CPU.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create Ridge Regression model
    model = Ridge(alpha=alpha, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
    }

    return {
        "model": model,
        "metrics": metrics,
        "predictions": y_pred
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.rand(1000)  # Regression target

    # Build and evaluate Ridge Regression model
    result = ridge_regression(
        X=X,
        y=y,
        gpu_optimized='yes',
        test_size=0.2,
        alpha=0.5
    )

    print("Metrics:", result["metrics"])
"""

# 34.

def lasso_regression(
    X,
    y,
    gpu_optimized='no',
    test_size=0.2,
    random_state=42,
    alpha=1.0,
    max_iter=1000
):
    """
    Builds and evaluates a Lasso Regression model for regression tasks.

    Args:
        X (np.array): Input features.
        y (np.array): Target labels.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility. Default is 42.
        alpha (float): Regularization strength. Default is 1.0.
        max_iter (int): Maximum number of iterations. Default is 1000.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled.")
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('GPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    else:
        print("Running on CPU.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create Lasso Regression model
    model = Lasso(alpha=alpha, max_iter=max_iter, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "mean_squared_error": mean_squared_error(y_test, y_pred),
        "r2_score": r2_score(y_test, y_pred)
    }

    return {
        "model": model,
        "metrics": metrics,
        "predictions": y_pred
    }


"""
# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.rand(1000)  # Regression target

    # Build and evaluate Lasso Regression model
    result = lasso_regression(
        X=X,
        y=y,
        gpu_optimized='yes',
        test_size=0.2,
        alpha=0.5,
        max_iter=500
    )

    print("Metrics:", result["metrics"])
"""

# 35.

def gaussian_naive_bayes(
    X,
    y,
    gpu_optimized='no',
    test_size=0.2,
    random_state=42
):
    """
    Builds and evaluates a Gaussian Naive Bayes model for classification tasks.

    Args:
        X (np.array): Input features.
        y (np.array): Target labels.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility. Default is 42.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled.")
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('GPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    else:
        print("Running on CPU.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create Gaussian Naive Bayes model
    model = GaussianNB()

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return {
        "model": model,
        "metrics": metrics,
        "predictions": y_pred
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification

    # Build and evaluate Gaussian Naive Bayes model
    result = gaussian_naive_bayes(
        X=X,
        y=y,
        gpu_optimized='yes',
        test_size=0.2
    )

    print("Metrics:", result["metrics"])
"""

# 36.

def linear_perceptron(
    X,
    y,
    gpu_optimized='no',
    test_size=0.2,
    random_state=42,
    max_iter=1000,
    eta0=1.0
):
    """
    Builds and evaluates a Linear Perceptron model for binary classification tasks.

    Args:
        X (np.array): Input features.
        y (np.array): Target labels.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility. Default is 42.
        max_iter (int): Maximum number of iterations. Default is 1000.
        eta0 (float): Learning rate. Default is 1.0.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled.")
        tf.config.set_logical_device_configuration(
            tf.config.list_physical_devices('GPU')[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]
        )
    else:
        print("Running on CPU.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create Linear Perceptron model
    model = Perceptron(max_iter=max_iter, eta0=eta0, random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return {
        "model": model,
        "metrics": metrics,
        "predictions": y_pred
    }


"""
# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y = np.random.randint(0, 2, 1000)  # Binary classification

    # Build and evaluate Linear Perceptron model
    result = linear_perceptron(
        X=X,
        y=y,
        gpu_optimized='yes',
        test_size=0.2,
        max_iter=500,
        eta0=0.01
    )

    print("Metrics:", result["metrics"])
"""

# 37.

def neural_collaborative_filtering(
    user_input,
    item_input,
    ratings,
    gpu_optimized='no',
    embedding_dim=50,
    hidden_units=[128, 64],
    activation='relu',
    dropout_rate=0.2,
    learning_rate=0.001,
    epochs=10,
    batch_size=32,
    test_size=0.2,
    random_state=42
):
    """
    Builds and evaluates a Neural Collaborative Filtering model for recommendation tasks.

    Args:
        user_input (np.array): User IDs.
        item_input (np.array): Item IDs.
        ratings (np.array): Corresponding ratings.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        embedding_dim (int): Dimensionality of user and item embeddings. Default is 50.
        hidden_units (list): List of hidden layer sizes. Default is [128, 64].
        activation (str): Activation function for hidden layers. Default is 'relu'.
        dropout_rate (float): Dropout rate for regularization. Default is 0.2.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        epochs (int): Number of training epochs. Default is 10.
        batch_size (int): Batch size for training. Default is 32.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility. Default is 42.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled.")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Running on CPU.")
    else:
        print("Running on CPU.")

    # Split the data
    user_train, user_test, item_train, item_test, ratings_train, ratings_test = train_test_split(
        user_input, item_input, ratings, test_size=test_size, random_state=random_state
    )

    # Define input layers
    user_input_layer = Input(shape=(1,), name='user_input')
    item_input_layer = Input(shape=(1,), name='item_input')

    # Embedding layers
    user_embedding = Embedding(input_dim=np.max(user_input) + 1, output_dim=embedding_dim, name='user_embedding')(user_input_layer)
    item_embedding = Embedding(input_dim=np.max(item_input) + 1, output_dim=embedding_dim, name='item_embedding')(item_input_layer)

    # Flatten embeddings
    user_flatten = Flatten()(user_embedding)
    item_flatten = Flatten()(item_embedding)

    # Concatenate embeddings
    concatenated = Concatenate()([user_flatten, item_flatten])

    # Hidden layers
    x = concatenated
    for units in hidden_units:
        x = Dense(units, activation=activation)(x)
        x = Dropout(dropout_rate)(x)

    # Output layer
    output = Dense(1, activation='linear', name='rating_output')(x)

    # Build the model
    model = Model(inputs=[user_input_layer, item_input_layer], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')

    # Train the model
    model.fit(
        [user_train, item_train], ratings_train,
        validation_data=([user_test, item_test], ratings_test),
        epochs=epochs,
        batch_size=batch_size
    )

    # Predictions
    predictions = model.predict([user_test, item_test]).flatten()

    # Compute metrics
    mse = mean_squared_error(ratings_test, predictions)

    return {
        "model": model,
        "predictions": predictions,
        "metrics": {"mean_squared_error": mse}
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    n_users = 1000
    n_items = 500
    user_input = np.random.randint(0, n_users, size=10000)
    item_input = np.random.randint(0, n_items, size=10000)
    ratings = np.random.uniform(1, 5, size=10000)

    # Build and evaluate Neural Collaborative Filtering model
    result = neural_collaborative_filtering(
        user_input=user_input,
        item_input=item_input,
        ratings=ratings,
        gpu_optimized='yes',
        embedding_dim=64,
        hidden_units=[128, 64, 32],
        learning_rate=0.001,
        epochs=5
    )

    print("Metrics:", result["metrics"])
"""

# 38.

class RestrictedBoltzmannMachine:
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, epochs=10):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.normal(0, 0.01, (self.n_visible, self.n_hidden))
        self.hidden_bias = np.zeros(self.n_hidden)
        self.visible_bias = np.zeros(self.n_visible)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def train(self, data):
        for epoch in range(self.epochs):
            for sample in data:
                v0 = sample
                h0 = self.sigmoid(np.dot(v0, self.weights) + self.hidden_bias)
                h0_states = (h0 > np.random.rand(self.n_hidden)).astype(float)
                v1 = self.sigmoid(np.dot(h0_states, self.weights.T) + self.visible_bias)
                h1 = self.sigmoid(np.dot(v1, self.weights) + self.hidden_bias)

                self.weights += self.learning_rate * (
                    np.outer(v0, h0) - np.outer(v1, h1)
                )
                self.visible_bias += self.learning_rate * (v0 - v1)
                self.hidden_bias += self.learning_rate * (h0 - h1)

class DeepBeliefNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=10):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.rbms = [
            RestrictedBoltzmannMachine(layer_sizes[i], layer_sizes[i + 1], learning_rate, epochs)
            for i in range(len(layer_sizes) - 1)
        ]

    def pretrain(self, data):
        for i, rbm in enumerate(self.rbms):
            print(f"Training RBM {i + 1}/{len(self.rbms)}")
            rbm.train(data)
            data = self.sigmoid(np.dot(data, rbm.weights) + rbm.hidden_bias)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def fine_tune(self, X_train, y_train, epochs, batch_size):
        model = Sequential()
        model.add(InputLayer(input_shape=(self.layer_sizes[0],)))
        for rbm in self.rbms:
            model.add(Dense(rbm.n_hidden, activation='sigmoid', weights=[rbm.weights, rbm.hidden_bias]))
        model.add(Dense(len(np.unique(y_train)), activation='softmax'))
        model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return model


def deep_belief_network(
    X,
    y,
    layer_sizes,
    gpu_optimized='no',
    pretrain_epochs=10,
    fine_tune_epochs=10,
    batch_size=32,
    test_size=0.2,
    learning_rate=0.01,
    random_state=42
):
    """
    Builds and evaluates a Deep Belief Network for tasks like dimensionality reduction and signal classification.

    Args:
        X (np.array): Input features.
        y (np.array): Target labels.
        layer_sizes (list): List of layer sizes for the DBN.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        pretrain_epochs (int): Number of pretraining epochs for RBMs. Default is 10.
        fine_tune_epochs (int): Number of fine-tuning epochs for the entire DBN. Default is 10.
        batch_size (int): Batch size for training. Default is 32.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        learning_rate (float): Learning rate for training. Default is 0.01.
        random_state (int): Seed for reproducibility. Default is 42.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        print("GPU optimization enabled.")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Running on CPU.")
    else:
        print("Running on CPU.")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Build the DBN
    dbn = DeepBeliefNetwork(layer_sizes=layer_sizes, learning_rate=learning_rate, epochs=pretrain_epochs)

    # Pretrain the DBN
    dbn.pretrain(X_train)

    # Fine-tune the DBN
    model = dbn.fine_tune(X_train, y_train, epochs=fine_tune_epochs, batch_size=batch_size)

    # Evaluate the model
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    metrics = {
        "accuracy": accuracy_score(y_test, predicted_classes),
        "mean_squared_error": mean_squared_error(y_test, predicted_classes)
    }

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(1000, 100)  # 1000 samples, 100 features
    y = np.random.randint(0, 10, 1000)  # Multiclass classification with 10 classes

    # Build and evaluate Deep Belief Network
    result = deep_belief_network(
        X=X,
        y=y,
        layer_sizes=[100, 64, 32],
        gpu_optimized='yes',
        pretrain_epochs=5,
        fine_tune_epochs=10
    )

    print("Metrics:", result["metrics"])
"""


# 39.

def gradient_boosted_decision_trees(
    X,
    y,
    task_type='classification',
    gpu_optimized='no',
    test_size=0.2,
    random_state=42,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8
):
    """
    Builds and evaluates a Gradient Boosted Decision Trees (GBDT) model for tasks like classification and regression.

    Args:
        X (np.array): Input features.
        y (np.array): Target labels.
        task_type (str): Type of task, 'classification' or 'regression'.
        gpu_optimized (str): If 'yes', utilizes GPU for computation. Default is 'no'.
        test_size (float): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int): Seed for reproducibility. Default is 42.
        n_estimators (int): Number of boosting rounds. Default is 100.
        learning_rate (float): Learning rate for boosting. Default is 0.1.
        max_depth (int): Maximum depth of trees. Default is 3.
        subsample (float): Fraction of samples used for training each tree. Default is 0.8.
        colsample_bytree (float): Fraction of features used for training each tree. Default is 0.8.

    Returns:
        dict: Contains the trained model, predictions, and evaluation metrics.
    """
    # GPU Optimization setup
    tree_method = 'gpu_hist' if gpu_optimized == 'yes' else 'auto'

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Choose model type
    if task_type == 'classification':
        model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            random_state=random_state
        )
    elif task_type == 'regression':
        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method=tree_method,
            random_state=random_state
        )
    else:
        raise ValueError("Invalid task_type. Choose 'classification' or 'regression'.")

    # Train the model
    model.fit(X_train, y_train)

    # Predictions
    predictions = model.predict(X_test)

    # Metrics
    metrics = {}
    if task_type == 'classification':
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
            "recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, predictions, average='weighted', zero_division=0)
        }
    elif task_type == 'regression':
        mse = np.mean((y_test - predictions) ** 2)
        metrics = {"mean_squared_error": mse}

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data
    X = np.random.rand(1000, 10)  # 1000 samples, 10 features
    y_classification = np.random.randint(0, 2, 1000)  # Binary classification
    y_regression = np.random.rand(1000)  # Regression task

    # Classification example
    classification_result = build_gradient_boosted_decision_trees(
        X=X,
        y=y_classification,
        task_type='classification',
        gpu_optimized='yes'
    )
    print("Classification Metrics:", classification_result["metrics"])

    # Regression example
    regression_result = gradient_boosted_decision_trees(
        X=X,
        y=y_regression,
        task_type='regression',
        gpu_optimized='yes'
    )
    print("Regression Metrics:", regression_result["metrics"])
"""


# 40.

class MarkovRandomField:
    def __init__(self, num_nodes, num_states, gpu_optimized='no'):
        """
        Initializes the Markov Random Field.

        Args:
            num_nodes (int): Number of nodes in the graph.
            num_states (int): Number of states each node can take.
            gpu_optimized (str): If 'yes', utilizes GPU for computations. Default is 'no'.
        """
        self.num_nodes = num_nodes
        self.num_states = num_states
        self.gpu_optimized = gpu_optimized
        self.pairwise_potentials = np.random.rand(num_states, num_states)
        self.node_potentials = np.random.rand(num_nodes, num_states)

        if self.gpu_optimized == 'yes':
            self.pairwise_potentials = tf.convert_to_tensor(self.pairwise_potentials, dtype=tf.float32)
            self.node_potentials = tf.convert_to_tensor(self.node_potentials, dtype=tf.float32)

    def infer(self, max_iterations=10):
        """
        Performs inference on the MRF using belief propagation.

        Args:
            max_iterations (int): Number of iterations for belief propagation.

        Returns:
            np.array or tf.Tensor: Marginal probabilities for each node.
        """
        messages = np.random.rand(self.num_nodes, self.num_states)

        if self.gpu_optimized == 'yes':
            messages = tf.convert_to_tensor(messages, dtype=tf.float32)

        for iteration in range(max_iterations):
            new_messages = []

            for i in range(self.num_nodes):
                incoming_messages = np.sum(messages, axis=0) - messages[i]

                if self.gpu_optimized == 'yes':
                    incoming_messages = tf.reduce_sum(messages, axis=0) - messages[i]

                updated_message = self.node_potentials[i] + incoming_messages
                new_messages.append(updated_message)

            if self.gpu_optimized == 'yes':
                messages = tf.stack(new_messages)
            else:
                messages = np.array(new_messages)

        marginals = self.node_potentials + np.sum(messages, axis=0)

        if self.gpu_optimized == 'yes':
            marginals = tf.nn.softmax(marginals, axis=1)
        else:
            marginals = np.exp(marginals) / np.sum(np.exp(marginals), axis=1, keepdims=True)

        return marginals

    def evaluate(self, true_labels, predicted_labels):
        """
        Evaluates the model's predictions.

        Args:
            true_labels (np.array): Ground truth labels.
            predicted_labels (np.array): Predicted labels from the MRF.

        Returns:
            dict: Evaluation metrics.
        """
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        return {
            "accuracy": accuracy,
            "f1_score": f1
        }

def markov_random_field(num_nodes, num_states, gpu_optimized='no', max_iterations=10, true_labels=None):
    """
    Wrapper function for Markov Random Field (MRF) to perform inference and evaluation.

    Args:
        num_nodes (int): Number of nodes in the graph.
        num_states (int): Number of states each node can take.
        gpu_optimized (str): If 'yes', utilizes GPU for computations. Default is 'no'.
        max_iterations (int): Number of iterations for belief propagation. Default is 10.
        true_labels (np.array): Ground truth labels for evaluation. Default is None.

    Returns:
        dict: Contains marginals, predicted labels, and evaluation metrics.
    """
    # Initialize MRF model
    mrf = MarkovRandomField(num_nodes=num_nodes, num_states=num_states, gpu_optimized=gpu_optimized)

    # Perform inference
    marginals = mrf.infer(max_iterations=max_iterations)

    predicted_labels = np.argmax(marginals, axis=1) if isinstance(marginals, np.ndarray) else tf.argmax(marginals, axis=1).numpy()

    # Evaluation metrics
    metrics = {}
    if true_labels is not None:
        metrics = mrf.evaluate(true_labels, predicted_labels)

    return {
        "model": mrf,
        "marginals": marginals,
        "predicted_labels": predicted_labels,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    num_nodes = 10
    num_states = 5
    true_labels = np.random.randint(0, num_states, num_nodes)

    result = markov_random_field(num_nodes=num_nodes, num_states=num_states, gpu_optimized='yes', max_iterations=20, true_labels=true_labels)

    print("Marginals:", result["marginals"])
    print("Predicted Labels:", result["predicted_labels"])
    print("Evaluation Metrics:", result["metrics"])
"""

# 41.

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim, gpu_optimized='no'):
        """
        Initializes a Kalman Filter.

        Args:
            state_dim (int): Dimensionality of the state vector.
            measurement_dim (int): Dimensionality of the measurement vector.
            gpu_optimized (str): If 'yes', utilizes GPU for computations. Default is 'no'.
        """
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.gpu_optimized = gpu_optimized

        # Initialize state and covariance
        self.state = np.zeros((state_dim, 1))
        self.covariance = np.eye(state_dim)

        # Initialize Kalman Filter matrices
        self.transition_matrix = np.eye(state_dim)
        self.control_matrix = np.zeros((state_dim, state_dim))
        self.measurement_matrix = np.eye(measurement_dim, state_dim)
        self.process_noise = np.eye(state_dim)
        self.measurement_noise = np.eye(measurement_dim)

        if gpu_optimized == 'yes':
            self.state = tf.convert_to_tensor(self.state, dtype=tf.float32)
            self.covariance = tf.convert_to_tensor(self.covariance, dtype=tf.float32)

    def predict(self, control_input=None):
        """
        Performs the prediction step of the Kalman Filter.

        Args:
            control_input (np.array): Optional control input vector.

        Returns:
            tuple: Predicted state and covariance.
        """
        if control_input is None:
            control_input = np.zeros((self.state_dim, 1))

        if self.gpu_optimized == 'yes':
            control_input = tf.convert_to_tensor(control_input, dtype=tf.float32)
            self.state = tf.matmul(self.transition_matrix, self.state) + tf.matmul(self.control_matrix, control_input)
            self.covariance = tf.matmul(tf.matmul(self.transition_matrix, self.covariance), tf.transpose(self.transition_matrix)) + self.process_noise
        else:
            self.state = np.dot(self.transition_matrix, self.state) + np.dot(self.control_matrix, control_input)
            self.covariance = np.dot(np.dot(self.transition_matrix, self.covariance), self.transition_matrix.T) + self.process_noise

        return self.state, self.covariance

    def update(self, measurement):
        """
        Performs the update step of the Kalman Filter.

        Args:
            measurement (np.array): Measurement vector.

        Returns:
            tuple: Updated state and covariance.
        """
        if self.gpu_optimized == 'yes':
            measurement = tf.convert_to_tensor(measurement, dtype=tf.float32)
            innovation = measurement - tf.matmul(self.measurement_matrix, self.state)
            innovation_covariance = tf.matmul(self.measurement_matrix, tf.matmul(self.covariance, tf.transpose(self.measurement_matrix))) + self.measurement_noise
            kalman_gain = tf.matmul(tf.matmul(self.covariance, tf.transpose(self.measurement_matrix)), tf.linalg.inv(innovation_covariance))
            self.state = self.state + tf.matmul(kalman_gain, innovation)
            self.covariance = tf.matmul(tf.eye(self.state_dim) - tf.matmul(kalman_gain, self.measurement_matrix), self.covariance)
        else:
            innovation = measurement - np.dot(self.measurement_matrix, self.state)
            innovation_covariance = np.dot(self.measurement_matrix, np.dot(self.covariance, self.measurement_matrix.T)) + self.measurement_noise
            kalman_gain = np.dot(np.dot(self.covariance, self.measurement_matrix.T), np.linalg.inv(innovation_covariance))
            self.state = self.state + np.dot(kalman_gain, innovation)
            self.covariance = np.dot(np.eye(self.state_dim) - np.dot(kalman_gain, self.measurement_matrix), self.covariance)

        return self.state, self.covariance


def kalman_filter(state_dim, measurement_dim, gpu_optimized='no', num_steps=10, true_states=None, measurements=None):
    """
    Wrapper function to utilize the Kalman Filter for a specific task.

    Args:
        state_dim (int): Dimensionality of the state vector.
        measurement_dim (int): Dimensionality of the measurement vector.
        gpu_optimized (str): If 'yes', utilizes GPU for computations. Default is 'no'.
        num_steps (int): Number of time steps for prediction and update.
        true_states (np.array): Ground truth states for evaluation. Default is None.
        measurements (np.array): Observed measurements. Default is None.

    Returns:
        dict: Contains predicted states, metrics, and the final Kalman Filter object.
    """
    kf = KalmanFilter(state_dim=state_dim, measurement_dim=measurement_dim, gpu_optimized=gpu_optimized)

    predicted_states = []
    if true_states is not None:
        mse_scores = []

    for t in range(num_steps):
        control_input = None  # Optional control input can be added here

        # Prediction step
        state, _ = kf.predict(control_input)

        # Update step if measurements are provided
        if measurements is not None:
            measurement = measurements[t].reshape(-1, 1)
            state, _ = kf.update(measurement)

        predicted_states.append(state)

        # Evaluate if true states are provided
        if true_states is not None:
            mse_scores.append(mean_squared_error(true_states[t], state))

    metrics = {}
    if true_states is not None:
        metrics = {
            "mean_mse": np.mean(mse_scores)
        }

    return {
        "predicted_states": predicted_states,
        "metrics": metrics,
        "model": kf
    }


"""
# Example usage
if __name__ == "__main__":
    state_dim = 4
    measurement_dim = 2
    num_steps = 10

    true_states = [np.random.rand(state_dim, 1) for _ in range(num_steps)]
    measurements = [np.random.rand(measurement_dim, 1) for _ in range(num_steps)]

    result = kalman_filter(state_dim=state_dim, measurement_dim=measurement_dim, gpu_optimized='yes', num_steps=num_steps, true_states=true_states, measurements=measurements)

    print("Predicted States:", result["predicted_states"])
    print("Metrics:", result["metrics"])
"""

# 42.

class VisionTransformer:
    def __init__(self, image_size, patch_size, num_classes, 
                 d_model, num_heads, num_layers, mlp_dim, 
                 dropout_rate=0.1, gpu_optimized='no'):
        """
        Initializes the Vision Transformer (ViT).

        Args:
            image_size (int): Size of the input image (assumes square images).
            patch_size (int): Size of each patch (assumes square patches).
            num_classes (int): Number of output classes.
            d_model (int): Dimensionality of the transformer model.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer layers.
            mlp_dim (int): Dimensionality of the MLP head.
            dropout_rate (float): Dropout rate. Default is 0.1.
            gpu_optimized (str): If 'yes', enables GPU optimization. Default is 'no'.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.gpu_optimized = gpu_optimized

        if gpu_optimized == 'yes':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")

    def build_model(self):
        """Builds the Vision Transformer model."""
        num_patches = (self.image_size // self.patch_size) ** 2

        inputs = Input(shape=(self.image_size, self.image_size, 3))

        # Create patches
        patches = Conv2D(filters=self.d_model, kernel_size=self.patch_size, strides=self.patch_size)(inputs)
        patches = Reshape((num_patches, self.d_model))(patches)

        # Positional encoding
        pos_encoding = tf.Variable(tf.random.normal([1, num_patches, self.d_model]), trainable=True)
        x = patches + pos_encoding

        # Transformer layers
        for _ in range(self.num_layers):
            x = self._transformer_block(x)

        # Classification head
        x = GlobalAveragePooling1D()(x)
        x = Dense(self.mlp_dim, activation='relu')(x)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        return Model(inputs=inputs, outputs=outputs)

    def _transformer_block(self, x):
        """Creates a single transformer block."""
        attn_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.d_model)(x, x)
        attn_output = Dropout(self.dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn_output = Dense(self.mlp_dim, activation='relu')(out1)
        ffn_output = Dense(self.d_model)(ffn_output)
        ffn_output = Dropout(self.dropout_rate)(ffn_output)
        return LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

def vision_transformer(image_size, patch_size, num_classes, 
                            d_model, num_heads, num_layers, mlp_dim, 
                            dropout_rate=0.1, gpu_optimized='no', 
                            train_data=None, val_data=None, epochs=10):
    """
    Wrapper function to train and evaluate the Vision Transformer model.

    Args:
        image_size (int): Size of the input image (assumes square images).
        patch_size (int): Size of each patch (assumes square patches).
        num_classes (int): Number of output classes.
        d_model (int): Dimensionality of the transformer model.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        mlp_dim (int): Dimensionality of the MLP head.
        dropout_rate (float): Dropout rate. Default is 0.1.
        gpu_optimized (str): If 'yes', enables GPU optimization. Default is 'no'.
        train_data (tf.data.Dataset): Training dataset.
        val_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs. Default is 10.

    Returns:
        dict: Contains the trained model, metrics, and predictions on the validation set.
    """
    vit = VisionTransformer(image_size, patch_size, num_classes, d_model, num_heads, num_layers, mlp_dim, dropout_rate, gpu_optimized)
    model = vit.build_model()
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_data, validation_data=val_data, epochs=epochs)

    # Evaluate
    val_loss, val_accuracy = model.evaluate(val_data)

    # Collect predictions
    predictions = model.predict(val_data)

    metrics = {
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy
    }

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions
    }

"""
# Example usage
if __name__ == "__main__":
    image_size = 224
    patch_size = 16
    num_classes = 10
    d_model = 64
    num_heads = 4
    num_layers = 6
    mlp_dim = 128

    # Placeholder datasets (replace with actual datasets)
    train_data = tf.data.Dataset.from_tensor_slices((np.random.rand(100, image_size, image_size, 3), np.random.randint(0, num_classes, 100)))
    train_data = train_data.batch(16)
    val_data = tf.data.Dataset.from_tensor_slices((np.random.rand(20, image_size, image_size, 3), np.random.randint(0, num_classes, 20)))
    val_data = val_data.batch(16)

    result = vision_transformer(image_size=image_size, patch_size=patch_size, num_classes=num_classes,
                                     d_model=d_model, num_heads=num_heads, num_layers=num_layers, 
                                     mlp_dim=mlp_dim, dropout_rate=0.1, gpu_optimized='yes', 
                                     train_data=train_data, val_data=val_data, epochs=5)

    print("Metrics:", result["metrics"])
"""


# 43.

class ContrastiveLearning:
    def __init__(self, feature_dim, temperature=0.5, gpu_optimized='no'):
        """
        Initializes the Contrastive Learning framework.

        Args:
            feature_dim (int): Dimensionality of the feature embedding.
            temperature (float): Temperature scaling for contrastive loss. Default is 0.5.
            gpu_optimized (str): If 'yes', enables GPU optimization. Default is 'no'.
        """
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.gpu_optimized = gpu_optimized

        if gpu_optimized == 'yes':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")

    def build_model(self, input_shape):
        """Builds the contrastive learning model."""
        inputs = Input(shape=input_shape)

        # Encoder network
        x = Conv2D(64, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(self.feature_dim)(x)

        # Normalize the feature embeddings
        outputs = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

        return Model(inputs=inputs, outputs=outputs)

    def contrastive_loss(self, z_i, z_j):
        """Computes the contrastive loss between two feature embeddings."""
        batch_size = tf.shape(z_i)[0]

        # Concatenate positive pairs
        z = tf.concat([z_i, z_j], axis=0)

        # Similarity scores
        sim_matrix = tf.matmul(z, z, transpose_b=True)
        sim_matrix = tf.exp(sim_matrix / self.temperature)

        # Mask to remove self-similarity
        mask = tf.eye(2 * batch_size)
        mask = 1 - mask

        # Compute loss
        positive_pairs = tf.concat([tf.range(batch_size), batch_size + tf.range(batch_size)], axis=0)
        sim_pos = tf.gather_nd(sim_matrix, tf.stack([positive_pairs, tf.range(2 * batch_size)], axis=1))
        sim_sum = tf.reduce_sum(sim_matrix * mask, axis=1)

        loss = -tf.reduce_mean(tf.math.log(sim_pos / sim_sum))
        return loss

def contrastive_learning(feature_dim, temperature, input_shape, gpu_optimized='no',
                               train_data=None, val_data=None, epochs=10):
    """
    Wrapper function to train and evaluate a Contrastive Learning model.

    Args:
        feature_dim (int): Dimensionality of the feature embedding.
        temperature (float): Temperature scaling for contrastive loss.
        input_shape (tuple): Shape of the input data.
        gpu_optimized (str): If 'yes', enables GPU optimization. Default is 'no'.
        train_data (tf.data.Dataset): Training dataset.
        val_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs. Default is 10.

    Returns:
        dict: Contains the trained model and metrics.
    """
    clf = ContrastiveLearning(feature_dim, temperature, gpu_optimized)
    model = clf.build_model(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training loop
    for epoch in range(epochs):
        for (x_i, x_j) in train_data:
            with tf.GradientTape() as tape:
                z_i = model(x_i, training=True)
                z_j = model(x_j, training=True)
                loss = clf.contrastive_loss(z_i, z_j)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

    # Evaluate
    metrics = {}
    if val_data is not None:
        all_embeddings = []
        all_labels = []
        for (x, y) in val_data:
            embeddings = model(x, training=False)
            all_embeddings.append(embeddings.numpy())
            all_labels.append(y.numpy())

        # Example metrics
        metrics['embedding_variance'] = np.var(np.concatenate(all_embeddings, axis=0))

    return {
        "model": model,
        "metrics": metrics
    }


"""
# Example usage
if __name__ == "__main__":
    feature_dim = 128
    temperature = 0.5
    input_shape = (64, 64, 3)

    # Placeholder datasets (replace with actual datasets)
    x_train = np.random.rand(100, 64, 64, 3)
    x_train_pairs = (x_train[:50], x_train[50:])  # Example positive pairs
    train_data = tf.data.Dataset.from_tensor_slices(x_train_pairs).batch(16)

    val_data = tf.data.Dataset.from_tensor_slices((x_train, np.random.randint(0, 10, 100))).batch(16)

    result = contrastive_learning(feature_dim=feature_dim, temperature=temperature,
                                       input_shape=input_shape, gpu_optimized='yes',
                                       train_data=train_data, val_data=val_data, epochs=5)

    print("Metrics:", result["metrics"])
"""


# 44.

class SimCLR:
    def __init__(self, feature_dim, temperature=0.5, gpu_optimized='no'):
        """
        Initializes the SimCLR framework.

        Args:
            feature_dim (int): Dimensionality of the feature embedding.
            temperature (float): Temperature scaling for contrastive loss. Default is 0.5.
            gpu_optimized (str): If 'yes', enables GPU optimization. Default is 'no'.
        """
        self.feature_dim = feature_dim
        self.temperature = temperature
        self.gpu_optimized = gpu_optimized

        if gpu_optimized == 'yes':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")

    def build_encoder(self, input_shape):
        """Builds the encoder network."""
        inputs = Input(shape=input_shape)

        # Encoder network
        x = Conv2D(64, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(self.feature_dim)(x)

        # Normalize the feature embeddings
        outputs = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)

        return Model(inputs=inputs, outputs=outputs)

    def contrastive_loss(self, z_i, z_j):
        """Computes the contrastive loss between two feature embeddings."""
        batch_size = tf.shape(z_i)[0]

        # Concatenate positive pairs
        z = tf.concat([z_i, z_j], axis=0)

        # Similarity scores
        sim_matrix = tf.matmul(z, z, transpose_b=True)
        sim_matrix = tf.exp(sim_matrix / self.temperature)

        # Mask to remove self-similarity
        mask = tf.eye(2 * batch_size)
        mask = 1 - mask

        # Compute loss
        positive_pairs = tf.range(batch_size)
        sim_pos = tf.gather_nd(sim_matrix, tf.stack([positive_pairs, positive_pairs + batch_size], axis=1))
        sim_sum = tf.reduce_sum(sim_matrix * mask, axis=1)

        loss = -tf.reduce_mean(tf.math.log(sim_pos / sim_sum))
        return loss


def simclr(feature_dim, temperature, input_shape, gpu_optimized='no', train_data=None, val_data=None, epochs=10):
    """
    Wrapper function to train and evaluate a SimCLR model.

    Args:
        feature_dim (int): Dimensionality of the feature embedding.
        temperature (float): Temperature scaling for contrastive loss.
        input_shape (tuple): Shape of the input data.
        gpu_optimized (str): If 'yes', enables GPU optimization. Default is 'no'.
        train_data (tf.data.Dataset): Training dataset.
        val_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs. Default is 10.

    Returns:
        dict: Contains the trained model and metrics.
    """
    simclr = SimCLR(feature_dim, temperature, gpu_optimized)
    encoder = simclr.build_encoder(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training loop
    for epoch in range(epochs):
        for (x_i, x_j) in train_data:
            with tf.GradientTape() as tape:
                z_i = encoder(x_i, training=True)
                z_j = encoder(x_j, training=True)
                loss = simclr.contrastive_loss(z_i, z_j)

            grads = tape.gradient(loss, encoder.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

    # Evaluate
    metrics = {}
    if val_data is not None:
        all_embeddings = []
        all_labels = []
        for (x, y) in val_data:
            embeddings = encoder(x, training=False)
            all_embeddings.append(embeddings.numpy())
            all_labels.append(y.numpy())

        # Example metrics
        metrics['embedding_variance'] = np.var(np.concatenate(all_embeddings, axis=0))

    return {
        "encoder": encoder,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    feature_dim = 128
    temperature = 0.5
    input_shape = (64, 64, 3)

    # Placeholder datasets (replace with actual datasets)
    x_train = np.random.rand(100, 64, 64, 3)
    x_train_pairs = (x_train[:50], x_train[50:])  # Example positive pairs
    train_data = tf.data.Dataset.from_tensor_slices(x_train_pairs).batch(16)

    val_data = tf.data.Dataset.from_tensor_slices((x_train, np.random.randint(0, 10, 100))).batch(16)

    result = simclr(feature_dim=feature_dim, temperature=temperature,
                         input_shape=input_shape, gpu_optimized='yes',
                         train_data=train_data, val_data=val_data, epochs=5)

    print("Metrics:", result["metrics"])
"""


# 45.

class BYOL:
    def __init__(self, input_shape, projection_dim, predictor_dim, gpu_optimized='no'):
        """
        Initializes the BYOL framework.

        Args:
            input_shape (tuple): Shape of input data.
            projection_dim (int): Dimensionality of projection layer.
            predictor_dim (int): Dimensionality of the predictor network.
            gpu_optimized (str): If 'yes', enables GPU optimization. Default is 'no'.
        """
        self.input_shape = input_shape
        self.projection_dim = projection_dim
        self.predictor_dim = predictor_dim
        self.gpu_optimized = gpu_optimized

        if gpu_optimized == 'yes':
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")

    def build_encoder(self):
        """Builds the encoder network."""
        inputs = Input(shape=self.input_shape)
        x = Conv2D(64, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(self.projection_dim, activation='relu')(x)
        return Model(inputs, x, name="encoder")

    def build_predictor(self):
        """Builds the predictor network."""
        inputs = Input(shape=(self.projection_dim,))
        x = Dense(self.predictor_dim, activation='relu')(inputs)
        x = Dense(self.projection_dim)(x)
        return Model(inputs, x, name="predictor")

    def loss_function(self, z1, z2):
        """Computes the BYOL loss."""
        z1 = tf.math.l2_normalize(z1, axis=1)
        z2 = tf.math.l2_normalize(z2, axis=1)
        return 2 - 2 * tf.reduce_mean(tf.reduce_sum(z1 * z2, axis=1))

def byol(input_shape, projection_dim, predictor_dim, gpu_optimized='no', train_data=None, val_data=None, epochs=10):
    """
    Wrapper function to train and evaluate a BYOL model.

    Args:
        input_shape (tuple): Shape of input data.
        projection_dim (int): Dimensionality of projection layer.
        predictor_dim (int): Dimensionality of predictor network.
        gpu_optimized (str): If 'yes', enables GPU optimization. Default is 'no'.
        train_data (tf.data.Dataset): Training dataset.
        val_data (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs. Default is 10.

    Returns:
        dict: Contains the trained model and metrics.
    """
    byol = BYOL(input_shape, projection_dim, predictor_dim, gpu_optimized)
    encoder = byol.build_encoder()
    predictor = byol.build_predictor()

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training loop
    for epoch in range(epochs):
        for (x1, x2) in train_data:
            with tf.GradientTape() as tape:
                z1 = encoder(x1, training=True)
                z2 = encoder(x2, training=True)

                p1 = predictor(z1, training=True)
                p2 = predictor(z2, training=True)

                loss = byol.loss_function(p1, z2) + byol.loss_function(p2, z1)

            grads = tape.gradient(loss, encoder.trainable_variables + predictor.trainable_variables)
            optimizer.apply_gradients(zip(grads, encoder.trainable_variables + predictor.trainable_variables))

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

    # Evaluation
    metrics = {}
    if val_data is not None:
        embeddings = []
        labels = []
        for (x, y) in val_data:
            embedding = encoder(x, training=False)
            embeddings.append(embedding.numpy())
            labels.append(y.numpy())

        # Example metrics
        metrics['embedding_variance'] = np.var(np.concatenate(embeddings, axis=0))

    return {
        "encoder": encoder,
        "predictor": predictor,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    input_shape = (64, 64, 3)
    projection_dim = 128
    predictor_dim = 64

    # Placeholder datasets (replace with actual datasets)
    x_train = np.random.rand(100, 64, 64, 3)
    x_train_pairs = (x_train[:50], x_train[50:])  # Example positive pairs
    train_data = tf.data.Dataset.from_tensor_slices(x_train_pairs).batch(16)

    val_data = tf.data.Dataset.from_tensor_slices((x_train, np.random.randint(0, 10, 100))).batch(16)

    result = byol(input_shape=input_shape, projection_dim=projection_dim,
                       predictor_dim=predictor_dim, gpu_optimized='yes',
                       train_data=train_data, val_data=val_data, epochs=5)

    print("Metrics:", result["metrics"])
"""


# 46.

def build_clip_model(
    image_input_shape=(224, 224, 3),
    text_embedding_dim=512,
    projection_dim=256,
    learning_rate=0.001,
    gpu_optimized='no'
):
    """
    Builds a CLIP (Contrastive Language-Image Pretraining) model for associating images with text.

    Args:
        image_input_shape (tuple): Shape of the input images (default: (224, 224, 3)).
        text_embedding_dim (int): Dimensionality of the text embeddings (default: 512).
        projection_dim (int): Dimensionality of the projection space (default: 256).
        learning_rate (float): Learning rate for the optimizer (default: 0.001).
        gpu_optimized (str): If 'yes', uses a GPU for computations. Default is 'no'.

    Returns:
        dict: Contains the CLIP model, metrics, and training function.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")

    # Image encoder
    image_input = tf.keras.Input(shape=image_input_shape, name="image_input")
    image_base = tf.keras.applications.ResNet50(
        weights='imagenet', include_top=False, input_shape=image_input_shape
    )
    image_features = GlobalAveragePooling2D()(image_base(image_input))
    image_projection = tf.keras.Dense(projection_dim, activation='linear', name="image_projection")(image_features)

    # Text encoder
    text_input = Input(shape=(text_embedding_dim,), name="text_input")
    text_projection = Dense(projection_dim, activation='linear', name="text_projection")(text_input)

    # Normalize projections
    image_projection = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(image_projection)
    text_projection = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(text_projection)

    # Contrastive similarity score
    similarity = Dot(axes=1, normalize=True, name="similarity")([image_projection, text_projection])

    # Compile model
    clip_model = Model(inputs=[image_input, text_input], outputs=similarity, name="CLIP")
    clip_model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return {
        "model": clip_model,
        "image_encoder": image_base,
        "training_function": train_clip_model,
        "evaluation_function": evaluate_clip_model,
    }


def train_clip_model(model, image_data, text_data, labels, batch_size=32, epochs=10):
    """
    Trains the CLIP model.

    Args:
        model (tf.keras.Model): The CLIP model to train.
        image_data (np.array): Array of image data.
        text_data (np.array): Array of text embeddings.
        labels (np.array): Array of labels for training.
        batch_size (int): Batch size for training (default: 32).
        epochs (int): Number of epochs to train (default: 10).

    Returns:
        tf.keras.callbacks.History: Training history.
    """
    history = model.fit(
        [image_data, text_data],
        labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.2
    )
    return history


def evaluate_clip_model(model, image_data, text_data, true_labels):
    """
    Evaluates the CLIP model.

    Args:
        model (tf.keras.Model): The trained CLIP model.
        image_data (np.array): Array of image data.
        text_data (np.array): Array of text embeddings.
        true_labels (np.array): Array of true labels.

    Returns:
        dict: Evaluation metrics.
    """
    predictions = model.predict([image_data, text_data])
    predicted_labels = np.round(predictions).astype(int)

    metrics = {
        "accuracy": accuracy_score(true_labels, predicted_labels),
        "precision": precision_score(true_labels, predicted_labels, average='weighted'),
        "recall": recall_score(true_labels, predicted_labels, average='weighted'),
        "f1_score": f1_score(true_labels, predicted_labels, average='weighted')
    }
    return metrics


def clip(
    image_data,
    text_data,
    labels,
    image_input_shape=(224, 224, 3),
    text_embedding_dim=512,
    projection_dim=256,
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
    gpu_optimized='no'
):
    """
    Encapsulates the process of building, training, and evaluating the CLIP model.

    Args:
        image_data (np.array): Array of image data.
        text_data (np.array): Array of text embeddings.
        labels (np.array): Array of labels for training and evaluation.
        image_input_shape (tuple): Shape of the input images.
        text_embedding_dim (int): Dimensionality of the text embeddings.
        projection_dim (int): Dimensionality of the projection space.
        learning_rate (float): Learning rate for training.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        gpu_optimized (str): If 'yes', uses GPU for computations.

    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    # Build the model
    clip_model = build_clip_model(
        image_input_shape=image_input_shape,
        text_embedding_dim=text_embedding_dim,
        projection_dim=projection_dim,
        learning_rate=learning_rate,
        gpu_optimized=gpu_optimized
    )

    # Train the model
    history = train_clip_model(clip_model, image_data, text_data, labels, batch_size=batch_size, epochs=epochs)

    # Evaluate the model
    metrics = evaluate_clip_model(clip_model, image_data, text_data, labels)

    return {
        "model": clip_model,
        "history": history,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Simulate data
    num_samples = 1000
    image_data = np.random.rand(num_samples, 224, 224, 3)
    text_data = np.random.rand(num_samples, 512)
    labels = np.random.randint(0, 2, size=num_samples)

    # Run the CLIP pipeline
    result = clip(
        image_data=image_data,
        text_data=text_data,
        labels=labels,
        batch_size=32,
        epochs=10,
        gpu_optimized='yes'
    )

    print("Evaluation Metrics:", result["metrics"])
"""
    

# 47.

class DinoModel:
    def __init__(self, input_shape=(224, 224, 3), patch_size=16, num_heads=8, projection_dim=64, num_layers=4):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.projection_dim = projection_dim
        self.num_layers = num_layers

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        num_patches = (self.input_shape[0] // self.patch_size) * (self.input_shape[1] // self.patch_size)
        
        # Patch embedding
        patches = Conv2D(filters=self.projection_dim, kernel_size=self.patch_size, strides=self.patch_size)(inputs)
        patches = Reshape((num_patches, self.projection_dim))(patches)

        # Positional encoding
        position_embedding = Embedding(input_dim=num_patches, output_dim=self.projection_dim)
        positions = tf.range(start=0, limit=num_patches, delta=1)
        encoded_patches = patches + position_embedding(positions)

        # Transformer encoder
        for _ in range(self.num_layers):
            # Multi-head attention
            attention_output = MultiHeadAttention(num_heads=self.num_heads, key_dim=self.projection_dim)(encoded_patches, encoded_patches)
            attention_output = Add()([attention_output, encoded_patches])
            attention_output = LayerNormalization()(attention_output)

            # Feed-forward network
            ffn = Sequential([
                Dense(units=self.projection_dim * 4, activation="relu"),
                Dense(units=self.projection_dim)
            ])
            ffn_output = ffn(attention_output)
            encoded_patches = Add()([ffn_output, attention_output])
            encoded_patches = LayerNormalization()(encoded_patches)

        # Classification head
        representation = GlobalAveragePooling1D()(encoded_patches)
        outputs = Dense(units=1, activation="sigmoid")(representation)

        model = Model(inputs=inputs, outputs=outputs)
        return model

def dino(data, labels, input_shape=(224, 224, 3), patch_size=16, num_heads=8, projection_dim=64, num_layers=4, 
         batch_size=32, epochs=10, gpu_optimized='no', validation_split=0.2):
    """
    Builds, trains, and evaluates a DINO model.

    Args:
        data (np.array): Input image data.
        labels (np.array): Corresponding labels for the image data.
        input_shape (tuple): Shape of the input images. Default is (224, 224, 3).
        patch_size (int): Size of each patch for the vision transformer. Default is 16.
        num_heads (int): Number of attention heads. Default is 8.
        projection_dim (int): Dimension of the projection layer. Default is 64.
        num_layers (int): Number of transformer encoder layers. Default is 4.
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of training epochs. Default is 10.
        gpu_optimized (str): If 'yes', enables GPU acceleration. Default is 'no'.
        validation_split (float): Fraction of data to use for validation. Default is 0.2.

    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled using TensorFlow.")
            except RuntimeError as e:
                print("GPU setup failed:", e)
        else:
            print("No GPU detected. Using CPU.")

    # Prepare the data
    data = data.astype('float32') / 255.0
    labels = labels.astype('float32')

    # Build the model
    dino_instance = DinoModel(input_shape, patch_size, num_heads, projection_dim, num_layers)
    model = dino_instance.build_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(data, labels, batch_size=batch_size, epochs=epochs, validation_split=validation_split, verbose=1)

    # Evaluate the model
    predictions = (model.predict(data) > 0.5).astype(int).flatten()
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "f1_score": f1_score(labels, predictions, average='weighted')
    }

    return {
        "model": model,
        "history": history.history,
        "metrics": metrics
    }


"""
# Example usage
if __name__ == "__main__":
    # Dummy data
    num_samples = 100
    data = np.random.rand(num_samples, 224, 224, 3)
    labels = np.random.randint(0, 2, size=(num_samples,))

    result = dino(data, labels, gpu_optimized='yes', epochs=5)

    print("Metrics:", result["metrics"])
"""


# 48.

def build_swin_transformer(input_shape=(224, 224, 3), num_classes=10):
    """
    Builds a Swin Transformer model for image analysis tasks.

    Args:
        input_shape (tuple): The shape of the input image data.
        num_classes (int): Number of output classes for classification.

    Returns:
        Model: A Swin Transformer model.
    """
    inputs = Input(shape=input_shape)
    x = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)(inputs)
    x = GlobalAveragePooling2D()(x)
    x = LayerNormalization()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

def train_swin_transformer(model, train_data, train_labels, batch_size=32, epochs=10, learning_rate=0.001):
    """
    Compiles and trains the Swin Transformer model.

    Args:
        model (Model): The Swin Transformer model.
        train_data (np.array): Training image data.
        train_labels (np.array): Labels corresponding to the training data.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        History: Training history object.
    """
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, verbose=1)
    return history

def evaluate_swin_transformer(model, test_data, test_labels):
    """
    Evaluates the Swin Transformer model on test data.

    Args:
        model (Model): The Swin Transformer model.
        test_data (np.array): Test image data.
        test_labels (np.array): Labels corresponding to the test data.

    Returns:
        dict: Evaluation metrics including accuracy, precision, recall, and F1 score.
    """
    predictions = np.argmax(model.predict(test_data), axis=1)
    metrics = {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(test_labels, predictions, average='weighted'),
        "recall": recall_score(test_labels, predictions, average='weighted'),
        "f1_score": f1_score(test_labels, predictions, average='weighted')
    }
    return metrics

def swin_transformer(input_shape=(224, 224, 3), num_classes=10, train_data=None, train_labels=None, test_data=None, test_labels=None,
                     batch_size=32, epochs=10, learning_rate=0.001, gpu_optimized='no'):
    """
    Builds, trains, and evaluates a Swin Transformer model.

    Args:
        input_shape (tuple): The shape of the input image data.
        num_classes (int): Number of output classes for classification.
        train_data (np.array): Training image data.
        train_labels (np.array): Labels corresponding to the training data.
        test_data (np.array): Test image data.
        test_labels (np.array): Labels corresponding to the test data.
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train the model.
        learning_rate (float): Learning rate for the optimizer.
        gpu_optimized (str): If 'yes', enables GPU optimizations.

    Returns:
        dict: Contains the trained model, training history, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print("GPU setup failed:", e)

    model = build_swin_transformer(input_shape=input_shape, num_classes=num_classes)
    history = train_swin_transformer(model, train_data, train_labels, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate)
    metrics = evaluate_swin_transformer(model, test_data, test_labels)

    return {
        "model": model,
        "history": history,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Dummy data for demonstration
    train_data = np.random.rand(100, 224, 224, 3)
    train_labels = np.random.randint(0, 10, 100)
    test_data = np.random.rand(20, 224, 224, 3)
    test_labels = np.random.randint(0, 10, 20)

    result = swin_transformer(
        input_shape=(224, 224, 3),
        num_classes=10,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        batch_size=16,
        epochs=5,
        learning_rate=0.001,
        gpu_optimized='yes'
    )

    print("Metrics:", result["metrics"])
"""


# 49.

def build_mlp_mixer_model(input_shape, num_classes, patch_size, hidden_dim, num_blocks):
    """
    Builds an MLP-Mixer model for image classification.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        patch_size (int): Size of patches to split the image into.
        hidden_dim (int): Dimension of the hidden 
        num_blocks (int): Number of MLP-Mixer blocks.

    Returns:
        tf.keras.Model: Compiled MLP-Mixer model.
    """
    inputs = Input(shape=input_shape)

    # Split image into patches
    num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
    patch_proj = Conv2D(hidden_dim, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
    x = Reshape((num_patches, hidden_dim))(patch_proj)

    # MLP-Mixer blocks
    for _ in range(num_blocks):
        # Token mixing
        x_skip = x
        x = LayerNormalization()(x)
        x = Permute((2, 1))(x)
        x = Dense(hidden_dim, activation='gelu')(x)
        x = Dense(num_patches)(x)
        x = Permute((2, 1))(x)
        x = Add()([x, x_skip])

        # Channel mixing
        x_skip = x
        x = LayerNormalization()(x)
        x = Dense(hidden_dim, activation='gelu')(x)
        x = Dense(hidden_dim)(x)
        x = Add()([x, x_skip])

    # Classification head
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_mlp_mixer_model(model, train_data, train_labels, batch_size, epochs, validation_data=None):
    """
    Trains the MLP-Mixer model.

    Args:
        model (tf.keras.Model): The MLP-Mixer model.
        train_data (np.array): Training data.
        train_labels (np.array): Training labels.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        validation_data (tuple): Validation data and labels (optional).

    Returns:
        tf.keras.callbacks.History: Training history.
    """
    history = model.fit(
        train_data, train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data
    )
    return history

def evaluate_mlp_mixer_model(model, test_data, test_labels):
    """
    Evaluates the MLP-Mixer model and computes metrics.

    Args:
        model (tf.keras.Model): The MLP-Mixer model.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.

    Returns:
        dict: Evaluation metrics.
    """
    predictions = model.predict(test_data)
    predicted_labels = np.argmax(predictions, axis=1)

    metrics = {
        "accuracy": accuracy_score(test_labels, predicted_labels),
        "f1_score": f1_score(test_labels, predicted_labels, average='weighted')
    }
    return metrics

def mlp_mixer(input_shape, num_classes, patch_size=16, hidden_dim=128, num_blocks=4, batch_size=32, epochs=10, gpu_optimized='no', train_data=None, train_labels=None, test_data=None, test_labels=None):
    """
    Builds, trains, and evaluates an MLP-Mixer model.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        patch_size (int): Size of patches to split the image into.
        hidden_dim (int): Dimension of the hidden layers.
        num_blocks (int): Number of MLP-Mixer blocks.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        gpu_optimized (str): If 'yes', enables GPU optimization.
        train_data (np.array): Training data.
        train_labels (np.array): Training labels.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.

    Returns:
        dict: Contains model, training history, and evaluation metrics.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print("GPU setup failed:", e)

    model = build_mlp_mixer_model(input_shape, num_classes, patch_size, hidden_dim, num_blocks)
    history = train_mlp_mixer_model(model, train_data, train_labels, batch_size, epochs, validation_data=(test_data, test_labels))
    metrics = evaluate_mlp_mixer_model(model, test_data, test_labels)

    return {
        "model": model,
        "history": history,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Example data (randomly generated for demonstration purposes)
    input_shape = (128, 128, 3)
    num_classes = 10
    train_data = np.random.rand(100, 128, 128, 3)
    train_labels = np.random.randint(0, num_classes, 100)
    test_data = np.random.rand(20, 128, 128, 3)
    test_labels = np.random.randint(0, num_classes, 20)

    result = mlp_mixer(input_shape, num_classes, train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels, gpu_optimized='yes')

    print("Evaluation Metrics:", result["metrics"])
"""


# 50.

def build_deep_lab(input_shape=(256, 256, 3), num_classes=21, learning_rate=0.001, gpu_optimized='no'):
    """
    Builds and compiles a DeepLab model for semantic segmentation.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels). Default is (256, 256, 3).
        num_classes (int): Number of segmentation classes. Default is 21.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        gpu_optimized (str): If 'yes', enables GPU for computations. Default is 'no'.

    Returns:
        model: Compiled DeepLab model.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print("Failed to set GPU optimization:", e)
        else:
            print("No GPU found. Running on CPU.")

    # Input layer
    inputs = Input(shape=input_shape)

    # Initial convolution block
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    # Atrous Spatial Pyramid Pooling (ASPP) module
    aspp = []
    for dilation_rate in [1, 6, 12, 18]:
        y = Conv2D(64, (3, 3), padding='same', activation='relu', dilation_rate=dilation_rate)(x)
        y = BatchNormalization()(y)
        aspp.append(y)

    x = Add()(aspp)

    # Upsampling block
    x = UpSampling2D(size=(4, 4))(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)

    # Final classification layer
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(x)

    # Model
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', MeanIoU(num_classes=num_classes)])

    return model


def train_deep_lab_model(model, train_data, train_labels, val_data, val_labels, batch_size=32, epochs=10):
    """
    Trains the DeepLab model on the provided dataset.

    Args:
        model: Compiled DeepLab model.
        train_data (np.array): Training images.
        train_labels (np.array): One-hot encoded training labels.
        val_data (np.array): Validation images.
        val_labels (np.array): One-hot encoded validation labels.
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of training epochs. Default is 10.

    Returns:
        history: Training history object.
    """
    history = model.fit(train_data, train_labels,
                        validation_data=(val_data, val_labels),
                        batch_size=batch_size,
                        epochs=epochs)
    return history


def evaluate_deep_lab_model(model, test_data, test_labels):
    """
    Evaluates the DeepLab model on the test dataset.

    Args:
        model: Trained DeepLab model.
        test_data (np.array): Test images.
        test_labels (np.array): One-hot encoded test labels.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    results = model.evaluate(test_data, test_labels, verbose=0)
    metrics = {"loss": results[0], "accuracy": results[1], "mean_iou": results[2]}
    return metrics


def deep_lab(input_shape, num_classes, train_images, train_labels, val_images, val_labels, test_images, test_labels, 
             learning_rate=0.001, gpu_optimized='no', batch_size=32, epochs=10):
    """
    Builds, trains, and evaluates a DeepLab model for semantic segmentation.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of segmentation classes.
        train_images (np.array): Training images.
        train_labels (np.array): One-hot encoded training labels.
        val_images (np.array): Validation images.
        val_labels (np.array): One-hot encoded validation labels.
        test_images (np.array): Test images.
        test_labels (np.array): One-hot encoded test labels.
        learning_rate (float): Learning rate for the optimizer.
        gpu_optimized (str): If 'yes', enables GPU for computations.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.

    Returns:
        dict: Dictionary containing the model, training history, and evaluation metrics.
    """
    # Build the model
    model = build_deep_lab(input_shape=input_shape, num_classes=num_classes, learning_rate=learning_rate, gpu_optimized=gpu_optimized)

    # Train the model
    history = train_deep_lab_model(model, train_images, train_labels, val_images, val_labels, batch_size=batch_size, epochs=epochs)

    # Evaluate the model
    metrics = evaluate_deep_lab_model(model, test_images, test_labels)

    return {
        "model": model,
        "history": history,
        "metrics": metrics
    }

"""
# Example usage
if __name__ == "__main__":
    # Dummy data for demonstration
    input_shape = (256, 256, 3)
    num_classes = 21
    train_images = tf.random.uniform((100, *input_shape))
    train_labels = to_categorical(tf.random.uniform((100, 256, 256), maxval=num_classes, dtype=tf.int32), num_classes)
    val_images = tf.random.uniform((20, *input_shape))
    val_labels = to_categorical(tf.random.uniform((20, 256, 256), maxval=num_classes, dtype=tf.int32), num_classes)
    test_images = tf.random.uniform((20, *input_shape))
    test_labels = to_categorical(tf.random.uniform((20, 256, 256), maxval=num_classes, dtype=tf.int32), num_classes)

    # Build, train, and evaluate model
    result = deep_lab(input_shape=input_shape, num_classes=num_classes, 
                      train_images=train_images, train_labels=train_labels, 
                      val_images=val_images, val_labels=val_labels, 
                      test_images=test_images, test_labels=test_labels, 
                      learning_rate=0.001, gpu_optimized='yes', batch_size=32, epochs=5)

    print("Evaluation Metrics:", result["metrics"])

"""


# 51.

def build_mask_rcnn(input_shape=(256, 256, 3), num_classes=21, learning_rate=0.001, gpu_optimized='no'):
    """
    Builds and compiles a Mask R-CNN model for instance segmentation.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels). Default is (256, 256, 3).
        num_classes (int): Number of segmentation classes. Default is 21.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
        gpu_optimized (str): If 'yes', enables GPU for computations. Default is 'no'.

    Returns:
        model: Compiled Mask R-CNN model.
    """
    if gpu_optimized == 'yes':
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print("Failed to set GPU optimization:", e)
        else:
            print("No GPU found. Running on CPU.")

    # Input layer
    inputs = Input(shape=input_shape)

    # Backbone (Feature Extraction)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)

    # Region Proposal Network (RPN) simulation (simplified for demonstration)
    rpn = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    rpn = BatchNormalization()(rpn)

    # Mask Prediction Branch
    mask_branch = UpSampling2D(size=(2, 2))(rpn)
    mask_branch = Conv2D(128, (3, 3), padding='same', activation='relu')(mask_branch)
    mask_branch = BatchNormalization()(mask_branch)
    masks = Conv2D(num_classes, (1, 1), activation='softmax', name='masks')(mask_branch)

    # Model
    model = Model(inputs, masks)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy', MeanIoU(num_classes=num_classes)])

    return model


def train_mask_rcnn_model(model, train_data, train_labels, val_data, val_labels, batch_size=32, epochs=10):
    """
    Trains the Mask R-CNN model on the provided dataset.

    Args:
        model: Compiled Mask R-CNN model.
        train_data (np.array): Training images.
        train_labels (np.array): One-hot encoded training labels.
        val_data (np.array): Validation images.
        val_labels (np.array): One-hot encoded validation labels.
        batch_size (int): Batch size for training. Default is 32.
        epochs (int): Number of training epochs. Default is 10.

    Returns:
        history: Training history object.
    """
    history = model.fit(train_data, train_labels,
                        validation_data=(val_data, val_labels),
                        batch_size=batch_size,
                        epochs=epochs)
    return history


def evaluate_mask_rcnn_model(model, test_data, test_labels):
    """
    Evaluates the Mask R-CNN model on the test dataset.

    Args:
        model: Trained Mask R-CNN model.
        test_data (np.array): Test images.
        test_labels (np.array): One-hot encoded test labels.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    results = model.evaluate(test_data, test_labels, verbose=0)
    metrics = {"loss": results[0], "accuracy": results[1], "mean_iou": results[2]}
    return metrics


def mask_rcnn(input_shape=(256, 256, 3), num_classes=21, train_data=None, train_labels=None,
              val_data=None, val_labels=None, test_data=None, test_labels=None,
              learning_rate=0.001, batch_size=32, epochs=10, gpu_optimized='no'):
    """
    Builds, trains, and evaluates a Mask R-CNN model for instance segmentation.

    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of segmentation classes.
        train_data (np.array): Training images.
        train_labels (np.array): One-hot encoded training labels.
        val_data (np.array): Validation images.
        val_labels (np.array): One-hot encoded validation labels.
        test_data (np.array): Test images.
        test_labels (np.array): One-hot encoded test labels.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        gpu_optimized (str): If 'yes', enables GPU for computations.

    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    model = build_mask_rcnn(input_shape=input_shape, num_classes=num_classes,
                            learning_rate=learning_rate, gpu_optimized=gpu_optimized)

    history = train_mask_rcnn_model(model, train_data, train_labels, val_data, val_labels,
                                    batch_size=batch_size, epochs=epochs)

    metrics = evaluate_mask_rcnn_model(model, test_data, test_labels)

    return {
        "model": model,
        "history": history,
        "metrics": metrics
    }


"""
# Example usage
if __name__ == "__main__":
    # Dummy data for demonstration
    input_shape = (256, 256, 3)
    num_classes = 21
    train_images = tf.random.uniform((100, *input_shape))
    train_labels = tf.random.uniform((100, 256, 256), maxval=num_classes, dtype=tf.int32)
    val_images = tf.random.uniform((20, *input_shape))
    val_labels = tf.random.uniform((20, 256, 256), maxval=num_classes, dtype=tf.int32)
    test_images = tf.random.uniform((20, *input_shape))
    test_labels = tf.random.uniform((20, 256, 256), maxval=num_classes, dtype=tf.int32)

    # Convert labels to one-hot encoding
    train_labels = tf.one_hot(train_labels, depth=num_classes)
    val_labels = tf.one_hot(val_labels, depth=num_classes)
    test_labels = tf.one_hot(test_labels, depth=num_classes)

    # Build, train, and evaluate model
    result = mask_rcnn(input_shape=input_shape, num_classes=num_classes,
                       train_data=train_images, train_labels=train_labels,
                       val_data=val_images, val_labels=val_labels,
                       test_data=test_images, test_labels=test_labels,
                       learning_rate=0.001, batch_size=16, epochs=5,
                       gpu_optimized='yes')

    print("Evaluation Metrics:", result["metrics"])
"""


# 52.

def build_efficient_net(
    version="efficientnet",  # Options: 'efficientnet', 'efficientnet_v2'
    variant="B0",           # Model variant: B0, B1, etc.
    input_shape=(224, 224, 3),
    num_classes=10,
    include_top=True,
    dropout_rate=0.2,
    learning_rate=0.001,
    gpu_optimized='yes',
    metrics=['accuracy'],
    task_type='classification'  # Options: 'classification', 'regression'
):
    """
    Builds a robust EfficientNet-based model optimized for GPU and customizable for various tasks.

    Parameters:
    - version: String, 'efficientnet' or 'efficientnet_v2'.
    - variant: String, model variant such as 'B0', 'B1', etc.
    - input_shape: Tuple, shape of the input images (default: (224, 224, 3)).
    - num_classes: Integer, number of output classes (default: 10).
    - include_top: Boolean, whether to include the final Dense layer (default: True).
    - dropout_rate: Float, dropout rate for regularization (default: 0.2).
    - learning_rate: Float, learning rate for the optimizer (default: 0.001).
    - gpu_optimized: String, 'yes' to enable GPU optimization using CUDA (default: 'yes').
    - metrics: List, performance metrics to evaluate the model (default: ['accuracy']).
    - task_type: String, type of task ('classification' or 'regression').

    Returns:
    - A compiled EfficientNet-based model.
    """
    # Configure GPU settings if needed
    if gpu_optimized.lower() == 'yes':
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU detected, running on CPU.")

    # Select EfficientNet version
    if version == "efficientnet":
        base_model_func = getattr(efficientnet, f"EfficientNet{variant}", None)
    elif version == "efficientnet_v2":
        base_model_func = getattr(efficientnet_v2, f"EfficientNetV2{variant}", None)
    else:
        raise ValueError("Invalid version. Choose 'efficientnet' or 'efficientnet_v2'.")

    if not base_model_func:
        raise ValueError(f"Invalid variant '{variant}' for version '{version}'.")

    # Build the base model
    base_model = base_model_func(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Add classification or task-specific layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    if include_top:
        if task_type == 'classification':
            output = Dense(num_classes, activation="softmax")(x)
        elif task_type == 'regression':
            output = Dense(1, activation="linear")(x)
        else:
            raise ValueError("Unsupported task_type. Choose 'classification' or 'regression'.")
    else:
        output = x

    model = Model(inputs=base_model.input, outputs=output)

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    loss = "categorical_crossentropy" if task_type == 'classification' else "mse"
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def efficient_net(
    version="efficientnet",
    variant="B0",
    input_shape=(224, 224, 3),
    num_classes=10,
    train_data=None,
    train_labels=None,
    val_data=None,
    val_labels=None,
    test_data=None,
    test_labels=None,
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
    gpu_optimized='yes',
    metrics=['accuracy'],
    task_type='classification',
    patience=5
):
    """
    Builds, trains, and evaluates an EfficientNet model.

    Args:
        version (str): EfficientNet version ('efficientnet' or 'efficientnet_v2').
        variant (str): Model variant (e.g., 'B0', 'B1').
        input_shape (tuple): Shape of input data.
        num_classes (int): Number of output classes.
        train_data (np.array): Training data.
        train_labels (np.array): Training labels.
        val_data (np.array): Validation data.
        val_labels (np.array): Validation labels.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        gpu_optimized (str): Whether to use GPU ('yes' or 'no').
        metrics (list): List of evaluation metrics.
        task_type (str): Task type ('classification' or 'regression').
        patience (int): Patience to break out
    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    model = build_efficient_net(
        version=version,
        variant=variant,
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
        gpu_optimized=gpu_optimized,
        metrics=metrics,
        task_type=task_type
    )

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    evaluation_metrics = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return {
        "model": model,
        "history": history.history,
        "metrics": dict(zip(model.metrics_names, evaluation_metrics))
    }


"""
# Example usage:
if __name__ == "__main__":
    # Assuming train_data, train_labels, val_data, val_labels, test_data, and test_labels are defined
    result = efficient_net(
        version="efficientnet_v2",
        variant="B0",
        input_shape=(224, 224, 3),
        num_classes=10,
        train_data=None,  # Replace with actual data
        train_labels=None,  # Replace with actual labels
        val_data=None,  # Replace with actual data
        val_labels=None,  # Replace with actual labels
        test_data=None,  # Replace with actual data
        test_labels=None,  # Replace with actual labels
        learning_rate=0.0001,
        batch_size=32,
        epochs=10,
        gpu_optimized='yes',
        metrics=['accuracy', 'Precision', 'Recall'],
        task_type='classification',
        patience=5
    )
    print("Evaluation Metrics:", result["metrics"])
"""

# 53

def build_res_net(
    variant="ResNet50",  # Options: 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2'
    input_shape=(224, 224, 3),
    num_classes=10,
    include_top=True,
    dropout_rate=0.2,
    learning_rate=0.001,
    gpu_optimized='yes',
    metrics=['accuracy'],
    task_type='classification'  # Options: 'classification', 'regression'
):
    """
    Builds a robust ResNet-based model optimized for GPU and customizable for various tasks.

    Parameters:
    - variant: String, ResNet variant such as 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', or 'ResNet152V2'.
    - input_shape: Tuple, shape of the input images (default: (224, 224, 3)).
    - num_classes: Integer, number of output classes (default: 10).
    - include_top: Boolean, whether to include the final Dense layer (default: True).
    - dropout_rate: Float, dropout rate for regularization (default: 0.2).
    - learning_rate: Float, learning rate for the optimizer (default: 0.001).
    - gpu_optimized: String, 'yes' to enable GPU optimization using CUDA (default: 'yes').
    - metrics: List, performance metrics to evaluate the model (default: ['accuracy']).
    - task_type: String, type of task ('classification' or 'regression').

    Returns:
    - A compiled ResNet-based model.
    """
    # Configure GPU settings if needed
    if gpu_optimized.lower() == 'yes':
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU detected, running on CPU.")

    # Select ResNet variant
    base_model_func = globals().get(variant, None)
    if not base_model_func:
        raise ValueError(f"Invalid variant '{variant}'. Choose 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', or 'ResNet152V2'.")

    # Build the base model
    base_model = base_model_func(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    # Add classification or task-specific layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)

    if include_top:
        if task_type == 'classification':
            output = Dense(num_classes, activation="softmax")(x)
        elif task_type == 'regression':
            output = Dense(1, activation="linear")(x)
        else:
            raise ValueError("Unsupported task_type. Choose 'classification' or 'regression'.")
    else:
        output = x

    model = Model(inputs=base_model.input, outputs=output)

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    loss = "categorical_crossentropy" if task_type == 'classification' else "mse"
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def res_net(
    variant="ResNet50",
    input_shape=(224, 224, 3),
    num_classes=10,
    train_data=None,
    train_labels=None,
    val_data=None,
    val_labels=None,
    test_data=None,
    test_labels=None,
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
    gpu_optimized='yes',
    metrics=['accuracy'],
    task_type='classification',
    patience=5
):
    """
    Builds, trains, and evaluates a ResNet model.

    Args:
        variant (str): ResNet variant ('ResNet50', 'ResNet101', 'ResNet152', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2').
        input_shape (tuple): Shape of input data.
        num_classes (int): Number of output classes.
        train_data (np.array): Training data.
        train_labels (np.array): Training labels.
        val_data (np.array): Validation data.
        val_labels (np.array): Validation labels.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        gpu_optimized (str): Whether to use GPU ('yes' or 'no').
        metrics (list): List of evaluation metrics.
        task_type (str): Task type ('classification' or 'regression').
        patience (int): patience
    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    model = build_res_net(
        variant=variant,
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate,
        gpu_optimized=gpu_optimized,
        metrics=metrics,
        task_type=task_type
    )

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    evaluation_metrics = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return {
        "model": model,
        "history": history.history,
        "metrics": dict(zip(model.metrics_names, evaluation_metrics))
    }

"""

# Example usage:
if __name__ == "__main__":
    # Assuming train_data, train_labels, val_data, val_labels, test_data, and test_labels are defined
    result = res_net(
        variant="ResNet50",
        input_shape=(224, 224, 3),
        num_classes=10,
        train_data=None,  # Replace with actual data
        train_labels=None,  # Replace with actual labels
        val_data=None,  # Replace with actual data
        val_labels=None,  # Replace with actual labels
        test_data=None,  # Replace with actual data
        test_labels=None,  # Replace with actual labels
        learning_rate=0.0001,
        batch_size=32,
        epochs=10,
        gpu_optimized='yes',
        metrics=['accuracy', 'Precision', 'Recall'],
        task_type='classification',
        patience=5
    )
    print("Evaluation Metrics:", result["metrics"])
"""

# 54

def build_u_net(
    input_shape=(128, 128, 3),
    num_classes=1,  # For binary segmentation, 1 output class; for multi-class, adjust accordingly
    base_filters=64,  # Number of filters in the first convolutional layer
    dropout_rate=0.2,
    learning_rate=0.001,
    gpu_optimized='yes',
    metrics=['accuracy'],
):
    """
    Builds a U-Net model optimized for GPU and customizable for segmentation tasks.

    Parameters:
    - input_shape: Tuple, shape of the input images (default: (128, 128, 3)).
    - num_classes: Integer, number of output classes (default: 1 for binary segmentation).
    - base_filters: Integer, number of filters in the first convolutional layer (default: 64).
    - dropout_rate: Float, dropout rate for regularization (default: 0.2).
    - learning_rate: Float, learning rate for the optimizer (default: 0.001).
    - gpu_optimized: String, 'yes' to enable GPU optimization using CUDA (default: 'yes').
    - metrics: List, performance metrics to evaluate the model (default: ['accuracy']).

    Returns:
    - A compiled U-Net model.
    """
    # Configure GPU settings if needed
    if gpu_optimized.lower() == 'yes':
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU detected, running on CPU.")

    def conv_block(input_tensor, num_filters):
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(input_tensor)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(num_filters, (3, 3), activation='relu', padding='same')(x)
        return x

    def encoder_block(input_tensor, num_filters):
        x = conv_block(input_tensor, num_filters)
        p = MaxPooling2D((2, 2))(x)
        return x, p

    def decoder_block(input_tensor, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
        x = concatenate([x, skip_features])
        x = conv_block(x, num_filters)
        return x

    # Input layer
    inputs = Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, base_filters)
    s2, p2 = encoder_block(p1, base_filters * 2)
    s3, p3 = encoder_block(p2, base_filters * 4)
    s4, p4 = encoder_block(p3, base_filters * 8)

    # Bottleneck
    b1 = conv_block(p4, base_filters * 16)

    # Decoder
    d1 = decoder_block(b1, s4, base_filters * 8)
    d2 = decoder_block(d1, s3, base_filters * 4)
    d3 = decoder_block(d2, s2, base_filters * 2)
    d4 = decoder_block(d3, s1, base_filters)

    # Output layer
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid' if num_classes == 1 else 'softmax')(d4)

    model = Model(inputs, outputs)

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    loss = 'binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy'
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

def u_net(
    input_shape=(128, 128, 3),
    num_classes=1,
    train_data=None,
    train_labels=None,
    val_data=None,
    val_labels=None,
    test_data=None,
    test_labels=None,
    base_filters=64,
    dropout_rate=0.2,
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
    gpu_optimized='yes',
    metrics=['accuracy'],
    patience=5
):
    """
    Builds, trains, and evaluates a U-Net model.

    Args:
        input_shape (tuple): Shape of input data.
        num_classes (int): Number of output classes.
        train_data (np.array): Training data.
        train_labels (np.array): Training labels.
        val_data (np.array): Validation data.
        val_labels (np.array): Validation labels.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.
        base_filters (int): Number of filters in the first convolutional layer.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        gpu_optimized (str): Whether to use GPU ('yes' or 'no').
        metrics (list): List of evaluation metrics.
        patience (int): patience
    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    model = build_u_net(
        input_shape=input_shape,
        num_classes=num_classes,
        base_filters=base_filters,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
        gpu_optimized=gpu_optimized,
        metrics=metrics
    )

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    evaluation_metrics = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return {
        "model": model,
        "history": history.history,
        "metrics": dict(zip(model.metrics_names, evaluation_metrics))
    }

"""
# Example usage:
if __name__ == "__main__":
    # Assuming train_data, train_labels, val_data, val_labels, test_data, and test_labels are defined
    result = u_net(
        input_shape=(128, 128, 3),
        num_classes=1,
        train_data=None,  # Replace with actual data
        train_labels=None,  # Replace with actual labels
        val_data=None,  # Replace with actual data
        val_labels=None,  # Replace with actual labels
        test_data=None,  # Replace with actual data
        test_labels=None,  # Replace with actual labels
        base_filters=64,
        dropout_rate=0.2,
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        gpu_optimized='yes',
        metrics=['accuracy', 'precision'],
        patience=5
    )
    print("Evaluation Metrics:", result["metrics"])
"""

# 55.

def build_yolo(
    input_shape=(416, 416, 3),
    num_classes=80,  # Default for COCO dataset
    anchors=None,  # Predefined anchors for bounding boxes
    learning_rate=0.001,
    gpu_optimized='yes',
    metrics=['accuracy']
):
    """
    Builds a YOLO model optimized for GPU and customizable for object detection tasks.

    Parameters:
    - input_shape: Tuple, shape of the input images (default: (416, 416, 3)).
    - num_classes: Integer, number of object classes (default: 80 for COCO dataset).
    - anchors: List of tuples, predefined anchors for bounding boxes (default: None).
    - learning_rate: Float, learning rate for the optimizer (default: 0.001).
    - gpu_optimized: String, 'yes' to enable GPU optimization using CUDA (default: 'yes').
    - metrics: List, performance metrics to evaluate the model (default: ['accuracy']).

    Returns:
    - A compiled YOLO model.
    """
    # Configure GPU settings if needed
    if gpu_optimized.lower() == 'yes':
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU detected, running on CPU.")

    if anchors is None:
        anchors = [
            (10, 13), (16, 30), (33, 23),  # Default YOLOv3 small anchors
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)
        ]

    # Input layer
    inputs = Input(shape=input_shape)

    # YOLO Backbone (simplified, can replace with a pre-trained model like Darknet-53)
    x = Conv2D(32, (3, 3), strides=(1, 1), padding="same", activation="relu")(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Additional convolutional layers (example, can be expanded for deeper networks)
    x = Conv2D(128, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), padding="same", activation="relu")(x)

    # Output layer (bounding box predictions)
    outputs = Conv2D(
        len(anchors) * (num_classes + 5),  # 5 = tx, ty, tw, th, object confidence
        (1, 1),
        strides=(1, 1),
        padding="same",
        activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

    return model

def yolo(
    input_shape=(416, 416, 3),
    num_classes=80,
    train_data=None,
    train_labels=None,
    val_data=None,
    val_labels=None,
    test_data=None,
    test_labels=None,
    anchors=None,
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
    gpu_optimized='yes',
    metrics=['accuracy'],
    patience=5
):
    """
    Builds, trains, and evaluates a YOLO model.

    Args:
        input_shape (tuple): Shape of input data.
        num_classes (int): Number of object classes.
        train_data (np.array): Training data.
        train_labels (np.array): Training labels.
        val_data (np.array): Validation data.
        val_labels (np.array): Validation labels.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.
        anchors (list): Predefined anchors for bounding boxes.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        gpu_optimized (str): Whether to use GPU ('yes' or 'no').
        metrics (list): List of evaluation metrics.
        patience (int): patience
    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    model = build_yolo(
        input_shape=input_shape,
        num_classes=num_classes,
        anchors=anchors,
        learning_rate=learning_rate,
        gpu_optimized=gpu_optimized,
        metrics=metrics
    )

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    evaluation_metrics = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return {
        "model": model,
        "history": history.history,
        "metrics": dict(zip(model.metrics_names, evaluation_metrics))
    }

"""
# Example usage:
if __name__ == "__main__":
    # Assuming train_data, train_labels, val_data, val_labels, test_data, and test_labels are defined
    result = yolo(
        input_shape=(416, 416, 3),
        num_classes=80,
        train_data=None,  # Replace with actual data
        train_labels=None,  # Replace with actual labels
        val_data=None,  # Replace with actual data
        val_labels=None,  # Replace with actual labels
        test_data=None,  # Replace with actual data
        test_labels=None,  # Replace with actual labels
        anchors=None,
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        gpu_optimized='yes',
        metrics=['accuracy', 'precision'],
        patience=5
    )
    print("Evaluation Metrics:", result["metrics"])
"""

# 56

def build_ssd(
    input_shape=(300, 300, 3),
    num_classes=21,  # Default for Pascal VOC dataset
    base_model_filters=64,
    learning_rate=0.001,
    gpu_optimized='yes',
    metrics=['accuracy']
):
    """
    Builds an SSD model optimized for GPU and customizable for object detection tasks.

    Parameters:
    - input_shape: Tuple, shape of the input images (default: (300, 300, 3)).
    - num_classes: Integer, number of object classes including background (default: 21 for Pascal VOC).
    - base_model_filters: Integer, number of filters in the base model (default: 64).
    - learning_rate: Float, learning rate for the optimizer (default: 0.001).
    - gpu_optimized: String, 'yes' to enable GPU optimization using CUDA (default: 'yes').
    - metrics: List, performance metrics to evaluate the model (default: ['accuracy']).

    Returns:
    - A compiled SSD model.
    """
    # Configure GPU settings if needed
    if gpu_optimized.lower() == 'yes':
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU detected, running on CPU.")

    # Input layer
    inputs = Input(shape=input_shape)

    # Base model: a simple feature extractor (can be replaced with pre-trained models like VGG16 or MobileNet)
    x = Conv2D(base_model_filters, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(base_model_filters * 2, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Additional feature maps for multi-scale detection
    f1 = Conv2D(base_model_filters * 4, (3, 3), activation='relu', padding='same')(x)
    f2 = Conv2D(base_model_filters * 8, (3, 3), activation='relu', padding='same')(f1)

    # Detection layers (bounding box regression and class prediction)
    loc_pred = Conv2D(num_classes * 4, (3, 3), activation='linear', padding='same')(f2)  # Bounding box regression
    cls_pred = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(f2)  # Class prediction

    model = Model(inputs, [loc_pred, cls_pred])

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=['mean_squared_error', 'categorical_crossentropy'],
        metrics=metrics
    )

    return model

def ssd(
    input_shape=(300, 300, 3),
    num_classes=21,
    train_data=None,
    train_labels=None,
    val_data=None,
    val_labels=None,
    test_data=None,
    test_labels=None,
    base_model_filters=64,
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
    gpu_optimized='yes',
    metrics=['accuracy'],
    patience=5
):
    """
    Builds, trains, and evaluates an SSD model.

    Args:
        input_shape (tuple): Shape of input data.
        num_classes (int): Number of object classes including background.
        train_data (np.array): Training data.
        train_labels (np.array): Training labels.
        val_data (np.array): Validation data.
        val_labels (np.array): Validation labels.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.
        base_model_filters (int): Number of filters in the base model.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        gpu_optimized (str): Whether to use GPU ('yes' or 'no').
        metrics (list): List of evaluation metrics.
        patience (int): patience
    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    model = build_ssd(
        input_shape=input_shape,
        num_classes=num_classes,
        base_model_filters=base_model_filters,
        learning_rate=learning_rate,
        gpu_optimized=gpu_optimized,
        metrics=metrics
    )

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    evaluation_metrics = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return {
        "model": model,
        "history": history.history,
        "metrics": dict(zip(model.metrics_names, evaluation_metrics))
    }

"""
# Example usage:
if __name__ == "__main__":
    # Assuming train_data, train_labels, val_data, val_labels, test_data, and test_labels are defined
    result = ssd(
        input_shape=(300, 300, 3),
        num_classes=21,
        train_data=None,  # Replace with actual data
        train_labels=None,  # Replace with actual labels
        val_data=None,  # Replace with actual data
        val_labels=None,  # Replace with actual labels
        test_data=None,  # Replace with actual data
        test_labels=None,  # Replace with actual labels
        base_model_filters=64,
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        gpu_optimized='yes',
        metrics=['accuracy', 'precision'],
        patience=5
    )
    print("Evaluation Metrics:", result["metrics"])
"""

# 57

def build_vit_gpt2(
    input_shape=(224, 224, 3),
    num_classes=10,
    vit_model_name="google/vit-base-patch16-224",
    gpt2_model_name="gpt2",
    learning_rate=0.001,
    gpu_optimized='yes',
    metrics=['accuracy']
):
    """
    Builds a multi-modal model combining Vision Transformer (ViT) for image features
    and GPT-2 for text features.

    Parameters:
    - input_shape: Tuple, shape of the input images (default: (224, 224, 3)).
    - num_classes: Integer, number of output classes (default: 10).
    - vit_model_name: String, pretrained ViT model name (default: "google/vit-base-patch16-224").
    - gpt2_model_name: String, pretrained GPT-2 model name (default: "gpt2").
    - learning_rate: Float, learning rate for the optimizer (default: 0.001).
    - gpu_optimized: String, 'yes' to enable GPU optimization using CUDA (default: 'yes').
    - metrics: List, performance metrics to evaluate the model (default: ['accuracy']).

    Returns:
    - A compiled multi-modal model.
    """
    # Configure GPU settings if needed
    if gpu_optimized.lower() == 'yes':
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            try:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                print("GPU optimization enabled.")
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
        else:
            print("No GPU detected, running on CPU.")

    # Load Vision Transformer (ViT) model
    vit = ViTModel.from_pretrained(vit_model_name)
    vit.trainable = False

    # Load GPT-2 model
    gpt2 = GPT2Model.from_pretrained(gpt2_model_name)
    gpt2.trainable = False

    # Image input branch
    image_input = Input(shape=input_shape, name="image_input")
    vit_features = vit(image_input.pixel_values).last_hidden_state  # Extract features
    vit_features = Flatten()(vit_features)

    # Text input branch
    text_input = Input(shape=(None,), dtype="int32", name="text_input")
    gpt2_features = gpt2(text_input.input_ids).last_hidden_state  # Extract features
    gpt2_features = Flatten()(gpt2_features)

    # Concatenate features
    combined_features = concatenate([vit_features, gpt2_features])

    # Fully connected layers for classification
    x = Dense(512, activation="relu")(combined_features)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax", name="output_layer")(x)

    # Define the model
    model = Model(inputs=[image_input, text_input], outputs=output)

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics)

    return model

def vit_gpt2(
    input_shape=(224, 224, 3),
    num_classes=10,
    train_data=None,
    train_labels=None,
    val_data=None,
    val_labels=None,
    test_data=None,
    test_labels=None,
    vit_model_name="google/vit-base-patch16-224",
    gpt2_model_name="gpt2",
    learning_rate=0.001,
    batch_size=32,
    epochs=10,
    gpu_optimized='yes',
    metrics=['accuracy'],
    patience=5
):
    """
    Builds, trains, and evaluates a ViT-GPT2 multi-modal model.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.
        train_data (np.array): Training data.
        train_labels (np.array): Training labels.
        val_data (np.array): Validation data.
        val_labels (np.array): Validation labels.
        test_data (np.array): Test data.
        test_labels (np.array): Test labels.
        vit_model_name (str): Pretrained ViT model name.
        gpt2_model_name (str): Pretrained GPT-2 model name.
        learning_rate (float): Learning rate.
        batch_size (int): Batch size.
        epochs (int): Number of epochs.
        gpu_optimized (str): Whether to use GPU ('yes' or 'no').
        metrics (list): List of evaluation metrics.
        patience (int): patience
    Returns:
        dict: Contains the model, training history, and evaluation metrics.
    """
    model = build_vit_gpt2(
        input_shape=input_shape,
        num_classes=num_classes,
        vit_model_name=vit_model_name,
        gpt2_model_name=gpt2_model_name,
        learning_rate=learning_rate,
        gpu_optimized=gpu_optimized,
        metrics=metrics
    )

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_data, train_labels,
        validation_data=(val_data, val_labels),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    evaluation_metrics = model.evaluate(test_data, test_labels, batch_size=batch_size)

    return {
        "model": model,
        "history": history.history,
        "metrics": dict(zip(model.metrics_names, evaluation_metrics))
    }

"""
# Example usage:
if __name__ == "__main__":
    # Assuming train_data, train_labels, val_data, val_labels, test_data, and test_labels are defined
    result = vit_gpt2(
        input_shape=(224, 224, 3),
        num_classes=10,
        train_data=None,  # Replace with actual data
        train_labels=None,  # Replace with actual labels
        val_data=None,  # Replace with actual data
        val_labels=None,  # Replace with actual labels
        test_data=None,  # Replace with actual data
        test_labels=None,  # Replace with actual labels
        vit_model_name="google/vit-base-patch16-224",
        gpt2_model_name="gpt2",
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        gpu_optimized='yes',
        metrics=['accuracy', 'precision'],
        patience=5
    )
    print("Evaluation Metrics:", result["metrics"])
"""