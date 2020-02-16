import re
import sys
import time
import nltk
import pandas
import pickle
import argparse
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix


# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")


def load_data(database_filepath):
    """Load the data from the database into a pandas dataframe"""

    # Import the data
    engine = create_engine("sqlite:///{}".format(database_filepath))
    query = "select * from Message"
    df = pandas.read_sql(query, engine)

    # Split the data into the X, Y, and cateogry_names sets
    cols = ["id", "message", "original", "genre"]
    category_names = list(filter(lambda x: x not in cols, df.columns.values))
    X = df.message.values
    Y = df[category_names].values

    return X, Y, category_names


def tokenize(text):
    """Tokenize the input text"""

    # Remove punctuation and convert to all lowercase
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    text = text.lower()

    # Split into tokens and lemmatize
    lemmatizer = WordNetLemmatizer()
    words = stopwords.words("english")
    tokens = word_tokenize(text)
    tokens = filter(lambda t: t not in words, tokens)
    tokens = list(map(lambda t: lemmatizer.lemmatize(t).strip(), tokens))

    return tokens


def build_model():
    """Build the machine learning pipeline"""

    # Configure the pipeline
    pipeline = Pipeline([
        ("text_pipeline", Pipeline([
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
        ])),
       ("clf", MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1))
    ])

    # Optimize the pipeline with grid search
    params_text = {
        "text_pipeline__vect__ngram_range": ((1, 1), (1, 2), (2, 2)),
        "text_pipeline__vect__max_df": (0.9, 1.0),
        "text_pipeline__vect__min_df": (0.0, 0.1),
        "text_pipeline__vect__max_features": (None, 10000, 20000),
        "text_pipeline__tfidf__use_idf": (True, False),
        "text_pipeline__tfidf__sublinear_tf": (True, False),
    }

    # Random forest classifier parameters
    params_randomforest = {
        "clf__estimator": [RandomForestClassifier()],
        "clf__estimator__n_estimators": [100, 500, 2000],
        "clf__estimator__max_features": ["auto", "log2"],
        "clf__estimator__max_depth": [None, 10, 20],
        "clf__estimator__bootstrap": [True, False],
        "clf__estimator__min_samples_leaf": [1, 2, 4],
        "clf__estimator__min_samples_split": [2, 5, 10],
    }

    # Complement NB classifier parameters
    params_complementnb = {
        "clf__estimator": [ComplementNB()],
        "clf__estimator__alpha": [0.1, 0.5, 1.0],
    }

    # Multinomial NB classifier parameters
    params_multinomialnb = {
        "clf__estimator": [MultinomialNB()],
        "clf__estimator__alpha": [0.1, 0.5, 1.0],
    }

    # MLP classifier parameters
    params_mlp = {
        "clf__estimator": [MLPClassifier()],
    }


    # Set the parameter grid
    params_randomforest.update(params_text)
    params_complementnb.update(params_text)
    params_multinomialnb.update(params_text)

    parameters = [
        params_randomforest,
#        params_complementnb,
#        params_multinomialnb,
#        params_mlp,
    ]

    return GridSearchCV(pipeline, parameters)


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model"""
   
    # Print the best parameters
    print(model.best_params_)

    # For each label, compute the metrics and store in the dataframe
    columns = [
        "label", 
        "accuracy", 
        "precision", 
        "recall", 
        "f1-score", 
        "support"
    ]
    averages = pandas.DataFrame(columns=columns)
    y_pred = model.predict(X_test)
    for i, label in enumerate(category_names):
        yi_true = Y_test[:, i]
        yi_pred = y_pred[:, i]
        report = classification_report(
            yi_true, 
            yi_pred, 
            output_dict=True, 
            zero_division=0
        )

        values = report["weighted avg"]
        values["accuracy"] = report["accuracy"]
        values["label"] = label
        averages = averages.append(values, ignore_index=True)
    
    # Compute the mean values and append to the dataframe
    mean = { "label": "AVERAGE" }
    mean.update( averages[columns[1:]].mean().to_dict() )
    averages = averages.append(mean, ignore_index=True)
    averages.to_csv("evaluation.csv", index=False)
    print(averages)


def save_model(model, model_filepath):
    """Save the model to a pickle file"""
    
    with open(model_filepath, "wb") as f:
        pickle.dump(model, f)


def main():
    """Main script program"""

    # Configure the argument parser
    parser = argparse.ArgumentParser(
        description="Build, train, and evaluate the machine learning model"
    )
    parser.add_argument(
        "database_filepath", 
        help="File path to the messages/categories database (*.db)"
    )
    parser.add_argument(
        "model_filepath", 
        help="File path to the saved model (*.pkl)"
    )

    # Parse the input arguments
    args = parser.parse_args()
    database_filepath = args.database_filepath
    model_filepath = args.model_filepath

    # Execute the machine learning pipeline
    start = time.time()
    print("Loading data...\n    DATABASE: {}".format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    print("Building model...")
    model = build_model()

    print("Training model...")
    model.fit(X_train, Y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, Y_test, category_names)

    print("Saving model...\n    MODEL: {}".format(model_filepath))
    save_model(model, model_filepath)

    print("Trained model saved!")
    
    end = time.time()
    print("\n\tModel trained in {:.3f}s\n".format(end-start))


# Execute the main program
if __name__ == "__main__":
    main()
