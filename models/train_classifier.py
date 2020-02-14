import re
import nltk
import pandas
import pickle
import argparse
import multiprocessing
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Download NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# Configure global constants
MAX_CPU = multiprocessing.cpu_count()


def load_data(database_filepath):
    """Load the data from the database into a pandas dataframe"""

    # Import the data
    engine  = create_engine("sqlite:///{}".format(database_filepath))
    query   = "select * from Message"
    df      = pandas.read_sql(query, engine)

    # Split the data into the X, Y, and cateogry_names sets
    columns         = ["id", "message", "original", "genre"]
    category_names  = list(filter(lambda x: x not in columns, df.columns.values))
    X               = df.message.values
    Y               = df[category_names].values

    return X, Y, category_names


def tokenize(text):
    """Tokenize the input text"""

    # Remove punctuation and convert to all lowercase
    text        = re.sub("[^a-zA-Z0-9]", " ", text)
    text        = text.lower()

    # Split into tokens and lemmatize
    tokens      = word_tokenize(text)
    lemmatizer  = WordNetLemmatizer()
    tokens      = list(map(lambda token: lemmatizer.lemmatize(token).strip(), tokens))

    return tokens


def build_model():
    """Build the machine learning pipeline"""

    # Configure the pipeline
    pipeline    = Pipeline([
        ("text_pipeline", Pipeline([
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
        ])),
        ("clf", MultiOutputClassifier(KNeighborsClassifier()))
    ])

    # Optimize the pipeline with grid search
    parameters  = {
        "text_pipeline__vect__ngram_range": ((1, 1), (1, 2), (2, 2)),
        "text_pipeline__vect__max_df": (0.5, 0.75, 1.0),
        "text_pipeline__vect__min_df": (0.0, 0.25, 0.5),
        "text_pipeline__vect__max_features": (None, 10000, 20000),
        "text_pipeline__tfidf__use_idf": (True, False),
        "text_pipeline__tfidf__sublinear_tf": (True, False),
        "clf__estimator": (KNeighborsClassifier(), RandomForestClassifier()),
    }

    cv  = GridSearchCV(pipeline, parameters, n_jobs=MAX_CPU)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the model"""
    
    #y_pred  = model.predict(X_test)
    #matrix  = confusion_matrix(Y_test, y_pred, labels=category_names)
    #print(matrix)

    accuracy_score   = model.score(X_test, Y_test)
    print(accuracy_score)



def save_model(model, model_filepath):
    """Save the model to a pickle file"""

    pickle.dump(model, model_filepath)


def main():
    """Main script program"""

    # Configure the argument parser
    parser  = argparse.ArgumentParser(description="Build, train, and evaluate the machine learning model")
    parser.add_argument("database_filepath", help="File path to the messages/categories database (*.db)")
    parser.add_argument("model_filepath", help="File path to the saved model (*.pkl)")

    # Parse the input arguments
    args                = parser.parse_args()
    database_filepath   = args.database_filepath
    model_filepath      = args.model_filepath

    # Execute the machine learning pipeline
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


# Execute the main program
if __name__ == "__main__":
    main()
