import pandas
import argparse
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Load in the messages and categories data sets"""

    df_messages = pandas.read_csv(messages_filepath)
    df_categories = pandas.read_csv(categories_filepath)
    df = df_messages.join(df_categories.set_index("id"), on="id")

    return df


def clean_data(df):
    """Clean the messages and categories dataframe"""
    
    # Rename the category columns and extract the values
    categories = df.categories.str.split(";", expand=True)
    category_names = categories.iloc[0].str.split("-", expand=True)[0].values
    categories.columns = category_names
    categories = categories.apply(lambda q: q.str.split("-").str[-1], axis=0)
    categories = categories.astype(int)

    # Drop the categories and join the cleaned data
    df.drop(columns=["categories"], inplace=True)
    df = df.join(categories)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """Save the dataframe to the database"""

    engine = create_engine("sqlite:///{}".format(database_filename))
    df.to_sql("Message", engine, index=False, if_exists="replace")


def main():
    """Main script program"""

    # Configure the argument parser
    parser  = argparse.ArgumentParser(
        description="Disaster response ETL pipeline"
    )
    parser.add_argument(
        "messages_filepath", 
        help="File path to the messages data set (CSV)"
    )
    parser.add_argument(
        "categories_filepath", 
        help="File path to the categories data set (CSV)"
    )
    parser.add_argument(
        "database_filepath", 
        help="File path to the database for storage"
    )

    # Parse the arguments
    args = parser.parse_args()
    messages_filepath = args.messages_filepath
    categories_filepath = args.categories_filepath
    database_filepath = args.database_filepath

    # Process the ETL pipeline
    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
          .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
    return

    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)


# Execute the main program
if __name__ == "__main__":
    main()

