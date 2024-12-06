import kagglehub
import pandas as pd

# Download latest version
path = "creditcard.csv"
# prompt: i need a code to read and analyze the data in a csv file


# Replace 'your_file.csv' with the actual path to your CSV file
try:
    df = pd.read_csv(path)
    print(df.info())
    
    # Display some basic information about the DataFrame
    print("Shape of the DataFrame:", df.shape)  # (number of rows, number of columns)
    print("\nFirst 5 rows:\n", df.head())
    print("\nLast 5 rows:\n", df.tail())
    print("\nColumn names:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nDescriptive statistics:\n", df.describe(include='all')) # Includes non-numeric data
    # For specific columns: df[['column1', 'column2']].describe()
    
    # Check for missing values
    print("\nMissing values per column:\n", df.isnull().sum())
    
    df.fillna(df.mean(), inplace=True)  # Replace missing values with the mean of each column

    print("\nMissing values per column:\n", df.isnull().sum())
    #Identify duplicates
    print(df.duplicated().sum())
    #Remove duplicates
    df_no_duplicates = df.drop_duplicates()
    print(df_no_duplicates.duplicated().sum())
    

except FileNotFoundError:
    print(f"Error: File 'your_file.csv' not found. Please check the file path.")
except pd.errors.EmptyDataError:
    print(f"Error: 'your_file.csv' is empty.")
except pd.errors.ParserError:
    print(f"Error: Unable to parse 'your_file.csv'. Please ensure it's a valid CSV file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

