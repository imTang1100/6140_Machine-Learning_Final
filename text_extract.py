import pandas as pd

# Load the Parquet file
df = pd.read_parquet('train-00000-of-00001-023bf2eb9de82a2c.parquet')

# Check the first few entries to ensure it's loaded correctly
print(df.head())

# Save the 'text' column to a CSV file
df[['text']].to_csv('output_descriptions.csv', index=False)

