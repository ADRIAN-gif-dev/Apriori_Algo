import pandas as pd

# Load the file Adrian put in the repo
df = pd.read_csv('data/Groceries_dataset.csv')

# 1. Group the data by Member and Date (creating one "basket" per visit)
baskets = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list).reset_index()

# 2. Identify every unique item sold in the store
all_items = sorted(df['itemDescription'].unique())

# 3. Create the Weka File (.arff)
# This format is a "Checklist" that Weka's Apriori algorithm needs
with open('groceries_for_weka.arff', 'w') as f:
    f.write("@relation groceries\n\n")
    for item in all_items:
        # We tell Weka that for each item, it's either 't' (true) or '?' (missing)
        f.write(f"@attribute '{item}' {{t}}\n")
    
    f.write("\n@data\n")
    
    for items_in_basket in baskets['itemDescription']:
        # Create a row of 't' and '?' based on what was in the basket
        row = [('t' if item in items_in_basket else '?') for item in all_items]
        f.write(",".join(row) + "\n")

print("Success! You created 'groceries_for_weka.arff'.")