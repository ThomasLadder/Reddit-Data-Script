import pandas as pd

file = "recovery-labeled-dataset-8-27.csv"
df = pd.read_csv(file)


posts = df['selftext']
pattern = r'[0-9]'

numerical_posts = df.loc[posts.str.contains(pattern) == True]
pd.to_datetime(numerical_posts['created_utc'])
sorteddf = numerical_posts.sort_values(by=['created_utc'], ascending=False)
top_hundred = sorteddf[:100]

top_hundred.to_csv("100_recovery_posts_with_numbers.csv")




