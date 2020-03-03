import numpy as np
import pandas as pd
from collections import Counter
from matplotlib import pyplot as plt

file = 'reddit-sw-2018.csv'

data = pd.read_csv(file, delimiter=',', dtype={"saved": str, "author_flair_css_class": str, "name": str,
                                                         "link_flair_text": str, "distinguished": str,
                                                         "selftext": str})

# Plots bar chart of the top words used in posts
def visualizeTopWords(topwords, counts):
    y_pos = np.arange(len(topwords))
    plt.bar(y_pos, counts)
    plt.xticks(y_pos, topwords)
    plt.ylabel('Number of times used')
    plt.xlabel('Words')
    plt.title('Top Words Used')
    plt.show()

author_col = data['author']
text_col = data['selftext']

# Get number of posts, insert in new column in DataFrame
total = author_col.count()
total_col = np.zeros(total)
total_col[0] = total
data['total'] = total_col

# Get unique users, insert in new column
unique = author_col.nunique()
unique_col = np.zeros(total)
unique_col[0] = unique
data['unique'] = unique_col

# Remove floats from column -
# https://stackoverflow.com/questions/38091444/how-to-drop-rows-not-containing-string-type-in-a-column-in-pandas - piRSquared
non_float_text = text_col.loc[text_col.apply(type) != float]

# Get word count for posts, then calculate mean, median, and range and inserts each in a new column. Currently counts
# [deleted] as a word.
word_counts = non_float_text.str.split().str.len()
mean_words = word_counts.mean()
median_words = word_counts.median()
maximum = word_counts.max()
minimum = word_counts.min()
count_range = maximum - minimum

mean_col = np.zeros(total)
mean_col[0] = mean_words
data['mean'] = mean_col

median_col = np.zeros(total)
median_col[0] = median_words
data['median'] = median_col

range_col = np.zeros(total)
range_col[0] = count_range
data['range'] = range_col

data['word_count'] = word_counts

# Get top words and their respective counts, inserts in DataFrame

# Counts words -
# https://stackoverflow.com/questions/29903025/count-most-frequent-100-words-from-sentences-in-dataframe-pandas - Joran Beasley
top_words = Counter(" ".join(non_float_text).split()).most_common(20)

word_list = []
count_list = []
for x in range(len(top_words)):
    word_list.append(top_words[x][0])
    count_list.append(top_words[x][1])

top_words_col = np.empty(total, dtype="<U20")
top_words_col[:20] = word_list
top_counts_col = np.zeros(total, dtype=int)
top_counts_col[:20] = count_list
data['top_words'] = top_words_col
data['top_counts'] = top_counts_col


# Overwrites original csv with DataFrame with additional columns for total words, unique users, word count of each post,
# mean, median, and range of word counts, and the top 20 words used with the amount of times each was used.
data.to_csv(r'reddit-sw-2018.csv')


visualizeTopWords(word_list, count_list)







