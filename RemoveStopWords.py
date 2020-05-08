import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

file = "100_recovery_posts_with_numbers.csv"
df = pd.read_csv(file)
posts = df['selftext']

# custom stop word list
stop = ['ve','im','gt','dont','don','did','didn',
        'didnt','doing','does','doesnt','lot','ive','got','id',
        'feel', 'youre','thats','like','just','going','ill','let',
        'tl','theyre','youll','non','isnt','get','getting','able','wasn','whats',
        'amp','wont','having','wouldnt','havent','hes','wasnt','went','gonna','ll','m','t','s','r', 'i']

# store stop words in set for efficiency
stop_words = set(stop)

# lowercase words in posts so that stopwords takes out all instances
lower = posts.str.lower()

tokenized = lower.apply(word_tokenize)

# remove stop words, return posts to string format
tokens_without_sw = tokenized.apply(lambda x: [word for word in x if not word in stop_words])
detokenized = tokens_without_sw.apply(lambda x: TreebankWordDetokenizer().detokenize(x))

detokenized.to_csv("100_posts_final", header=False, index=False)