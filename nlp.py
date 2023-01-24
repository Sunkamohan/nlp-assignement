import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords

# Load the data from the csv file
data = pd.read_csv('news.csv',dtype = "unicode")

#drop rows with NaN in content column
data = data.dropna(subset=['content'])

# Access the content column
content = data['content']

# Perform test-train split
train_data, test_data = train_test_split(content, test_size=0.1)

# text cleaning
def clean_text(text):
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # convert to lowercase
    text = text.lower()
    # remove numbers
    text = ''.join([i for i in text if not i.isdigit()])
    return text

# Clean the train data and test data
train_data = train_data.apply(lambda x: clean_text(x))
test_data = test_data.apply(lambda x: clean_text(x))

# Use TF-IDF to identify and eliminate unnecessary and redundant sentences
tfidf = TfidfVectorizer()
train_tfidf = tfidf.fit_transform(train_data)
test_tfidf = tfidf.transform(test_data)

# Use cosine similarity to compare the vector representations of sentences and identify similar sentences
cosine_similarity = cosine_similarity(test_tfidf, train_tfidf)

# Set threshold for similarity
threshold = 0.8

# Remove the identified similar sentences from the test data
cleaned_test_data = []
for index, (row, content) in enumerate(zip(cosine_similarity, test_data)):
    if max(row) < threshold:
        cleaned_test_data.append(content)

# Convert cleaned_test_data to a Series
cleaned_test_data = pd.Series(cleaned_test_data)

# Create a table with the specified columns for the test set
result_table = pd.DataFrame(columns=['Original Content', 'New Content', 'Removed Lines', 'Further Metrics'])
result_table = result_table[result_table['New Content'].isin(cleaned_test_data)]
result_table['Original Content'] = test_data
result_table['New Content'] = cleaned_test_data
result_table['Removed Lines'] = [i for i in test_data if i not in cleaned_test_data]

#Counting the number of stop words in the cleaned test data and store it in the 'Further Metrics' column
stop_words = set(stopwords.words('english'))
result_table['Further Metrics'] = result_table['New Content'].apply(lambda x: len([word for word in x.split() if word in stop_words]) if type(x) == str else None)

# Export the result table to a csv file
result_table.to_csv('cleaned_news_data.csv', index=False)
