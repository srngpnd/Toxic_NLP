# Toxic_Comment_Classification_NLP

### Project for ISMT S-117 Text Analytics and Natural Language Processing at Harvard University

### Table of Content
- Overview
- Dataset Overview
- Data Preprocessing and EDA
- Model Fitting 
- Results

### Overview
Discussing daily events or important topic you care about can be difficult. The threat of abuse and harassment online means that many people stop expressing themselves and give up on seeking different opinions. Platforms struggle to effectively facilitate conversations, leading many communities to limit or completely shut down user comments.

As a student with great interest in Natural Language Processing, as well as a strong believer in making online discussion more productive and respectful, my aim is to build a model that is capable of detecting different types of toxicity like threats, obscenity, insults, and identity-based hate to control the hatred over the internet. This model would help various platforms to create a safe environment for a healthy conversation.

### Dataset Overview

The dataset we are using consists of comments from Wikipedia’s talk page edits. These comments have been labeled by human raters for toxic behavior. The types of toxicity are:
- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

There are 159,571 observations in the training dataset and 153,164 observations in the testing dataset. 

### Data Preprocessing and EDA

Since all of our data are text comments, we wrote our own `simple_tokenizer()` function, removing punctuations and special characters and lemmatizing the comments. After benchmarking between different vectorizers (TFIDFVectorizer and CountVectorizer), we chose TFIDFVectorizer, which provides us with better performance.

So Tf-idf stands for term frequency-inverse document frequency, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query. 

Due to aforementioned reason, tfidfVectorizer is the best vector generator in our scenario.

![alt text](https://github.com/srngpnd/Toxic_NLP/blob/master/Images/Distribution_1.png?raw=true) 

![alt text](https://github.com/srngpnd/Toxic_NLP/blob/master/Images/Distribution_2.png?raw=true) 

### Model Fitting

