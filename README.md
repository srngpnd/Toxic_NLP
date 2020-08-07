# Toxic_Comment_Classification_NLP

### Project for ISMT S-117 Text Analytics and Natural Language Processing at Northeastern University

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

Since all of our data are text comments, we wrote our own `tokenize()` function, removing punctuations and special characters, stemming and/or lemmatizing the comments, and filtering out comments with length below 3. After benchmarking between different vectorizers (TFIDFVectorizer and CountVectorizer), we chose TFIDFVectorizer, which provides us with better performance.

The major concern of the data is that most of the comments are clean (i.e., non-toxic). There are only a few observations in the training data for Labels like `threat`. This indicates that we need to deal with imbalanced classes later on and indeed, we use different methods, such as resampling, choosing appropriate evaluation metrics, and choosing robust models to address this problem.

### Model Fitting

