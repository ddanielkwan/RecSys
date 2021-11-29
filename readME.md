# Group 7 Recommendation System

## **Table of Contents**
1. [Approach](#approach)
2. [Data Parcing](#data-parcing)
3. [NLP Tokenizing](#nlp-token)
4. [Test](#test)
5. [How to run](#run)

## **Approach**<a name="approach"></a>
***
For our approach we used a hybrid (collaborative-filtering & content based) approach to recommend a user, movies. 
We first ask the user for a user_id and movie_id. The algorithm then collects the data required data it needs to accurately recommend movies. 
The algorithm then constructs a "user character" to represent the types of movies the user likes (movies the user has watched and rated 3.0 stars or more) by combining their tags. 
The algorithm then finds the cosine similarity of the "user character" and movies the user has not yet watched and returns the first k (variable) similar movies.

## **Data Parcing**<a name="data-parcing"></a>
***
To accurately recommend a movie to a user, we needed specific data to analyze. The algorithm creates a **self.friends** dataframe to represent the other users with the same gender and within the same age range OR the same occupation as user.

```python
    ...
    self.friends=self.get_friends()
    ...
    def get_friends(self):
        same_gen_age_range = (self.user_df["gender"]==self.user["gender"])&(self.user_df["age"]<self.user["age"]+5)&(self.user_df["age"]>self.user["age"]-5)
        same_occ = (self.user_df["occupation"]==self.user["occupation"])
        relevent_users = self.user_df[same_gen_age_range | same_occ]
        '''
            return only 1 since returning all friends will trigger 
            a kernel kill
        '''
        return relevent_users.iloc[:1]
```

This will be usefull later to accurately recommend users.

We also have a 

```python
def get_not_watched_tags():
    ...
    return not_watched_movies
```
function that that creates the "user character" and appends it to a dataframe called **not_watched_movies** of movies the user has not yet watched. The "user character" is created by grouping all tags of the movies that the user has watched with a rating of 3.0 or higher. The **not_watched_movies** is a dataframe of 100 movies that the user has not yet watched (the algorithm with recommend a movie based on these 100 movies since having more movie data triggers a kernel kill) and the "user character" at the end. 

## **NLP Tokenizing**<a name="nlp-token"></a> 
***
To calculate the similarity between users, our first approach was to calculate the similarity between words. For example if one movie had the tags "funny pixar" and another movie had the tags "funny pixar" we would return a cosine similarity of 1.00. Though this is not very accurate. If one movie had the tags "blood" and another had the tags "bloody" when we compare the words, we get a cosine similarity of 0.00 since the words "blood" and "bloody" don't have no similarity in words. 

To bypass this error, we tried a different approach to find the levenshtiens distance between each word. The movies with the tags with the smallest levenshtiens distance to the tags of the "user character" were the most similar movie. This approach would work, but it is very inefficient since it has a time complexity of O(n^3).

We decided to go with a different approach and that was using tokenization. We thought this was our best approach since some tags are sentences, some are synonyms of other tags, so using NLP and tokenizers would accurately vectorize tags so that we can find the cosine similarity easier. We start by vecorizing the tags of each movie using BERT and pytorch to get the semantic embeddings of each tag. Then we use mean_pooling to represent each tag as a vector (This solves our issue with the "blood" and "bloody" mentioned above). We then find the cosine simalarity with the "user character" and the k=5 nearest movies. The algorithm then returns the movies and recommends them to the user.


## **Test**<a name="test"></a>
***
To test if our algorithm is recommending movies our user would like. We created an algorithm to predict the rating of a movie watched or not yet watched. 

We first ask the user to input a user_id and a movie_id. This movie ID can be a movie that has or hasn't not been rated yet.

In this algorithm we use a user and their friends (mentioned above) previously rated movies to predict the rating of the inputted movie id. We then use the same NLP Tokenizer model as the recommender to vectorize the tags of the movies. We then find the cosine_similarity between the inputted movie and the rated movies in the ___ variable. 

We calculated different predictions. We used weighted mean and weighted meadian to predict the rating of the inputted movie. You can compare the returned value with it's rated value if the movie has been rated by the user. If the movie has not been rated by the user, our algorithm can still give an educated prediction on what they would rate the movie.

## **How to run**<a name="run"></a>
***
Navigate in the terminal into the directory that contains main.py and run the following line
```bsh
pip3 source venv/bin/activate
```
To activate the virtual environment. Then run the following to install all the required dependencies
```bsh
pip3 install -r requirements.txt
```
After you have activated the venv and installed all the dependencies, you can run the following code to run the recommendatio system
```bsh
python3 main.py 
```
This will prompt you to input a user id. Input a user id of your choice.

The code will run a loading animation and tell you which stage of the recommendation system the system is at currently. Sit tight.

When the code has executed it will print the recommended movies.

To predict the rating of a movie run the following code in the same directory

```bsh
python3 main.py TEST
```
This will prompt you to input a user id. Input a user id of your choice.

Then it will prompt you to enter a movie id. Input a movie id of your choice.

When the code has executed it will print the predicted rating of the movie.
