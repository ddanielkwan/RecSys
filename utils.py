from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import pandas as pd
import animation
import csv
import time
from heapq import heappush, heappop
import os.path
import torch

from transformers import AutoTokenizer, AutoModel

class RecomendationSystem:

    @animation.wait(animation=["Initialize System.", "Initialize System..", "Initialize System...", 'Initialize System     '], color='green')
    def __init__(self, user_id):
        self.movies_df = pd.read_csv("./ml-latest-small/movies.csv")
        self.tags_df = pd.read_csv("./ml-latest-small/tags.csv")
        self.user_df = self.extract_users()
        self.ratings_df = self.extract_ratings()
        self.user = self.get_user(user_id)
        self.user_id = user_id
        self.friends = self.get_friends()
        self.model_name = "sentence-transformers/bert-base-nli-mean-tokens"
        self.token = None

    def get_user(self, user_id):
        user = self.user_df[self.user_df.index==user_id][["gender","age","occupation"]]
        return {"gender": user["gender"].to_list()[0], "age":user["age"].to_list()[0], "occupation":user["occupation"].to_list()[0]}

    def get_friends(self):
        same_gen_age_range = (self.user_df["gender"]==self.user["gender"])&(self.user_df["age"]<self.user["age"]+5)&(self.user_df["age"]>self.user["age"]-5)
        same_occ = (self.user_df["occupation"]==self.user["occupation"])
        relevent_users = self.user_df[same_gen_age_range | same_occ]
        
        return relevent_users.iloc[:1]

    def extract_users(self):
        columns_to_keep = ['userId', 'gender', "age", "occupation"]

        if os.path.isfile("./users.csv"):
            return pd.read_csv("./users.csv", usecols=columns_to_keep).set_index("userId")

        user_dat = open("./ml-1m/users.dat")
        datContent=[]
        for idx,val in enumerate(user_dat.readlines()):
            if idx==0:
                datContent.append(['userId', 'gender', "age", "occupation"])
            datContent.append(val.strip().split("::") )

        with open("./users.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(datContent)

        columns_to_keep = ['userId', 'gender', "age", "occupation"]
        return pd.read_csv("./users.csv", usecols=columns_to_keep).set_index(
        "userId")


    def extract_ratings(self):
        columns_to_keep = ['userId', 'movieId', "rating"]

        if os.path.isfile("./ratings.csv"):
            return pd.read_csv("./ratings.csv", usecols=columns_to_keep)

        ratings_dat = open("./ml-1m/ratings.dat")

        datContent=[]
        for idx,val in enumerate(ratings_dat.readlines()):
            if idx==0:
                datContent.append(['userId', 'movieId', "rating"])
            datContent.append(val.strip().split("::") )

        with open("./ratings.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(datContent)
        
        return pd.read_csv("./ratings.csv", usecols=columns_to_keep)


    def get_movie_id(self, movie_id):
        return self.movies_df[self.movies_df["movie_id"]==movie_id]["title"].to_string(index=False)

    
    def generate_bucket(self):
        bucket_of_tags = self.tags_df[["movieId","tag"]]
        bucket_of_tags = bucket_of_tags.groupby('movieId',as_index=False).agg(lambda x: ' '.join(map(str,x)))

        movies_tags = pd.merge(bucket_of_tags, self.movies_df, how="outer", on='movieId')
    
        del movies_tags["genres"]
    
        return movies_tags
    
    @animation.wait(animation=["Getting Movie Tags.", "Getting Movie Tags..", "Getting Movie Tags...", "Getting Movie Tags    "], color='green')
    def get_watched_tags(self, movie_id):
        bucket_of_tags = self.generate_bucket()
    
        friends_data = pd.merge(self.friends, self.ratings_df, how="inner", on='userId')
        users_friends_data = pd.merge(friends_data, bucket_of_tags, how="outer", on='movieId')[["movieId","rating","tag","title"]]
        users_friends_data = users_friends_data.dropna(how='any',subset=['rating',"tag"])

        users_friends_data = users_friends_data[users_friends_data["rating"]>=3.0]
    
        candidate = bucket_of_tags[bucket_of_tags["movieId"]==movie_id]
        _data = [candidate["movieId"].to_string(index=False), 0, candidate["tag"].to_string(index=False),candidate["title"].to_string(index=False)]
    
    
        if movie_id in users_friends_data["movieId"].to_list():
            users_friends_data = users_friends_data.reset_index()
            index  = users_friends_data[users_friends_data["movieId"]==movie_id].index.to_list()[0]-2
            indices = users_friends_data.index.to_list()
            indices[index], indices[-1] = indices[-1], indices[index]
            users_friends_data = users_friends_data.reindex(indices)
        else:
            
            users_friends_data.loc[-1] =_data
        
        return users_friends_data

    @animation.wait(animation=["Getting Not Watched Movie Tags.", "Getting Not Watched Movie Tags..", "Getting Not Watched Movie Tags...", "Getting Not Watched Movie Tags    "], color='green')
    def get_not_watched_tags(self):
        user_watched = self.ratings_df[self.ratings_df["userId"]==self.user_id]
        bucket_of_tags = self.tags_df[["movieId","tag"]]
        bucket_of_tags = bucket_of_tags.groupby('movieId').agg(lambda x: ' '.join(map(str,x))).reset_index()

        movies = []
        for i in bucket_of_tags["movieId"]:
            movies.append(self.movies_df[self.movies_df["movieId"]==i]["title"].to_string()[4:].strip())
    
        titles = pd.DataFrame(movies, columns=["title"])

        bucket_of_tags["title"] = titles.values

        not_watched_movies = bucket_of_tags[~bucket_of_tags["movieId"].isin(user_watched["movieId"])].dropna()
        watched_movies_tags = pd.merge(user_watched, bucket_of_tags, how="outer", on="movieId")

        watched_movies_tags = watched_movies_tags.dropna(how='any',subset=['rating',"tag", "userId", "title"])

        user_custom_movie = watched_movies_tags.groupby('userId',as_index=False).agg(lambda x: ' '.join(map(str,x)))
        user_custom_movie = user_custom_movie.loc[0]
        del user_custom_movie["rating"], 
        del user_custom_movie["userId"]
        not_watched_movies.loc[99]=user_custom_movie
        not_watched_movies = not_watched_movies[:99-2]

        return not_watched_movies

    def set_tokens(self, tokens):
        self.tokens = tokens

    
    @animation.wait(animation=["Tokenize Tags.", "Tokenize Tags..", "Tokenize Tags...", "Tokenize Tags     "], color='green')
    def train(self, movie_tags):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)

        tokens = {"input_ids": [], "attention_mask": []}

        for tag in movie_tags["tag"].to_list():
            new_tokens = tokenizer.encode_plus(tag, max_length=128, truncation=True, padding="max_length",return_tensors='pt')

            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])
        
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

        self.tokens = tokens

        output = model(**self.tokens)

        return output

    @animation.wait(animation=["Calculate Cosine Similarity.", "Calculate Cosine Similarity..", "Calculate Cosine Similarity...", '\\'], color='green')
    def cosine_sim(self, model):
        embedding = model.last_hidden_state
        attention = self.tokens["attention_mask"]
        mask = attention.unsqueeze(-1).expand(embedding.shape).float()
        mask_embeddings = embedding * mask


        summed = torch.sum(mask_embeddings, 1)

        counts = torch.clamp(mask.sum(1), min=1e-9)

        mean_pooled = summed / counts
        mean_pooled = mean_pooled.detach().numpy()

        sim = cosine_similarity([mean_pooled[-1]],mean_pooled)
        
        return sim

    @animation.wait(animation=["Calculating Weighted Mean.", "Calculating Weighted Mean..", "Calculating Weighted Mean..."], color='green')
    def get_weighted_average(self, rated_sim_movies):
        average = {}

        weight = [int(i*100) for i in rated_sim_movies["cosine_similarity"].to_list()]
        rating = rated_sim_movies["rating"].to_list()
        arr=[]

        for idx, val in enumerate(weight):
            arr.extend([[rating[idx]] * val])
    
        arr.sort(key = lambda x:x[1], reverse=True)
        mid = len(arr)//2
        
        # rated_sim_movies = rated_sim_movies.sort_values(by=['cosine_similarity'], ascending=False)
        average["weighted_mean"] = sum(rated_sim_movies["cosine_similarity"] * rated_sim_movies["rating"])/(sum(rated_sim_movies["cosine_similarity"]))

        average["weighted_median"] = arr[mid][0]
        return average

    @animation.wait(animation=["Getting Recommendations.", "Getting Recommendations..", "Getting Recommendations...","Getting Recommendations    "], color='green')
    def get_top_k_movies(self, movie_tags, k):
        movie_tags = movie_tags.sort_values(by=['cosine_similarity'], ascending=False)
        
        recommendations = [movie_tags["title"].to_list()[i] for i in range(1, k+1)]

        return recommendations