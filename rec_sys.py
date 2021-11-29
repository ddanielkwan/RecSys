from termcolor import colored, cprint
from utils import RecomendationSystem

class RecSys:
    def __init__(self, user):
        self.rec_sys = RecomendationSystem(user)
        
    def predict_rating(self, movie_id):
        
        # Find tags
        movie_tags = self.rec_sys.get_watched_tags(movie_id)

        # Train model
        model = self.rec_sys.train(movie_tags)

        # Generate cosine_sim
        cosin_sim = self.rec_sys.cosine_sim(model)

        movie_tags["cosine_similarity"] = cosin_sim[0]

        # Get weighted average
        rating = self.rec_sys.get_weighted_average(movie_tags)
        # Round to the nearest 0.5
        # rating = 0.5 * round(rating/0.5)
        return rating

    def find_recommendations(self):
        
        # Find tags of non-watched movies
        movie_tags = self.rec_sys.get_not_watched_tags()

        # Train model
        model = self.rec_sys.train(movie_tags)

        # Generate cosine_sim
        cosin_sim = self.rec_sys.cosine_sim(model)

        movie_tags["cosine_similarity"] = cosin_sim[0]

        # Get top k movies
        k = 5
        recommendations = self.rec_sys.get_top_k_movies(movie_tags, k)
        return recommendations

