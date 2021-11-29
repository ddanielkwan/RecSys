from rec_sys import RecSys
import sys

try:
    arg = sys.argv[1]
except:
    arg = None

if __name__=="__main__":

    user = input("Enter user id: ")
    
    print()
    
    system = RecSys(int(user))

    if arg=="TEST":
        movie = int(input("Enter movie id: "))
        print()
        rating = system.predict_rating(movie)

        print("=====Predicted Rating=====")
        print("User {} weighted mean for movie {}: {}".format(user, movie, rating["weighted_mean"]))
        print("User {} weighted median for movie {}: {}".format(user, movie, rating["weighted_median"]))
        

    else:
        recommendations = system.find_recommendations()
        
        print("=====Recommendations=====")
        for movie in recommendations:
            print(movie)

    print()
