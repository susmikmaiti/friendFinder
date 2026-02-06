import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from flask import Flask, request, render_template, session, url_for
#from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = "supersecretkey"

#client = MongoClient("mongodb://localhost:27017/")
#db = client["matchmaking_db"]
#users_collection = db["users"]

# Loading the saved model from pickle file
with open("kmeans_matchmaking_model.pkl", "rb") as file:
    kmeans_model, scaler, df = pickle.load(file)

encoding_maps = {
    "status": {0: "single", 1: "available", 2: "seeing someone", 3: "married", 4: "unknown"},
    "sex": {0: "m", 1: "f"},
    "orientation": {0: "straight", 1: "bisexual", 2: "gay"},
    "drinks": {0: "not at all", 1: "rarely", 2: "socially", 3: "often", 4: "very often", 5: "desperately"},
    "drugs": {0: "never", 1: "sometimes", 2: "often"},
    "smokes": {0: "no", 1: "sometimes", 2: "when drinking", 3: "yes", 4: "trying to quit"},
}

def decode_match(match):
    # convert the numerical values back to categorical values(human readable form)
    for key, mapping in encoding_maps.items():
        if key in match:
            match[key] = mapping.get(match[key], "unknown")
    return match

def find_matches(test_user):
    user_orientation = test_user["orientation"]
    user_sex = test_user["sex"]

    def is_match(candidate):
        candidate_sex = candidate["sex"]
        candidate_orientation = candidate["orientation"]
        if user_orientation == 0:
            return (user_sex == 0 and candidate_sex == 1 and candidate_orientation in [0, 1]) or \
                   (user_sex == 1 and candidate_sex == 0 and candidate_orientation in [0, 1])
        elif user_orientation == 1:
            return candidate_orientation in [0, 1]
        elif user_orientation == 2:
            return user_sex == candidate_sex and candidate_orientation == 2
        return False

    # Converting input features to array and scale
    test_features = np.array([[test_user.get(key, 0) for key in ["age", "status", "orientation", "drinks", "drugs", "height", "smokes"]]])
    test_features_scaled = scaler.transform(test_features)

    # Predicting which cluster the input data belongs to
    cluster = kmeans_model.predict(test_features_scaled)[0]

    # Filtering potential matches based on orientation 
    cluster_candidates = df[df["Cluster"] == cluster].copy()
    candidates = cluster_candidates[cluster_candidates.apply(is_match, axis=1)].copy()

    if candidates.empty:
        return []  # Return an empty list if no candidates found

    # Scaling candidate features
    candidate_features = scaler.transform(candidates[["age", "status", "orientation", "drinks", "drugs", "height", "smokes"]])
    # Finding the candidates that lie the nearest with the input!
    match_distances = euclidean_distances(candidate_features, test_features_scaled).flatten()

    # Adding distance score and sort matches
    candidates.loc[:, "distance_score"] = match_distances
    candidates = candidates.sort_values(by=["distance_score"], ascending=True)

    top_matches = candidates.head(10).copy()
    top_matches["id"] = top_matches.index

    return [decode_match(match) for match in top_matches.to_dict(orient="records")]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            user_data = {key: int(request.form[key]) for key in ["age", "status", "sex", "orientation", "drinks", "drugs", "height", "smokes"]}
            session["user"] = user_data
            matches = find_matches(user_data)
            session["matches"] = matches
            return render_template("matches.html", matches=matches)
        except ValueError:
            return "Invalid input. Please enter valid numerical values.", 400
    return render_template("index.html")

@app.route("/matches")
def show_matches():
    try:
        if "matches" in session and session["matches"]:
            return render_template("matches.html", matches=session["matches"])
        else:
            return render_template("matches.html", message="No matches found. Please try again later.")
    except Exception as e:
        return render_template("error.html", message=f"An error occurred: {str(e)}"), 500

@app.route("/profile/<name>")
def profile(name):
    try:
        match = df.loc[df["Name"] == name].iloc[0].to_dict()
        match = decode_match(match)
        return render_template("profile.html", match=match)
    except IndexError:
        return "Match not found", 404

if __name__ == "__main__":
    app.run(debug=True)
