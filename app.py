import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from flask import Flask, request, render_template, session, url_for, redirect

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ── Load model bundle ──────────────────────────────────────────────────────
with open("kmeans_matchmaking_model.pkl", "rb") as f:
    kmeans_model, scaler, df_model = pickle.load(f)

# ── Load full CSV for rich profile data ────────────────────────────────────
df_full = pd.read_csv("okCupidCleanedAndNamed.csv")

# Attach Name column from model df into full df (aligned by original index)
# df_model was built from the cleaned working subset; merge on positional alignment
# We'll look up profiles from df_full using the row index stored in df_model
df_model = df_model.reset_index(drop=False)   # keeps original CSV index as 'index' col

CLUSTER_FEATURES = ["age", "status", "sex", "orientation", "drinks", "drugs", "height", "smokes"]

# ── Encoding / decoding maps ───────────────────────────────────────────────
ENCODING_MAPS = {
    "status":      {"single": 0, "available": 1, "seeing someone": 2, "married": 3, "unknown": 4},
    "sex":         {"m": 0, "f": 1},
    "orientation": {"straight": 0, "bisexual": 1, "gay": 2},
    "drinks":      {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5},
    "drugs":       {"never": 0, "sometimes": 1, "often": 2},
    "smokes":      {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4},
}

DECODING_MAPS = {k: {v: k2 for k2, v in m.items()} for k, m in ENCODING_MAPS.items()}
# Override sex decoding to be readable
DECODING_MAPS["sex"] = {0: "Male", 1: "Female"}
DECODING_MAPS["orientation"] = {0: "Straight", 1: "Bisexual", 2: "Gay"}

TOP_N_CLUSTERS = 5   # search nearest N cluster centres
MAX_MATCHES    = 50  # return up to this many matches


def compatible(user_sex, user_ori, cand_sex, cand_ori):
    """Return True if candidate is a compatible match for the user."""
    if user_ori == 0:   # straight — wants opposite sex, open to straight/bi
        return (
            (user_sex == 0 and cand_sex == 1 and cand_ori in [0, 1]) or
            (user_sex == 1 and cand_sex == 0 and cand_ori in [0, 1])
        )
    elif user_ori == 1:  # bisexual — open to any orientation
        return cand_ori in [0, 1, 2]
    elif user_ori == 2:  # gay — same sex, gay or bisexual
        return user_sex == cand_sex and cand_ori in [1, 2]
    return False


def find_matches(user_dict):
    """
    Given an encoded user dict, return up to MAX_MATCHES decoded match records.
    Each record contains clustering features (decoded) + Name + distance_score.
    """
    user_vec    = np.array([[user_dict[k] for k in CLUSTER_FEATURES]])
    user_scaled = scaler.transform(user_vec)

    # ── Find TOP_N_CLUSTERS nearest centres ───────────────────────────────
    centre_dists = euclidean_distances(user_scaled, kmeans_model.cluster_centers_).flatten()
    top_clusters = np.argsort(centre_dists)[:TOP_N_CLUSTERS].tolist()

    # ── Filter by cluster membership ──────────────────────────────────────
    nearby = df_model[df_model["Cluster"].isin(top_clusters)].copy()

    # ── Compatibility filter ───────────────────────────────────────────────
    u_sex = user_dict["sex"]
    u_ori = user_dict["orientation"]

    mask = nearby.apply(
        lambda row: compatible(u_sex, u_ori, int(row["sex"]), int(row["orientation"])),
        axis=1
    )
    candidates = nearby[mask].copy()

    if candidates.empty:
        return []

    # ── Rank by Euclidean distance in scaled space ─────────────────────────
    cand_scaled = scaler.transform(candidates[CLUSTER_FEATURES].values)
    dists       = euclidean_distances(cand_scaled, user_scaled).flatten()
    candidates  = candidates.copy()
    candidates["distance_score"] = dists
    candidates  = candidates.sort_values("distance_score").head(MAX_MATCHES)

    # ── Decode for display ─────────────────────────────────────────────────
    results = []
    for _, row in candidates.iterrows():
        record = {
            "Name":           row.get("Name", "—"),
            "age":            int(row["age"]),
            "status":         DECODING_MAPS["status"].get(int(row["status"]), "—"),
            "sex":            DECODING_MAPS["sex"].get(int(row["sex"]), "—"),
            "orientation":    DECODING_MAPS["orientation"].get(int(row["orientation"]), "—"),
            "drinks":         DECODING_MAPS["drinks"].get(int(row["drinks"]), "—"),
            "drugs":          DECODING_MAPS["drugs"].get(int(row["drugs"]), "—"),
            "smokes":         DECODING_MAPS["smokes"].get(int(row["smokes"]), "—"),
            "height":         float(row["height"]),
            "distance_score": round(float(row["distance_score"]), 4),
            "csv_index":      int(row["index"]),   # original CSV row — used for profile lookup
        }
        results.append(record)

    return results


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            user_data = {
                k: int(request.form[k])
                for k in ["age", "status", "sex", "orientation", "drinks", "drugs", "height", "smokes"]
            }
            session["user"]    = user_data
            matches            = find_matches(user_data)
            session["matches"] = matches
            return render_template("matches.html", matches=matches, count=len(matches))
        except (ValueError, KeyError) as e:
            return render_template("index.html", error=f"Invalid input: {e}")
    return render_template("index.html")


@app.route("/matches")
def show_matches():
    matches = session.get("matches", [])
    if not matches:
        return render_template("matches.html", matches=[], message="No matches found. Go back and try again.")
    return render_template("matches.html", matches=matches, count=len(matches))


@app.route("/profile/<int:csv_index>")
def profile(csv_index):
    try:
        row   = df_full.iloc[csv_index].to_dict()
        # Decode the 8 clustering fields so they display as text
        for field, mp in DECODING_MAPS.items():
            if field in row:
                encoded_val = ENCODING_MAPS[field].get(
                    str(row[field]).strip().lower(), None
                )
                if encoded_val is not None:
                    row[field] = mp.get(encoded_val, row[field])
        # Attach the Name from the model df if it exists
        name_row = df_model[df_model["index"] == csv_index]
        row["Name"] = name_row["Name"].values[0] if not name_row.empty else "Profile"
        return render_template("profile.html", match=row)
    except (IndexError, KeyError):
        return render_template("error.html", message="Profile not found."), 404


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="Page not found."), 404

@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", message=f"Server error: {e}"), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    