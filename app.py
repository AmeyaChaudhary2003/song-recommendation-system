from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# read the data
df = pd.read_csv("data/song-dataset.csv", low_memory=False)[:11085]

# remove duplicates
df = df.drop_duplicates(subset="Song Name")

# drop Null values
df = df.dropna(axis=0)

# Combine more features
df["data"] = df["Artist Name"] + " " + df["Song Name"]

# Use TF-IDF vectorization
vectorizer = TfidfVectorizer()
vectorized = vectorizer.fit_transform(df["data"])
similarities = cosine_similarity(vectorized)

# Assign the new dataframe with `similarities` values
df_tmp = pd.DataFrame(similarities, columns=df["Song Name"], index=df["Song Name"]).reset_index()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        input_song = request.form["input_song"]
        if input_song in df_tmp.columns:
            recommendation = df_tmp.nlargest(10, input_song).reset_index()[["Song Name", "index"]]
            recommendation.columns = ["Song Name", "Artist Name"]
            return render_template("index.html", recommendations=recommendation.values[1:])
        else:
            return render_template("index.html", message="Sorry, the song is not in our database.")
    return render_template("index.html", recommendations=None, message=None)



if __name__ == "__main__":
    app.run(debug=True)

