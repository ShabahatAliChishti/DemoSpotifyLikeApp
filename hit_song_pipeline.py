import kfp
from kfp import dsl

# Replace this with your registry image (must contain faiss, xgboost, pandas, numpy, sklearn, kfp)
BASE_IMAGE = "local-kfp-faiss-xgb:1.0"

# -------------------------------
# Step 1: Data Generation
# -------------------------------
@dsl.component(base_image=BASE_IMAGE)
def generate_song_data(output_csv: dsl.OutputPath()):
    import pandas as pd
    import numpy as np

    artists = ["Ariana Grande", "The Weeknd", "Drake", "Billie Eilish", "Ed Sheeran"]
    songs = []
    for i in range(300):
        artist = np.random.choice(artists)
        songs.append({
            "song_id": i,
            "song_name": f"Song_{i}",
            "artist": artist,
            "tempo": int(np.random.randint(60, 180)),
            "energy": float(np.round(np.random.rand(), 4)),
            "danceability": float(np.round(np.random.rand(), 4)),
            "valence": float(np.round(np.random.rand(), 4)),
            # popularity simulated as binary (0/1)
            "popularity": int(np.random.randint(0, 2))
        })

    df = pd.DataFrame(songs)
    df.to_csv(output_csv, index=False)
    print(f"✅ Song dataset saved to: {output_csv} (rows={len(df)})")


# -------------------------------
# Step 2: Build FAISS index
# -------------------------------
@dsl.component(base_image=BASE_IMAGE)
def build_faiss_index(data_csv: dsl.InputPath(), index_out: dsl.OutputPath()):
    import pandas as pd
    import numpy as np
    import faiss

    df = pd.read_csv(data_csv)
    features = df[["tempo", "energy", "danceability", "valence"]].to_numpy().astype("float32")

    # L2 flat index
    d = features.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(features)

    # Save index directly to file
    faiss.write_index(index, index_out)

    print(f"✅ FAISS index built and saved to: {index_out} (n={index.ntotal})")

# -------------------------------
# Step 3: Train XGBoost model
# -------------------------------
@dsl.component(base_image=BASE_IMAGE)
def train_xgb_model(data_csv: dsl.InputPath(), model_out: dsl.OutputPath()):
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(data_csv)
    X = df[["tempo", "energy", "danceability", "valence"]]
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Trained XGBoost — test accuracy: {acc:.4f}")

    # save model
    model.save_model(model_out)
    print(f"✅ XGBoost model saved to: {model_out}")


# -------------------------------
# Step 4: Evaluate Model (whole dataset)
# -------------------------------
@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(data_csv: dsl.InputPath(), model_path: dsl.InputPath(), accuracy_out: dsl.OutputPath()):
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(data_csv)
    X = df[["tempo", "energy", "danceability", "valence"]]
    y = df["popularity"]

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"✅ Model Accuracy (on full dataset): {acc:.4f}")

    with open(accuracy_out, "w") as f:
        f.write(str(acc))


# -------------------------------
# Step 5: Predict and show similar songs using FAISS
# -------------------------------
@dsl.component(base_image=BASE_IMAGE)
def predict_and_similar(
    data_csv: dsl.InputPath(),
    model_path: dsl.InputPath(),
    index_path: dsl.InputPath(),
    result_out: dsl.OutputPath()
):
    import pandas as pd
    import xgboost as xgb
    import numpy as np
    import faiss
    import json

    # Load data
    df = pd.read_csv(data_csv)
    features = df[["tempo", "energy", "danceability", "valence"]].to_numpy().astype("float32")

    # Load FAISS index correctly
    index = faiss.read_index(index_path)

    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Choose a random song (or pick by id)
    song_idx = int(np.random.randint(0, len(df)))
    song_row = df.iloc[song_idx]
    song_feat = features[song_idx: song_idx+1]

    # Predict popularity
    pred = model.predict(song_feat)[0]
    pred_proba = float(model.predict_proba(song_feat)[0][1]) if hasattr(model, "predict_proba") else None

    # Top-5 similar songs via FAISS
    k = 5
    distances, indices = index.search(song_feat, k)
    similar = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[int(idx)]
        similar.append({
            "song_id": int(row["song_id"]),
            "song_name": row["song_name"],
            "artist": row["artist"],
            "distance": float(dist),
            "popularity": int(row["popularity"])
        })

    # Save result
    result = {
        "queried_song": {
            "song_id": int(song_row["song_id"]),
            "song_name": song_row["song_name"],
            "artist": song_row["artist"],
            "predicted_popularity": int(pred),
            "predicted_popularity_proba": pred_proba
        },
        "similar_songs": similar
    }

    with open(result_out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"✅ Prediction & similar songs saved to: {result_out}")
    print("Queried song:", result["queried_song"])
    print("Top similar songs:")
    for s in similar:
        print(f" - {s['song_name']} by {s['artist']} (dist={s['distance']:.4f}, popularity={s['popularity']})")

# -------------------------------
# Pipeline Definition
# -------------------------------
@dsl.pipeline(name="spotify-style-faiss-xgb-pipeline")
def spotify_pipeline():
    data_task = generate_song_data()
    faiss_task = build_faiss_index(data_csv=data_task.outputs["output_csv"])
    train_task = train_xgb_model(data_csv=data_task.outputs["output_csv"])
    eval_task = evaluate_model(
        data_csv=data_task.outputs["output_csv"],
        model_path=train_task.outputs["model_out"]
    )
    predict_task = predict_and_similar(
        data_csv=data_task.outputs["output_csv"],
        model_path=train_task.outputs["model_out"],
        index_path=faiss_task.outputs["index_out"]
    )


if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=spotify_pipeline,
        package_path="spotify_faiss_xgb_pipeline.yaml"
    )
    print("✅ Compiled pipeline to spotify_faiss_xgb_pipeline.yaml")
