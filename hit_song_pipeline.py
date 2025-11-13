import kfp
from kfp import dsl

BASE_IMAGE = "local-kfp-xgb:1.0"

# -------------------------------
# Step 1: Data Generation
# -------------------------------
@dsl.component(base_image=BASE_IMAGE)
def generate_data(output_csv: dsl.OutputPath()):
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({
        'tempo': np.random.randint(60, 180, 100),
        'energy': np.random.rand(100),
        'danceability': np.random.rand(100),
        'valence': np.random.rand(100),
        'hit': np.random.randint(0, 2, 100)
    })

    df.to_csv(output_csv, index=False)
    print(f"✅ Data generated and saved to: {output_csv}")

# -------------------------------
# Step 2: Model Training
# -------------------------------
@dsl.component(base_image=BASE_IMAGE)
def train_model(input_csv: dsl.InputPath(), model_out: dsl.OutputPath()):
    import pandas as pd
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_csv)
    X = df[['tempo', 'energy', 'danceability', 'valence']]
    y = df['hit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    model.save_model(model_out)
    print(f"✅ Model saved to: {model_out}")

# -------------------------------
# Step 3: Model Evaluation
# -------------------------------
@dsl.component(base_image=BASE_IMAGE)
def evaluate_model(data_csv: dsl.InputPath(), model_path: dsl.InputPath(), accuracy_out: dsl.OutputPath()):
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(data_csv)
    X = df[['tempo', 'energy', 'danceability', 'valence']]
    y = df['hit']

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    print(f"✅ Model Accuracy: {acc}")

    with open(accuracy_out, 'w') as f:
        f.write(str(acc))

# -------------------------------
# Step 4: Pipeline Definition
# -------------------------------
@dsl.pipeline(name="hit-song-pipeline")
def hit_song_pipeline():
    # Step 1: generate data
    generate_task = generate_data()

    # Step 2: train model
    train_task = train_model(input_csv=generate_task.outputs['output_csv'])

    # Step 3: evaluate model
    evaluate_task = evaluate_model(
        data_csv=generate_task.outputs['output_csv'],
        model_path=train_task.outputs['model_out']
    )

# -------------------------------
# Step 5: Compile Pipeline
# -------------------------------
if __name__ == "__main__":
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=hit_song_pipeline,
        package_path="hit_song_pipeline_fixed.yaml"
    )
    print("✅ Pipeline compiled successfully! Upload 'hit_song_pipeline_fixed.yaml' to Kubeflow UI.")
