"""Run every stage end to end: python pipeline/run_pipeline.py [--skip-training]"""
import sys

import build_features
import export_frontend
import generate_dummy_data
import train_model

if __name__ == "__main__":
    generate_dummy_data.main()
    build_features.main()
    if "--skip-training" not in sys.argv:
        train_model.main()
    else:
        print("Skipping training: front end will use the committed frontend/data/model.json")
    export_frontend.main()
