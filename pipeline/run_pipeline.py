"""Run the pipeline end to end: python pipeline/run_pipeline.py [--fetch]

Without --fetch, starts from the downloaded data in data/raw/ (stage 1 is slow and only needs
running once). Then: LTS -> features -> synthetic census -> model -> front-end export.
"""
import sys

import export_frontend
import features
import fetch_lsoas
import fetch_osm
import lts
import synthetic_census
import train_model

if __name__ == "__main__":
    if "--fetch" in sys.argv:
        sys.argv = [sys.argv[0]]  # fetch_osm parses its own arguments
        fetch_lsoas.main()
        fetch_osm.main()
    for stage in (lts, features, synthetic_census, train_model, export_frontend):
        print(f"\n=== {stage.__name__} ===")
        stage.main()
    print("\nDone. Check the browser model with: node tests/frontend_parity.mjs")
