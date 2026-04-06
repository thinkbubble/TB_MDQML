import sys
import types
import pandas as pd
import os

sys.modules['pyedflib'] = types.ModuleType('pyedflib')
sys.modules['pyedflib.highlevel'] = types.ModuleType('highlevel')

from preprocess_data import sensing_and_signal_ingestion
from load_data import load_preprocessed_csv
from .main import run_pipeline
from .visualize import generate_visualizations


def main():

    my_folder = 'team_sahith'
    raw_data = f'./{my_folder}/raw_data'
    cleaned_data = f'./{my_folder}/cleaned_data'
    checkpoint = f'./{my_folder}/job_data.csv'

    _, mapping, _ = sensing_and_signal_ingestion(raw_data, cleaned_data, checkpoint)

    csv_files = mapping.get("csv", {})
    all_results = []

    for file_name, path in csv_files.items():

        print("\nProcessing:", file_name)

        data = load_preprocessed_csv(path)

        final, results, cm, preview, nulls, corr, feature_importance = run_pipeline(data)

        if final is None:
            continue

        print("Best Model:", final["best_model"])
        print("Best Score:", round(final["score"], 2))

        print("\nModel Scores:")
        for r in results:
            print(f"{r['model']} -> {round(r['score'], 2)}")

        print("\nPreview:\n", preview)
        print("\nMissing Values:\n", nulls)

        # Save reports
        report_folder = f"{my_folder}/reports/{file_name.replace('.csv','')}"
        os.makedirs(report_folder, exist_ok=True)

        preview.to_csv(f"{report_folder}/preview.csv", index=False)
        nulls.to_csv(f"{report_folder}/missing_values.csv")
        corr.to_csv(f"{report_folder}/correlation.csv")
        pd.DataFrame(results).to_csv(f"{report_folder}/model_results.csv", index=False)

        if feature_importance is not None:
            pd.DataFrame(feature_importance).to_csv(
                f"{report_folder}/feature_importance.csv", index=False
            )

        for r in results:
            r["dataset"] = file_name
            all_results.append(r)

        generate_visualizations(
            file_name,
            my_folder,
            results,
            cm,
            nulls,
            corr,
            feature_importance
        )

    df = pd.DataFrame(all_results)

    if df.empty:
        print("No results generated")
        return

    df.to_csv(f"{my_folder}/model_results.csv", index=False)


if __name__ == "__main__":
    main()