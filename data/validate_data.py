"""
Runs a Great Expectations suite against reference.csv and bad_batch.csv
to prove clean data passes and corrupted data gets caught.
"""
import json
import pandas as pd
import great_expectations as gx


def build_suite(context):
    suite = gx.ExpectationSuite(name="wine_quality_suite")
    suite = context.suites.add(suite)

    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="alcohol"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="pH", min_value=2.5, max_value=4.5))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="quality", min_value=0, max_value=10))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="quality"))
    return suite


def validate_file(data_source, suite, csv_path, label):
    df = pd.read_csv(csv_path)
    data_asset = data_source.add_dataframe_asset(name=f"asset_{label}")
    batch_definition = data_asset.add_batch_definition_whole_dataframe(f"batch_{label}")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    results = batch.validate(suite)

    print(f"\n=== {label} ({csv_path}) ===")
    print(f"Overall success: {results.success}")
    for r in results.results:
        exp_type = r.expectation_config.type
        col = r.expectation_config.kwargs.get("column")
        status = "PASS" if r.success else "FAIL"
        unexpected = r.result.get("unexpected_count", 0)
        print(f"  [{status}] {exp_type} on '{col}' (unexpected rows: {unexpected})")

    return results


def main():
    context = gx.get_context()
    data_source = context.data_sources.add_pandas(name="wine_data_source")
    suite = build_suite(context)

    ref_results = validate_file(data_source, suite, "data/reference.csv", "reference")
    bad_results = validate_file(data_source, suite, "data/bad_batch.csv", "bad_batch")

    report = {
        "reference_passed": ref_results.success,
        "bad_batch_passed": bad_results.success,
    }
    with open("reports/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved summary to reports/validation_report.json")


if __name__ == "__main__":
    main()