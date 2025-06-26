from src.data_preprocessing import merge_and_clean_data, add_combined_reports

# Paths
reports_path = "data/indiana_reports.csv"
projections_path = "data/indiana_projections.csv"
images_dir = "data/images"
merged_csv = "data/merged_dataset.csv"
combined_txt = "data/processed_reports.txt"

# Merge and clean
df = merge_and_clean_data(reports_path, projections_path,
                          images_dir, save_path=merged_csv)
df = add_combined_reports(df, save_txt_path=combined_txt)
print("Preprocessing complete. Combined reports saved.")
