import os
import pandas as pd


def merge_and_clean_data(reports_path, projections_path, images_dir, save_path=None):
    """
    Merges reports and projections CSVs, adds full image paths, and saves merged CSV.
    """
    reports_df = pd.read_csv(reports_path)
    projections_df = pd.read_csv(projections_path)
    merged_df = pd.merge(reports_df, projections_df, on='uid', how='inner')
    merged_df['image_path'] = merged_df['filename'].apply(
        lambda fn: os.path.join(images_dir, fn))
    if save_path:
        merged_df.to_csv(save_path, index=False)
    return merged_df


def preprocess_text(text):
    """
    Cleans up report text: remove placeholders, lowercase, etc.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace('nan', '').replace('xxxx', '').replace(
        'x-xxxx', '').lower().replace('\n', ' ')
    return text


def add_combined_reports(df, save_txt_path=None):
    """
    Creates a combined report field: index [IDX] findings [SEP] impressions
    """
    df['findings'] = df['findings'].astype(str).apply(preprocess_text)
    df['impressions'] = df['impression'].astype(str).apply(preprocess_text)
    df['index'] = df.index
    df['combined_report'] = df['index'].astype(
        str) + " [IDX] " + df['findings'] + " [SEP] " + df['impressions']
    if save_txt_path:
        with open(save_txt_path, 'w') as f:
            for report in df['combined_report']:
                f.write(report.strip() + '\n')
    return df


def get_image_paths(df):
    """
    Returns list of image file paths from DataFrame.
    """
    return df['image_path'].tolist()
