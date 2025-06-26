from src.report_generation import load_gpt2_tokenizer_model, fine_tune_gpt2
import pandas as pd

processed_txt = "data/processed_reports.txt"
save_dir = "data/finetuned_distilgpt2"
with open(processed_txt, 'r') as f:
    reports = [line.strip() for line in f.readlines() if line.strip()]
tokenizer, model = load_gpt2_tokenizer_model(device='cuda')
losses = fine_tune_gpt2(model, tokenizer, reports, epochs=10,
                        batch_size=8, device='cuda', save_path=save_dir)
print("Training complete. Model saved.")
