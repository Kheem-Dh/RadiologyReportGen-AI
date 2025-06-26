from src.evaluation import compute_bleu, compute_rouge, compute_bertscore

# Example reports
reference = "The lungs are clear. Heart size is normal."
generated = "The heart is normal in size and the lungs are clear."

bleu = compute_bleu(reference, generated)
rouge = compute_rouge(reference, generated)
P, R, F1 = compute_bertscore(reference, generated)

print("BLEU:", bleu)
print("ROUGE:", rouge)
print(f"BERTScore: P={P:.3f}, R={R:.3f}, F1={F1:.3f}")
