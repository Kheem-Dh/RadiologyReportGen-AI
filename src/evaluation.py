from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score


def compute_bleu(reference, generated):
    ref_tokens = reference.lower().split()
    gen_tokens = generated.lower().split()
    return sentence_bleu([ref_tokens], gen_tokens)


def compute_rouge(reference, generated):
    rouge = Rouge()
    return rouge.get_scores(generated, reference)


def compute_bertscore(reference, generated):
    P, R, F1 = score([generated], [reference], lang="en", verbose=False)
    return float(P.mean()), float(R.mean()), float(F1.mean())
