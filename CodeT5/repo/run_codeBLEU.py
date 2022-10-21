from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu

def main():
  output_fn = "/content/prediction_output.txt"
  gold_fn = "/content/Review4Repair/Inference/C_sample_tgt-test.txt"
  codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, "java")
  bleu = round(_bleu(gold_fn, output_fn), 2)
  print(f"codeBLEU: {codebleu*100}")
  print(f"BLEU: {bleu}")


if __name__ == "__main__":
    main()