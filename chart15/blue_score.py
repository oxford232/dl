import sacrebleu
from inference import translate_sentence

with open('../chart14/en2cn/valid_en.txt', 'r', encoding='utf-8') as f:
    src_sentences = [line.strip() for line in f.readlines()]

with open('../chart14/en2cn/valid_zh.txt', 'r', encoding='utf-8') as f:
    ref_sentence = [line.strip() for line in f.readlines()]


assert len(src_sentences) == len(ref_sentence), "源语言和参考翻译句子数不匹配"

hypotheses = []
for i, src in enumerate(src_sentences):
    print(f"Translating {i+1}/{len(src_sentences)}...")
    translation = translate_sentence(src)
    print(ref_sentence[i], translation)
    hypotheses.append(translation.strip())

bleu = sacrebleu.corpus_bleu(hypotheses, [ref_sentence], tokenize="zh")

print("\n========== BLEU Evaluation Result ==========")
print(f"BLEU Score: {bleu.score:.2f}")