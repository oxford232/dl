import torch
import sentencepiece as spm
from transformer import build_transformer

sp_en = spm.SentencePieceProcessor()
sp_en.load('../chart14/en_bpe.model')
sp_cn = spm.SentencePieceProcessor()
sp_cn.load('../chart14/zh_bpe.model')


PAD_ID = sp_en.pad_id()
UNK_ID = sp_en.unk_id()
BOS_ID = sp_en.bos_id()
EOS_ID = sp_en.eos_id()

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

SRC_VOCAB_SIZE = 16000
TGT_VOCAB_SIZE = 16000
SRC_SEQ_LEN = 128
TGT_SEQ_LEN = 128


model = build_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN).to(DEVICE)
model.load_state_dict(torch.load('transformer.pt', map_location=DEVICE))
model.eval()


def create_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def translate_sentence(sentence, max_len=100):
    tokens = [BOS_ID] + sp_en.encode(sentence, out_type=int) + [EOS_ID]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(DEVICE)
    src_mask = create_mask(src_tensor, PAD_ID)

    tgt_indices = [BOS_ID]

    with torch.no_grad():
        encoder_output = model.encode(src_tensor, src_mask)

        for _ in range(max_len):
            tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(DEVICE)

            tgt_mask = torch.tril(torch.ones((len(tgt_indices), len(tgt_indices)), device=DEVICE)).bool()
            tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)

            decoder_output = model.decode(encoder_output, src_mask, tgt_tensor, tgt_mask)
            output = model.project(decoder_output)

            pred_token = output.argmax(dim=2)[:, -1].item()
            tgt_indices.append(pred_token)

            if pred_token == EOS_ID:
                break

    translated = sp_cn.decode(tgt_indices)
    return translated

if __name__ == '__main__':
    print("Transformer Translator (type 'quit' or 'exit' to end)")
    while True:
        src_sent = input("\nEnter English sentence: ")
        if src_sent.lower() in ['quit', 'exit']:
            break

        translation = translate_sentence(src_sent)
        print(f"Chinese Translation: {translation}")

