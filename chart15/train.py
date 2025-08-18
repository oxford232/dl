import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
from transformer import build_transformer

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
sp_en = spm.SentencePieceProcessor()
sp_en.load("../chart14/en_bpe.model")
sp_cn = spm.SentencePieceProcessor()
sp_cn.load("../chart14/zh_bpe.model")

def tokenize_en(text):
    return sp_en.encode(text, out_type=int)

def tokenize_cn(text):
    return sp_cn.encode(text, out_type=int)

PAD_ID = sp_en.pad_id()  # 1
UNK_ID = sp_en.unk_id()  # 0
BOS_ID = sp_en.bos_id()  # 2
EOS_ID = sp_en.eos_id()  # 3

class TranslationDataset(Dataset):
    def __init__(self, src_file, trg_file, src_tokenizer, trg_tokenizer, max_len=100):
        with open(src_file, encoding='utf-8') as f:
            src_lines = f.read().splitlines()

        with open(trg_file, encoding='utf-8') as f:
            trg_lines = f.read().splitlines()

        assert len(src_lines) == len(trg_lines)

        self.pairs = []
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer
        print("len: ", len(src_lines))
        index = 0
        for src, trg in zip(src_lines, trg_lines):
            index += 1
            if index % 100000 == 0:
                print(index)

            src_ids = [BOS_ID] + self.src_tokenizer(src) + [EOS_ID]
            trg_ids = [BOS_ID] + self.trg_tokenizer(trg) + [EOS_ID]

            if len(src_ids) <= max_len and len(trg_ids) <= max_len:
                self.pairs.append((src_ids, trg_ids))

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        src_ids, trg_ids = self.pairs[idx]
        return torch.LongTensor(src_ids), torch.LongTensor(trg_ids)
    
    @staticmethod
    def collate_fn(batch):
        src_batch, trg_batch = zip(*batch)
        src_lens = [len(x) for x in src_batch]
        trg_lens = [len(x) for x in trg_batch]
        src_pad = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=PAD_ID)
        trg_pad = nn.utils.rnn.pad_sequence(trg_batch, batch_first=True, padding_value=PAD_ID)
        return src_pad, trg_pad, src_lens, trg_lens
    

def create_mask(src, tgt, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    tgt_len = tgt.size(1)

    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return src_mask, tgt_mask

def train(model, dataloader, optimizer, criterion, pad_idx):
    model.train()
    total_loss = 0
    step = 0
    log_loss = 0

    for src, tgt, src_lens, tgt_lens in dataloader:
        step += 1

        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask, tgt_mask = create_mask(src, tgt_input, pad_idx)

        optimizer.zero_grad()
        encoder_output = model.encode(src, src_mask)
        decoder_output = model.decode(encoder_output, src_mask, tgt_input, tgt_mask)
        output = model.project(decoder_output)

        output = output.reshape(-1, output.shape[-1])
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        log_loss += loss.item()

        if step % 100 == 0:
            avg_log_loss = log_loss / 100
            print(f"Step {step}: Avg Loss = {avg_log_loss:.4f}")
            log_loss = 0

    return total_loss / len(dataloader)


def main():
    SRC_VOCAB_SIZE = 16000
    TGT_VOCAB_SIZE = 16000
    SRC_SEQ_LEN = 128
    TGT_SEQ_LEN = 128
    BATCH_SIZE = 2
    NUM_EPOCHS = 10
    LR = 1e-4

    train_dataset = TranslationDataset("../chart14/en2cn/train_en.txt", "../chart14/en2cn/train_zh.txt", tokenize_en, tokenize_cn)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_fn)

    model = build_transformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, SRC_SEQ_LEN, TGT_SEQ_LEN).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for epoch in range(NUM_EPOCHS):
        loss = train(model, train_dataloader, optimizer, criterion, PAD_ID)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")
        torch.save(model.state_dict(), "transformer.pt")

if __name__ == "__main__":
    main()

    


