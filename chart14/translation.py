import os
import torch 
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import torch.nn as nn
import torch.optim as optim

sp_en = spm.SentencePieceProcessor()
sp_en.load('./en_bpe_test.model')
sp_cn = spm.SentencePieceProcessor()
sp_cn.load('./zh_bpe_test.model')

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
        super().__init__()
        with open(src_file, encoding='utf-8') as f:
            src_lines = f.read().splitlines()
        with open(trg_file, encoding='utf-8') as f:
            trg_lines = f.read().splitlines()
        assert len(src_file) == len(trg_file)
        self.pairs = []
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

        for src, trg in zip(src_lines, trg_lines):
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
        src_pad = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_ID)
        trg_pad = nn.utils.rnn.pad_sequence(trg_batch, padding_value=PAD_ID)
        return src_pad, trg_pad, src_lens, trg_lens
    

dataset = TranslationDataset('./en2cn/train_en.txt', './en2cn/train_zh.txt', tokenize_en, tokenize_cn)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=TranslationDataset.collate_fn)

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.bi_lstm = nn.LSTM(emb_dim, hid_dim, n_layers, bidirectional=True)
        self.fc_hidden = nn.ModuleList([nn.Linear(hid_dim * 2, hid_dim) for _ in range(n_layers)])
        self.fc_cell = nn.ModuleList([nn.Linear(hid_dim * 2, hid_dim) for _ in range(n_layers)])

    def forward(self, src, src_len):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, enforce_sorted=False)
        outputs, (hidden, cell) = self.bi_lstm(packed)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)

        hidden = hidden.view(self.n_layers, 2, -1, hidden.size(2))
        cell = cell.view(self.n_layers, 2, -1, cell.size(2))

        final_hidden = []
        final_cell = []

        for layer in range(self.n_layers):
            h_cat = torch.cat((hidden[layer][-2], hidden[layer][-1]), dim=1)
            c_cat = torch.cat((cell[layer][-2], cell[layer][-1]), dim=1)

            h_layer = torch.tanh(self.fc_hidden[layer](h_cat)).unsqueeze(0)
            c_layer = torch.tanh(self.fc_cell[layer](c_cat)).unsqueeze(0)

            final_hidden.append(h_layer)
            final_cell.append(c_layer)

        hidden_concat = torch.cat(final_hidden, dim=0)
        cell_concat = torch.cat(final_cell, dim=0)
        print(outputs.shape, hidden_concat.shape, cell_concat.shape)
        return outputs, hidden_concat, cell_concat
    
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2 + hid_dim, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        hidden = hidden.permute(1, 0, 2)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        print("attention shape: ", hidden.shape, encoder_outputs.shape)

        src_len = encoder_outputs.shape[1]
        hidden = hidden.repeat(1, src_len, 1)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)

    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, attention, n_layers=3):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(hid_dim * 2 + emb_dim, hid_dim, num_layers=n_layers)
        self.fc_out = nn.Linear(hid_dim * 3, output_dim)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        input_token = input_token.unsqueeze(0)
        embedded = self.embedding(input_token)
        last_hidden = hidden[-1].unsqueeze(0)

        a = self.attention(last_hidden, encoder_outputs, mask)
        a = a.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(a, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)

        lstm_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(lstm_input, (hidden, cell))
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted), dim=1))

        return prediction, hidden, cell, a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, src_len, trg):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
        
        encoder_outputs, hidden, cell = self.encoder(src, src_len)

        input_token = trg[0]

        mask = (src != PAD_ID).permute(1, 0)

        for t in range(1, max_len):
            output, hidden, cell, _ = self.decoder(input_token, hidden, cell, encoder_outputs, mask)
            outputs[t] = output
            input_token = trg[t]

        return outputs
    

def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    step_loss = 0
    step_count = 0

    for i, (src, trg, src_len, _) in enumerate(iterator):
        src = src.to(model.device)
        trg = trg.to(model.device)
        optimizer.zero_grad()
        output = model(src, src_len, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        step_loss += loss.item()
        epoch_loss += loss.item()
        step_count += 1

        if (i + 1) % 100 == 0:
            avg_step_loss = step_loss / step_count
            print(f'Step [{i + 1}/{len(iterator)}] | Loss: {avg_step_loss:.4f}')
            step_loss = 0
            step_count = 0

    return epoch_loss / len(iterator)



if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dataset = TranslationDataset("./en2cn/train_en_test.txt", "./en2cn/train_zh_test.txt", tokenize_en, tokenize_cn)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=TranslationDataset.collate_fn)
    
    INPUT_DIM = sp_en.get_piece_size()
    OUTPUT_DIM = sp_cn.get_piece_size()
    ENC_EMB_DIM = 512
    DEC_EMB_DIM = 512
    HID_DIM = 512

    attention = Attention(HID_DIM).to(device)
    encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM).to(device)
    decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, attention).to(device)
    model = Seq2Seq(encoder, decoder, device).to(device)
    # model.load_state_dict(torch.load('seq2seq_bpe_attention.pt', map_location=device))

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    N_EPOCHS = 1

    for epoch in range(N_EPOCHS):
        loss = train(model, loader, optimizer, criterion)
        print(f'Epoch {epoch + 1}/{N_EPOCHS} | Loss: {loss:.4f}')
        torch.save(model.state_dict(), 'seq2seq_bpe_attention.pt')




