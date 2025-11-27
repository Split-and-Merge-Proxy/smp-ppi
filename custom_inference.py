import os
import re
import torch
import torch.nn as nn

from fairseq import checkpoint_utils, options, tasks, utils
from transformers import T5EncoderModel, T5Tokenizer
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import AlbertModel, AlbertTokenizer


def get_padding_mask(seq_len, max_len):
    """根据长度获取transformer中的mask"""
    padding_mask = torch.arange(max_len).view(1, -1).repeat(seq_len.size(0), 1) # B x L
    padding_mask = padding_mask.to(seq_len.device)
    padding_mask = padding_mask >= seq_len.view(-1, 1)
    padding_mask.requires_grad = False
    return padding_mask     # B x L


def get_pro_rep(encs, lens):
    """获取蛋白质序列的表示，使用AVGPool，将编码压缩成1"""
    padding_mask = get_padding_mask(lens, max_len=encs.size(1))
    rep = encs * (1.-padding_mask.type_as(encs)).unsqueeze(-1)
    rep = torch.sum(rep, dim=1)
    rep = torch.div(rep, lens.unsqueeze(-1))
    return rep


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_len = args.max_len
  
        self.projector = nn.Sequential(
            nn.Linear(args.emb_dim, args.hid_dim, bias=False),
            nn.Dropout(args.dropout),
            nn.LayerNorm(args.hid_dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=args.hid_dim, nhead=4, \
                dim_feedforward=args.hid_dim*4, dropout=args.dropout, batch_first=True), 
            num_layers=args.trans_layers)
        self.num_layers = self.transformer.num_layers

    def forward_projecter(self, embs):
        # embs: B x L x D
        embs = self.projector(embs)
        return embs

    def forward_transformer(self, encs, lens):
        padding_mask = get_padding_mask(lens, encs.size(1))
        encs = self.transformer(encs, src_key_padding_mask=padding_mask)
        return encs, lens

    def forward(self, fst_embs, fst_lens, sec_embs, sec_lens):
        """
            fst_embs: bsz x max_len x emb_dim
            sec_embs: bsz x max_len x emb_dim
        """
        fst_encs = self.forward_projecter(fst_embs)
        sec_encs = self.forward_projecter(sec_embs)

        fst_encs, fst_lens = self.forward_transformer(fst_encs, fst_lens)
        sec_encs, sec_lens = self.forward_transformer(sec_encs, sec_lens)

        return fst_encs, fst_lens, sec_encs, sec_lens

class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.projector = nn.Linear(args.hid_dim, 2)
        if hasattr(args, "fuse_out"):
            self.fuse_out = args.fuse_out
        else:
            self.fuse_out = False
        if self.fuse_out:
            self.rep_maker = nn.Sequential(
                nn.Linear(args.emb_dim+args.hid_dim, args.hid_dim),
                nn.ReLU())

    def forward(self, fst_encs, fst_lens, sec_encs, sec_lens):

        fst_reps = get_pro_rep(fst_encs, fst_lens)      # B x D
        sec_reps = get_pro_rep(sec_encs, sec_lens)
        reps = fst_reps * sec_reps

        logits = self.projector(reps)
        return {"logits": logits, "fst_reps": fst_reps, 'sec_reps': sec_reps, "reps": reps}


class PPITrans(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, fst_encs, fst_lens, sec_encs, sec_lens):
        
        fst_encs, fst_lens, sec_encs, sec_lens = self.encoder(fst_encs, fst_lens, sec_encs, sec_lens)
        output = self.decoder(fst_encs, fst_lens, sec_encs, sec_lens)

        return output


def build_pretrained_model(model_name):
    if "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = T5EncoderModel.from_pretrained(model_name)
    elif "albert" in model_name:
        tokenizer = AlbertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AlbertModel.from_pretrained(model_name)
    elif "bert" in model_name:
        tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = BertModel.from_pretrained(model_name)
    elif "xlnet" in model_name:
        tokenizer = XLNetTokenizer.from_pretrained(model_name, do_lower_case=False )
        model = XLNetModel.from_pretrained(model_name)
    else:
        raise ValueError(f"Unkown model name: {model_name}")
    return tokenizer, model

def extract_seq_feat(seq, tokenizer, embeder, device):
    seq = re.sub(r"[UZOB]", "X", seq)
    seq = " ".join(seq)
    seqs = [seq]
    inputs = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True)
    inputs = {k: torch.tensor(v).to(device) for k, v in inputs.items()}
    with torch.no_grad():
        embedding = embeder(**inputs)
    embedding = embedding.last_hidden_state.cpu().numpy()
    assert len(seqs) == len(embedding) == 1
    for idx in range(len(embedding)):
        seq_len = (inputs['attention_mask'][idx] == 1).sum()
        seq_emb = embedding[idx][:seq_len-1]
        assert seq_len - 1 == len(seqs[idx].strip().split())
        break

    return seq_emb

def read_fa(file_):
    f = open(file_, 'r')
    for line in f:
        if line.startswith('>'):
            continue
        
        seq = line.split('\n')[0]
        break
    return seq


def main():
    fa_file_path_1, fa_file_path_2 = './example/test1.fasta', './example/test2.fasta'
    seq1, seq2 = read_fa(fa_file_path_1), read_fa(fa_file_path_2)
    seq_len_1, seq_len_2 = len(seq1), len(seq2)
    seq_len_1, seq_len_2 = torch.tensor([seq_len_1]), torch.tensor([seq_len_2])
    
    pretrained_model = './prot_t5_xl_uniref50'
    print(">>>>> load pretrained language model")
    tokenizer, embeder = build_pretrained_model(pretrained_model)
    embeder = embeder.eval()
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embeder = embeder.to(device)
    seq_emb1 = extract_seq_feat(seq1, tokenizer, embeder, device)
    seq_emb2 = extract_seq_feat(seq2, tokenizer, embeder, device)
    seq_emb1, seq_emb2 = torch.Tensor(seq_emb1), torch.Tensor(seq_emb2)

    ckpt_path = './ckpts/smp_hippie.pt'
    print(">>>>> load model from {}".format(ckpt_path))
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    args = ckpt['args']
    model = PPITrans(args)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()

    # prepare input
    seq_len_1, seq_len_2 = seq_len_1.to(device), seq_len_2.to(device)
    seq_emb1, seq_emb2 = seq_emb1.unsqueeze(0).to(device), seq_emb2.unsqueeze(0).to(device)

    print(">>>>> Inference on your custom data")
    with torch.no_grad():
        output = model(seq_emb1, seq_len_1, seq_emb2, seq_len_2)

    probs = torch.softmax(output['logits'], dim=-1).cpu().tolist()
    if probs[0][1] > 0.5:
        print('this two proteins are interactive')
    else:
        print('this tow proteins are not interactive')
    # import pdb; pdb.set_trace()
    print(">>>>> Finish")

if __name__ == "__main__":
    main()