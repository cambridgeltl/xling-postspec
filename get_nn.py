import io
import numpy as np
import faiss
import argparse

parser = argparse.ArgumentParser(description='Get nearest neighbours of a list of source words in a target language')

parser.add_argument("--src_emb", type=str, required=True, help="Source embedding path")
parser.add_argument("--tgt_emb", type=str, required=True, help="Target embedding path")
parser.add_argument("--max_emb", type=int, default=200000, help="Maximum number of word embeddings to load")
parser.add_argument("--att_ctr", type=str, required=True, help="Source attract constraints path")
parser.add_argument("--rep_ctr", type=str, required=True, help="Source repel constraints path")

args = parser.parse_args()


def load_vec(emb_path, max_emb=200000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == max_emb:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors).astype("float32")
    embeddings = embeddings / np.linalg.norm(embeddings, 2, 1)[:, None]
    return embeddings, id2word, word2id


src_embeddings, src_id2word, src_word2id = load_vec(args.src_emb, max_emb=args.max_emb)
tgt_embeddings, tgt_id2word, tgt_word2id = load_vec(args.tgt_emb, max_emb=args.max_emb)

src_ctr_att = io.open(args.att_ctr)
src_ctr_rep = io.open(args.rep_ctr)
src_ctr_att = [p.split() for p in src_ctr_att.read().split("\n")]
src_ctr_rep = [p.split() for p in src_ctr_rep.read().split("\n")]

tgt_ctr_att = io.open(args.att_ctr + '.nns.txt', 'w')
tgt_ctr_rep = io.open(args.rep_ctr + '.nns.txt', 'w')


def get_nns(xq, src_id2word, xb, tgt_id2word, k=1):
    index = faiss.IndexFlatL2(300)   # build the index
    index.add(xb)                    # add vectors to the index
    D, Idx = index.search(xq, k)       # actual search
    w_tgt = Idx[:, 0]
    w_tgt = [tgt_id2word[w] for w in w_tgt]
    w_src = [src_id2word[w] for w in range(len(Idx))]
    src2tgt = dict(zip(w_src, w_tgt))
    return src2tgt


nns = get_nns(src_embeddings, src_id2word, tgt_embeddings, tgt_id2word)


def printout(fin, fout):
    for e in fin:
        if e:
            l, r = e
        else:
            continue
        if l in src_word2id and r in src_word2id:
            tl, tr = nns[l], nns[r]
            if tl != tr:
                fout.write(tl + " " + tr + "\n")
    fout.close()


printout(src_ctr_att, tgt_ctr_att)
printout(src_ctr_rep, tgt_ctr_rep)
