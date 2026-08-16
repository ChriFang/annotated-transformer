"""A small Chinese-to-English Transformer training example."""

from pathlib import Path
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator

from the_annotated_transformer import Batch, greedy_decode, make_model, run_epoch


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "cn2en_model.pt"
VOCAB_PATH = ROOT / "cn2en_vocab.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# This tiny dataset is for demonstrating the complete training flow only.
# Use a much larger Chinese-English parallel corpus for a useful translator.
PAIRS = [
    ("你好", "hello ."),
    ("早上好", "good morning ."),
    ("晚安", "good night ."),
    ("谢谢你", "thank you ."),
    ("再见", "goodbye ."),
    ("我爱你", "i love you ."),
    ("我喜欢学习", "i like studying ."),
    ("我喜欢音乐", "i like music ."),
    ("天气很好", "the weather is nice ."),
    ("今天天气很好", "the weather is nice today ."),
    ("我正在学习英语", "i am learning english ."),
    ("他正在读书", "he is reading a book ."),
    ("她喜欢咖啡", "she likes coffee ."),
    ("孩子们在公园玩耍", "the children are playing in the park ."),
    ("这个男孩在跑步", "the boy is running ."),
    ("那个女孩在唱歌", "the girl is singing ."),
    ("我住在北京", "i live in beijing ."),
    ("我们明天见", "we will meet tomorrow ."),
    ("请打开这扇门", "please open this door ."),
    ("我想喝水", "i want to drink water ."),
    ("下午好", "good afternoon ."),
    ("晚上好", "good evening ."),
    ("你好吗？", "how are you ?"),
    ("我很好，谢谢。", "i am fine , thank you ."),
    ("你叫什么名字？", "what is your name ?"),
    ("我叫小明。", "my name is xiaoming ."),
    ("很高兴认识你。", "nice to meet you ."),
    ("再见。", "goodbye ."),
    ("明天见。", "see you tomorrow ."),
    ("一会儿见。", "see you later ."),
    ("谢谢。", "thank you ."),
    ("不用谢。", "you are welcome ."),
    ("对不起。", "i am sorry ."),
    ("没关系。", "it is okay ."),
    ("请原谅我。", "please forgive me ."),
    ("请进。", "please come in ."),
    ("请坐。", "please sit down ."),
    ("请稍等。", "please wait a moment ."),
    ("请说慢一点。", "please speak more slowly ."),
    ("请再说一遍。", "please say it again ."),
    ("我听不懂。", "i do not understand ."),
    ("你会说英语吗？", "can you speak english ?"),
    ("我会说一点英语。", "i can speak a little english ."),
]

SPECIALS = ["<s>", "</s>", "<blank>", "<unk>"]
PAD_ID = SPECIALS.index("<blank>")


def tokenize_zh(text):
    """Split Chinese into characters and keep ASCII words together."""
    tokens = []
    word = ""
    for char in text.strip():
        if char.isascii() and char.isalnum():
            word += char.lower()
            continue
        if word:
            tokens.append(word)
            word = ""
        if not char.isspace():
            tokens.append(char)
    if word:
        tokens.append(word)
    return tokens


def tokenize_en(text):
    return text.lower().split()


def build_vocabs():
    src_vocab = build_vocab_from_iterator(
        (tokenize_zh(src) for src, _ in PAIRS),
        specials=SPECIALS,
    )
    tgt_vocab = build_vocab_from_iterator(
        (tokenize_en(tgt) for _, tgt in PAIRS),
        specials=SPECIALS,
    )
    src_vocab.set_default_index(src_vocab["<unk>"])
    tgt_vocab.set_default_index(tgt_vocab["<unk>"])
    return src_vocab, tgt_vocab


def encode(tokens, vocab):
    return torch.tensor(
        [vocab["<s>"], *vocab(tokens), vocab["</s>"]],
        dtype=torch.long,
        device=DEVICE,
    )


def make_batches(src_vocab, tgt_vocab):
    batches = []
    for src_text, tgt_text in PAIRS:
        src = encode(tokenize_zh(src_text), src_vocab).unsqueeze(0)
        tgt = encode(tokenize_en(tgt_text), tgt_vocab).unsqueeze(0)
        batches.append(Batch(src, tgt, PAD_ID))
    return batches


def train(src_vocab, tgt_vocab):
    model = make_model(
        len(src_vocab),
        len(tgt_vocab),
        N=2,
        d_model=128,
        d_ff=256,
        h=4,
        dropout=0.1,
    ).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batches = make_batches(src_vocab, tgt_vocab)

    for epoch in range(1, 101):
        random.shuffle(batches)
        model.train()
        loss, _ = run_epoch(
            iter(batches),
            model,
            lambda output, target, norm: loss_compute(
                output, target, norm, model.generator, criterion
            ),
            optimizer,
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
            mode="train",
        )
        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | loss: {loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    torch.save((src_vocab, tgt_vocab), VOCAB_PATH)
    return model


def loss_compute(output, target, norm, generator, criterion):
    logits = generator(output)
    logits = logits.contiguous().view(-1, logits.size(-1))
    loss = criterion(logits, target.contiguous().view(-1)) / norm
    return loss.data * norm, loss


def load_or_train():
    if MODEL_PATH.exists() and VOCAB_PATH.exists():
        src_vocab, tgt_vocab = torch.load(VOCAB_PATH, map_location="cpu")
        model = make_model(
            len(src_vocab), len(tgt_vocab), N=2, d_model=128, d_ff=256, h=4
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        return model, src_vocab, tgt_vocab

    src_vocab, tgt_vocab = build_vocabs()
    model = train(src_vocab, tgt_vocab)
    return model, src_vocab, tgt_vocab


def translate(text, model, src_vocab, tgt_vocab, max_len=30):
    model.eval()
    src = encode(tokenize_zh(text), src_vocab).unsqueeze(0)
    src_mask = (src != PAD_ID).unsqueeze(-2)

    with torch.no_grad():
        output = greedy_decode(
            model,
            src,
            src_mask,
            max_len=max_len,
            start_symbol=tgt_vocab["<s>"],
        )[0]

    words = []
    for token_id in output.tolist():
        token = tgt_vocab.get_itos()[token_id]
        if token == "</s>":
            break
        if token not in {"<s>", "<blank>"}:
            words.append(token)
    return " ".join(words)


def main():
    print(f"Using device: {DEVICE}")
    model, src_vocab, tgt_vocab = load_or_train()
    print("Model ready.")

    for text in ["你好", "今天天气很好", "我喜欢音乐", "我吃饱了", "我想去爬山"]:
        print(f"中文: {text}")
        print(f"英文: {translate(text, model, src_vocab, tgt_vocab)}")


if __name__ == "__main__":
    main()
