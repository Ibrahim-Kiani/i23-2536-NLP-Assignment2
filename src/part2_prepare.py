import argparse
import random
from collections import Counter, defaultdict

from common import classify_topic_from_title, load_metadata, parse_articles, save_json

POS_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "CONJ", "POST", "NUM", "PUNC", "UNK"]
NER_TAGS = ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC", "O"]


def split_sentences(tokens):
    sents, cur = [], []
    for t in tokens:
        cur.append(t)
        if t in {"۔", ".", "!", "?"}:
            if cur:
                sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return [s for s in sents if len(s) > 2]


def make_gazetteer():
    persons = [
        "عمران", "نواز", "مریم", "شہباز", "بلاول", "زرداری", "قائد", "اقبال", "جناح", "عارف", "عاصم", "خرم",
        "فاطمہ", "عائشہ", "سمینہ", "رضا", "حسن", "حسین", "احمد", "علی", "حمزہ", "عثمان", "سعد", "عبداللہ",
        "ماہم", "نور", "ریاض", "سلیم", "حارث", "دانش", "وہاب", "وسیم", "بابر", "سرمد", "زین", "زویا",
        "طاہر", "ندیم", "عمر", "سارہ", "ثنا", "عاطف", "فیصل", "چوہدری", "خان", "شریف", "قریشی", "اکرم", "فیاض", "مجید",
    ]
    locations = [
        "پاکستان", "لاہور", "کراچی", "اسلام آباد", "پنجاب", "سندھ", "خیبر", "بلوچستان", "راولپنڈی", "پشاور",
        "کوئٹہ", "ملتان", "فیصل آباد", "گوجرانوالہ", "سیالکوٹ", "حیدرآباد", "میرپور", "گلگت", "سکردو", "مردان",
        "چارسدہ", "سوات", "دیر", "بونیر", "چترال", "تھر", "بدین", "ٹھٹھہ", "مانسہرہ", "ایبٹ آباد", "ناران", "کاغان",
        "کشمیر", "آزاد کشمیر", "لندن", "دبئی", "دوحہ", "کابل", "دہلی", "بیجنگ", "واشنگٹن", "نیویارک", "استنبول", "مکہ", "مدینہ", "یورپ", "ایشیا", "افریقہ", "خلیج", "بلوچ", "سرائیکی",
    ]
    orgs = [
        "پی ٹی آئی", "پیپلز پارٹی", "ن لیگ", "سپریم کورٹ", "ہائی کورٹ", "ایف آئی اے", "نیب", "آئی ایس پی آر", "اقوام متحدہ", "آئی ایم ایف",
        "اسٹیٹ بینک", "وزارت صحت", "وزارت تعلیم", "سینیٹ", "قومی اسمبلی", "پولیس", "رینجرز", "فوج", "ہیلتھ ڈیپارٹمنٹ", "ڈان",
        "بی بی سی", "ہم نیوز", "جیو", "ایکس", "فیس بک", "یوٹیوب", "پی سی بی", "فیفا", "ورلڈ بینک", "ڈبلیو ایچ او",
    ]
    return {"PER": persons, "LOC": locations, "ORG": orgs}


def rule_pos(token, lex):
    if token in lex["VERB"]:
        return "VERB"
    if token in lex["ADJ"]:
        return "ADJ"
    if token in lex["ADV"]:
        return "ADV"
    if token.isdigit() or token == "<NUM>":
        return "NUM"
    if token in {"۔", ".", ",", "،", "!", "?", ";", ":"}:
        return "PUNC"
    if token in {"میں", "ہم", "آپ", "وہ", "یہ", "تو", "نے", "کو"}:
        return "PRON"
    if token in {"کا", "کی", "کے", "ایک", "اس", "ان"}:
        return "DET"
    if token in {"اور", "لیکن", "یا", "اگر", "بلکہ"}:
        return "CONJ"
    if token in {"میں", "پر", "سے", "تک", "کو"}:
        return "POST"
    if token in lex["NOUN"]:
        return "NOUN"
    return "UNK"


def build_lexicon(sentences):
    freq = Counter(t for s in sentences for t in s)
    common = [w for w, _ in freq.most_common(1200)]
    lex = {
        "NOUN": set(common[:300]),
        "VERB": set(w for w in common if w.endswith("ا") or w.endswith("ے") or w.endswith("ی")) | set(common[300:500]),
        "ADJ": set(common[500:750]),
        "ADV": set(common[750:950]),
    }
    return lex


def tag_ner_sentence(sent, gaz):
    tags = ["O"] * len(sent)
    for etype in ["PER", "LOC", "ORG"]:
        names = gaz[etype]
        for i, tok in enumerate(sent):
            if tok in names:
                tags[i] = f"B-{etype}"
    return tags


def stratified_pick(sentences, topics, n=500):
    by_topic = defaultdict(list)
    for i, t in enumerate(topics):
        by_topic[t].append(i)

    picked = []
    top3 = sorted(by_topic, key=lambda x: len(by_topic[x]), reverse=True)[:3]
    for t in top3:
        cand = by_topic[t]
        random.shuffle(cand)
        picked.extend(cand[:100])

    remain = [i for i in range(len(sentences)) if i not in picked]
    random.shuffle(remain)
    picked.extend(remain[: max(0, n - len(picked))])
    return picked[:n]


def split_ids(ids, topics):
    random.shuffle(ids)
    train_end = int(0.7 * len(ids))
    val_end = int(0.85 * len(ids))
    return ids[:train_end], ids[train_end:val_end], ids[val_end:]


def write_conll(path, sents, tags):
    with open(path, "w", encoding="utf-8") as f:
        for s, y in zip(sents, tags):
            for t, z in zip(s, y):
                f.write(f"{t}\t{z}\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleaned", default="cleaned.txt")
    ap.add_argument("--metadata", default="Metadata.json")
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    random.seed(42)
    docs = parse_articles(args.cleaned)
    meta = load_metadata(args.metadata)

    sentences, sentence_topics = [], []
    for i, d in enumerate(docs):
        topic = classify_topic_from_title(meta[i].get("title", "")) if i < len(meta) else "Politics"
        for s in split_sentences(d):
            sentences.append(s)
            sentence_topics.append(topic)

    chosen = stratified_pick(sentences, sentence_topics, n=500)
    chosen_s = [sentences[i] for i in chosen]
    chosen_t = [sentence_topics[i] for i in chosen]

    lex = build_lexicon(chosen_s)
    gaz = make_gazetteer()

    pos_tags = [[rule_pos(t, lex) for t in s] for s in chosen_s]
    ner_tags = [tag_ner_sentence(s, gaz) for s in chosen_s]

    ids = list(range(len(chosen_s)))
    tr, va, te = split_ids(ids, chosen_t)

    write_conll(f"{args.out_dir}/pos_train.conll", [chosen_s[i] for i in tr], [pos_tags[i] for i in tr])
    write_conll(f"{args.out_dir}/pos_val.conll", [chosen_s[i] for i in va], [pos_tags[i] for i in va])
    write_conll(f"{args.out_dir}/pos_test.conll", [chosen_s[i] for i in te], [pos_tags[i] for i in te])

    write_conll(f"{args.out_dir}/ner_train.conll", [chosen_s[i] for i in tr], [ner_tags[i] for i in tr])
    write_conll(f"{args.out_dir}/ner_val.conll", [chosen_s[i] for i in va], [ner_tags[i] for i in va])
    write_conll(f"{args.out_dir}/ner_test.conll", [chosen_s[i] for i in te], [ner_tags[i] for i in te])

    dist = {
        "pos_train": Counter(t for i in tr for t in pos_tags[i]),
        "pos_val": Counter(t for i in va for t in pos_tags[i]),
        "pos_test": Counter(t for i in te for t in pos_tags[i]),
        "ner_train": Counter(t for i in tr for t in ner_tags[i]),
        "ner_val": Counter(t for i in va for t in ner_tags[i]),
        "ner_test": Counter(t for i in te for t in ner_tags[i]),
    }
    save_json(f"{args.out_dir}/part2_label_distribution.json", {k: dict(v) for k, v in dist.items()})


if __name__ == "__main__":
    main()
