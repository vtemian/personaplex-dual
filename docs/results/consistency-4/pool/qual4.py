import json, sys, collections, statistics
W = sys.argv[1]; sys.path.insert(0, W + "/src")
from dual import sbs_report as sr
from dual.constants import PAD
L = lambda p: [json.loads(l) for l in open(p) if l.strip()]
R = W + "/docs/results/"
s1 = L(R + "side-by-side-1/answers.jsonl"); s2 = L(R + "side-by-side-2/answers.jsonl"); s3 = L(R + "side-by-side-3/answers.jsonl")
print(collections.Counter(r['model'] for r in s1), collections.Counter(r['model'] for r in s2), collections.Counter(r['model'] for r in s3))
rows = ([dict(r, model="base") for r in s1 if r['model'] == 'trained'] + [dict(r, model="base") for r in s2 if r['model'] == 'baseline-e2' and r['rendering'].endswith('-swap')]
        + [dict(r, model="e1") for r in s3] + [dict(r, model="e2") for r in s2 if r['model'] == 'cons1-e2'])
EXTRA = sys.argv[2:]  # label=path:src
for sp in EXTRA:
    lab, rest = sp.split("=", 1); p, src = rest.rsplit(":", 1)
    rows += [dict(r, model=lab) for r in L(p) if r['model'] == src]
MODELS = ["base", "e1", "e2"] + list(dict.fromkeys(sp.split("=", 1)[0] for sp in EXTRA))
print(collections.Counter((r['model'], r['rendering']) for r in rows))
print(sr.quality_table(rows, MODELS))
print(sr.first_word_table(rows, MODELS))
def last_nonpad(ids):
    idx = [i for i, t in enumerate(ids) if int(t) != PAD]
    return idx[-1] if idx else None
words = sr.words_of
out = collections.defaultdict(list)
for r in rows:
    if r['rendering'].endswith('-swap'): continue
    assert len(r['ids']) == r['window'], (len(r['ids']), r['window'])
    lp = last_nonpad(r['ids'])
    spoke = bool(words(r['answer']))
    span = None if lp is None else (lp + 1) / r['window']
    stop = spoke and lp is not None and lp + 1 < 0.8 * r['window'] and not r['answer'].strip().endswith(('.', '?', '!'))
    stop2 = spoke and lp is not None and lp < 0.8 * r['window'] and not r['answer'].strip().endswith(('.', '?', '!'))
    q = spoke and r['answer'].strip().endswith('?')
    out[(r['model'], r['rendering'], 't0' if r['turn'] == 0 else 't2-8')].append((stop, stop2, span, q, len(words(r['answer'])), sr.onset(r['ids']), r['stem']))
for k in sorted(out):
    v = out[k]
    sp = [x[2] for x in v if x[2] is not None]
    short = sum(1 for x in v if 0 < x[4] <= 10)
    print(k, "n", len(v), "stop", sum(x[0] for x in v), "stop(lp<.8w)", sum(x[1] for x in v), "q", sum(x[3] for x in v),
          "medspan %.3f" % statistics.median(sp) if sp else '-', "medwords", statistics.median([x[4] for x in v if x[4]]) if any(x[4] for x in v) else '-',
          "<=10w", short, "onsets", sorted(collections.Counter(x[5] for x in v).items(), key=lambda z: (z[0] is None, z[0] or 0)))
