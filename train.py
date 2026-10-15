#!/usr/bin/env python3
"""train.py — hufundisha model ndogo ya Zilizotafsiriwa kwa data ya uchezaji wa app yako, kisha kuipakia kwenye Hugging Face (Model).

Data:   GET <TR_URL>?action=plays (uchezaji wa siku 90) + GET <TR_URL> (filamu, kwa tarehe ya kuongezwa) — zote kwa x-admin-key.
Model:  1) popularity = uchezaji wa kila kifaa/siku (kila kifaa mara 1 kwa siku) wenye uzito unaopungua na umri
           + bonasi ya uchanga wa filamu (mchanganyiko wa "zinazotazamwa" na "zilizoongezwa karibuni")
        2) similar   = item-item collaborative filtering (SVD ya matrix ya kifaa x filamu, cosine kwenye vipengele fiche)
Pato:   model.json -> https://huggingface.co/<MODEL_REPO>/resolve/main/model.json (app inaisoma kwa Trending)
Env:    TR_KEY, HF_TOKEN (lazima) | TR_URL, MODEL_REPO, HALF_LIFE_DAYS, FRESH_DAYS, FRESH_WEIGHT, LATENT_K, TOP_N (hiari)
        PLAYS_FILE / MOVIES_FILE = jaribio la ndani bila mtandao; bila HF_TOKEN model.json huandikwa hapa tu (--dry).
"""
import json, os, sys, urllib.request
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np

TR_URL = os.environ.get("TR_URL", "https://ooqvsqmsnwjqvuzbhczx.supabase.co/functions/v1/movie-zilizotafsriwa2").strip()
TR_KEY = os.environ.get("TR_KEY", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
MODEL_REPO = os.environ.get("MODEL_REPO", "jaynesmaxtv/translated-recommender").strip()
HALF_LIFE_DAYS = float(os.environ.get("HALF_LIFE_DAYS", "7"))   # uchezaji wa zamani unapungua uzito nusu kila siku hizi
FRESH_DAYS = float(os.environ.get("FRESH_DAYS", "14"))          # filamu mpya inapata bonasi inayopungua nusu kila siku hizi
FRESH_WEIGHT = float(os.environ.get("FRESH_WEIGHT", "1.5"))     # nguvu ya bonasi ya uchanga (1 = uchezaji mmoja wa leo)
LATENT_K = int(os.environ.get("LATENT_K", "16"))
TOP_N = int(os.environ.get("TOP_N", "10"))


def get_json(url):
    req = urllib.request.Request(url, headers={"x-admin-key": TR_KEY} if TR_KEY else {})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def ts(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def fit(plays, movies, now):
    # kila (kifaa, filamu, siku) hesabiwa mara moja — inazuia spam na kubonyeza mara nyingi
    seen = {}
    for p in plays:
        mid, dev, t = str(p.get("movie_id") or ""), str(p.get("device_id") or ""), ts(p["created_at"])
        if mid:
            seen[(dev or f"anon-{len(seen)}", mid, t.date())] = t
    score = defaultdict(float)
    for (_, mid, _), t in seen.items():
        age = max(0.0, (now - t).total_seconds() / 86400)
        score[mid] += 0.5 ** (age / HALF_LIFE_DAYS)
    for m in movies:
        if m.get("status", "public") != "public" or m.get("active") is False or not m.get("created_at"):
            continue
        age = max(0.0, (now - ts(m["created_at"])).total_seconds() / 86400)
        score["n2-" + str(m["id"])] += FRESH_WEIGHT * 0.5 ** (age / FRESH_DAYS)
    top = max(score.values(), default=0) or 1.0
    popularity = {k: round(v / top * 100, 2) for k, v in sorted(score.items(), key=lambda kv: -kv[1]) if v > 0}

    # item-item CF: matrix binary kifaa x filamu -> SVD -> cosine
    pairs = {(d, m) for (d, m, _) in seen if not d.startswith("anon-")}
    devs = sorted({d for d, _ in pairs})
    count = defaultdict(int)
    for _, m in pairs:
        count[m] += 1
    items = sorted(m for m, c in count.items() if c >= 2)  # inahitaji angalau vifaa 2
    similar = {}
    if len(devs) >= 3 and len(items) >= 3:
        di, ii = {d: i for i, d in enumerate(devs)}, {m: i for i, m in enumerate(items)}
        X = np.zeros((len(devs), len(items)), dtype=np.float32)
        for d, m in pairs:
            if m in ii:
                X[di[d], ii[m]] = 1.0
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        k = max(1, min(LATENT_K, len(S)))
        V = Vt[:k].T * S[:k]
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        sim = V @ V.T
        np.fill_diagonal(sim, -1)
        for m, i in ii.items():
            order = [j for j in np.argsort(-sim[i])[:TOP_N] if sim[i, j] > 0.2]
            if order:
                similar[m] = [items[j] for j in order]
    return {"version": 1, "updated": now.isoformat(timespec="seconds"),
            "stats": {"plays": len(seen), "devices": len(devs), "movies": len(popularity)},
            "popularity": popularity, "similar": similar}


def main():
    now = datetime.now(timezone.utc)
    if os.environ.get("PLAYS_FILE"):
        plays = json.load(open(os.environ["PLAYS_FILE"], encoding="utf-8"))
        movies = json.load(open(os.environ["MOVIES_FILE"], encoding="utf-8")) if os.environ.get("MOVIES_FILE") else []
    else:
        if not TR_KEY:
            print("❌ Secret TR_KEY haijawekwa."); return 2
        plays = get_json(f"{TR_URL}?action=plays")
        movies = get_json(TR_URL)
    if isinstance(plays, dict):
        raise RuntimeError(plays.get("error", "jibu la plays si orodha"))
    model = fit(plays, movies if isinstance(movies, list) else [], now)
    print(f"Model: {model['stats']} | trending 5 za juu: {list(model['popularity'].items())[:5]}", flush=True)
    data = json.dumps(model, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if not HF_TOKEN:
        open("model.json", "wb").write(data); print("(--dry) model.json imeandikwa hapa; HF_TOKEN haijawekwa."); return 0
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.create_repo(MODEL_REPO, repo_type="model", private=False, exist_ok=True)
    card = ("---\nlicense: mit\ntags: [recommender, collaborative-filtering]\n---\n"
            "# Translated recommender\nModel ndogo (popularity + item-item CF) ya Zilizotafsiriwa. `model.json`: `popularity`, `similar`.\n").encode()
    api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=MODEL_REPO, repo_type="model")
    api.upload_file(path_or_fileobj=data, path_in_repo="model.json", repo_id=MODEL_REPO, repo_type="model",
                    commit_message=f"train {now:%Y-%m-%d %H:%M}")
    print(f"✅ https://huggingface.co/{MODEL_REPO}/resolve/main/model.json ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
