import os, re, json, argparse, glob, hashlib, random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Sequence

import math
import requests
import psycopg2
import psycopg2.extras
from bs4 import BeautifulSoup

# ---------------------------
# Defaults & Config handling
# ---------------------------

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/blogsimilarity.cfg")

DEFAULTS: Dict[str, Any] = {
    "site_root": "_site",
    "data_out": "_data/related.json",
    "exclude_globs": ["tags/**", "drafts/**", "private/**", "admin/**"],
    "content_ids": ["content"],  # list of element IDs, first found is used
    "neighbors": {
        "ksample" : 16,
        "k": 8,
        "temperature" : 0.7,
        "pin_top" : True,
        "seed" : None,
        "seealso": 4
    },
    "chunk": { "max_tokens": 800, "overlap_tokens": 80 },
    "embedding": {
        "provider": "ollama",              # "ollama" or "openai"
        "model": "nomic-embed-text",       # ollama model OR openai embedding model
        "ollama_url": "http://127.0.0.1:11434/api/embeddings",
        "openai_api_base": "https://api.openai.com/v1/embeddings",
        "openai_api_key": "OPENAI_API_KEY"
        # vector dim will be auto-detected from provider during (re)init
    },
    "db": {
        "host": "127.0.0.1",
        "port": 5432,
        "user": "blog",
        "password": "blog",
        "dbname": "blog"
    },
    "strip_image_hosts" : None
}

def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = DEFAULTS
    p = path or DEFAULT_CONFIG_PATH
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            user = json.load(f)
        cfg = deep_merge(DEFAULTS, user)
    return cfg

# ---------------------------
# Small utils
# ---------------------------

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def canonical_path(site_root: Path, file_path: Path) -> str:
    # e.g. _site/blog/my-post/index.html -> /blog/my-post/
    rel = "/" + str(file_path.relative_to(site_root)).replace(os.sep, "/")
    rel = re.sub(r"index\.html$", "", rel)
    rel = re.sub(r"\.html$", "", rel)
    if not rel.endswith("/"):
        rel += "/"
    return rel

def should_exclude(site_root: Path, file_path: Path, exclude_globs: List[str]) -> bool:
    rel = str(file_path.relative_to(site_root)).replace(os.sep, "/")
    for pattern in exclude_globs:
        if glob.fnmatch.fnmatch(rel, pattern):
            return True
    return False

def select_content_element(soup, ids_or_classes):
    """
    Return inner HTML of the first element matched by the provided list.
    Each entry can be:
      - 'name'       → try id='name' then class='name'
      - '#name'      → CSS id
      - '.classname' → CSS class
      - any CSS selector (e.g., 'main article#post .entry')
    """
    for key in ids_or_classes:
        key = (key or "").strip()
        if not key:
            continue

        # If the key looks like a CSS selector, try it directly
        if key.startswith(("#", ".")) or any(ch in key for ch in " >:+~[]"):
            el = soup.select_one(key)
            if el:
                return el.decode_contents()

        # Fallback: plain token -> try id then class
        el = soup.find(id=key)
        if el:
            return el.decode_contents()
        el = soup.find(class_=key)
        if el:
            return el.decode_contents()

    return None

def html_to_markdown(inner_html: str) -> str:
    try:
        import markdownify
        return markdownify.markdownify(inner_html, heading_style="ATX")
    except Exception:
        soup = BeautifulSoup(inner_html, "lxml")
        for a in soup.find_all("a"):
            a.replace_with(f"{a.get_text(strip=True)} ({a.get('href','')})")
        return soup.get_text("\n", strip=True)

def rough_token_count(text: str) -> int:
    return int(len(text.split()) * 1.3)

def chunk_markdown(md: str, max_tokens=800, overlap=80) -> List[str]:
    words = md.split()
    approx_ratio = 1/1.3
    max_words = int(max_tokens * approx_ratio)
    overlap_words = int(overlap * approx_ratio)
    out, i = [], 0
    while i < len(words):
        j = min(len(words), i + max_words)
        out.append(" ".join(words[i:j]))
        if j == len(words): break
        i = max(0, j - overlap_words)
    return out

def extract_og_meta(soup: BeautifulSoup, strip_hosts = None) -> Dict[str, Optional[str]]:
    def meta(name, prop=False):
        tag = soup.find("meta", property=name) if prop else soup.find("meta", attrs={"name": name})
        return tag["content"].strip() if tag and tag.has_attr("content") else None

    title = meta("og:title", prop=True) or meta("twitter:title") or (soup.title.get_text(strip=True) if soup.title else None)
    desc  = meta("og:description", prop=True) or meta("description")
    image = meta("og:image:secure_url", prop=True) or meta("og:image", prop=True) or meta("twitter:image")
    if image:
        if strip_hosts:
            for hst in strip_hosts:
                if image.startswith(hst):
                    image = image[len(hst):] or "/"
    return {"title": title, "description": desc, "image": image}

def vec_to_literal(v: List[float]) -> str:
    # pgvector input text: '[1,2,3]'
    return "[" + ",".join(str(float(x)) for x in v) + "]"

def average_vectors(vectors: List[List[float]]) -> List[float]:
    if not vectors: return []
    dim = len(vectors[0])
    sums = [0.0]*dim
    for v in vectors:
        for i, x in enumerate(v):
            sums[i] += float(x)
    n = float(len(vectors))
    return [s/n for s in sums]

def rendered_rel(site_root: Path, file_path: Path) -> str:
    # e.g. _site/blog/my-post/index.html -> /blog/my-post/index.html
    return "/" + str(file_path.relative_to(site_root)).replace(os.sep, "/")

def extract_source_path_meta(soup: BeautifulSoup) -> Optional[str]:
    # Read <meta name="page-srcpath" content="{{ page.path }}">
    tag = soup.find("meta", attrs={"name": "page-srcpath"})
    return tag["content"].strip() if tag and tag.has_attr("content") else None

# ---------------------------
# Embeddings
# ---------------------------

def embed_texts_ollama(texts: List[str], model: str, url: str) -> List[List[float]]:
    embs = []

    for txt in texts:
        r = requests.post(url, json={"model": model, "prompt": txt}, timeout=(20, 600))
        r.raise_for_status()
        r = r.json()
        embs.append(r["embedding"])
    return embs

def embed_texts_openai(texts: List[str], model: str, base: str, api_key: str) -> List[List[float]]:
    if not api_key:
        raise RuntimeError(f"OpenAI API key missing)")
    headers = {"Authorization": f"Bearer {api_key}"}
    r = requests.post(base, json={"model": model, "input": texts}, headers=headers, timeout=600)
    r.raise_for_status()
    js = r.json()
    return [d["embedding"] for d in js["data"]]

def embeddings_for(texts: List[str], cfg: Dict[str, Any]) -> List[List[float]]:
    emb = cfg["embedding"]
    if emb["provider"] == "ollama":
        return embed_texts_ollama(texts, emb["model"], emb["ollama_url"])
    return embed_texts_openai(texts, emb["model"], emb["openai_api_base"], emb["openai_api_key"])

def detect_embedding_dim(cfg: Dict[str, Any]) -> int:
    vecs = embeddings_for(["dimension probe"], cfg)
    if not vecs or not vecs[0]:
        raise RuntimeError("Failed to detect embedding dimension from provider.")
    return len(vecs[0])

# ---------------------------
# DB access
# ---------------------------

def db_connect(db_cfg: Dict[str, Any]):
    return psycopg2.connect(
        host=db_cfg["host"],
        port=db_cfg["port"],
        user=db_cfg["user"],
        password=db_cfg["password"],
        dbname=db_cfg["dbname"],
    )

def create_tables(cur, dim: int):
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS pages (
      path           TEXT PRIMARY KEY,
      content_hash   TEXT NOT NULL,
      title          TEXT,
      description    TEXT,
      image          TEXT,
      centroid       vector({dim}),
      updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
      is_public      BOOLEAN NOT NULL DEFAULT true
    );
    """)

    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS chunks (
      id        BIGSERIAL PRIMARY KEY,
      path      TEXT NOT NULL REFERENCES pages(path) ON DELETE CASCADE,
      ord       INTEGER NOT NULL,
      text_md   TEXT NOT NULL,
      embedding vector({dim}) NOT NULL
    );
    """)

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_path ON pages(path);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_updated_at ON pages(updated_at);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_is_public ON pages(is_public);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_centroid ON pages USING ivfflat (centroid vector_cosine_ops) WITH (lists = 100);")

def drop_tables(cur):
    cur.execute("DROP TABLE IF EXISTS chunks CASCADE;")
    cur.execute("DROP TABLE IF EXISTS pages CASCADE;")

def page_hash_changed(cur, path: str, new_hash: str) -> bool:
    cur.execute("SELECT content_hash FROM pages WHERE path=%s", (path,))
    row = cur.fetchone()
    return (row is None) or (row[0] != new_hash)

def upsert_page(cur, path: str, content_hash: str, og: Dict[str, Optional[str]], centroid_vec: Optional[List[float]], is_public: bool):
    centroid_sql = vec_to_literal(centroid_vec) if centroid_vec else None
    cur.execute("""
      INSERT INTO pages (path, content_hash, title, description, image, centroid, is_public)
      VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
      ON CONFLICT (path) DO UPDATE SET
        content_hash = EXCLUDED.content_hash,
        title        = COALESCE(EXCLUDED.title, pages.title),
        description  = COALESCE(EXCLUDED.description, pages.description),
        image        = COALESCE(EXCLUDED.image, pages.image),
        centroid     = COALESCE(EXCLUDED.centroid, pages.centroid),
        updated_at   = now(),
        is_public    = EXCLUDED.is_public
    """, (path, content_hash, og.get("title"), og.get("description"), og.get("image"), centroid_sql, is_public))

def replace_chunks(cur, path: str, chunks_md: List[str], embeddings: List[List[float]]):
    cur.execute("DELETE FROM chunks WHERE path=%s", (path,))
    for i, (md, emb) in enumerate(zip(chunks_md, embeddings)):
        cur.execute("""
          INSERT INTO chunks (path, ord, text_md, embedding)
          VALUES (%s, %s, %s, %s::vector)
        """, (path, i, md, vec_to_literal(emb)))

def get_all_public_pages_with_meta(cur) -> Dict[str, Dict[str, Optional[str]]]:
    cur.execute("""
      SELECT path, title, description, image
      FROM pages
      WHERE is_public AND centroid IS NOT NULL
    """)
    return {row[0]: {"title": row[1], "desc": row[2], "image": row[3]} for row in cur.fetchall()}

def knn_pages(cur, query_vec: List[float], k: int, exclude_path: str) -> List[str]:
    cur.execute("""
      SELECT path
      FROM pages
      WHERE centroid IS NOT NULL
        AND is_public
        AND path <> %s
      ORDER BY centroid <=> %s::vector
      LIMIT %s
    """, (exclude_path, vec_to_literal(query_vec), k))
    return [r[0] for r in cur.fetchall()]

def random_pages(cur, limit: int, exclude: List[str]) -> List[str]:
    cur.execute("""
      SELECT path FROM pages
      WHERE is_public AND centroid IS NOT NULL AND path <> ALL(%s)
      ORDER BY random()
      LIMIT %s
    """, (exclude or ["__none__"], limit))
    return [r[0] for r in cur.fetchall()]


def boltzmann_sample(cands, k, temperature=0.7, pin_top=True, seed=None):
    """
    cands: list[(path, dist)] sorted by ascending dist
    returns: list[path] of length k, sampled without replacement
    """
    if not cands:
        return []
    k = min(k, len(cands))
    rng = random.Random(seed) if seed is not None else random

    # Optionally pin the top-1 deterministic neighbor
    chosen = []
    rest = cands[:]
    if pin_top and k > 0:
        chosen.append(rest[0][0])
        rest = rest[1:]
        k -= 1
        if k == 0:
            return chosen

    if not rest:
        return chosen

    # Compute Boltzmann weights from distances
    dmin = min(d for _, d in rest)
    T = max(float(temperature), 1e-6)
    weights = [math.exp(- (d - dmin) / T) for _, d in rest]

    # Weighted sampling without replacement (simple renormalization loop)
    selected = []
    items = list(zip([p for p, _ in rest], weights))  # [(path, w), ...]
    for _ in range(k):
        total = sum(w for _, w in items)
        if total <= 0:
            # fallback to uniform
            idx = rng.randrange(len(items))
        else:
            # draw in [0, total)
            r = rng.random() * total
            acc = 0.0
            idx = 0
            for i, (_, w) in enumerate(items):
                acc += w
                if acc >= r:
                    idx = i
                    break
        selected.append(items[idx][0])
        # remove chosen and continue
        items.pop(idx)

    return chosen + selected

# ---------------------------
# Commands
# ---------------------------

def cmd_initdb(cfg: Dict[str, Any]):
    dim = detect_embedding_dim(cfg)
    with db_connect(cfg["db"]) as conn:
        with conn.cursor() as cur:
            create_tables(cur, dim)
        conn.commit()
    print(f"Initialized database (vector dim = {dim}).")

def cmd_resetdb(cfg: Dict[str, Any]):
    dim = detect_embedding_dim(cfg)
    with db_connect(cfg["db"]) as conn:
        with conn.cursor() as cur:
            drop_tables(cur)
            create_tables(cur, dim)
        conn.commit()
    print(f"Reset database (vector dim = {dim}).")

def cmd_index(cfg: Dict[str, Any], page_dir: Optional[str]):
    site_root = Path(page_dir or cfg["site_root"]).resolve()
    exclude_globs = cfg["exclude_globs"]
    content_ids = cfg["content_ids"]
    chunk_cfg = cfg["chunk"]

    # Probe provider once to verify
    dim_probe = detect_embedding_dim(cfg)

    html_files = [p for p in site_root.rglob("*.html") if not should_exclude(site_root, p, exclude_globs)]

    with db_connect(cfg["db"]) as conn:
        conn.autocommit = False
        with conn.cursor() as cur:
            indexed, skipped = 0, 0
            for f in html_files:
                html = f.read_text("utf-8", errors="ignore")
                soup = BeautifulSoup(html, "lxml")

                inner_html = select_content_element(soup, content_ids)
                if not inner_html:
                    skipped += 1
                    continue

                #path = canonical_path(site_root, f)
                path = rendered_rel(site_root, f)
                chash = sha1(inner_html)
                og = extract_og_meta(soup, cfg["strip_image_hosts"])
                is_public = not any(path.lstrip("/").startswith(prefix.rstrip("/")) for prefix in ["private/", "drafts/"])

                # Only re-embed if content changed
                if not page_hash_changed(cur, path, chash):
                    # Update metadata (title/desc/image/public) if present
                    html_file = rendered_rel(site_root, f)
                    upsert_page(cur, path, chash, og, None, is_public)
                    conn.commit()
                    skipped += 1
                    continue

                # Convert to MD & chunk
                md = html_to_markdown(inner_html)
                chunks_md = chunk_markdown(md, chunk_cfg["max_tokens"], chunk_cfg["overlap_tokens"])
                if not chunks_md:
                    skipped += 1
                    continue

                # Embeddings
                embs = embeddings_for(chunks_md, cfg)
                if any(len(e) != dim_probe for e in embs):
                    raise RuntimeError("Embedding dimension mismatch; run 'resetdb' after changing model/provider.")

                centroid = average_vectors(embs)

                # Upsert page meta + centroid, then replace chunks
                html_file = rendered_rel(site_root, f)
                upsert_page(cur, path, chash, og, centroid, is_public)
                replace_chunks(cur, path, chunks_md, embs)
                conn.commit()
                indexed += 1

            print(f"Indexing complete. Indexed: {indexed}, Skipped (unchanged/no content): {skipped}")

def cmd_genrel(cfg: Dict[str, Any], out_path: Optional[str]):
    out_file = Path(out_path or cfg["data_out"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    k = int(cfg["neighbors"]["k"])
    ksample = int(cfg["neighbors"]["ksample"])
    m = int(cfg["neighbors"]["seealso"])

    with db_connect(cfg["db"]) as conn, conn.cursor() as cur:
        meta = get_all_public_pages_with_meta(cur)
        all_paths = list(meta.keys())

        result: Dict[str, Dict[str, Any]] = {}
        for path in all_paths:
            # Use this page's centroid as query
            cur.execute("SELECT centroid FROM pages WHERE path=%s", (path,))
            row = cur.fetchone()
            if not row or row[0] is None:
                continue
            # row[0] is a string like '[...]' when fetched as text? psycopg2 returns pgvector as str; we only need it to query neighbors, so reuse it
            # Easier: query neighbors using DB-side centroid directly:
            cur.execute("""
              SELECT path,
                    centroid <=> (SELECT centroid FROM pages WHERE path=%s) AS dist
              FROM pages
              WHERE centroid IS NOT NULL
                AND is_public
                AND path <> %s
              ORDER BY dist
              LIMIT %s
            """, (path, path, ksample))
            cands = cur.fetchall()

            # Sample k of them using temperature
            k = int(cfg["neighbors"]["k"])
            T = float(cfg["neighbors"].get("temperature", 0.7))
            pin_top = bool(cfg["neighbors"].get("pin_top", True))
            seed = cfg["neighbors"].get("seed")  # could be None or int

            related = boltzmann_sample(cands, k=k, temperature=T, pin_top=pin_top, seed=seed)
            #related = [r[0] for r in cur.fetchall()]

            exclude = set([path] + related)
            seealso = random_pages(cur, m, list(exclude))

            def pack(p: str) -> Dict[str, str]:
                mi = meta.get(p, {})
                title = mi.get("title") or p.strip("/").split("/")[-1].replace("-", " ").title()
                desc  = (mi.get("desc") or "") # [:240]
                img   = mi.get("image") or ""
                return {"url": p, "title": title, "desc": desc, "image": img}

            result[path] = {
                "related": [pack(p) for p in related],
                "seealso": [pack(p) for p in seealso]
            }

    out_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    print(f"Wrote {out_file}")

# ---------------------------
# Main / argparse
# ---------------------------

def run(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point used by the command-line interface."""
    ap = argparse.ArgumentParser(prog="blogsimi", description="Build related pages via embeddings and pgvector.")
    ap.add_argument("--config", "-c", default=DEFAULT_CONFIG_PATH,
                    help=f"Path to JSON config (default: {DEFAULT_CONFIG_PATH})")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp_init = sub.add_parser("initdb", help="Create DB tables (auto-detect embedding dimension).")

    sp_reset = sub.add_parser("resetdb", help="Drop & recreate DB tables (auto-detect embedding dimension).")

    sp_index = sub.add_parser("index", help="Index the rendered HTML into pgvector (only changed pages re-embedded).")
    sp_index.add_argument("--page", "-p", help="Path to rendered site (_site). Overrides config.site_root.")

    sp_gen = sub.add_parser("genrel", help="Generate _data/related.json from DB neighbors.")
    sp_gen.add_argument("--out", "-o", help="Output path for related JSON. Overrides config.data_out.")

    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.cmd == "initdb":
        cmd_initdb(cfg)
    elif args.cmd == "resetdb":
        cmd_resetdb(cfg)
    elif args.cmd == "index":
        cmd_index(cfg, args.page)
    elif args.cmd == "genrel":
        cmd_genrel(cfg, args.out)
    else:
        ap.print_help()
        return 1

    return 0
