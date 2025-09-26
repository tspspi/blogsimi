# Blog similarity index

`blogsimi` builds semantically related article recommendations for static blogs.
It walks your rendered site, turns the content into embeddings, stores them in
PostgreSQL via [pgvector](https://github.com/pgvector/pgvector), and then exports
a recommendation JSON that you can drop straight into Jekyll or any static site
generator.  The design and rationale are described in [a blog post](https://www.tspi.at/2025/09/25/simi.html).

## Features

- Extracts rendered HTML, strips boilerplate (extracts only specific content ids or classes) and
  chunks content into embedding-friendly blocks.
- Supports Ollama (default) and OpenAI embedding providers; switching providers is a config change and
  requires a rebuild of the index.
- Persists embeddings, metadata, and recommendations in PostgreSQL with pgvector distance queries.
  Pages are only re-indexed when the content has changed.
- A very simple CLI

## Installation

The project can be installed from [PyPi](https://pypi.org/project/blogsimi/):

```bash
pip install blogsimi                # from PyPI once published
# or
pip install .                       # from a local checkout
```

## Configuration

Configuration lives in `~/.config/blogsimilarity.cfg` by default (overridable
with `--config`).  The file is JSON and mirrors the defaults baked into the package:

```json
{
  "site_root": "_site",
  "data_out": "_data/related.json",
  "exclude_globs": ["tags/**", "drafts/**", "private/**", "admin/**"],
  "content_ids": ["content"],
  "neighbors": {
    "ksample": 16,
    "k": 8,
    "temperature": 0.7,
    "pin_top": true,
    "seed": null,
    "seealso": 4
  },
  "chunk": {
    "max_tokens": 800,
    "overlap_tokens": 100
  },
  "embedding": {
    "provider": "ollama",
    "model": "nomic-embed-text",
    "ollama_url": "http://127.0.0.1:11434/api/embeddings",
    "openai_api_base": "https://api.openai.com/v1/embeddings",
    "openai_api_key_env": "OPENAI_API_KEY"
  },
  "db": {
    "host": "127.0.0.1",
    "port": 5432,
    "user": "blog",
    "password": "blog",
    "dbname": "blog"
  },
  "strip_image_hosts": null
}
```

Set `OPENAI_API_KEY` (or the environment variable you configure) when using the
OpenAI provider.  Ensure your PostgreSQL instance has the `pgvector` extension
installed. You have to manually enable the extension as a superuser in the database!

## CLI Usage

All functionality is exposed via the `blogsimi` command:

- `blogsimi initdb` – create the required tables and infer the embedding dimension from your provider.
  Note that the ```VECTOR``` extension has to be already enabled.
- `blogsimi resetdb` – drop and recreate the tables (useful when switching embedding dimensions).
- `blogsimi index [--page PATH]` – walk the rendered site (defaults to `site_root`), compute
  embeddings where content changed, and persist them.
- `blogsimi genrel [--out PATH]` – produce the recommendation JSON ready for your
  static site.

A typical run after rendering your blog might look like:

```bash
blogsimi index --page _site
blogsimi genrel --out _data/related.json
```

## Development

The repository uses a `src/` layout.  For local development, install in editable mode:

```bash
pip install -e .
PYTHONPATH=src python -m blogsimi.cli --help
```

## License

This project is released under the MIT License.
