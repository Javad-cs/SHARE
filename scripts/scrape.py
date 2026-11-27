"""
Image scraper for cultural food clusters.
Reads search_terms from clusters.json and downloads images for each term
into separate subfolders. No country-level queries — terms are cluster-level
to avoid country bias in scraping.

Output structure:
    data/raw/{cluster_id}/{term_slug}/Image_1.jpg ...

Usage:
    # Scrape all clusters
    python scripts/scrape.py

    # Scrape a specific cluster
    python scripts/scrape.py --cluster central_asian_plov

    # Override images per term
    python scripts/scrape.py --cluster central_asian_plov --limit 50

    # Dry run
    python scripts/scrape.py --cluster central_asian_plov --dry-run
"""

import json
import os
import re
import argparse
import time
from pathlib import Path
from bing_image_downloader import downloader


def load_clusters(path="notes/clusters.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def term_to_slug(term):
    """Convert search term to safe folder name slug.
    Preserves Unicode characters (Cyrillic, Devanagari, Arabic, Korean etc.)
    Only replaces spaces with underscores and removes filesystem-unsafe chars.
    """
    slug = term.strip()
    slug = re.sub(r'[<>:"/\\|?*]', '', slug)
    slug = re.sub(r'\s+', '_', slug)
    return slug or "unknown"


def scrape_term(term, term_dir, limit=30):
    """Download images for a single search term into term_dir."""
    os.makedirs(term_dir.parent, exist_ok=True)
    print(f"    Scraping: '{term}' -> {term_dir}")
    try:
        downloader.download(
            term,
            limit=limit,
            output_dir=str(term_dir.parent),
            adult_filter_off=False,
            force_replace=False,
            timeout=60,
        )
        # Bing downloader saves to output_dir/{query}/ — rename to slug
        bing_dir = term_dir.parent / term
        if bing_dir.exists() and bing_dir != term_dir:
            bing_dir.rename(term_dir)
    except Exception as e:
        print(f"    ERROR scraping '{term}': {e}")
    time.sleep(2)


def scrape_cluster(cluster, raw_dir, limit=30, dry_run=False):
    """Scrape all search terms for a cluster into term-slug subfolders."""
    cluster_id = cluster["cluster_id"]
    search_terms = cluster.get("search_terms", [])

    print(f"\n{'='*60}")
    print(f"Cluster: {cluster['cluster_name']}")
    print(f"Terms: {search_terms}")
    print(f"Limit per term: {limit}")
    print(f"{'='*60}")

    if not search_terms:
        print("  No search_terms found in cluster definition.")
        return

    for term in search_terms:
        slug = term_to_slug(term)
        term_dir = raw_dir / cluster_id / slug

        if dry_run:
            print(f"  [DRY RUN] Would scrape '{term}' -> {term_dir}")
            continue

        if term_dir.exists():
            existing = len(list(term_dir.glob("*")))
            if existing > 0:
                print(f"    [skip] '{term}' already has {existing} images")
                continue

        scrape_term(term, term_dir, limit=limit)

    if not dry_run:
        print(f"\n  Summary for {cluster_id}:")
        for term in search_terms:
            slug = term_to_slug(term)
            term_dir = raw_dir / cluster_id / slug
            count = len(list(term_dir.glob("*"))) if term_dir.exists() else 0
            print(f"    '{term}': {count} images")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape food images per search term into term-slug folders"
    )
    parser.add_argument("--clusters-file", default="notes/clusters.json")
    parser.add_argument("--raw-dir", default="data/raw",
                        help="Output directory for raw images (default: data/raw)")
    parser.add_argument("--cluster", default=None,
                        help="Scrape only this cluster")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max images per search term (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be scraped without downloading")
    args = parser.parse_args()

    data = load_clusters(args.clusters_file)
    clusters = data["clusters"]

    if args.cluster:
        clusters = [c for c in clusters if c["cluster_id"] == args.cluster]
        if not clusters:
            print(f"Cluster '{args.cluster}' not found.")
            return

    if args.dry_run:
        print("DRY RUN — no images will be downloaded.\n")

    for cluster in clusters:
        scrape_cluster(
            cluster,
            raw_dir=Path(args.raw_dir),
            limit=args.limit,
            dry_run=args.dry_run,
        )

    print("\n\nDone! Next step: run filter.py on data/raw/ to clean images.")


if __name__ == "__main__":
    main()