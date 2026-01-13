import arxiv
import requests
import feedparser
from habanero import Crossref
import bibtexparser
from collections import Counter
from difflib import SequenceMatcher

import re
import time
import itertools
import pickle

# ========== Utility function: Boolean query parsing ==========
def parse_boolean_query(query):
    """
    Parse boolean queries and return all sub-query combinations
    Input example:
      ("precision medicine" OR "digital health") AND ("interpretable machine learning" OR "explainable artificial intelligence")
    Output:
      ['"precision medicine" "interpretable machine learning"',
       '"precision medicine" "explainable artificial intelligence"',
       '"digital health" "interpretable machine learning"',
       '"digital health" "explainable artificial intelligence"']
    """
    # Split by AND blocks
    blocks = [b.strip(" ()") for b in query.split("AND")]
    option_lists = []
    for b in blocks:
        parts = [p.strip(" ()\"") for p in b.split("OR")]
        option_lists.append(parts)

    # Cartesian product combination + preserve double quotes
    combos = list(itertools.product(*option_lists))
    return [" ".join([f"\"{term}\"" for term in c]) for c in combos]


# ========== Data source functions ==========
def fetch_arxiv(keyword, max_results=20):
    """
    Search articles from arXiv and save results as BibTeX file.
    """
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    bib_entries = []
    try:
        for r in client.results(search):
            bib_entry = {
                "ENTRYTYPE": "article",
                "ID": r.get_short_id(),
                "title": r.title,
                "author": " and ".join([str(author) for author in r.authors]),
                "year": str(r.published.year),
                "journal": "arXiv preprint",
                "eprint": r.get_short_id(),
                "url": r.entry_id,
                "abstract": r.summary,
                "doi": r.doi if r.doi else "",
                "comment": r.comment if r.comment else "", 
				"citations": None,
				"source": "arXiv"
            }
            bib_entries.append(bib_entry)
    except arxiv.UnexpectedEmptyPageError:
        print(f"Keyword [{query}] has insufficient results, skipping empty pages.")
    print(f"Keyword [{query}]:")
    print(f"Total {len(bib_entries)} records.")
    return bib_entries

def fetch_crossref(keyword, max_results=20):
    """Fetch literature from Crossref"""
    cr = Crossref(timeout=60)  # Set timeout during initialization

    try:
        results = cr.works(query=keyword, limit=10)
        papers = []
        for item in results['message']['items']:
            papers.append({
                "title": item.get("title", [""])[0],
                "authors": [a.get("family", "") for a in item.get("author", []) if "family" in a],
                "year": item.get("issued", {}).get("date-parts", [[None]])[0][0],
                "doi": item.get("DOI", None),
                "url": item.get("URL", None),
                "citations": item.get("is-referenced-by-count", None),
                "source": "Crossref"
            })
    except requests.exceptions.Timeout:
        print("Request timed out, please retry later or check network connection.")
        papers = []
    except Exception as e:
        print(f"An error occurred: {e}")
        papers = []
    return papers

def fetch_semantic(doi=None, title=None):
    """Get citation count from Semantic Scholar"""
    base = "https://api.semanticscholar.org/graph/v1/paper/"
    if doi:
        url = base + f"DOI:{doi}?fields=citationCount"
    elif title:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={title}&limit=1&fields=citationCount"
    else:
        return None

    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if doi:
                return data.get("citationCount", None)
            elif "data" in data and data["data"]:
                return data["data"][0].get("citationCount", None)
    except Exception:
        return None
    return None

# ========== Deduplication & Enrichment ==========
def deduplicate_and_enrich(papers):
    """Deduplication: Prioritize Crossref/DOI"""
    seen = {}
    for p in papers:
        key = p["doi"].lower() if p["doi"] else p["title"].lower()
        if key in seen:
            if p["source"] != "arXiv":  # Keep officially published
                seen[key] = p
        else:
            seen[key] = p

    # Fill in citation counts
    for k, p in seen.items():
        if p["citations"] is None:
            p["citations"] = fetch_semantic(p["doi"], p["title"])
            time.sleep(0.2)
    return list(seen.values())

# ========== BibTeX & Analysis ==========
def to_bibtex(papers, filename="output.bib"):
    """Save as BibTeX"""
    db = []
    for i, p in enumerate(papers):
        # entry = {
        #     "ENTRYTYPE": "article",
        #     "ID": p["ID"] if "ID" in p else f"paper{i+1}",
        #     "title": p["title"],
        #     "author": p["author"],
        #     "year": str(p["year"]),
        #     "url": p["url"],
        #     "note": f"Cited {p['citations']} times" if p["citations"] is not None else "Citations: NA"
        # }
        p["citations"] = str(p["citations"]) if p["citations"] is not None else ""
        db.append(p)

    bib_db = bibtexparser.bibdatabase.BibDatabase()
    bib_db.entries = db

    with open(filename, "w", encoding="utf-8") as bibfile:
        bibtexparser.dump(bib_db, bibfile)
        
    print(f"✅ Generated {filename}, total {len(db)} references")

def keyword_analysis(papers):
    text = " ".join(p["title"] for p in papers)
    words = re.findall(r"\w+", text.lower())
    counter = Counter(words)
    print("\n📊 Top 15 Frequent Words:")
    for word, freq in counter.most_common(15):
        print(f"{word}: {freq}")

def author_analysis(papers):
    authors = []
    for p in papers:
        authors.extend(p["author"])
    counter = Counter(authors)
    print("\n👥 Top 10 Frequent Authors:")
    for author, freq in counter.most_common(10):
        print(f"{author}: {freq}")

# ========== New functions ==========
def find_doi_by_title(title, authors=None, topn=5):
    """
    Find DOI for a title using Crossref, verified by title similarity + author matching
    """
    cr = Crossref(timeout=60)
    try:
        results = cr.works(query=title, limit=topn)
        time.sleep(1)
    except Exception as e:
        print(f"Crossref query failed: {e}")
        return None

    best_match = None
    best_score = 0
    for item in results['message']['items']:
        candidate_title = item.get("title", [""])[0]
        candidate_authors = [a.get("family", "").lower() for a in item.get("author", []) if "family" in a]
        candidate_doi = item.get("DOI", None)

        # Title similarity
        score = SequenceMatcher(None, title.lower(), candidate_title.lower()).ratio()

        # Author verification (bonus points for overlap)
        if authors:
            overlap = len(set(a.lower() for a in authors) & set(candidate_authors))
            if overlap > 0:
                score += 0.1

        if score > best_score:
            best_score = score
            best_match = candidate_doi

    return best_match if best_score > 0.75 else None


# ========== Main program modification ==========
if __name__ == "__main__":
    query = '("Artificial Intelligence" OR "Machine Learning" OR "Deep Learning") AND ("Explainable AI" OR "XAI" OR "Interpretability" OR "Transparency") AND ("Precision Medicine" OR "Personalized Medicine" OR "Healthcare" OR "Medical Diagnosis")'
    sub_queries = parse_boolean_query(query)
    print(f"🔎 Decomposed into {len(sub_queries)} sub-queries:")
    for sq in sub_queries:
        print("   ", sq)

    papers = []
    for sq in sub_queries:  # Only test first sub-query
        # First fetch arXiv
        print(f"=== Processing sub-query: {sq} ===")
        arxiv_entries = fetch_arxiv(sq, 1000)

        # For each arXiv result, try to find DOI
        entrys = []
        for entry in arxiv_entries:
            print(f"🔍 Finding DOI: {entry['title'][:60]}...")
            authors = entry["author"].split(" and ")
            doi_guess = find_doi_by_title(entry["title"], authors)
            if doi_guess:
                entry["doi"] = doi_guess
                print(f"✅ Matched DOI: {doi_guess} | {entry['title'][:60]}...")

            entrys.append(entry)
        papers += entrys
    # Persist papers in pickle format
    with open("arxiv_raw.pkl", "wb") as f:
        pickle.dump(papers, f)
    print(f"\n📥 Total {len(papers)} arXiv entries fetched and saved as arxiv_raw.pkl")

    # Example of reading pickle file
    with open("arxiv_raw.pkl", "rb") as f:
        papers = pickle.load(f)
    
    enriched = deduplicate_and_enrich(papers)

    to_bibtex(enriched, "arxiv1.bib")
    # keyword_analysis(enriched)
    # author_analysis(enriched)

