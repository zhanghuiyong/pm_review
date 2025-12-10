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


class ArxivCrossrefImput:
    """
    A class for querying and imputing DOI information for arXiv papers using Crossref.
    """

    def __init__(self, timeout=60):
        """
        Initialize the ArxivCrossrefImput instance.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self.crossref_client = Crossref(timeout=timeout)

    def _parse_boolean_query(self, query: str) -> list:
        """
        Parse boolean query, returning all sub-query combinations.
        
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

    def fetch_arxiv(self, query: str, max_results: int = 20) -> list:
        """
        Fetch papers from arXiv based on query and convert to BibTeX format.

        Args:
            query: Search query string
            max_results: Maximum number of results to fetch

        Returns:
            List of paper entries in BibTeX format
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
            print(f"Query '{query}' returned insufficient results, skipped empty pages.")
        print(f"Query '{query}':")
        print(f"Total {len(bib_entries)} records found.")
        return bib_entries

    def fetch_crossref(self, keyword: str, max_results: int = 20) -> list:
        """
        Fetch papers from Crossref based on keyword.

        Args:
            keyword: Search keyword
            max_results: Maximum number of results to fetch

        Returns:
            List of paper entries from Crossref
        """
        try:
            results = self.crossref_client.works(query=keyword, limit=10)
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
            print("Request timed out. Please retry later or check network connection.")
            papers = []
        except Exception as e:
            print(f"An error occurred: {e}")
            papers = []
        return papers

    def fetch_semantic(self, doi: str = None, title: str = None) -> int:
        """
        Get citation count from Semantic Scholar.

        Args:
            doi: Paper DOI
            title: Paper title

        Returns:
            Citation count or None if not found
        """
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

    def deduplicate_and_enrich(self, papers: list) -> list:
        """
        Deduplicate papers prioritizing Crossref/DOI sources and enrich with citation counts.

        Args:
            papers: List of paper entries

        Returns:
            Deduplicated and enriched list of papers
        """
        seen = {}
        for p in papers:
            key = p["doi"].lower() if p["doi"] else p["title"].lower()
            if key in seen:
                if p["source"] != "arXiv":  # Prefer non-arXiv sources
                    seen[key] = p
            else:
                seen[key] = p

        # Enrich with citation counts
        for k, p in seen.items():
            if p["citations"] is None:
                p["citations"] = self.fetch_semantic(p["doi"], p["title"])
                time.sleep(0.2)
        return list(seen.values())

    def to_bibtex(self, papers: list, filename: str = "output.bib"):
        """
        Save papers to BibTeX file.

        Args:
            papers: List of paper entries
            filename: Output filename
        """
        db = []
        for i, p in enumerate(papers):
            p["citations"] = str(p["citations"]) if p["citations"] is not None else ""
            db.append(p)

        bib_db = bibtexparser.bibdatabase.BibDatabase()
        bib_db.entries = db

        with open(filename, "w", encoding="utf-8") as bibfile:
            bibtexparser.dump(bib_db, bibfile)

        print(f"✅ Saved {filename}, total {len(db)} entries")

    def find_doi_by_title(self, title: str, authors: list = None, topn: int = 5) -> str:
        """
        Find DOI for a given title using Crossref, validated by title similarity and author verification.

        Args:
            title: Paper title
            authors: List of authors (optional)
            topn: Number of top results to consider

        Returns:
            DOI if found, otherwise None
        """
        try:
            results = self.crossref_client.works(query=title, limit=topn)
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

            # Author validation (bonus points for overlap)
            if authors:
                overlap = len(set(a.lower() for a in authors) & set(candidate_authors))
                if overlap > 0:
                    score += 0.1

            if score > best_score:
                best_score = score
                best_match = candidate_doi

        return best_match if best_score > 0.75 else None

    def impute_dois_for_arxiv(self, query: str, max_results: int = 1000) -> list:
        """
        Query arXiv and attempt to find DOI for each result using Crossref.

        Args:
            query: Search query for arXiv
            max_results: Maximum results to fetch

        Returns:
            List of papers with potentially imputed DOIs
        """
        print(f"=== Processing query: {query} ===")
        arxiv_entries = self.fetch_arxiv(query, max_results)

        # For each arXiv entry, try to find DOI
        updated_entries = []
        for entry in arxiv_entries:
            print(f"🔍 Looking for DOI: {entry['title'][:60]}...")
            authors = entry["author"].split(" and ")
            doi_guess = self.find_doi_by_title(entry["title"], authors)
            if doi_guess:
                entry["doi"] = doi_guess
                print(f"✅ Found DOI: {doi_guess} | {entry['title'][:60]}...")

            updated_entries.append(entry)
        
        return updated_entries

    def save_papers_pickle(self, papers: list, filename: str = "papers.pkl"):
        """
        Save papers to a pickle file.

        Args:
            papers: List of paper entries
            filename: Output filename
        """
        with open(filename, "wb") as f:
            pickle.dump(papers, f)
        print(f"📥 Saved {len(papers)} papers to {filename}")

    def load_papers_pickle(self, filename: str = "papers.pkl") -> list:
        """
        Load papers from a pickle file.

        Args:
            filename: Input filename

        Returns:
            List of paper entries
        """
        with open(filename, "rb") as f:
            return pickle.load(f)


def main():
    """
    Example usage of the ArxivCrossrefImput class.
    """
    # Initialize the imputer
    imputer = ArxivCrossrefImput(timeout=60)

    # Define a complex query
    query = '("Artificial Intelligence" OR "Machine Learning" OR "Deep Learning") AND ("Explainable AI" OR "XAI" OR "Interpretability" OR "Transparency") AND ("Precision Medicine" OR "Personalized Medicine" OR "Healthcare" OR "Medical Diagnosis")'

    # Parse the boolean query into sub-queries
    sub_queries = imputer._parse_boolean_query(query)
    print(f"🔎 Parsed into {len(sub_queries)} sub-queries:")
    for sq in sub_queries:
        print("   ", sq)

    # Process each sub-query and collect papers
    all_papers = []
    for sq in sub_queries[:1]:  # Limit to first sub-query for demonstration
        papers = imputer.impute_dois_for_arxiv(sq, 1000)
        all_papers.extend(papers)

    # Save raw papers to pickle
    imputer.save_papers_pickle(all_papers, "arxiv_raw.pkl")

    # Load and process papers
    loaded_papers = imputer.load_papers_pickle("arxiv_raw.pkl")
    enriched_papers = imputer.deduplicate_and_enrich(loaded_papers)

    # Save to BibTeX
    imputer.to_bibtex(enriched_papers, "arxiv_with_doi_imputed.bib")

    print(f"\nSummary: Processed {len(all_papers)} initial papers, {len(enriched_papers)} after enrichment")


if __name__ == "__main__":
    main()



