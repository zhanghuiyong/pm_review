import arxiv
import bibtexparser
import re
import itertools
from typing import List, Dict, Any


class ArxivBatchSearch:
    """
    A class for batch searching arXiv papers using combined keywords and downloading metadata.
    """

    def __init__(self, max_results_per_query: int = 1000):
        """
        Initialize the ArxivBatchSearch instance.

        Args:
            max_results_per_query: Maximum number of results to fetch per individual query
        """
        self.max_results_per_query = max_results_per_query
        self.client = arxiv.Client()

    def _arxiv_to_bibtex(self, query: str) -> List[Dict[str, Any]]:
        """
        Search arXiv for papers matching the query and convert results to BibTeX format.

        Args:
            query: Search query string

        Returns:
            List of BibTeX entries as dictionaries
        """
        search = arxiv.Search(
            query=query,
            max_results=self.max_results_per_query,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        bib_entries = []
        try:
            for result in self.client.results(search):
                bib_entry = {
                    "ENTRYTYPE": "article",
                    "ID": result.get_short_id(),
                    "title": result.title,
                    "author": " and ".join([str(author) for author in result.authors]),
                    "year": str(result.published.year),
                    "journal": "arXiv preprint",
                    "eprint": result.get_short_id(),
                    "url": result.entry_id,
                    "abstract": result.summary,
                    "doi": result.doi if result.doi else "",
                    "comment": result.comment if result.comment else "",
                }
                bib_entries.append(bib_entry)
        except arxiv.UnexpectedEmptyPageError:
            print(f"Query '{query}' returned insufficient results, skipped empty pages.")
        
        print(f"Query '{query}':")
        print(f"Total {len(bib_entries)} records found.")
        return bib_entries

    def _parse_boolean_query(self, query: str) -> List[str]:
        """
        Parse boolean logic query like (A OR B) AND (C OR D) into sub-query combinations.

        Args:
            query: Boolean query string

        Returns:
            List of sub-queries as strings
        """
        # Split by AND parts
        groups = re.split(r"\s+AND\s+", query, flags=re.IGNORECASE)

        parsed_groups = []
        for group in groups:
            # Extract content within parentheses or split directly by OR
            group = group.strip()
            if group.startswith("(") and group.endswith(")"):
                group = group[1:-1]  # Remove outer parentheses
            terms = re.split(r"\s+OR\s+", group, flags=re.IGNORECASE)
            terms = [term.strip('" ').strip() for term in terms if term.strip()]
            parsed_groups.append(terms)

        # Generate Cartesian product of all groups
        combinations = list(itertools.product(*parsed_groups))

        # Format each combination as a query string
        subqueries = []
        for combo in combinations:
            formatted_combo = " AND ".join(['"{}"'.format(term) for term in combo])
            subqueries.append(formatted_combo)
        
        return subqueries

    def search_combined_keywords(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Perform batch search using a list of combined keyword queries.

        Args:
            queries: List of query strings to search

        Returns:
            Combined list of BibTeX entries from all queries
        """
        all_entries = []
        for query in queries:
            print(f"\nProcessing query: {query}")
            entries = self._arxiv_to_bibtex(query)
            all_entries.extend(entries)
        
        return all_entries

    def search_boolean_query(self, boolean_query: str) -> List[Dict[str, Any]]:
        """
        Parse and search a boolean query expression.

        Args:
            boolean_query: Boolean query string in format "(A OR B) AND (C OR D)"

        Returns:
            Combined list of BibTeX entries from all sub-queries
        """
        subqueries = self._parse_boolean_query(boolean_query)
        print("Expanded sub-queries:")
        for subquery in subqueries:
            print(f" - {subquery}")
        
        all_entries = []
        for subquery in subqueries:
            entries = self._arxiv_to_bibtex(subquery)
            all_entries.extend(entries)
        
        return all_entries

    def save_results(self, entries: List[Dict[str, Any]], filename: str = "arxiv_results.bib"):
        """
        Save BibTeX entries to a .bib file.

        Args:
            entries: List of BibTeX entries to save
            filename: Output filename
        """
        bib_db = bibtexparser.bibdatabase.BibDatabase()
        bib_db.entries = entries

        with open(filename, "w", encoding="utf-8") as bibfile:
            bibtexparser.dump(bib_db, bibfile)
        
        print(f"Results saved to {filename}")


def main():
    """
    Example usage of the ArxivBatchSearch class.
    """
    # Initialize the searcher
    searcher = ArxivBatchSearch(max_results_per_query=1000)

    # Example 1: Using predefined list of keyword combinations
    print("Example 1: Predefined keyword combinations")
    keywords = [
        '"precision medicine" AND "interpretable machine learning"', 
        '"causal inference" AND "machine learning"', 
        '"reinforcement learning" AND "healthcare"',
        '"digital health" AND "interpretable machine learning"',
        '"precision medicine" AND "explainable artificial intelligence"',
        '"digital health" AND "explainable artificial intelligence"'
    ]
    
    results1 = searcher.search_combined_keywords(keywords)
    searcher.save_results(results1, "arxiv_predefined_results.bib")

    # Example 2: Using boolean query parsing
    print("\nExample 2: Boolean query parsing")
    boolean_query = '("precision medicine" OR "digital health") AND ("interpretable machine learning" OR "explainable artificial intelligence")'
    results2 = searcher.search_boolean_query(boolean_query)
    searcher.save_results(results2, "arxiv_boolean_results.bib")

    # Summary
    print(f"\nTotal results from predefined queries: {len(results1)}")
    print(f"Total results from boolean query: {len(results2)}")


if __name__ == "__main__":
    main()



