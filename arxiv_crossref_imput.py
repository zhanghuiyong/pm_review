import re
import requests
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from habanero import Crossref
import bibtexparser
import pandas as pd
from urllib.parse import quote
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PaperRecord:
    """Unified data structure for paper records"""
    title: str
    authors: List[str]
    year: Optional[int] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    journal: Optional[str] = None
    citations: Optional[int] = None
    source: str = "unknown"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return asdict(self)


class ArxivCrossrefImput:
    """
    A robust tool for imputing missing DOI information for arXiv papers using Crossref API.
    Handles rate limiting, validation, and cross-referencing with multiple data sources.
    """
    
    def __init__(self, 
                 timeout: int = 30,
                 rate_limit_delay: float = 1.0,
                 similarity_threshold: float = 0.85):
        """
        Initialize the DOI imputation service.
        
        Args:
            timeout: Request timeout in seconds
            rate_limit_delay: Delay between API calls to respect rate limits
            similarity_threshold: Minimum title similarity score for DOI matching
        """
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self.similarity_threshold = similarity_threshold
        self.crossref = Crossref(mailto="your.email@example.com")  # Replace with your email
        self.session = requests.Session()
        
    def normalize_title(self, title: str) -> str:
        """Normalize title for comparison: lowercase, remove punctuation, normalize whitespace"""
        import re
        title = title.lower()
        title = re.sub(r'[^\w\s]', ' ', title)  # Remove punctuation
        title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
        return title
    
    def calculate_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        norm1 = self.normalize_title(title1)
        norm2 = self.normalize_title(title2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def extract_arxiv_id(self, entry: Dict) -> Optional[str]:
        """Extract arXiv ID from arXiv entry"""
        arxiv_id = entry.get('id', '')
        if 'arxiv.org' in arxiv_id:
            # Extract format like: http://arxiv.org/abs/2001.12345v2
            return arxiv_id.split('/')[-1].replace('v', '').split('v')[0]
        return None
    
    def query_crossref_by_title(self, title: str, max_results: int = 5) -> List[Dict]:
        """
        Query Crossref API by title
        
        Returns:
            List of matching Crossref entries
        """
        try:
            # Encode query string
            clean_query = self.clean_title_for_search(title)
            clean_title = str(clean_query) if clean_query is not None else ''
            clean_query = clean_title[:200] if len(clean_title) > 200 else clean_title
            query = quote(clean_query)  # Limit query length
            time.sleep(self.rate_limit_delay)  # Respect rate limits
            
            results = self.crossref.works(
                query={'title': title},
                limit=max_results,
                select='DOI,title,author,issued,container-title,type'
            )
            
            return results['message']['items'] if 'items' in results['message'] else []
            
        except Exception as e:
            logger.warning(f"Crossref query failed for title '{title[:50]}...': {e}")
            return []
    
    def match_authors_between_arxiv_and_crossref(self, arxiv_entry, crossref_entry):
        """
        Match authors between arXiv and CrossRef entries by surname (robust implementation)
        
        Args:
            arxiv_entry (dict): arXiv metadata entry containing author information
            crossref_entry (dict): CrossRef metadata entry containing author information
        
        Returns:
            bool: True if at least one common author (by surname) is found, False otherwise
        """
        # Early return if author field is missing in either entry
        if 'author' not in arxiv_entry or 'author' not in crossref_entry:
            return False

        def extract_surname_from_full_name(full_name):
            """
            Extract surname from a full name string (handles edge cases)
            
            Args:
                full_name (str): Full name (e.g., "John Doe", "Einstein", "Mary Ann Smith")
            
            Returns:
                str: Lowercase surname (empty string if input is invalid)
            """
            # Handle empty/None input
            if not full_name or not isinstance(full_name, str):
                return ""
            
            # Clean whitespace and split name parts
            name_parts = full_name.strip().split()
            
            # Return last part as surname (handles single-name authors)
            return name_parts[-1].lower() if name_parts else ""

        # -------------------------- Process arXiv Authors --------------------------
        # arXiv authors format: "Author1 and Author2 and Author3"
        arxiv_author_full_names = arxiv_entry['author'].split(' and ')
        arxiv_surnames = set()
        
        for full_name in arxiv_author_full_names:
            surname = extract_surname_from_full_name(full_name)
            if surname:  # Only add non-empty surnames
                arxiv_surnames.add(surname)

        # -------------------------- Process CrossRef Authors --------------------------
        # CrossRef authors format: list of dicts with 'family' (surname) and 'given' (first name)
        crossref_surnames = set()
        
        for author_dict in crossref_entry['author']:
            # Extract and clean surname from CrossRef author dict
            surname = author_dict.get('family', '').strip().lower()
            if surname:  # Only add non-empty surnames
                crossref_surnames.add(surname)

        # -------------------------- Author Matching Logic --------------------------
        # Check for at least one common surname (handle empty sets to avoid false positives)
        if not arxiv_surnames or not crossref_surnames:
            return False
        
        # Find intersection of surname sets
        common_surnames = arxiv_surnames.intersection(crossref_surnames)
        
        # Return True if any common authors found
        return len(common_surnames) > 0
    
    def validate_match(self, arxiv_entry: Dict, crossref_entry: Dict) -> Tuple[bool, float]:
        """Validate if arXiv entry matches Crossref entry"""
        arxiv_title = arxiv_entry.get('title', '')
        crossref_title = crossref_entry.get('title', [''])[0]
        
        # Calculate title similarity
        similarity = self.calculate_similarity(arxiv_title, crossref_title)
        
        # Basic validation conditions
        has_doi = 'DOI' in crossref_entry
        similarity_ok = similarity >= self.similarity_threshold
        
        # Optional: author matching validation
        author_match = self.match_authors_between_arxiv_and_crossref(arxiv_entry, crossref_entry)
        return (has_doi and similarity_ok and author_match, similarity)
    
    def clean_title_for_search(self, title):
        """Clean title for better search matching"""
        # Remove quotes, colons, etc.
        cleaned = re.sub(r'[:"“”‘’]', ' ', title)
        # Keep only alphanumeric and spaces
        cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
        return ' '.join(cleaned.split())  # normalize whitespace



    def impute_single_doi(self, arxiv_entry: Dict) -> Optional[str]:
        """Find DOI for a single arXiv entry"""
        title = arxiv_entry.get('title', '')
        if not title:
            return None
        
        logger.info(f"Searching DOI for: {title[:60]}...")
        
        # Query Crossref
        crossref_results = self.query_crossref_by_title(title)
        
        # Find best match
        best_doi = None
        best_score = 0.0
        
        for result in crossref_results:
            is_valid, similarity = self.validate_match(arxiv_entry, result)
            
            if is_valid and similarity > best_score:
                best_score = similarity
                best_doi = result.get('DOI')
                
                # Extract additional metadata
                if 'issued' in result:
                    arxiv_entry['year'] = result['issued']['date-parts'][0][0]
                if 'container-title' in result:
                    arxiv_entry['journal'] = result['container-title'][0]
        
        if best_doi:
            logger.info(f"  ✓ Found DOI: {best_doi} (similarity: {best_score:.2f})")
        else:
            logger.info(f"  ✗ No DOI found (best similarity: {best_score:.2f})")
        
        return best_doi
    
    def impute_dois_batch(self, 
                         arxiv_entries: List[Dict],
                         batch_size: int = 50,
                         progress_callback = None) -> List[PaperRecord]:
        """
        Batch process arXiv entries to find and impute DOIs
        
        Args:
            arxiv_entries: List of arXiv entries (output from ArxivBatchSearch)
            batch_size: Batch processing size
            progress_callback: Progress callback function
            
        Returns:
            List of PaperRecord objects with DOI information
        """
        total = len(arxiv_entries)
        results = []
        
        logger.info(f"Starting DOI imputation for {total} arXiv entries...")
        
        for i, entry in enumerate(arxiv_entries, 1):
            # Create PaperRecord object
            record = PaperRecord(
                title=entry.get('title', ''),
                authors=entry.get('author', '').split(' and ') if entry.get('author') else [],
                arxiv_id=self.extract_arxiv_id(entry),
                source='arXiv'
            )
            
            # If DOI already exists, use it directly
            existing_doi = entry.get('doi')
            if existing_doi and existing_doi.startswith('10.'):
                record.doi = existing_doi
                logger.info(f"[{i}/{total}] Using existing DOI: {existing_doi}")
            else:
                # Find DOI
                found_doi = self.impute_single_doi(entry)
                record.doi = found_doi
            
            results.append(record)
            
            # Update progress
            if progress_callback and i % 10 == 0:
                progress_callback(i, total)
            
            # Small batch delay to avoid API limits
            if i % batch_size == 0:
                time.sleep(self.rate_limit_delay * 2)
        
        # Calculate statistics
        doi_count = sum(1 for r in results if r.doi)
        logger.info(f"DOI imputation complete. Found {doi_count} DOIs out of {total} entries ({doi_count/total*100:.1f}%)")
        
        return results
    
    def export_to_bibtex(self, 
                        records: List[PaperRecord], 
                        filename: str = "arxiv_with_doi.bib") -> None:
        """Export to BibTeX format"""
        bib_entries = []
        
        for record in records:
            if not record.doi:
                continue  # Skip records without DOI
            
            entry = {
                'ENTRYTYPE': 'article',
                'ID': record.doi.replace('/', '_'),
                'title': record.title,
                'author': ' and '.join(record.authors),
                'doi': record.doi,
                'year': str(record.year) if record.year else '',
                'journal': record.journal if record.journal else 'arXiv',
                'note': f"arXiv:{record.arxiv_id}" if record.arxiv_id else ''
            }
            
            # Remove empty values
            entry = {k: v for k, v in entry.items() if v}
            bib_entries.append(entry)
        
        # Create BibTeX database
        bib_db = bibtexparser.bibdatabase.BibDatabase()
        bib_db.entries = bib_entries
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            bibtexparser.dump(bib_db, f)
        
        logger.info(f"Exported {len(bib_entries)} entries to {filename}")
    
    def export_to_dataframe(self, records: List[PaperRecord]) -> pd.DataFrame:
        """Export to Pandas DataFrame"""
        data = [r.to_dict() for r in records]
        df = pd.DataFrame(data)
        
        # Add status column
        df['has_doi'] = df['doi'].notna()
        
        return df


# ==================== USAGE EXAMPLE ====================
if __name__ == "__main__":
    # Example: Load arXiv search results from CSV file and impute DOIs
    
    # 1. Initialize DOI imputer
    doi_finder = ArxivCrossrefImput(
        timeout=30,
        rate_limit_delay=1.5,  # Conservative rate limiting
        similarity_threshold=0.82
    )
    
    # 2. Load previously saved arXiv search results
    # Assume we have a CSV file from ArxivBatchSearch output
    try:
        arxiv_df = pd.read_csv("arxiv_search_results.csv")
        arxiv_entries = arxiv_df.to_dict('records')
        print(f"Loaded {len(arxiv_entries)} arXiv entries from CSV")
    except FileNotFoundError:
        # If no CSV file exists, create sample dataset
        print("Creating sample arXiv data...")
        arxiv_entries = [
            {
                'author': 'Jiaming Qu and Jaime Arguello and Yue Wang',
                'comment': '',
                'doi': '',
                'eprint': '2406.03594v1',
                'journal': 'arXiv preprint',
                'title': 'Why is "Problems" Predictive of Positive Sentiment? A Case Study of Explaining Unintuitive Features in Sentiment Classification',
                'url': 'http://arxiv.org/abs/2406.03594v1',
                'year': 2024
            }
        ]
    
    # 3. Define progress callback function
    def print_progress(current, total):
        percent = (current / total) * 100
        print(f"\rProgress: {current}/{total} ({percent:.1f}%)", end='', flush=True)
    
    # 4. Batch DOI imputation
    print("\nStarting DOI imputation process...")
    enriched_records = doi_finder.impute_dois_batch(
        arxiv_entries=arxiv_entries,
        batch_size=20,
        progress_callback=print_progress
    )
    
    # 5. Export results
    print("\n\nExporting results...")
    
    # Export to BibTeX
    doi_finder.export_to_bibtex(enriched_records, "output/enriched_library.bib")
    
    # Export to DataFrame and save as CSV
    result_df = doi_finder.export_to_dataframe(enriched_records)
    result_df.to_csv("output/arxiv_with_dois.csv", index=False, encoding='utf-8')
    
    # Export statistics
    stats = {
        'total_papers': len(result_df),
        'papers_with_doi': result_df['has_doi'].sum(),
        'doi_recovery_rate': f"{result_df['has_doi'].mean()*100:.1f}%",
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("\n" + "="*50)
    print("DOI IMPUTATION SUMMARY")
    print("="*50)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Display first few records
    print("\nFirst 5 records:")
    print(result_df[['title', 'doi', 'has_doi']].head().to_string(index=False))