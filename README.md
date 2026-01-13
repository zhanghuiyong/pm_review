

# Literature Research Toolkit

A comprehensive Python toolkit for systematic literature research, featuring multi-database integration, cross-platform DOI imputation, and batch search capabilities.

**paper**:Paradigm Evolution of Explainable Artificial Intelligence in Precision Medicine: A Systematic Review Mapping the Path from Feature Attribution to Causal Intervention

## Overview

This toolkit provides three core modules designed to streamline academic literature research:

- **ArxivBatchSearch**: Batch search arXiv papers using combined keywords and download metadata
- **ArxivCrossrefImput**: Query and impute DOI information for arXiv papers using Crossref
- **UnifiedMultiDatabase**: Manage literature from multiple databases (PubMed, Web of Science, Embase, IEEE Xplore, arXiv, etc.) and unify them to BibTeX format

## Installation

```bash
pip install arxiv bibtexparser habanero rapidfuzz matplotlib seaborn biopython rispy numpy pandas
```

## Usage Examples

### 1. ArxivBatchSearch Example

Batch search arXiv papers using combined keywords and download metadata in BibTeX format:

```python
from arxiv_batch_search import ArxivBatchSearch

# Initialize the searcher with 1000 results per query
searcher = ArxivBatchSearch(max_results_per_query=1000)

# Example 1: Using predefined list of keyword combinations
keywords = [
    '"precision medicine" AND "interpretable machine learning"', 
    '"causal inference" AND "machine learning"', 
    '"reinforcement learning" AND "healthcare"'
]

results1 = searcher.search_combined_keywords(keywords)
searcher.save_results(results1, "arxiv_predefined_results.bib")

# Example 2: Using boolean query parsing
boolean_query = '("precision medicine" OR "digital health") AND ("interpretable machine learning" OR "explainable artificial intelligence")'
results2 = searcher.search_boolean_query(boolean_query)
searcher.save_results(results2, "arxiv_boolean_results.bib")
```

### 2. ArxivCrossrefImput Example

Query arXiv papers and attempt to find DOI information using Crossref:

```python
from arxiv_crossref_imput import ArxivCrossrefImput

# Initialize the imputer
imputer = ArxivCrossrefImput(timeout=60)

# Query arXiv and attempt to find DOI for each result using Crossref
query = '("Artificial Intelligence" OR "Machine Learning") AND ("Healthcare" OR "Medical Diagnosis")'
papers = imputer.impute_dois_for_arxiv(query, max_results=500)

# Save raw papers to pickle
imputer.save_papers_pickle(papers, "arxiv_raw.pkl")

# Load and process papers
loaded_papers = imputer.load_papers_pickle("arxiv_raw.pkl")
enriched_papers = imputer.deduplicate_and_enrich(loaded_papers)

# Save to BibTeX
imputer.to_bibtex(enriched_papers, "arxiv_with_doi_imputed.bib")
```

### 3. UnifiedMultiDatabase Example

Manage literature from different platforms and unify them to BibTeX format:

```python
from unified_multi_database import UnifiedMultiDatabase

# Initialize the manager
manager = UnifiedMultiDatabase()

# Load data from different sources
df_pubmed = manager.load_pubmed_data("data/pubmed_Artificial.nbib")
df_wos = manager.load_wos_data(["data/wos_savedrecs.ris"])
df_embase = manager.load_embase_data("data/embase_records.ris")
df_ieee = manager.load_ieee_data("data/ieee/")
df_arxiv = manager.load_arxiv_data("data/arxiv1.bib")
df_myref = manager.load_custom_bibtex("data/Explainability.bib")

# Merge all sources with deduplication
merged_df = manager.merge_all_sources([
    df_pubmed, df_wos, df_embase, df_ieee, df_arxiv, df_myref
])

# Save results
manager.save_results(
    merged_df,
    excel_filename="consolidated_literature.xlsx",
    json_filename="consolidated_literature.json"
)
```

## Features

- **Batch Search**: Search arXiv using complex boolean queries and keyword combinations
- **DOI Imputation**: Automatically find DOI information for arXiv papers through Crossref
- **Multi-Database Integration**: Support for PubMed, Web of Science, Embase, IEEE Xplore, and arXiv
- **Deduplication**: Smart deduplication based on title similarity and DOI
- **Standardized Format**: Unified field mapping to standard BibTeX format
- **Keyword Filtering**: Automatic exclusion of review articles and surveys based on keywords
- **Export Options**: Export to both Excel and JSON formats for further analysis
- **Year Filtering**: Built-in support for temporal filtering (2016-2025)

## File Structure

```
├── arxiv_batch_search.py       # Batch search arXiv papers
├── arxiv_crossref_imput.py     # DOI imputation for arXiv papers
├── unified_multi_database.py   # Multi-database literature management
├── data/                       # Sample data files
└── LICENSE                     # MIT License
```

## Requirements

- Python 3.7+
- arxiv
- bibtexparser
- habanero
- rapidfuzz
- matplotlib
- seaborn
- biopython
- rispy
- numpy
- pandas

## License

Apache License Version 2.0, License - see LICENSE file for details.