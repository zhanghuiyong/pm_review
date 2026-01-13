# Literature Research Toolkit

A comprehensive Python toolkit for systematic literature research, featuring multi-database integration, cross-platform DOI imputation, and batch search capabilities.

## Overview

This toolkit provides three core modules designed to streamline academic literature research:

- **ArxivBatchSearch**: Batch search arXiv papers using combined keywords and download metadata
- **ArxivCrossrefImput**: Query and impute DOI information for arXiv papers using Crossref
- **UnifiedMultiDatabase**: Manage literature from multiple databases (PubMed, Web of Science, Embase, IEEE Xplore, arXiv, etc.) and unify them to BibTeX format

## Prerequisites

*   **System & Environment**: It is recommended to use **Miniconda** or **Anaconda** for environment management.
*   **Conda**: Ensure Conda is installed and available in your terminal. You can verify this by running:
    ```bash
    conda --version
    ```

## Installation

1.  Create and activate a new Conda environment (e.g., named `review_paper`):
    ```bash
    conda create -n review_paper python=3.10 -y
    conda activate review_paper
    ```
2.  Install the required Python packages from PyPI:
    ```bash
    pip install arxiv bibtexparser habanero rapidfuzz matplotlib seaborn biopython rispy numpy pandas
    ```
3.  *(Optional)* For development or contribution, clone the repository and install in editable mode:
    ```bash
    git clone https://github.com/zhanghuiyong/pm_review.git
    cd pm_review
    pip install -e .
    ```

## Quick Start / Usage Example

Here is a basic workflow demonstrating the toolkit's capabilities:

### 1. ArxivBatchSearch Example

Like PubMed, Web of Science (WoS), and IEEE Xplore, this module accepts advanced Boolean query expressions for batch searching. The following example demonstrates batch querying and result export in BibTeX format.

**Complete Workflow Example:**

```python
from arxiv_batch_search import ArxivBatchSearch
import pandas as pd

# Initialize the searcher (limits align with arXiv API's practical constraints)
searcher = ArxivBatchSearch(max_results_per_query=500) # Note: arXiv API may cap single queries

# Define keyword combinations for a systematic review
keyword_combinations = '("explainable AI" OR "XAI") AND "Machine Learning"'

out_iterm = "output/arxiv_data.bib"  
searcher.fetch_from_arxiv(keyword_combinations, out_iterm)
```

This module provides the foundational batch retrieval layer. The subsequent `ArxivCrossrefImput` module can then enrich these records with missing DOI metadata, and the `UnifiedMultiDatabase` can merge them with results from other scholarly databases.

---

### 2. ArxivCrossrefImput Example

Query arXiv papers and attempt to find DOI information using Crossref:

```python
    # Example: Load arXiv search results from CSV file and impute DOIs
    import ArxivCrossrefImput
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
    excel_filename="output/consolidated_literature.xlsx",
    json_filename="output/consolidated_literature.json"
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