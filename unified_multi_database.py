"""
Unified Multi-Database Literature Management System
Author: Huiyong Zhang
Description:
    A class for managing literature from multiple databases (PubMed, Web of Science, 
    Embase, IEEE Xplore, arXiv, etc.) and unifying them to BibTeX format.
"""

import datetime
import json
from pathlib import Path
from time import sleep
import time
import bibtexparser
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Bio import Medline
from datetime import datetime
import rispy


class UnifiedMultiDatabase:
    """
    A class for managing literature from different platforms and unifying them to BibTeX format.
    """

    def __init__(self):
        """
        Initialize the UnifiedMultiDatabase instance.
        """
        # Define unified field mappings
        self.unified_columns = {
            # PubMed/nbib
            "TI": "title",
            "AB": "abstract",
            "AU": "authors",
            "DP": "year",
            "JT": "journal",
            "VI": "volume",
            "IP": "issue",
            "PG": "pages",
            "LID": "doi",
            # RIS/WOS/IEEE/Embase
            "type": "type",
            "title": "title",
            "abstract": "abstract",
            "author": "authors",
            "authors": "authors",
            "year": "year",
            "journal": "journal",
            "volume": "volume",
            "issue": "issue",
            "pages": "pages",
            "doi": "doi",
            "source": "source",
            "citations": "citations",
            "ENTRYTYPE": "entry_type",
            "ID": "ID",
            "eprint": "eprint"
        }

        self.unify_list = list(set(self.unified_columns.values()))

    def extract_year(self, date_str):
        """
        Extract year from date string.

        Args:
            date_str: Date string

        Returns:
            Year as integer or None
        """
        try:
            # Try direct conversion
            return int(date_str)
        except (ValueError, TypeError):
            # If failed, try extracting year from date string
            if isinstance(date_str, str):
                import re
                year_match = re.search(r'(\d{4})', date_str)
                if year_match:
                    return int(year_match.group(1))
        return None

    def load_ris(self, file_path, mapping=None):
        """
        Load RIS format data and return pandas DataFrame.

        Args:
            file_path: Path to RIS file
            mapping: Field mapping dictionary

        Returns:
            DataFrame containing RIS data
        """
        with open(file_path, encoding="utf-8") as f:
            entries = rispy.load(f, mapping=mapping)
        return pd.DataFrame(entries)

    def nbib_to_dataframe(self, nbib_file_path):
        """
        Convert PubMed exported NBIB file to pandas DataFrame.

        Args:
            nbib_file_path: Path to NBIB file

        Returns:
            DataFrame containing literature information
        """
        # Read NBIB file
        with open(nbib_file_path, 'r', encoding='utf-8') as f:
            # Use Medline parser to read records
            records = Medline.parse(f)
            
            # Convert records to list
            records_list = list(records)
        
        # Convert to DataFrame
        df = pd.DataFrame(records_list)
        
        return df

    def load_bibtex(self, file_path):
        """
        Load BibTeX file and return DataFrame.

        Args:
            file_path: Path to BibTeX file

        Returns:
            DataFrame containing BibTeX entries
        """
        with open(file_path, encoding="utf-8") as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)
        return pd.DataFrame(bib_database.entries)

    def deduplicate(self, df):
        """
        Deduplicate DataFrame based on title similarity and DOI.

        Args:
            df: Input DataFrame

        Returns:
            Deduplicated DataFrame
        """
        # Deduplicate based on title fuzzy matching
        if "title" in df.columns:
            titles = df["title"].fillna("").tolist()
            seen, keep_idx = set(), []
            for i, title in enumerate(titles):
                matched = False
                for s in seen:
                    if fuzz.ratio(title.lower(), s.lower()) > 90:  # Similarity threshold adjustable
                        matched = True
                        break
                if not matched:
                    keep_idx.append(i)
                    seen.add(title)
            df = df.iloc[keep_idx]
        
        # Remove duplicate DOI
        if "doi" in df.columns:
            df = df.drop_duplicates(subset="doi")

        return df.reset_index(drop=True)

    def keyword_filter(self, df, keywords):
        """
        Filter DataFrame by keywords in title and abstract.

        Args:
            df: Input DataFrame
            keywords: List of keywords to filter

        Returns:
            Filtered DataFrame
        """
        # Return directly if title and abstract do not exist
        if "title" not in df.columns and "abstract" not in df.columns:
            return df

        mask = pd.Series([False] * len(df))
        if "title" in df.columns:
            mask = mask | df["title"].str.contains("|".join(keywords), case=False, na=False)
        if "abstract" in df.columns:
            mask = mask | df["abstract"].str.contains("|".join(keywords), case=False, na=False)
        
        # Exclude those containing keywords, keep those without
        return df[~mask].reset_index(drop=True)

    def export_for_screening(self, df, output_excel="screening_list.xlsx"):
        """
        Export DataFrame to Excel for manual screening.

        Args:
            df: Input DataFrame
            output_excel: Output Excel filename
        """
        df.to_excel(output_excel, index=False)
        print(f"[INFO] Exported to {output_excel} for manual screening.")

    def import_screening_results(self, screened_excel):
        """
        Import manual screening results from Excel.

        Args:
            screened_excel: Path to screened Excel file

        Returns:
            Tuple of included and excluded DataFrames
        """
        df = pd.read_excel(screened_excel)
        included = df[df["include"] == 1]
        excluded = df[df["include"] == 0]
        print(f"[INFO] Included: {len(included)}, Excluded: {len(excluded)}")
        return included, excluded

    def analyze_and_plot(self, df):
        """
        Analyze and plot statistics of DataFrame.

        Args:
            df: Input DataFrame
        """
        if "year" in df.columns:
            # Year distribution
            plt.figure(figsize=(8,4))
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            sns.countplot(x="year", data=df, order=sorted(df["year"].dropna().unique()))
            plt.xticks(rotation=45)
            plt.title("Publications by Year")
            plt.tight_layout()
            plt.savefig("year_distribution.png")
            plt.show()

        if "journal" in df.columns:
            # Top 10 journals distribution
            top_journals = df["journal"].value_counts().head(10)
            top_journals.plot(kind="barh", figsize=(6,4))
            plt.title("Top Journals")
            plt.tight_layout()
            plt.savefig("top_journals.png")
            plt.show()

    def unify_columns(self, df):
        """
        Unify column names according to standardized mapping.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with unified column names
        """
        # Only keep fields that appear in mapping and rename them
        cols = {col: self.unified_columns[col] for col in df.columns if col in self.unified_columns}
        df = df.rename(columns=cols)
        df = df[list(set(self.unified_columns.values()) & set(df.columns))]
        return df

    def load_pubmed_data(self, file_path):
        """
        Load and process PubMed NBIB data.

        Args:
            file_path: Path to NBIB file

        Returns:
            Processed DataFrame
        """
        df = self.nbib_to_dataframe(file_path)
        print(f"[INFO] Loaded {len(df)} entries from {file_path}")
        
        # If 'Journal Article' is in df["PT"], generate df['type'] = 'Journal'
        if "PT" in df.columns:
            df.loc[df["PT"].astype(str).str.contains("Journal Article", case=False, na=False), "type"] = "Journal"
        
        # Keep only journal or conference papers, remove reviews or editorials
        if "PT" in df.columns:
            df = df[
                df["PT"].astype(str).str.contains("Journal Article|Conference Paper", case=False, na=False)
                & ~df["PT"].astype(str).str.contains("review|editorial", case=False, na=False)
            ]

        df["ID"] = df["PMID"]
        
        df = self.unify_columns(df)
        df['year'] = df['year'].str.extract(r'(\d{4})').astype(int)
        
        # Deduplicate
        df = self.deduplicate(df)
        print(f"[INFO] After deduplication: {len(df)} entries")

        # Keyword filter
        keywords = ["review", "PRIMSA", "survey", "overview"]
        df = self.keyword_filter(df, keywords)
        print(f"[INFO] After keyword filtering: {len(df)} entries")
        
        df.loc[:, "source"] = "pubmed"
        return df

    def load_wos_data(self, file_paths, mapper=None):
        """
        Load and process Web of Science RIS data.

        Args:
            file_paths: List of paths to RIS files
            mapper: Field mapping dictionary

        Returns:
            Processed DataFrame
        """
        if mapper is None:
            mapper = {
                'TY': 'type',
                'T2': 'journal',
                'JT': 'journal',
                'SP': 'start_page',
                'EP': 'end_page',
                'SP': 'pages',
                'PY': 'year',
                'TI': 'title',
                'AB': 'abstract',
                'AU': 'authors',
                'VL': 'volume',
                'IS': 'issue',
                'IP': 'issue',
                'PG': 'pages',
                'LID': 'doi',
                'DO': 'doi',
                'N1': 'citations',
                'UK': 'unknown',   # Fallback
            }

        dfs = []
        for path in file_paths:
            df = self.load_ris(path, mapping=mapper)
            dfs.append(df)
        
        df = pd.concat(dfs, ignore_index=True)
        print(f"[INFO] Loaded {len(df)} entries from WOS files")

        df = self.unify_columns(df)
        
        # Extract numbers after "Total Times Cited:", store in citations field (overwrite original field)
        if "citations" in df.columns:
            df["citations"] = df["citations"].astype(str).str.extract(r"Total Times Cited:\s*(\d+)", expand=False)
        
        df['year'] = df['year'].str.extract(r'(\d{4})').astype(int)
        
        # Change 'JOUR' in type field to 'Journal'
        df["type"] = df["type"].replace({"JOUR": "Journal"})
        
        # Deduplicate
        df = self.deduplicate(df)
        print(f"[INFO] After deduplication: {len(df)} entries")
        
        df.loc[:, "source"] = "wos"
        return df

    def load_embase_data(self, file_path, mapper=None):
        """
        Load and process Embase RIS data.

        Args:
            file_path: Path to RIS file
            mapper: Field mapping dictionary

        Returns:
            Processed DataFrame
        """
        if mapper is None:
            mapper = {
                'TY': 'type',
                'M3': 'document_type',
                'Y1': 'year',
                'T2': 'journal',
                'JT': 'journal',
                'JF': 'journal',
                'JO': 'journal_abbrev',
                'SP': 'start_page',
                'EP': 'end_page',
                'SP': 'pages',
                'PY': 'year',
                'TI': 'title',
                'T1': 'title',
                'AB': 'abstract',
                'N2': 'abstract',
                'AU': 'authors',
                'A1': 'authors',
                'VL': 'volume',
                'IS': 'issue',
                'IP': 'issue',
                'PG': 'pages',
                'LID': 'doi',
                'DO': 'doi',
                'LA': 'language',
                'KW': 'keywords',
                'UK': 'unknown',   # Fallback
            }

        df = self.load_ris(file_path, mapping=mapper)
        print(f"[INFO] Loaded {len(df)} entries from {file_path}")
        
        # Keep only records with document_type as Article or Preprint
        if "document_type" in df.columns:
            df = df[df["document_type"].astype(str).str.lower().isin(["article", "preprint"])]
        
        # Keep only records with type field as 'JOUR', change to 'Journal'
        if "type" in df.columns:
            df = df[df["type"].astype(str).str.upper() == "JOUR"]
            df.loc[:, "type"] = "Journal"
        
        df = self.unify_columns(df)
        df['year'] = df['year'].str.extract(r'(\d{4})').astype(int)
        
        # Deduplicate
        df = self.deduplicate(df)
        print(f"[INFO] After deduplication: {len(df)} entries")
        
        # Keyword filter
        keywords = ["review", "PRIMSA", "survey", "overview"]
        df = self.keyword_filter(df, keywords)
        print(f"[INFO] After keyword filtering: {len(df)} entries")
        
        df.loc[:, "source"] = "embase"
        return df

    def load_ieee_data(self, directory_path, mapper=None):
        """
        Load and process IEEE Xplore RIS data.

        Args:
            directory_path: Path to directory containing RIS files
            mapper: Field mapping dictionary

        Returns:
            Processed DataFrame
        """
        if mapper is None:
            mapper = {
                'type_of_reference': 'type',
                'TI': 'title',
                'T2': 'journal',  # For conferences, this maps to conference name; fallback for journals too
                'JO': 'journal',  # Abbreviated form
                'JA': 'journal_abbrev',
                'SP': 'start_page',
                'SP': 'pages',
                'EP': 'end_page',
                'AU': 'authors',
                'PY': 'year',
                'Y1': 'date',
                'KW': 'keywords',
                'DO': 'doi',
                'AB': 'abstract',
                'VL': 'volume',
                'VO': 'volume',
                'IS': 'issue',
                'IP': 'issue',  # Fallback for issue
                'SN': 'issn',
                'UK': 'unknown',   # Fallback
            }

        ris_files = [f for f in os.listdir(directory_path) if f.endswith(".ris")]
        print("Found RIS files:", ris_files)
        
        # Load
        df_list = []
        for ris_file in ris_files:
            file_path = os.path.join(directory_path, ris_file)
            print(f"Loading file: {file_path}")
            df_list.append(self.load_ris(file_path))

        # Combine into one DataFrame
        df = pd.concat(df_list, ignore_index=True)
        print("Combined DataFrame shape df_ieee:", df.shape)

        df = df.rename(columns={'type_of_reference': 'type'})
        type_mapping = {
            'CONF': 'Conference',
            'JOUR': 'Journal',
            'CHAP': 'Book',
            'BOOK': 'Book'
        }
        keys_to_keep = list(type_mapping.keys())
        df = df[df['type'].isin(keys_to_keep)].copy()
        df['type'] = df['type'].replace(type_mapping)
        
        df.loc[:, 'journal'] = df['journal_name']
        df.loc[:, 'issue'] = df['issn']

        df = self.unify_columns(df)
        df['year'] = df['year'].str.extract(r'(\d{4})').astype(int)

        # Deduplicate
        df = self.deduplicate(df)
        print(f"[INFO] After deduplication: {len(df)} entries")
        
        # Keyword filter
        keywords = ["review", "PRIMSA", "survey", "overview"]
        df = self.keyword_filter(df, keywords)
        print(f"[INFO] After keyword filtering: {len(df)} entries")
        
        df.loc[:, "source"] = "ieee"
        return df

    def load_arxiv_data(self, file_path):
        """
        Load and process arXiv BibTeX data.

        Args:
            file_path: Path to BibTeX file

        Returns:
            Processed DataFrame
        """
        df = self.load_bibtex(file_path)
        df["type"] = df["ENTRYTYPE"]
        df["type"] = df["type"].replace({"article": "Journal"})
        
        df = self.unify_columns(df)
        
        # Deduplicate
        df = self.deduplicate(df)
        print(f"[INFO] After deduplication: {len(df)} entries")
        
        # Keyword filter
        keywords = ["review", "PRIMSA", "survey", "overview"]
        df = self.keyword_filter(df, keywords)
        print(f"[INFO] After keyword filtering: {len(df)} entries")
        
        # Rename column to match other data sources
        df["source"] = "arXiv"
        return df

    def load_custom_bibtex(self, file_path):
        """
        Load and process custom BibTeX data.

        Args:
            file_path: Path to BibTeX file

        Returns:
            Processed DataFrame
        """
        df = self.load_bibtex(file_path)

        type_mapping = {
            'article': 'Journal',          # Journal article
            'inproceedings': 'Conference', # Conference paper
            'book': 'Book',                # Book
            'inbook': 'Book',              # Chapter in book
            'incollection': 'Book'
        }
        
        keys_to_keep = list(type_mapping.keys())
        df = df[df['ENTRYTYPE'].isin(keys_to_keep)].copy()
        df['type'] = df['ENTRYTYPE'].replace(type_mapping)
        
        df = self.unify_columns(df)
        df = df[df['abstract'].notna()]
        df['doi'] = df['doi'].fillna(df['ID'])
        
        # Deduplicate
        df = self.deduplicate(df)
        print(f"[INFO] After deduplication: {len(df)} entries")
        
        # Keyword filter
        keywords = ["review", "PRIMSA", "survey", "overview"]
        df = self.keyword_filter(df, keywords)
        print(f"[INFO] After keyword filtering: {len(df)} entries")
        
        # Rename column to match other data sources
        df["source"] = "myref"
        return df

    def merge_all_sources(self, dataframes):
        """
        Merge all data sources into a single DataFrame.

        Args:
            dataframes: List of DataFrames from different sources

        Returns:
            Merged DataFrame
        """
        df = pd.concat(dataframes, ignore_index=True)
        print(f"[INFO] Total records after merging: {len(df)}")
        
        # Apply year extraction function
        df['year'] = df['year'].apply(self.extract_year)

        # Filter data between 2016-2025
        df = df[(df['year'] >= 2016) & (df['year'] <= 2025)]
        print(f"[INFO] Total records after year filtering (2016-2025): {len(df)}")
        
        # Define data source priority, lower value means higher priority
        source_priority = {
            "pubmed": 1,
            "wos": 2,
            "embase": 3,
            "ieee": 4,
            "arXiv": 5,
            "myref": 6
        }
        df["source_priority"] = df["source"].map(source_priority).fillna(99)

        # Sort by source priority then deduplicate by title (keep highest priority source)
        if "title" in df.columns:
            df = df.sort_values("source_priority")
            df['title'] = df['title'].str.strip()
            df = df.drop_duplicates(subset="title", keep="first")

        # First deduplicate by DOI (keep highest priority source)
        if "doi" in df.columns:
            df = df.sort_values("source_priority")
            df['doi'] = df['doi'].str.strip()
            df = df.drop_duplicates(subset="doi", keep="first")

        df = df.reset_index(drop=True)
        df.drop(columns=["source_priority"], inplace=True, errors='ignore')
        
        # Fill missing ID field
        mask = df['ID'].isna()
        df.loc[mask, 'ID'] = df.index[mask]
        
        return df

    def save_results(self, df, excel_filename="screening_list.xlsx", json_filename="screening_data.json"):
        """
        Save results to Excel and JSON files.

        Args:
            df: DataFrame to save
            excel_filename: Output Excel filename
            json_filename: Output JSON filename
        """
        result_dir = Path("result")
        result_dir.mkdir(exist_ok=True)

        self.export_for_screening(df, output_excel=f"result/{excel_filename}")
        df.to_json(f"result/{json_filename}", orient="records", lines=True)
        print(f"[INFO] Results saved to result/{excel_filename} and result/{json_filename}")


def main():
    """
    Example usage of the UnifiedMultiDatabase class.
    """
    # Initialize the manager
    manager = UnifiedMultiDatabase()

    # Load data from different sources
    print("Loading PubMed data...")
    df_pubmed = manager.load_pubmed_data("data/pubmed_Artificial.nbib")

    print("Loading Web of Science data...")
    df_wos = manager.load_wos_data([
        "data/wof_savedrecs.ris",
        "data/wof_savedrecs1.ris"
    ])

    print("Loading Embase data...")
    df_embase = manager.load_embase_data("data/embase_records.ris")

    print("Loading IEEE data...")
    df_ieee = manager.load_ieee_data("data/ieee")

    print("Loading arXiv data...")
    df_arxiv = manager.load_arxiv_data("data/arxiv1.bib")

    print("Loading custom BibTeX data...")
    df_myref = manager.load_custom_bibtex("data/Explainability.bib")

    # Merge all sources
    print("Merging all data sources...")
    merged_df = manager.merge_all_sources([
        df_pubmed, df_wos, df_embase, df_ieee, df_arxiv, df_myref
    ])

    # Save results
    manager.save_results(
        merged_df,
        excel_filename="screening_citations_20251113.xlsx",
        json_filename="screening_citations_20251113.json"
    )

    print(f"Final dataset contains {len(merged_df)} unique records from all sources")
    print(f"Data sources distribution:\n{merged_df['source'].value_counts()}")


if __name__ == "__main__":
    main()



