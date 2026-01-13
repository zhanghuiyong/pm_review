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





import importlib.metadata

packages = [
    "arxiv",
    "bibtexparser",
    "habanero",
    "rapidfuzz",
    "matplotlib",
    "seaborn",
    "biopython",
    "rispy",
    "numpy",
    "pandas"
]

for pkg in packages:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg}=={version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} is not installed")
        