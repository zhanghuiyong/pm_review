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

# ========== 工具函数：布尔解析 ==========
def parse_boolean_query(query):
    """
    解析布尔查询，返回所有子查询组合
    输入示例：
      ("precision medicine" OR "digital health") AND ("interpretable machine learning" OR "explainable artificial intelligence")
    输出：
      ['"precision medicine" "interpretable machine learning"',
       '"precision medicine" "explainable artificial intelligence"',
       '"digital health" "interpretable machine learning"',
       '"digital health" "explainable artificial intelligence"']
    """
    # 按 AND 拆分块
    blocks = [b.strip(" ()") for b in query.split("AND")]
    option_lists = []
    for b in blocks:
        parts = [p.strip(" ()\"") for p in b.split("OR")]
        option_lists.append(parts)

    # 笛卡尔积组合 + 保留双引号
    combos = list(itertools.product(*option_lists))
    return [" ".join([f"\"{term}\"" for term in c]) for c in combos]


# ========== 数据源函数 ==========
def fetch_arxiv(keyword, max_results=20):
    """
    从 arXiv 搜索文章并将结果保存为 BibTeX 文件。
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
        print(f"关键词【{query}】结果不足，已跳过空页。")
    print(f"关键词【{query}】：")
    print(f"共 {len(bib_entries)} 条记录。")
    return bib_entries

def fetch_crossref(keyword, max_results=20):
    """从 Crossref 抓取文献"""
    cr = Crossref(timeout=60)  # 在初始化时设置超时时间

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
        print("请求超时，请稍后重试或检查网络。")
        papers = []
    except Exception as e:
        print(f"发生错误：{e}")
        papers = []
    return papers

def fetch_semantic(doi=None, title=None):
    """从 Semantic Scholar 获取引用数"""
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

# ========== 去重与补全 ==========
def deduplicate_and_enrich(papers):
    """去重：优先保留 Crossref/DOI"""
    seen = {}
    for p in papers:
        key = p["doi"].lower() if p["doi"] else p["title"].lower()
        if key in seen:
            if p["source"] != "arXiv":  # 保留正式出版
                seen[key] = p
        else:
            seen[key] = p

    # 补全引用数
    for k, p in seen.items():
        if p["citations"] is None:
            p["citations"] = fetch_semantic(p["doi"], p["title"])
            time.sleep(0.2)
    return list(seen.values())

# ========== BibTeX & 分析 ==========
def to_bibtex(papers, filename="output.bib"):
    """保存为 BibTeX"""
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
        
    print(f"✅ 已生成 {filename}，共 {len(db)} 条文献")

def keyword_analysis(papers):
    text = " ".join(p["title"] for p in papers)
    words = re.findall(r"\w+", text.lower())
    counter = Counter(words)
    print("\n📊 Top 15 高频词：")
    for word, freq in counter.most_common(15):
        print(f"{word}: {freq}")

def author_analysis(papers):
    authors = []
    for p in papers:
        authors.extend(p["author"])
    counter = Counter(authors)
    print("\n👥 Top 10 高频作者：")
    for author, freq in counter.most_common(10):
        print(f"{author}: {freq}")

# ========== 新增函数 ==========
def find_doi_by_title(title, authors=None, topn=5):
    """
    用 Crossref 查找某个标题的 DOI，按标题相似度 + 作者验证
    """
    cr = Crossref(timeout=60)
    try:
        results = cr.works(query=title, limit=topn)
        time.sleep(1)
    except Exception as e:
        print(f"Crossref 查询失败：{e}")
        return None

    best_match = None
    best_score = 0
    for item in results['message']['items']:
        candidate_title = item.get("title", [""])[0]
        candidate_authors = [a.get("family", "").lower() for a in item.get("author", []) if "family" in a]
        candidate_doi = item.get("DOI", None)

        # 标题相似度
        score = SequenceMatcher(None, title.lower(), candidate_title.lower()).ratio()

        # 作者验证（有交集加分）
        if authors:
            overlap = len(set(a.lower() for a in authors) & set(candidate_authors))
            if overlap > 0:
                score += 0.1

        if score > best_score:
            best_score = score
            best_match = candidate_doi

    return best_match if best_score > 0.75 else None


# ========== 主程序修改 ==========
if __name__ == "__main__":
    query = '("Artificial Intelligence" OR "Machine Learning" OR "Deep Learning") AND ("Explainable AI" OR "XAI" OR "Interpretability" OR "Transparency") AND ("Precision Medicine" OR "Personalized Medicine" OR "Healthcare" OR "Medical Diagnosis")'
    sub_queries = parse_boolean_query(query)
    print(f"🔎 已拆解为 {len(sub_queries)} 个子查询：")
    for sq in sub_queries:
        print("   ", sq)

    papers = []
    for sq in sub_queries:  # 仅测试第一个子查询
        # 先抓取 arXiv
        print(f"=== 处理子查询：{sq} ===")
        arxiv_entries = fetch_arxiv(sq, 1000)

        # 对每一条 arXiv 结果，尝试查找 DOI
        entrys = []
        for entry in arxiv_entries:
            print(f"🔍 查找 DOI: {entry['title'][:60]}...")
            authors = entry["author"].split(" and ")
            doi_guess = find_doi_by_title(entry["title"], authors)
            if doi_guess:
                entry["doi"] = doi_guess
                print(f"✅ 匹配到 DOI: {doi_guess} | {entry['title'][:60]}...")

            entrys.append(entry)
        papers += entrys
    # 持久化 papers 为 pickle 格式
    with open("arxiv_raw.pkl", "wb") as f:
        pickle.dump(papers, f)
    print(f"\n📥 共抓取 {len(papers)} 条 arXiv，并已保存为 arxiv_raw.pkl")

    # 读取 pickle 文件示例
    with open("arxiv_raw.pkl", "rb") as f:
        papers = pickle.load(f)
    
    enriched = deduplicate_and_enrich(papers)

    to_bibtex(enriched, "arxiv1.bib")
    # keyword_analysis(enriched)
    # author_analysis(enriched)

