
##################################100%####################################################################
#!/usr/bin/env python3
"""
AI Firm Website Finder – v5.3 (Debug Ready)
-------------------------------------------
✅ Async + multi-threaded
✅ Dict-safe DDG search results
✅ Domain + location filters
✅ AI reasoning + justification
✅ Optional debug mode to save full logs
"""

import os
import re
import json
import asyncio
import hashlib
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from dotenv import load_dotenv

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

# --------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = "search_cache_v5"
DEBUG_DIR = "debug_logs"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

ddgs = DDGS()
API_KEY = os.getenv("GROQ_API_KEY")
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AI-FirmFinder/5.3)"}

# --------------------------------------------------------------------
def fetch_html(url: str, limit: int = 2000) -> str:
    if not isinstance(url, str) or not url.startswith("http"):
        return ""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code < 400:
            soup = BeautifulSoup(r.text, "html.parser")
            text = soup.get_text(" ", strip=True)
            return text[:limit]
    except Exception:
        pass
    return ""

def cached_search(query: str, max_results: int = 15) -> List[str]:
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    path = os.path.join(CACHE_DIR, f"{key}.json")
    if os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(data, list):
                return [d["href"] if isinstance(d, dict) and "href" in d else d for d in data]
        except Exception:
            pass
    results = list(ddgs.text(query, max_results=max_results))
    urls = []
    for r in results:
        if isinstance(r, dict) and r.get("href"):
            urls.append(r["href"])
        elif isinstance(r, str):
            urls.append(r)
    json.dump(urls, open(path, "w", encoding="utf-8"), indent=2)
    return urls

def extract_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(
            m.group(0).replace("True", "true").replace("False", "false")
        )
    except Exception:
        return {}

def normalize_urls(urls: List[Any]) -> List[str]:
    clean = []
    for u in urls:
        if isinstance(u, dict) and "href" in u:
            clean.append(u["href"])
        elif isinstance(u, str):
            clean.append(u)
    return list(dict.fromkeys(clean))

def filter_by_domain_and_location(firm: str, address: str, urls: List[str]) -> List[str]:
    urls = normalize_urls(urls)
    tokens = [t for t in re.split(r"[^a-z]", firm.lower()) if len(t) > 2]
    city_tokens = [t for t in re.split(r"[^a-z]", address.lower()) if len(t) > 2]

    filtered = []
    for u in urls:
        if not isinstance(u, str):
            continue
        d = re.sub(r"https?://(www\.)?", "", u.lower()).split("/")[0]
        if any(t in d for t in tokens) or any(t in u.lower() for t in city_tokens):
            filtered.append(u)

    if len(filtered) < 10:
        filtered = urls[:15]
    return filtered[:15]

# --------------------------------------------------------------------
def ai_chat(prompt: str) -> str:
    if not GROQ_AVAILABLE or not API_KEY:
        return ""
    try:
        client = Groq(api_key=API_KEY)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"⚠️ AI error: {e}")
        return ""

# --------------------------------------------------------------------
def ai_select_best_site(firm: str, address: str, candidates: List[str], debug: bool=False) -> Dict[str, Any]:
    candidates = normalize_urls(candidates)
    snippets = {}
    for u in candidates:
        snippets[u] = fetch_html(u)
    candidates = [u for u in candidates if snippets.get(u)]

    if not candidates:
        return {"best_url": None, "confidence": 0, "reason": "no_content"}

    joined = "\n\n".join(
        f"URL: {u}\nTEXT:\n{snippets[u][:1000]}"
        for u in candidates
    )

    prompt = f"""
You are a digital investigator identifying the *official website* of a firm.

Firm: {firm}
Address: {address}

Below are 15 candidate URLs with their page text:
{joined}

Your task:
- Identify which URL is the firm's official website (main corporate domain or homepage).
- Consider firm name, domain match, and address/location mentions.
- Ignore LinkedIn, Wikipedia, directories, or unrelated sites.

Return STRICT JSON:
{{
  "best_url": "https://...",
  "confidence": 0–1,
  "reason": "Why this is official (short).",
  "summary": "Brief explanation comparing chosen URL with others (why others are less likely)."
}}
"""

    text = ai_chat(prompt)
    data = extract_json(text)
    if not data:
        for u in candidates:
            if re.search(rf"{re.escape(firm.split()[0].lower())}", u.lower()):
                data = {"best_url": u, "confidence": 0.6, "reason": "token fallback", "summary": ""}
                break
        else:
            data = {"best_url": candidates[0], "confidence": 0.5, "reason": "fallback", "summary": ""}

    # Save debug info if requested
    if debug:
        debug_log = {
            "timestamp": datetime.now().isoformat(),
            "firm": firm,
            "address": address,
            "candidates": candidates,
            "snippets": {u: snippets[u][:400] for u in candidates},
            "ai_output": data,
            "raw_ai_text": text,
        }
        with open(os.path.join(DEBUG_DIR, "debug_logs.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(debug_log, ensure_ascii=False) + "\n")

    return data

# --------------------------------------------------------------------
async def website_selector(firm: str, address: str, debug: bool=False) -> Dict[str, Any]:
    query = f"{firm} {address} official website"
    logger.info(f"\n🔍 Searching: {query}")
    urls = cached_search(query, 20)
    if not urls:
        return {"best_url": None, "reason": "no_candidates"}

    urls = filter_by_domain_and_location(firm, address, urls)
    logger.info(f"Found {len(urls)} filtered candidates.")

    result = ai_select_best_site(firm, address, urls, debug)
    logger.info(f"✅ AI Selected: {result.get('best_url')}")
    logger.info(f"Reason: {result.get('reason')}\n")
    return result

# --------------------------------------------------------------------
async def process_excel(path: str, concurrent_tasks: int = 5, debug: bool=False):
    df = pd.read_excel(path)
    out_rows = []

    async def handle_row(i, row):
        firm = str(row.get("Representative") or "").strip()
        address = str(row.get("Representative address") or "").strip()
        if not firm:
            return None
        res = await website_selector(firm, address, debug)
        return {
            "Firm": firm,
            "Address": address,
            "Official Website": res.get("best_url"),
            "Confidence": res.get("confidence", ""),
            "Reason": res.get("reason", ""),
            "AI Summary": res.get("summary", ""),
        }

    sem = asyncio.Semaphore(concurrent_tasks)
    async def bounded_task(i, row):
        async with sem:
            return await handle_row(i, row)

    tasks = [bounded_task(i, row) for i, row in df.iterrows()]
    results = await asyncio.gather(*tasks)
    out_rows = [r for r in results if r]

    out_df = pd.DataFrame(out_rows)
    out_path = "Website_Results_AI_v5.3.xlsx"
    out_df.to_excel(out_path, index=False)
    logger.info(f"\n✅ Saved to: {out_path}")
    if debug:
        logger.info(f"🐞 Debug logs written to: {DEBUG_DIR}/debug_logs.jsonl")



# --------------------------------------------------------------------
def find_official_website(firm: str, address: str, debug: bool = False) -> Dict[str, Any]:
    """
    Public API wrapper to allow main.py to call website_finder_ai as a module.
    Handles caching, filtering, and AI scoring synchronously.
    """
    try:
        urls = cached_search(f"{firm} {address} official website", 20)
        if not urls:
            return {"best_url": None, "reason": "no_candidates"}

        urls = filter_by_domain_and_location(firm, address, urls)
        result = ai_select_best_site(firm, address, urls, debug)
        return {
            "best_url": result.get("best_url"),
            "confidence": result.get("confidence", 0),
            "reason": result.get("reason", ""),
            "summary": result.get("summary", "")
        }
    except Exception as e:
        logger.error(f"Website finder failed for {firm}: {e}")
        return {"best_url": None, "reason": str(e), "confidence": 0}


# --------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (saves all AI reasoning)")
    args = parser.parse_args()

    fp = input("\n📁 Enter Excel file path: ").strip()
    if not os.path.exists(fp):
        print("❌ File not found.")
    else:
        asyncio.run(process_excel(fp, debug=args.debug))
