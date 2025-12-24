

#######################################################################almost
#!/usr/bin/env python3
"""
smart_email_finder_failover.py

Universal email finder with a robust LLM-powered failover layer.

Usage:
    python smart_email_finder_failover.py <homepage_url> "<person_name>"

Behavior:
 - Normal flow: detect directory, choose profile, extract email (as in v5).
 - FAILOVER: If profile not found or no emails extracted, run a site-wide search:
     * Render homepage and a short list of internal pages (links),
     * Extract text + run regex,
     * Ask the LLM (strict JSON output) to find explicit emails that are actually present and tied to the person.
 - LLM is instructed NOT to guess. If it cannot find explicit emails, it returns [].

Note: Run responsibly. Respect robots.txt & site rate limits when using at scale.
"""

import os, re, sys, json, time, requests, difflib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from groq import Groq

# -------------- config --------------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    print("Missing GROQ_API_KEY in .env")
    sys.exit(1)

MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_KEY)

UA = "SmartEmailFinder/1.0 (+https://github.com/)"
REQUEST_TIMEOUT = 15
MAX_SITE_PAGES = 25           # how many internal pages to fetch in failover
PLAYWRIGHT_SCROLL_TRIES = 12  # when scrolling dynamic directories
# ------------------------------------

# ---------- utility functions ----------
def render_page_html_and_links(url, headless=True, scroll=False, scroll_tries=5):
    """Render page with Playwright, return HTML and absolute links."""
    print(f"🌐 Render: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        page = browser.new_page(user_agent=UA)
        page.goto(url, wait_until="networkidle", timeout=45000)
        
        if scroll:
            last_h = 0
            for _ in range(scroll_tries):
                page.mouse.wheel(0, 3000)
                page.wait_for_timeout(800)
                new_h = page.evaluate("document.body.scrollHeight")
                if new_h == last_h:
                    break
                last_h = new_h
        html = page.content()
        # gather links
        els = page.query_selector_all("a[href]")
        links = []
        for a in els:
            try:
                href = a.get_attribute("href")
                if href:
                    links.append(urljoin(url, href.split("#")[0]))
            except Exception:
                continue
        browser.close()
    return html, list(dict.fromkeys(links))  # preserve unique order

def simple_fetch_html_links(url):
    """Fast HTTP GET + parse links (fallback)"""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return "", []
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.select("a[href]"):
            href = a.get("href")
            if href:
                links.append(urljoin(url, href.split("#")[0]))
        return r.text, list(dict.fromkeys(links))
    except Exception:
        return "", []

def extract_emails_from_text(text):
    """Return list of unique emails found via regex."""
    emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    return list(dict.fromkeys([e.lower() for e in emails]))

def domain_only(url):
    try:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}"
    except:
        return url

# ---------- AI helpers (strict JSON) ----------
def llm_sitewide_search(name, pages): 
    """
    pages: list of dict {url, text_snippet}
    Instruct LLM to search these snippets for explicit emails belonging to `name`.
    IMPORTANT: LLM MUST return strict JSON:
      {"results": [{"email": "...", "url": "...", "context": "...", "confidence": 0.0}, ...]}
    - It must NOT invent any email.
    - If nothing found, return {"results": []}
    """
    # Build compact context (truncate each snippet)
    sample_data = []
    for p in pages:
        snippet = p["text"][:2000].replace("\n", " ")
        sample_data.append({"url": p["url"], "text": snippet})
    prompt = f"""
You are a careful web-extraction assistant. You will be given a list of page url + text snippets.
Task: Find any e-mail addresses that are explicitly present in the provided text snippets that clearly belong to the person named: "{name}".

Rules (must follow):
1) Do NOT guess or invent. Only return emails that literally appear in the supplied text snippets.
2) Return output ONLY as JSON matching exactly this schema:
   {{
     "results": [
       {{
         "email": "<email address exactly as found>",
         "url": "<page url where it appears>",
         "context": "<short text (<=200 chars) around the email>",
         "confidence": <float 0.0-1.0 — 1.0 means exact match / email shown with name nearby>
       }},
       ...
     ]
   }}
3) If you find no explicit emails in the provided snippets, return: {"results": []}
4) When deciding confidence:
   - If the snippet contains the person's full name near the email, use confidence 0.95-1.0.
   - If the snippet contains only last name or initials near the email, use 0.6-0.85.
   - If email is present but no nearby name, use 0.3-0.6.
5) For context include up to ~200 characters surrounding the email.

Here are the pages (url + snippet). Keep JSON machine-parseable only.
Pages:
{json.dumps(sample_data, ensure_ascii=False)[:38000]}
"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
    )
    raw = resp.choices[0].message.content.strip()
    # extract JSON object from raw
    try:
        jtext = re.search(r"\{.*\}", raw, re.S).group(0)
        data = json.loads(jtext)
        # normalize emails lower-case
        for r in data.get("results", []):
            r["email"] = r["email"].strip().lower()
            r["context"] = r.get("context","")[:300]
            r["url"] = r.get("url","")
            r["confidence"] = float(r.get("confidence", 0.0))
        return data.get("results", [])
    except Exception as e:
        print("LLM failed to return strict JSON or parsing failed:", e)
        return []

# ---------- main failover routine ----------
def sitewide_failover_search(home_url, person_name, max_pages=MAX_SITE_PAGES):
    """
    1) Render homepage, gather internal links (breadth-first up to max_pages)
    2) Render each candidate page (Playwright) and collect text snippets
    3) Run regex to gather any emails across pages
    4) If regex finds candidate emails, pass snippets+emails to LLM for final decision
    """
    base = domain_only(home_url)
    print("🔎 Failover: sitewide search starting from homepage...")

    # 1) render home quickly, collect candidate links
    html, links = render_page_html_and_links(home_url, headless=True, scroll=False)
    internal_links = [l for l in links if urlparse(l).netloc == urlparse(home_url).netloc]
    # keep order, limit
    candidate_links = internal_links[: max_pages]
    # always include home_url first
    if home_url not in candidate_links:
        candidate_links.insert(0, home_url)

    pages = []
    seen = set()
    # 2) render each candidate and extract text
    for url in candidate_links:
        if url in seen:
            continue
        seen.add(url)
        try:
            # for directories/likely dynamic pages, scroll briefly
            scroll = any(k in url.lower() for k in ("/people", "/professionals", "/team", "/about"))
            html, _ = render_page_html_and_links(url, headless=True, scroll=scroll, scroll_tries=4)
            # extract visible text
            soup = BeautifulSoup(html, "html.parser")
            for s in soup(["script","style","noscript","svg"]):
                s.decompose()
            text = soup.get_text(" ", strip=True)
            snippet = text[:5000]
            pages.append({"url": url, "text": snippet})
        except Exception as e:
            # fallback fetch
            t, _ = simple_fetch_html_links(url)
            pages.append({"url": url, "text": (t or "")[:5000]})
        # polite delay
        time.sleep(0.6)

    # 3) quick regex pass across page texts
    found_emails = {}
    for p in pages:
        es = extract_emails_from_text(p["text"])
        for e in es:
            found_emails.setdefault(e, []).append(p["url"])

    # If regex found emails, prepare pages with context for LLM; else still run LLM (on text) to search for mentions
    if found_emails:
        print(f"🔎 Regex discovered {len(found_emails)} unique email(s) across site.")
    else:
        print("🔎 Regex found 0 emails — will still run LLM to search text for explicit emails (if present).")

    # 4) Ask LLM (strict JSON) to identify which emails belong to person_name
    results = llm_sitewide_search(person_name, pages)
    # results are verified by LLM to be present in the snippets
    if results:
        # sort by confidence desc
        results_sorted = sorted(results, key=lambda r: r.get("confidence",0), reverse=True)
        return results_sorted

    # if LLM returns nothing but regex found some emails, do local heuristics:
    if found_emails:
        local_candidates = []
        for e, urls in found_emails.items():
            # compute heuristic score: presence of name in page text near email?
            score = 0.0
            candidate_context = ""
            for u in urls:
                page = next((p for p in pages if p["url"] == u), None)
                if not page: continue
                idx = page["text"].lower().find(e.lower())
                context = page["text"][max(0, idx-60): idx+len(e)+60] if idx>=0 else page["text"][:200]
                candidate_context = context.strip()
                if person_name.lower() in context.lower():
                    score = max(score, 0.95)
                elif any(part.lower() in context.lower() for part in person_name.split()):
                    score = max(score, 0.7)
                else:
                    score = max(score, 0.4)
            local_candidates.append({"email": e, "url": urls[0], "context": candidate_context, "confidence": score})
        # sort by heuristic confidence
        return sorted(local_candidates, key=lambda r: r["confidence"], reverse=True)

    # nothing found
    return []

# ---------- main agent (integrates with prior v5 flow) ----------
# For brevity, this file assumes you already have the v5 main functions:
# - ai_decide_directory, detect_structure, fetch_aem_profiles, fetch_html_profiles,
# - ai_pick_profile, ai_choose_email, etc.
# We'll include a small main that tries the usual flow and then calls failover when needed.

# Minimal re-implementations / lightweight copy of detection + selection (keeps script self-contained)
def render_page(url):
    html, links = render_page_html_and_links(url, headless=True, scroll=False, scroll_tries=3)
    return html, links

def detect_structure_quick(url):
    """Quick detect by HEAD/GET text."""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
        text = (r.text or "").lower()
        if "_jcr_content.search.json" in text or "aemassets" in text:
            return "aem"
        if "wp-json" in text or "wp-content" in text:
            return "wordpress"
        if "graphql" in text or "/api/" in text:
            return "graphql"
    except:
        pass
    return "html"

def ai_decide_directory_quick(home_url, links):
    """Very small heuristic + LLM picking (keeps concise)."""
    # prefer anything that looks like people/team/professional/attorney
    candidates = [l for l in links if any(k in l.lower() for k in ("people","professional","team","attorney","lawyer"))]
    if not candidates:
        candidates = links[:30]
    # Ask LLM but force JSON output
    prompt = f"""
Select the ONE URL most likely to be the people/professionals directory for the site {home_url}.
Return ONLY JSON: {{ "directory": "<url or none>" }}
Candidates:
{candidates[:60]}
"""
    try:
        r = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.0)
        raw = r.choices[0].message.content
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        return data.get("directory")
    except:
        return candidates[0] if candidates else None

def ai_pick_profile_quick(name, urls):
    """Ask LLM to pick profile url with strict JSON, fallback fuzzy."""
    prompt = f"""
From this list of profile URLs, pick the one that most likely belongs to '{name}'.
Return JSON only: {{ "profile_url": "<url or 'none'>" }}
URLs:
{urls[:150]}
"""
    try:
        r = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.0)
        raw = r.choices[0].message.content
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        url = data.get("profile_url", "none")
    except:
        url = "none"
    if url in ("none", None) or "http" not in str(url):
        name_parts = name.lower().replace(".", "").split()
        matches = [u for u in urls if all(part in u.lower() for part in name_parts[:2])]
        if matches:
            url = matches[0]
        else:
            best = difflib.get_close_matches(name.lower().replace(" ", "-"), urls, n=1)
            url = best[0] if best else "none"
    return url

def ai_choose_email_quick(name, emails, text):
    """LLM picks email or returns nothing. Strict output not enforced here for brevity."""
    if not emails:
        return None
    prompt = f"""
Given these emails {emails} and this page text snippet, which email belongs to {name}? Return the single email or 'none'.
TEXT:
{text[:2000]}
"""
    r = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0.0)
    return r.choices[0].message.content.strip()

# ---------- main ----------
def main():
    if len(sys.argv) < 3:
        print("Usage: python smart_email_finder_failover.py <homepage_url> <person_name>")
        sys.exit(1)
    home_url = sys.argv[1].rstrip("/")
    person_name = sys.argv[2].strip()
    print(f"🔤 Searching for: {person_name} @ {home_url}")

    # 1) Render home and collect links
    home_html, home_links = render_page(home_url)

    # 2) Decide directory
    directory = ai_decide_directory_quick(home_url, home_links)
    if not directory:
        print("⚠️ Could not identify directory page; proceeding to sitewide failover.")
        results = sitewide_failover_search(home_url, person_name)
        if results:
            print_results(results, person_name)
            return
        print("❌ Nothing found.")
        return

    print(f"➡️ Directory: {directory}")

    # 3) detect structure and gather profile links
    structure = detect_structure_quick(directory)
    print(f"🧩 Detected: {structure}")
    profile_urls = []
    if "aem" in structure:
        # full AEM pagination (simple)
        for start in range(0,1000,50):
            api = urljoin(directory, f"_jcr_content.search.json?c=professional&q=&start={start}")
            try:
                r = requests.get(api, headers={"User-Agent":UA}, timeout=REQUEST_TIMEOUT)
                if r.status_code != 200: break
                j = r.json()
                hits = j.get("hits", [])
                if not hits: break
                for h in hits:
                    path = h.get("path") or h.get("url") or ""
                    if path:
                        profile_urls.append(urljoin(directory, path))
            except: break
    elif structure == "graphql":
        profile_urls = fetch_graphql_profiles(directory)  # use earlier defined function from v5 flow
    elif structure == "wordpress":
        profile_urls = fetch_wordpress_profiles(directory)
    else:
        profile_urls = fetch_html_profiles(directory)

    profile_urls = list(dict.fromkeys(profile_urls))
    print(f"🔎 Collected {len(profile_urls)} profile URLs")

    # 4) Ask AI to pick the person profile
    chosen = ai_pick_profile_quick(person_name, profile_urls)
    if chosen and chosen != "none":
        print(f"➡️ Chosen profile: {chosen}")
        # render and extract
        html, _ = render_page_html_and_links(chosen, headless=True, scroll=True, scroll_tries=4)
        emails = extract_emails_from_text(BeautifulSoup(html,'html.parser').get_text(" ",strip=True))
        if emails:
            # let LLM pick
            final = ai_choose_email_quick(person_name, emails, html)
            print("\nRESULT (direct):")
            print({"profile": chosen, "emails_found": emails, "llm_choice": final})
            return

    # 5) If we reach here, run failover sitewide LLM-backed search
    print("↪️ Primary profile extraction failed — running intelligent sitewide failover.")
    results = sitewide_failover_search(home_url, person_name)
    if results:
        print_results(results, person_name)
    else:
        print("❌ No explicit emails found on site.")

def print_results(results, name):
    print("\n==============================")
    print(f"👤 {name}")
    for r in results:
        print(f"📧 {r['email']}  (confidence={r.get('confidence',0):.2f})")
        print(f"🔗 {r.get('url')}")
        print(f"✂ context: {r.get('context')[:160]}")
        print("------------------------------")
    print("==============================")

# helpers from v5 reused (minimal)
def fetch_graphql_profiles(directory_url):
    # Playwright scroll method (kept minimal)
    print("⚙️ Scroll-fetch (Playwright) for dynamic lists...")
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(user_agent=UA)
        page.goto(directory_url, wait_until="networkidle", timeout=45000)
        last_h=0
        for _ in range(PLAYWRIGHT_SCROLL_TRIES):
            page.mouse.wheel(0,3000)
            page.wait_for_timeout(800)
            new_h=page.evaluate("document.body.scrollHeight")
            if new_h==last_h: break
            last_h=new_h
        links=[urljoin(directory_url,a.get_attribute("href")) for a in page.query_selector_all("a[href]")]
        browser.close()
    profiles=[u for u in links if any(k in u.lower() for k in ("people","professional","team","attorney","person"))]
    return list(dict.fromkeys(profiles))

def fetch_wordpress_profiles(base_url):
    print("⚙️ WordPress: simple REST fallback")
    urls=[]
    try:
        r = requests.get(urljoin(base_url, "/wp-json/wp/v2/pages"), headers={"User-Agent":UA}, timeout=REQUEST_TIMEOUT)
        if r.status_code==200:
            for e in r.json():
                link = e.get("link") or e.get("slug")
                if link and any(k in (link or "").lower() for k in ("team","people","attorney","professional")):
                    urls.append(link if link.startswith("http") else urljoin(base_url, link))
    except: pass
    return list(dict.fromkeys(urls))

def fetch_html_profiles(directory_url):
    print("⚙️ HTML fallback: fetch & small render hybrid")
    # first try fast GET
    try:
        r = requests.get(directory_url, headers={"User-Agent":UA}, timeout=REQUEST_TIMEOUT)
        soup=BeautifulSoup(r.text,'html.parser')
        links=[urljoin(directory_url,a['href']) for a in soup.select("a[href]")]
        profiles=[u for u in links if any(k in u.lower() for k in ("people","professional","team","attorney","bio"))]
        if len(profiles)>150:
            return list(dict.fromkeys(profiles))
    except: profiles=[]
    # fallback to Playwright scroll + collect links
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True)
        page=browser.new_page(user_agent=UA)
        page.goto(directory_url, wait_until="networkidle", timeout=45000)
        last_h=0
        for _ in range(PLAYWRIGHT_SCROLL_TRIES):
            page.mouse.wheel(0,3000)
            page.wait_for_timeout(800)
            new_h=page.evaluate("document.body.scrollHeight")
            if new_h==last_h: break
            last_h=new_h
        links=[urljoin(directory_url,a.get_attribute("href")) for a in page.query_selector_all("a[href]")]
        browser.close()
    profiles=[u for u in links if any(k in u.lower() for k in ("people","professional","team","attorney","bio"))]
    return list(dict.fromkeys(profiles))


def find_email_from_site(home_url: str, person_name: str):
    """
    Reusable version of main() for integration with other systems.
    Runs full extraction workflow (not just failover).
    Returns a normalized result list like:
      [{'email': 'john.doe@firm.com', 'url': '...', 'context': '', 'confidence': 0.95}]
    """
    print(f"🔍 Integrated call: Searching for {person_name} @ {home_url}")

    home_html, home_links = render_page(home_url)
    directory = ai_decide_directory_quick(home_url, home_links)

    if not directory:
        print("⚠️ No directory detected, running sitewide failover...")
        return sitewide_failover_search(home_url, person_name)

    print(f"➡️ Directory: {directory}")

    structure = detect_structure_quick(directory)
    print(f"🧩 Detected: {structure}")

    # Gather profile URLs
    profile_urls = []
    if "aem" in structure:
        # AEM API style
        for start in range(0, 1000, 50):
            api = urljoin(directory, f"_jcr_content.search.json?c=professional&q=&start={start}")
            try:
                r = requests.get(api, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
                if r.status_code != 200:
                    break
                j = r.json()
                hits = j.get("hits", [])
                if not hits:
                    break
                for h in hits:
                    path = h.get("path") or h.get("url") or ""
                    if path:
                        profile_urls.append(urljoin(directory, path))
            except:
                break
    elif structure == "graphql":
        profile_urls = fetch_graphql_profiles(directory)
    elif structure == "wordpress":
        profile_urls = fetch_wordpress_profiles(directory)
    else:
        profile_urls = fetch_html_profiles(directory)

    profile_urls = list(dict.fromkeys(profile_urls))
    print(f"🔎 Collected {len(profile_urls)} profile URLs")

    # Pick the right profile
    chosen = ai_pick_profile_quick(person_name, profile_urls)
    if chosen and chosen != "none":
        print(f"➡️ Chosen profile: {chosen}")
        html, _ = render_page_html_and_links(chosen, headless=True, scroll=True, scroll_tries=4)
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)
        emails = extract_emails_from_text(text)
        if emails:
            final = ai_choose_email_quick(person_name, emails, text)
            print(f"✅ Integrated result: {final}")
            # Normalize return
            return [{
                'email': emails[0].lower(),
                'url': chosen,
                'context': final[:200] if isinstance(final, str) else '',
                'confidence': 0.95
            }]

    # If that fails, fallback
    print("↪️ Fallback: running sitewide failover search...")
    return sitewide_failover_search(home_url, person_name)

# ---------- run ----------
if __name__ == "__main__":
    main()




# #!/usr/bin/env python3
# """
# smart_email_finder_failover.py - Clean, minimal, human-readable version.

# Usage:
#     python smart_email_finder_failover.py <homepage_url> "<person_name>"
# """

# import os
# import re
# import sys
# import json
# import time
# import requests
# import difflib
# from urllib.parse import urljoin, urlparse

# from bs4 import BeautifulSoup
# from dotenv import load_dotenv
# from playwright.sync_api import sync_playwright
# from groq import Groq

# # Configuration
# load_dotenv()
# GROQ_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_KEY:
#     print("Missing GROQ_API_KEY in .env")
#     sys.exit(1)

# MODEL = "llama-3.1-8b-instant"
# client = Groq(api_key=GROQ_KEY)

# USER_AGENT = "SmartEmailFinder/1.0"
# REQUEST_TIMEOUT = 15
# MAX_SITE_PAGES = 25
# PLAYWRIGHT_SCROLL_TRIES = 12

# # Utilities

# def domain_base(url: str) -> str:
#     try:
#         p = urlparse(url)
#         return f"{p.scheme}://{p.netloc}"
#     except Exception:
#         return url

# def http_fetch_html_links(url: str):
#     """Fast GET + link parse fallback."""
#     try:
#         r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
#         if r.status_code != 200:
#             return "", []
#         soup = BeautifulSoup(r.text, "html.parser")
#         links = []
#         for a in soup.select("a[href]"):
#             href = a.get("href")
#             if href:
#                 links.append(urljoin(url, href.split("#")[0]))
#         return r.text, list(dict.fromkeys(links))
#     except Exception:
#         return "", []

# def extract_emails_from_text(text: str):
#     matches = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", text or "")
#     return list(dict.fromkeys([m.lower() for m in matches]))

# # Playwright rendering

# def render_with_playwright(url: str, headless=True, scroll=False, scroll_tries=5):
#     """Return (html, absolute_links) rendered with Playwright."""
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=headless)
#         page = browser.new_page(user_agent=USER_AGENT)
#         page.goto(url, wait_until="networkidle", timeout=45000)
#         if scroll:
#             last_h = 0
#             for _ in range(scroll_tries):
#                 page.mouse.wheel(0, 3000)
#                 page.wait_for_timeout(800)
#                 new_h = page.evaluate("document.body.scrollHeight")
#                 if new_h == last_h:
#                     break
#                 last_h = new_h
#         html = page.content()
#         elements = page.query_selector_all("a[href]")
#         links = []
#         for el in elements:
#             try:
#                 href = el.get_attribute("href")
#                 if href:
#                     links.append(urljoin(url, href.split("#")[0]))
#             except Exception:
#                 continue
#         browser.close()
#     return html, list(dict.fromkeys(links))

# # LLM-backed sitewide search (strict JSON expected)

# def build_llm_prompt_for_pages(name: str, pages: list):
#     sample = []
#     for p in pages:
#         snippet = (p.get("text") or "")[:2000].replace("\n", " ")
#         sample.append({"url": p.get("url", ""), "text": snippet})
#     pages_json = json.dumps(sample, ensure_ascii=False)[:38000]
#     prompt = (
#         "You are a careful web-extraction assistant. You will be given a list of page url + text snippets.\n"
#         f"Task: Find any e-mail addresses that are explicitly present in the provided text snippets that clearly belong to the person named: \"{name}\".\n\n"
#         "Rules (must follow):\n"
#         "1) Do NOT guess or invent. Only return emails that literally appear in the supplied text snippets.\n"
#         "2) Return output ONLY as JSON matching exactly this schema:\n"
#         '{ "results": [ { "email": "<email address exactly as found>", '
#         '"url": "<page url where it appears>", "context": "<short text (<=200 chars) around the email>", '
#         '"confidence": <float 0.0-1.0> }, ... ] }\n'
#         '3) If you find no explicit emails return: {"results": []}\n'
#         "4) Confidence guidance: if full name near email use 0.95-1.0, last name/initials 0.6-0.85, no name 0.3-0.6.\n"
#         "5) Context should be up to ~200 characters around the email.\n\n"
#         "Pages:\n"
#         f"{pages_json}"
#     )
#     return prompt

# def llm_sitewide_search(name: str, pages: list):
#     prompt = build_llm_prompt_for_pages(name, pages)
#     resp = client.chat.completions.create(
#         model=MODEL,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.0,
#         max_tokens=800,
#     )
#     raw = resp.choices[0].message.content.strip()
#     try:
#         jtext = re.search(r"\{.*\}", raw, re.S).group(0)
#         data = json.loads(jtext)
#         results = data.get("results", [])
#         for r in results:
#             r["email"] = r.get("email", "").strip().lower()
#             r["context"] = (r.get("context", "") or "")[:300]
#             r["url"] = r.get("url", "")
#             r["confidence"] = float(r.get("confidence", 0.0))
#         return results
#     except Exception as e:
#         print("LLM failed to return strict JSON or parsing failed:", e)
#         return []

# # Failover: sitewide search that uses Playwright + regex + LLM

# def sitewide_failover_search(home_url: str, person_name: str, max_pages: int = MAX_SITE_PAGES):
#     base = domain_base(home_url)
#     html, links = render_with_playwright(home_url, headless=True, scroll=False)
#     internal_links = [l for l in links if urlparse(l).netloc == urlparse(home_url).netloc]
#     candidates = internal_links[:max_pages]
#     if home_url not in candidates:
#         candidates.insert(0, home_url)

#     pages = []
#     seen = set()
#     for url in candidates:
#         if url in seen:
#             continue
#         seen.add(url)
#         try:
#             scroll = any(k in url.lower() for k in ("/people", "/professionals", "/team", "/about"))
#             page_html, _ = render_with_playwright(url, headless=True, scroll=scroll, scroll_tries=4)
#             soup = BeautifulSoup(page_html, "html.parser")
#             for el in soup(["script", "style", "noscript", "svg"]):
#                 el.decompose()
#             text = soup.get_text(" ", strip=True)
#             pages.append({"url": url, "text": text[:5000]})
#         except Exception:
#             t, _ = http_fetch_html_links(url)
#             pages.append({"url": url, "text": (t or "")[:5000]})
#         time.sleep(0.6)

#     found_emails = {}
#     for p in pages:
#         for e in extract_emails_from_text(p["text"]):
#             found_emails.setdefault(e, []).append(p["url"])

#     if found_emails:
#         print(f"Regex found {len(found_emails)} unique email(s).")
#     else:
#         print("Regex found 0 emails; LLM will still be asked to search provided text.")

#     results = llm_sitewide_search(person_name, pages)
#     if results:
#         results_sorted = sorted(results, key=lambda r: r.get("confidence", 0.0), reverse=True)
#         return results_sorted

#     if found_emails:
#         local_candidates = []
#         for e, urls in found_emails.items():
#             score = 0.0
#             context = ""
#             for u in urls:
#                 page = next((pp for pp in pages if pp["url"] == u), None)
#                 if not page:
#                     continue
#                 idx = page["text"].lower().find(e.lower())
#                 if idx >= 0:
#                     context = page["text"][max(0, idx - 60): idx + len(e) + 60].strip()
#                 else:
#                     context = page["text"][:200].strip()
#                 if person_name.lower() in context.lower():
#                     score = max(score, 0.95)
#                 elif any(part.lower() in context.lower() for part in person_name.split()):
#                     score = max(score, 0.7)
#                 else:
#                     score = max(score, 0.4)
#             local_candidates.append({"email": e, "url": urls[0], "context": context, "confidence": score})
#         return sorted(local_candidates, key=lambda r: r["confidence"], reverse=True)

#     return []

# # Quick site structure detection and simple AI helpers

# def detect_structure_quick(url: str):
#     try:
#         r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
#         text = (r.text or "").lower()
#         if "_jcr_content.search.json" in text or "aemassets" in text:
#             return "aem"
#         if "wp-json" in text or "wp-content" in text:
#             return "wordpress"
#         if "graphql" in text or "/api/" in text:
#             return "graphql"
#     except Exception:
#         pass
#     return "html"

# def ai_decide_directory_quick(home_url: str, links: list):
#     candidates = [l for l in links if any(k in l.lower() for k in ("people", "professional", "team", "attorney", "lawyer"))]
#     if not candidates:
#         candidates = links[:30]
#     prompt = (
#         f"Select the ONE URL most likely to be the people/professionals directory for the site {home_url}.\n"
#         "Return ONLY JSON: { \"directory\": \"<url or none>\" }\n\n"
#         "Candidates:\n" + "\n".join(candidates[:60])
#     )
#     try:
#         r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0)
#         raw = r.choices[0].message.content
#         data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
#         return data.get("directory")
#     except Exception:
#         return candidates[0] if candidates else None

# def ai_pick_profile_quick(name: str, urls: list):
#     if not urls:
#         return "none"
#     prompt = (
#         f"From this list of profile URLs, pick the one that most likely belongs to '{name}'.\n"
#         "Return JSON only: { \"profile_url\": \"<url or 'none'>\" }\n\n"
#         "URLs:\n" + "\n".join(urls[:150])
#     )
#     try:
#         r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0)
#         raw = r.choices[0].message.content
#         data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
#         url = data.get("profile_url", "none")
#     except Exception:
#         url = "none"
#     if url in ("none", None) or "http" not in str(url):
#         name_parts = name.lower().replace(".", "").split()
#         matches = [u for u in urls if all(part in u.lower() for part in name_parts[:2])]
#         if matches:
#             url = matches[0]
#         else:
#             best = difflib.get_close_matches(name.lower().replace(" ", "-"), urls, n=1)
#             url = best[0] if best else "none"
#     return url

# def ai_choose_email_quick(name: str, emails: list, text: str):
#     if not emails:
#         return None
#     prompt = (
#         f"Given these emails {emails} and this page text snippet, which email belongs to {name}? "
#         "Return the single email or 'none'.\n\nTEXT:\n" + (text or "")[:2000]
#     )
#     r = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.0)
#     return r.choices[0].message.content.strip()

# # Profile collection helpers

# def fetch_graphql_profiles(directory_url: str):
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=True)
#         page = browser.new_page(user_agent=USER_AGENT)
#         page.goto(directory_url, wait_until="networkidle", timeout=45000)
#         last_h = 0
#         for _ in range(PLAYWRIGHT_SCROLL_TRIES):
#             page.mouse.wheel(0, 3000)
#             page.wait_for_timeout(800)
#             new_h = page.evaluate("document.body.scrollHeight")
#             if new_h == last_h:
#                 break
#             last_h = new_h
#         links = [urljoin(directory_url, a.get_attribute("href")) for a in page.query_selector_all("a[href]")]
#         browser.close()
#     profiles = [u for u in links if any(k in u.lower() for k in ("people", "professional", "team", "attorney", "person"))]
#     return list(dict.fromkeys(profiles))

# def fetch_wordpress_profiles(base_url: str):
#     urls = []
#     try:
#         r = requests.get(urljoin(base_url, "/wp-json/wp/v2/pages"), headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
#         if r.status_code == 200:
#             for e in r.json():
#                 link = e.get("link") or e.get("slug")
#                 if link and any(k in (link or "").lower() for k in ("team", "people", "attorney", "professional")):
#                     urls.append(link if link.startswith("http") else urljoin(base_url, link))
#     except Exception:
#         pass
#     return list(dict.fromkeys(urls))

# def fetch_html_profiles(directory_url: str):
#     profiles = []
#     try:
#         r = requests.get(directory_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
#         soup = BeautifulSoup(r.text, "html.parser")
#         links = [urljoin(directory_url, a["href"]) for a in soup.select("a[href]")]
#         profiles = [u for u in links if any(k in u.lower() for k in ("people", "professional", "team", "attorney", "bio"))]
#         if len(profiles) > 150:
#             return list(dict.fromkeys(profiles))
#     except Exception:
#         profiles = []
#     if not profiles:
#         with sync_playwright() as p:
#             browser = p.chromium.launch(headless=True)
#             page = browser.new_page(user_agent=USER_AGENT)
#             page.goto(directory_url, wait_until="networkidle", timeout=45000)
#             last_h = 0
#             for _ in range(PLAYWRIGHT_SCROLL_TRIES):
#                 page.mouse.wheel(0, 3000)
#                 page.wait_for_timeout(800)
#                 new_h = page.evaluate("document.body.scrollHeight")
#                 if new_h == last_h:
#                     break
#                 last_h = new_h
#             links = [urljoin(directory_url, a.get_attribute("href")) for a in page.query_selector_all("a[href]")]
#             browser.close()
#         profiles = [u for u in links if any(k in u.lower() for k in ("people", "professional", "team", "attorney", "bio"))]
#     return list(dict.fromkeys(profiles))

# # Main extraction workflow and helpers

# def print_results(results: list, name: str):
#     print("\n==============================")
#     print(f"Person: {name}")
#     for r in results:
#         print(f"{r.get('email')}  (confidence={r.get('confidence', 0):.2f})")
#         print(r.get("url"))
#         print(r.get("context", "")[:160])
#         print("------------------------------")
#     print("==============================")

# def find_email_from_site(home_url: str, person_name: str):
#     home_html, home_links = render_with_playwright(home_url)
#     directory = ai_decide_directory_quick(home_url, home_links)
#     if not directory:
#         return sitewide_failover_search(home_url, person_name)

#     structure = detect_structure_quick(directory)
#     if "aem" in structure:
#         profile_urls = []
#         for start in range(0, 1000, 50):
#             api = urljoin(directory, f"_jcr_content.search.json?c=professional&q=&start={start}")
#             try:
#                 r = requests.get(api, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
#                 if r.status_code != 200:
#                     break
#                 j = r.json()
#                 hits = j.get("hits", [])
#                 if not hits:
#                     break
#                 for h in hits:
#                     path = h.get("path") or h.get("url") or ""
#                     if path:
#                         profile_urls.append(urljoin(directory, path))
#             except Exception:
#                 break
#     elif structure == "graphql":
#         profile_urls = fetch_graphql_profiles(directory)
#     elif structure == "wordpress":
#         profile_urls = fetch_wordpress_profiles(directory)
#     else:
#         profile_urls = fetch_html_profiles(directory)

#     profile_urls = list(dict.fromkeys(profile_urls))
#     chosen = ai_pick_profile_quick(person_name, profile_urls)
#     if chosen and chosen != "none":
#         html, _ = render_with_playwright(chosen, headless=True, scroll=True, scroll_tries=4)
#         soup = BeautifulSoup(html, "html.parser")
#         text = soup.get_text(" ", strip=True)
#         emails = extract_emails_from_text(text)
#         if emails:
#             final = ai_choose_email_quick(person_name, emails, text)
#             return [{
#                 "email": emails[0].lower(),
#                 "url": chosen,
#                 "context": (final[:200] if isinstance(final, str) else ""),
#                 "confidence": 0.95
#             }]

#     return sitewide_failover_search(home_url, person_name)

# # CLI

# def main():
#     if len(sys.argv) < 3:
#         print("Usage: python smart_email_finder_failover.py <homepage_url> <person_name>")
#         sys.exit(1)
#     home_url = sys.argv[1].rstrip("/")
#     person_name = sys.argv[2].strip()
#     print(f"Searching for: {person_name} @ {home_url}")

#     home_html, home_links = render_with_playwright(home_url)
#     directory = ai_decide_directory_quick(home_url, home_links)
#     if not directory:
#         print("Could not detect directory; running sitewide failover.")
#         results = sitewide_failover_search(home_url, person_name)
#         if results:
#             print_results(results, person_name)
#             return
#         print("No results found.")
#         return

#     print(f"Directory detected: {directory}")
#     structure = detect_structure_quick(directory)
#     print(f"Detected structure: {structure}")

#     # Gather profile URLs based on structure
#     profile_urls = []
#     if "aem" in structure:
#         for start in range(0, 1000, 50):
#             api = urljoin(directory, f"_jcr_content.search.json?c=professional&q=&start={start}")
#             try:
#                 r = requests.get(api, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
#                 if r.status_code != 200:
#                     break
#                 j = r.json()
#                 hits = j.get("hits", [])
#                 if not hits:
#                     break
#                 for h in hits:
#                     path = h.get("path") or h.get("url") or ""
#                     if path:
#                         profile_urls.append(urljoin(directory, path))
#             except Exception:
#                 break
#     elif structure == "graphql":
#         profile_urls = fetch_graphql_profiles(directory)
#     elif structure == "wordpress":
#         profile_urls = fetch_wordpress_profiles(directory)
#     else:
#         profile_urls = fetch_html_profiles(directory)

#     profile_urls = list(dict.fromkeys(profile_urls))
#     print(f"Collected {len(profile_urls)} profile URLs")

#     chosen = ai_pick_profile_quick(person_name, profile_urls)
#     if chosen and chosen != "none":
#         print(f"Chosen profile: {chosen}")
#         html, _ = render_with_playwright(chosen, headless=True, scroll=True, scroll_tries=4)
#         emails = extract_emails_from_text(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
#         if emails:
#             final = ai_choose_email_quick(person_name, emails, html)
#             print("RESULT (direct):")
#             print({"profile": chosen, "emails_found": emails, "llm_choice": final})
#             return

#     print("Primary profile extraction failed; running sitewide failover.")
#     results = sitewide_failover_search(home_url, person_name)
#     if results:
#         print_results(results, person_name)
#     else:
#         print("No explicit emails found on site.")

# if __name__ == "__main__":
#     main()
# python email_extraction_stage.py https://www.finnegan.com/en/professionals/ "anthony lombardi"