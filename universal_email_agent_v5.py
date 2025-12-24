#!/usr/bin/env python3
"""
Universal Email Agent v5 - Production Ready
AI-powered email extraction agent for any professional directory website.

Key Improvements in v5:
- Smart page ready detection (handles heavy JS sites)
- Better search result parsing with AI fallback
- Input validation before interaction
- Dynamic content waiting (AJAX/SPA support)
- Improved name matching algorithm
- Retry logic for transient failures

Version: 5.0.0
"""

import os
import sys
import json
import re
import asyncio
from urllib.parse import urljoin, urlparse
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import httpx
import dns.resolver

# ============== CONFIGURATION ==============
load_dotenv()

@dataclass
class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL: str = "llama-3.3-70b-versatile"
    MODEL_FAST: str = "llama-3.1-8b-instant"
    MAX_CANDIDATES: int = 10
    PAGE_TIMEOUT: int = 60000
    ELEMENT_TIMEOUT: int = 10000
    MIN_CONFIDENCE: int = 65
    HEADLESS: bool = True

CONFIG = Config()

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

GENERIC_EMAIL_PREFIXES = [
    'info@', 'contact@', 'support@', 'admin@', 'office@', 'enquiries@',
    'mail@', 'hello@', 'help@', 'sales@', 'marketing@', 'press@',
    'media@', 'careers@', 'jobs@', 'recruitment@', 'hr@', 'legal@',
    'privacy@', 'webmaster@', 'noreply@', 'no-reply@', 'donotreply@',
    'permissions@', 'feedback@', 'general@', 'enquiry@', 'reception@',
    'digitalmarketing@', 'communications@', 'pr@', 'news@', 'events@',
    'subscriptions@', 'subscribe@', 'unsubscribe@', 'newsletter@',
    'copyright@', 'licensing@', 'reprints@', 'editor@', 'editorial@',
    'reference@', 'library@', 'research@', 'alumni@', 'admissions@'
]

PROFILE_URL_PATTERNS = [
    '/people/', '/professionals/', '/attorneys/', '/lawyers/', '/team/',
    '/bio/', '/profile/', '/staff/', '/lawyer/', '/attorney/', '/our-people/',
    '/leadership/', '/partners/', '/associates/', '/counsel/', '/member/',
    '/expert/', '/advisor/', '/director/', '/employee/'
]

AVOID_URL_PATTERNS = [
    '/news/', '/article/', '/press/', '/insight/', '/publication/',
    '/blog/', '/event/', '/about/', '/contact/', '/careers/', '/search',
    'format=vcard', '.pdf', '.doc', 'mailto:', 'javascript:', 'tel:',
    '/subscribe', '/login', '/register', '/privacy', '/terms', '/cookie',
    '/sitemap', '/rss', '/feed', 'linkedin.com', 'twitter.com', 'facebook.com'
]

# ============== LOGGING ==============
def log(level: str, msg: str, indent: int = 0):
    prefix = " " * indent
    icons = {
        'start': '[START]', 'ok': '[OK]', 'fail': '[FAIL]', 'warn': '[WARN]',
        'info': '[INFO]', 'search': '[SEARCH]', 'ai': '[AI]', 'found': '[FOUND]',
        'check': '[CHECK]', 'success': '[SUCCESS]', 'skip': '[SKIP]',
        'type': '[TYPE]', 'wait': '[WAIT]', 'popup': '[POPUP]', 'nav': '[NAV]',
        'retry': '[RETRY]', 'debug': '[DEBUG]'
    }
    print(f"{prefix}{icons.get(level, '[*]')} {msg}")

# ============== LLM FUNCTIONS ==============
async def call_llm(system_prompt: str, user_prompt: str, max_tokens: int = 500, fast: bool = False) -> str:
    """Call LLM with retry logic"""
    if not CONFIG.GROQ_API_KEY:
        return "{}"

    model = CONFIG.MODEL_FAST if fast else CONFIG.MODEL

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {CONFIG.GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "temperature": 0,
                        "max_tokens": max_tokens
                    }
                )

                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                elif response.status_code == 429:
                    await asyncio.sleep(2 ** attempt)
                    continue
        except Exception as e:
            if attempt == 2:
                log('warn', f"LLM error: {str(e)[:40]}")
            await asyncio.sleep(1)

    return "{}"

def parse_json(response: str) -> dict:
    """Safely parse JSON from LLM response"""
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except:
        pass
    return {}

# ============== AI ANALYSIS FUNCTIONS ==============
async def ai_find_search_input(elements: List[dict]) -> dict:
    """Use AI to find the best search input for people directory"""

    # Pre-filter: Only visible INPUT elements with text/search types
    visible_inputs = []
    for el in elements:
        if el.get('tag') != 'INPUT':
            continue
        if not el.get('visible', False):
            continue
        el_type = el.get('type', '').lower()
        if el_type in ['hidden', 'checkbox', 'radio', 'submit', 'button', 'file', 'image', 'password', 'email']:
            continue
        if el_type in ['', 'text', 'search']:
            visible_inputs.append(el)

    if not visible_inputs:
        return {"found": False, "reason": "No visible text inputs"}

    # Limit for LLM context
    visible_inputs = visible_inputs[:20]

    elements_desc = "\n".join([
        f"[{i}] placeholder='{el.get('placeholder', '')[:40]}' | label='{el.get('label', '')[:30]}' | "
        f"name='{el.get('name', '')}' | id='{el.get('id', '')[:20]}' | in_header={el.get('in_header', False)}"
        for i, el in enumerate(visible_inputs)
    ])

    prompt = f"""Find the BEST input field for searching PEOPLE by NAME on this directory page.

INPUTS:
{elements_desc}

RULES:
- MUST be for searching people/attorneys/lawyers by name (not site search)
- PREFER inputs with "name", "keyword", "people", "attorney", "lawyer" in placeholder/label
- STRONGLY AVOID inputs with in_header=True (those are usually site-wide search)
- AVOID inputs that filter by location, practice area, or other non-name criteria

Return JSON: {{"index": <0-{len(visible_inputs)-1}>, "confidence": <50-100>, "reason": "<brief>"}}
If NONE suitable: {{"index": null, "confidence": 0, "reason": "<why>"}}"""

    response = await call_llm("Return ONLY valid JSON.", prompt, fast=True)
    result = parse_json(response)

    if result.get("index") is not None and result.get("confidence", 0) >= 50:
        idx = result["index"]
        if 0 <= idx < len(visible_inputs):
            return {
                "found": True,
                "element": visible_inputs[idx],
                "confidence": result.get("confidence", 50),
                "reason": result.get("reason", "")
            }

    return {"found": False, "reason": result.get("reason", "No suitable input found")}

async def ai_analyze_search_results(page: Page, target_name: str, base_url: str) -> List[str]:
    """Use AI to find profile links when standard parsing fails"""

    try:
        # Get all links with their text
        links_data = await page.evaluate("""() => {
            const links = [];
            document.querySelectorAll('a[href]').forEach(a => {
                const href = a.href;
                const text = (a.innerText || a.textContent || '').trim().slice(0, 100);
                if (href && text && !href.startsWith('javascript:') && !href.startsWith('mailto:')) {
                    // Remove URL fragments like #footer
                    const cleanHref = href.split('#')[0];
                    if (cleanHref) links.push({href: cleanHref, text});
                }
            });
            return links.slice(0, 100);
        }""")

        if not links_data:
            return []

        # Filter to same domain and deduplicate
        domain = urlparse(base_url).netloc
        seen_urls = set()
        same_domain_links = []
        for l in links_data:
            clean_url = l['href'].split('#')[0]  # Remove fragments
            if domain in clean_url and clean_url not in seen_urls:
                seen_urls.add(clean_url)
                same_domain_links.append({'href': clean_url, 'text': l['text']})

        if not same_domain_links:
            return []

        links_text = "\n".join([f"- {l['text'][:50]} -> {l['href']}" for l in same_domain_links[:30]])

        prompt = f"""Find links that lead to the profile page of "{target_name}".

LINKS ON PAGE:
{links_text}

Return JSON with array of URLs (most likely first):
{{"profile_urls": ["url1", "url2", ...], "confidence": <0-100>}}

If no matches: {{"profile_urls": [], "confidence": 0}}"""

        response = await call_llm("Return ONLY valid JSON.", prompt, max_tokens=300, fast=True)
        result = parse_json(response)

        # Clean URLs (remove fragments)
        urls = [url.split('#')[0] for url in result.get("profile_urls", [])]
        return urls[:5]

    except Exception as e:
        log('warn', f"AI link analysis failed: {str(e)[:30]}", 1)
        return []

async def ai_verify_profile(page_text: str, target_name: str) -> dict:
    """Use AI to verify if a page is the target person's profile"""

    parts = target_name.split()
    first_name = parts[0] if parts else ""
    last_name = parts[-1] if len(parts) > 1 else first_name

    # Handle suffixes
    suffixes = ['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'esq', 'esq.']
    if last_name.lower() in suffixes and len(parts) > 2:
        last_name = parts[-2]

    prompt = f"""Is this the professional profile page for "{target_name}"?

CRITICAL VERIFICATION RULES:
1. The page MUST prominently display the name "{target_name}" (or close variation)
2. BOTH "{first_name}" AND "{last_name}" must appear in the person's name on the page
3. Small variations OK: "Robert J. Giuffra Jr." matches "Robert Giuffra"
4. Different names do NOT match: "Samantha Chaifetz" does NOT match "Gina Durham"
5. If the page shows a DIFFERENT person's name prominently, return is_match: false
6. Only extract email that belongs to {target_name}, not other people mentioned

PAGE CONTENT (excerpt):
{page_text[:2500]}

Return JSON:
{{
    "is_match": true/false,
    "confidence": 0-100,
    "email": "email@domain.com" or null,
    "name_on_page": "<exact name shown on page>",
    "match_reason": "<why it matches or doesn't>"
}}"""

    response = await call_llm("Return ONLY valid JSON.", prompt, max_tokens=200)
    result = parse_json(response)

    return {
        "is_match": result.get("is_match", False),
        "confidence": result.get("confidence", 0),
        "email": result.get("email")
    }

# ============== UTILITY FUNCTIONS ==============
def validate_email_mx(email: str) -> bool:
    """Validate email domain has MX records"""
    if not email or "@" not in email:
        return False
    try:
        domain = email.split("@")[1]
        dns.resolver.resolve(domain, "MX", lifetime=5)
        return True
    except:
        return False

def is_valid_email(email: str) -> bool:
    """Check if email is valid and not generic"""
    if not email or "@" not in email:
        return False
    email_lower = email.lower()
    return not any(email_lower.startswith(prefix) for prefix in GENERIC_EMAIL_PREFIXES)

def is_profile_url(url: str) -> bool:
    url_lower = url.lower()
    return any(p in url_lower for p in PROFILE_URL_PATTERNS)

def should_skip_url(url: str) -> bool:
    url_lower = url.lower()
    return any(p in url_lower for p in AVOID_URL_PATTERNS)

def name_in_url(url: str, name: str) -> bool:
    """Check if name parts appear in URL"""
    url_lower = url.lower()
    parts = [p.lower() for p in name.split() if len(p) > 2]
    return any(p in url_lower for p in parts)

# ============== PAGE FUNCTIONS ==============
async def wait_for_page_ready(page: Page, timeout: int = 10000):
    """Wait for page to be fully interactive"""
    try:
        await page.wait_for_load_state("networkidle", timeout=timeout)
    except:
        pass

    try:
        await page.wait_for_function("() => document.readyState === 'complete'", timeout=5000)
    except:
        pass

    # Wait for any lazy-loaded content
    try:
        await page.wait_for_function(
            "() => !document.querySelector('.loading, .spinner, [class*=\"loading\"]')",
            timeout=5000
        )
    except:
        pass

    await page.wait_for_timeout(1500)

async def extract_page_elements(page: Page) -> List[dict]:
    """Extract all interactive elements from the page"""
    try:
        elements = await page.evaluate("""() => {
            const results = [];
            document.querySelectorAll('input, button, a[href], select, textarea, [role="button"], [role="search"]').forEach((el, index) => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);

                const visible = rect.width > 0 && rect.height > 0 &&
                               style.display !== 'none' &&
                               style.visibility !== 'hidden' &&
                               parseFloat(style.opacity) > 0 &&
                               rect.top < window.innerHeight &&
                               rect.bottom > 0;

                let label = '';
                if (el.id) {
                    const l = document.querySelector(`label[for="${el.id}"]`);
                    if (l) label = l.innerText?.trim() || '';
                }
                if (!label) {
                    const pl = el.closest('label');
                    if (pl) label = pl.innerText?.trim() || '';
                }
                if (!label) {
                    const prev = el.previousElementSibling;
                    if (prev && ['LABEL', 'SPAN', 'DIV'].includes(prev.tagName)) {
                        label = prev.innerText?.trim()?.slice(0, 50) || '';
                    }
                }

                let inHeader = false;
                let parent = el.parentElement;
                for (let i = 0; i < 10 && parent && parent !== document.body; i++) {
                    const tag = parent.tagName?.toLowerCase();
                    const cls = (parent.className || '').toString().toLowerCase();
                    const id = (parent.id || '').toLowerCase();

                    if (tag === 'header' || tag === 'nav' ||
                        cls.includes('header') || cls.includes('nav') || cls.includes('menu') ||
                        id.includes('header') || id.includes('nav') || id.includes('menu')) {
                        inHeader = true;
                        break;
                    }
                    parent = parent.parentElement;
                }

                const isDisabled = el.disabled || el.getAttribute('aria-disabled') === 'true';
                const isReadonly = el.readOnly || el.getAttribute('aria-readonly') === 'true';

                results.push({
                    index, tag: el.tagName, id: el.id || '',
                    class: (el.className || '').toString().slice(0, 80),
                    type: el.type || '', placeholder: el.placeholder || '',
                    'aria-label': el.getAttribute('aria-label') || '',
                    name: el.name || '',
                    text: (el.innerText || '').slice(0, 50).trim(),
                    label: label.slice(0, 50),
                    visible, in_header: inHeader,
                    disabled: isDisabled, readonly: isReadonly
                });
            });
            return results;
        }""")
        return elements or []
    except Exception as e:
        log('warn', f"Element extraction error: {str(e)[:30]}")
        return []

async def get_element_locator(page: Page, element: dict):
    """Get a Playwright locator for an element"""
    tag = element.get('tag', 'input').lower()

    strategies = [
        ('id', lambda: page.locator(f"#{element['id']}")),
        ('name', lambda: page.locator(f"{tag}[name='{element['name']}']")),
        ('placeholder', lambda: page.locator(f"{tag}[placeholder='{element['placeholder']}']")),
        ('aria-label', lambda: page.locator(f"[aria-label='{element['aria-label']}']")),
    ]

    for attr, get_loc in strategies:
        if element.get(attr):
            try:
                loc = get_loc()
                if await loc.count() > 0:
                    first = loc.first
                    if await first.is_visible(timeout=2000):
                        return first
            except:
                pass

    return None

async def handle_popups(page: Page, max_attempts: int = 3):
    """Handle cookie consent and other popups"""
    log('popup', "Checking for popups...")

    selectors = [
        "#onetrust-accept-btn-handler", "#accept-cookies", "#cookie-accept",
        "button[id*='accept' i]", "button[id*='agree' i]",
        "button[class*='accept' i]", "button[class*='agree' i]",
        "button:has-text('Accept All')", "button:has-text('Accept')",
        "button:has-text('OK')", "button:has-text('Agree')",
        "button:has-text('Got it')", "button:has-text('I understand')",
        "[aria-label='Close']", ".close-button", ".modal-close",
    ]

    count = 0
    for _ in range(max_attempts):
        found = False
        for sel in selectors:
            try:
                loc = page.locator(sel).first
                if await loc.count() > 0 and await loc.is_visible(timeout=1000):
                    await loc.click(timeout=3000)
                    await page.wait_for_timeout(500)
                    count += 1
                    found = True
                    log('ok', f"Clicked: {sel[:40]}", 1)
            except:
                pass

        try:
            await page.keyboard.press("Escape")
        except:
            pass

        if not found:
            break

    if count > 0:
        log('ok', f"Handled {count} popup(s)", 1)
    else:
        log('info', "No popups found", 1)

async def extract_emails_from_page(page: Page, page_text: str) -> List[str]:
    """Extract emails using multiple methods"""
    found_emails = set()

    # Method 1: mailto links
    try:
        for link in await page.locator("a[href^='mailto:']").all():
            href = await link.get_attribute("href")
            if href:
                email = href.replace("mailto:", "").split("?")[0].split("&")[0].strip()
                if "@" in email:
                    found_emails.add(email.lower())
    except:
        pass

    # Method 2: Regex on page text
    for email in EMAIL_PATTERN.findall(page_text):
        if "@" in email:
            found_emails.add(email.lower())

    # Method 3: Regex on HTML
    try:
        html = await page.content()
        html = html.replace("&#64;", "@").replace("&#46;", ".").replace("&#x40;", "@")
        for email in EMAIL_PATTERN.findall(html):
            if "@" in email:
                found_emails.add(email.lower())
    except:
        pass

    # Method 4: Data attributes
    try:
        for attr in ["data-email", "data-mail"]:
            for el in await page.locator(f"[{attr}]").all():
                email = await el.get_attribute(attr)
                if email and "@" in email:
                    found_emails.add(email.lower())
    except:
        pass

    # Filter out generic emails
    valid = [e for e in found_emails if is_valid_email(e)]
    return valid if valid else list(found_emails)

# ============== SEARCH RESULT ANALYSIS ==============
def analyze_search_results(html: str, target_name: str, base_url: str) -> List[str]:
    """Analyze search results to find candidate profile links"""
    soup = BeautifulSoup(html, "lxml")
    domain = urlparse(base_url).netloc

    name_parts = [p.lower() for p in target_name.split()]
    first_name = name_parts[0] if name_parts else ""
    last_name = name_parts[-1] if len(name_parts) > 1 else first_name

    suffixes = ['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'esq', 'esq.']
    if last_name.lower() in suffixes and len(name_parts) > 2:
        last_name = name_parts[-2]

    candidates = []

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)

        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue

        full_url = urljoin(base_url, href)
        full_url = full_url.split('#')[0]  # Remove fragment

        if urlparse(full_url).netloc != domain:
            continue

        if should_skip_url(full_url):
            continue

        href_lower = href.lower()
        text_lower = text.lower()

        score = 0

        # CRITICAL: At minimum, last name must be present somewhere
        has_last_name = last_name in text_lower or last_name in href_lower
        if not has_last_name:
            continue  # Skip if last name not found at all

        has_all = all(p in text_lower or p in href_lower for p in name_parts if len(p) > 2)
        has_first = first_name in text_lower or first_name in href_lower

        if has_all:
            score += 40  # All name parts match
        elif has_first and has_last_name:
            score += 30  # First and last match
        elif has_last_name:
            score += 15  # Only last name matches

        if is_profile_url(full_url):
            score += 15

        if name_in_url(full_url, target_name):
            score += 10

        if len(text.split()) >= 2 and len(text) < 50:
            score += 5

        if score >= 15:
            candidates.append({"url": full_url, "text": text[:60], "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)

    seen = set()
    unique = []
    for c in candidates:
        if c["url"] not in seen:
            seen.add(c["url"])
            unique.append(c["url"])

    return unique[:CONFIG.MAX_CANDIDATES]

# ============== FALLBACK SEARCH ==============
async def fallback_find_input(page: Page):
    """Fallback method to find search input using patterns"""

    selectors = [
        "main input[placeholder*='name' i]",
        "section input[placeholder*='name' i]",
        ".content input[placeholder*='name' i]",
        "[class*='filter'] input[placeholder*='name' i]",
        "main input[placeholder*='keyword' i]",
        "section input[placeholder*='keyword' i]",
        "input[placeholder*='people' i]",
        "input[placeholder*='attorney' i]",
        "input[placeholder*='lawyer' i]",
        "input[placeholder*='professional' i]",
        "main input[type='search']",
        "section input[type='search']",
        ".search-form:not(header *) input",
        "[class*='filter'] input[type='text']",
        "input[type='search']:not(header input):not(nav input)",
    ]

    for sel in selectors:
        try:
            loc = page.locator(sel)
            for i in range(min(await loc.count(), 3)):
                el = loc.nth(i)
                if await el.is_visible(timeout=2000):
                    tag = await el.evaluate("el => el.tagName")
                    if tag.upper() == "INPUT":
                        log('found', f"Fallback: {sel[:50]}", 1)
                        return el
        except:
            pass

    return None

# ============== MAIN AGENT CLASS ==============
class UniversalEmailAgent:
    """Production-ready email extraction agent"""

    def __init__(self, url: str, name: str):
        self.start_url = url.rstrip("/")
        self.domain = urlparse(url).netloc
        self.name = name
        self.browser = None
        self.page = None

    async def setup_browser(self) -> bool:
        try:
            pw = await async_playwright().start()
            self.browser = await pw.chromium.launch(
                headless=CONFIG.HEADLESS,
                args=['--disable-blink-features=AutomationControlled', '--no-sandbox']
            )
            ctx = await self.browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080}
            )
            self.page = await ctx.new_page()
            await self.page.route("**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2}", lambda r: r.abort())
            return True
        except Exception as e:
            log('fail', f"Browser setup failed: {e}")
            return False

    async def cleanup(self):
        try:
            if self.browser:
                await self.browser.close()
        except:
            pass

    def construct_profile_urls(self) -> List[str]:
        """Construct likely profile URLs when search fails"""
        name_parts = self.name.lower().split()
        first_name = name_parts[0] if name_parts else ""
        last_name = name_parts[-1] if len(name_parts) > 1 else first_name

        # Handle suffixes
        suffixes = ['jr', 'jr.', 'sr', 'sr.', 'ii', 'iii', 'iv', 'esq', 'esq.']
        if last_name in suffixes and len(name_parts) > 2:
            last_name = name_parts[-2]

        # Common URL patterns used by law firms
        patterns = [
            f"/people/{first_name}-{last_name}",
            f"/people/{last_name}-{first_name}",
            f"/people/{last_name[0]}/{last_name}-{first_name}",
            f"/lawyers/{first_name}-{last_name}",
            f"/lawyers/{last_name}-{first_name}",
            f"/professionals/{first_name}-{last_name}",
            f"/attorneys/{first_name}-{last_name}",
            f"/team/{first_name}-{last_name}",
        ]

        # Build full URLs
        base = self.start_url.split('/people')[0].split('/lawyers')[0].split('/professionals')[0]
        base = base.split('/attorneys')[0].split('/team')[0]

        urls = [f"{base}{p}" for p in patterns]
        return urls[:5]  # Return top 5 most likely

    async def load_page(self, url: str = None, retry: int = 2) -> bool:
        target = url or self.start_url

        for attempt in range(retry):
            try:
                log('info', f"Loading{' (retry)' if attempt > 0 else ''}...")
                await self.page.goto(target, timeout=CONFIG.PAGE_TIMEOUT, wait_until='domcontentloaded')
                await wait_for_page_ready(self.page)
                return True
            except Exception as e:
                if attempt < retry - 1:
                    log('warn', f"Load failed, retrying...", 1)
                    await asyncio.sleep(2)
                else:
                    log('fail', f"Page load failed: {str(e)[:40]}")

        return False

    async def find_search_input(self, elements: List[dict]):
        log('search', "Analyzing page structure...")

        result = await ai_find_search_input(elements)

        if result.get("found"):
            element = result["element"]

            if element.get('in_header'):
                log('warn', "AI found header input, trying fallback...", 1)
            elif element.get('disabled') or element.get('readonly'):
                log('warn', "AI found disabled input, trying fallback...", 1)
            else:
                locator = await get_element_locator(self.page, element)
                if locator:
                    try:
                        if await locator.is_enabled(timeout=3000):
                            desc = element.get('placeholder') or element.get('label') or element.get('name') or 'input'
                            log('ai', f"Found: {desc[:40]} (conf: {result.get('confidence')}%)", 1)
                            return locator
                    except:
                        pass

        log('search', "Trying fallback patterns...", 1)
        return await fallback_find_input(self.page)

    async def perform_search(self, input_loc) -> bool:
        log('type', f"Searching: {self.name}")

        try:
            await input_loc.scroll_into_view_if_needed()
            await self.page.wait_for_timeout(500)

            try:
                await input_loc.click(timeout=5000)
            except:
                log('warn', "Click blocked, using JS...", 1)
                try:
                    await input_loc.evaluate("el => { el.click(); el.focus(); }")
                except:
                    pass

            await self.page.wait_for_timeout(300)

            try:
                await input_loc.fill("", timeout=5000)
            except:
                await input_loc.press("Control+a")
                await input_loc.press("Delete")

            await input_loc.type(self.name, delay=80)

            log('info', "Pressing Enter", 1)
            await self.page.keyboard.press("Enter")

            log('wait', "Waiting for results...", 1)

            # Wait for AJAX/navigation
            await asyncio.sleep(2)

            try:
                await self.page.wait_for_load_state("networkidle", timeout=10000)
            except:
                pass

            # Additional wait for JS frameworks to update DOM
            await self.page.wait_for_timeout(2500)

            # Wait for any loading indicators to disappear
            try:
                await self.page.wait_for_function(
                    "() => !document.querySelector('.loading, .spinner, [class*=\"loading\"], [class*=\"searching\"]')",
                    timeout=5000
                )
            except:
                pass

            for sel in ["button:has-text('Apply')", "button:has-text('Search')",
                       "button:has-text('Filter')", "button[type='submit']"]:
                try:
                    btn = self.page.locator(sel).first
                    if await btn.count() > 0 and await btn.is_visible(timeout=1000):
                        text = await btn.inner_text()
                        if not any(w in text.lower() for w in ['menu', 'nav', 'clear', 'reset']):
                            log('info', f"Clicking: {text.strip()[:20]}", 1)
                            await btn.click(timeout=3000)
                            await self.page.wait_for_timeout(2000)
                            break
                except:
                    pass

            return True

        except Exception as e:
            log('fail', f"Search failed: {str(e)[:40]}")
            return False

    async def check_direct_profile(self) -> Optional[dict]:
        url = self.page.url

        if is_profile_url(url) and name_in_url(url, self.name):
            log('nav', "Landed directly on profile page")

            try:
                text = await self.page.inner_text("body")
                emails = await extract_emails_from_page(self.page, text)

                if emails:
                    verdict = await ai_verify_profile(text, self.name)
                    if verdict.get("is_match") and verdict.get("confidence", 0) >= CONFIG.MIN_CONFIDENCE:
                        email = verdict.get("email") or emails[0]
                        if validate_email_mx(email):
                            return {"email": email, "profile_url": url, "confidence": verdict.get("confidence")}
            except:
                pass

        return None

    async def process_candidate(self, url: str) -> Optional[dict]:
        # Clean URL (remove fragments)
        url = url.split('#')[0]

        try:
            await self.page.goto(url, timeout=CONFIG.PAGE_TIMEOUT)
            await self.page.wait_for_timeout(2000)
            await handle_popups(self.page, max_attempts=1)

            text = await self.page.inner_text("body")
            emails = await extract_emails_from_page(self.page, text)

            verdict = await ai_verify_profile(text, self.name)
            log('info', f"Verdict: match={verdict.get('is_match')}, conf={verdict.get('confidence')}", 1)

            if verdict.get("is_match") and verdict.get("confidence", 0) >= CONFIG.MIN_CONFIDENCE:
                email = verdict.get("email") or (emails[0] if emails else None)

                # If no email found yet, try additional extraction methods
                if not email:
                    log('info', "Trying additional email extraction methods...", 1)

                    # Method: Click any "Email" or "Contact" links to reveal email
                    email_reveal_selectors = [
                        "a:has-text('Email')",
                        "button:has-text('Email')",
                        "a[class*='email' i]",
                        ".contact-link",
                        "[data-toggle*='email' i]"
                    ]

                    for sel in email_reveal_selectors:
                        try:
                            link = self.page.locator(sel).first
                            if await link.count() > 0 and await link.is_visible(timeout=1000):
                                # Check if it's a mailto link first
                                href = await link.get_attribute("href")
                                if href and href.startswith("mailto:"):
                                    email = href.replace("mailto:", "").split("?")[0].strip()
                                    if email and "@" in email:
                                        break
                        except:
                            pass

                    # Method: Look for email pattern in any visible text
                    if not email:
                        try:
                            all_text = await self.page.evaluate("() => document.body.innerText")
                            found = EMAIL_PATTERN.findall(all_text)
                            valid = [e for e in found if is_valid_email(e)]
                            if valid:
                                email = valid[0]
                        except:
                            pass

                if email and validate_email_mx(email):
                    return {"email": email, "profile_url": url, "confidence": verdict.get("confidence")}
                elif not email:
                    log('warn', "Profile matched but no email found", 1)
        except Exception as e:
            log('warn', f"Process error: {str(e)[:30]}", 1)

        return None

    async def get_contact_page_email(self) -> Optional[dict]:
        """Fallback: Get general firm email from contact page when personal email not found"""
        log('info', "Trying contact page fallback...")

        # Get base URL (home page)
        parsed = urlparse(self.start_url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        # Extract main domain for matching (e.g., "dlapiper" from "www.dlapiper.com")
        domain_parts = parsed.netloc.replace("www.", "").split(".")
        main_domain = domain_parts[0] if domain_parts else ""

        all_emails_found = set()
        pages_to_try = []

        # First try to find contact/about links from homepage
        try:
            log('info', f"Loading homepage: {base_url}", 1)
            await self.page.goto(base_url, timeout=CONFIG.PAGE_TIMEOUT, wait_until='domcontentloaded')
            await self.page.wait_for_timeout(2000)
            await handle_popups(self.page, max_attempts=1)

            # Extract emails from homepage footer (often has contact info)
            await self._extract_emails_to_set(all_emails_found)

            # Look for contact/about links
            link_selectors = [
                ("a[href*='contact']", "contact"),
                ("a:has-text('Contact')", "contact"),
                ("a:has-text('Contact Us')", "contact"),
                ("a[href*='about']", "about"),
                ("a:has-text('About')", "about"),
                ("footer a[href*='contact']", "footer-contact"),
                ("footer a[href*='email']", "footer-email"),
                ("a[href*='offices']", "offices"),
                ("a[href*='locations']", "locations"),
            ]

            for sel, link_type in link_selectors:
                try:
                    links = await self.page.locator(sel).all()
                    for link in links[:2]:  # Check first 2 of each type
                        try:
                            if await link.is_visible(timeout=1000):
                                href = await link.get_attribute("href")
                                if href and not href.startswith("mailto:") and not href.startswith("tel:"):
                                    full_url = urljoin(base_url, href)
                                    if full_url not in pages_to_try and parsed.netloc in full_url:
                                        pages_to_try.append(full_url)
                        except:
                            pass
                except:
                    pass

            # Also try common paths
            common_paths = [
                "/contact", "/contact-us", "/contactus", "/contact.html",
                "/about/contact", "/en/contact", "/en/contact-us",
                "/about", "/about-us", "/about.html",
                "/offices", "/locations", "/our-offices",
            ]

            for path in common_paths:
                full_url = f"{base_url}{path}"
                if full_url not in pages_to_try:
                    pages_to_try.append(full_url)

            log('info', f"Found {len(pages_to_try)} pages to check for contact email", 1)

            # Visit each page and extract emails
            for page_url in pages_to_try[:5]:  # Limit to 5 pages
                try:
                    log('check', f"Checking: {page_url[:50]}...", 1)
                    response = await self.page.goto(page_url, timeout=15000, wait_until='domcontentloaded')
                    if response and response.status == 200:
                        await self.page.wait_for_timeout(1500)
                        await handle_popups(self.page, max_attempts=1)
                        await self._extract_emails_to_set(all_emails_found)

                        # If we found good emails, we can stop
                        good_emails = [e for e in all_emails_found if self._is_good_contact_email(e, main_domain)]
                        if good_emails:
                            log('info', f"Found {len(good_emails)} potential contact emails", 1)
                            break
                except:
                    continue

            # Now filter and return best email
            if all_emails_found:
                log('info', f"Total emails found: {len(all_emails_found)}", 1)

                # Priority 1: Preferred prefixes from firm domain
                preferred_prefixes = ['info@', 'contact@', 'enquiries@', 'enquiry@', 'office@',
                                     'mail@', 'hello@', 'general@', 'reception@', 'admin@']

                for prefix in preferred_prefixes:
                    for email in all_emails_found:
                        if email.startswith(prefix) and main_domain in email:
                            if validate_email_mx(email):
                                log('found', f"General contact email: {email}", 1)
                                return {
                                    "email": email,
                                    "profile_url": self.page.url,
                                    "confidence": 50,
                                    "is_general_contact": True
                                }

                # Priority 2: Any email from firm domain
                for email in all_emails_found:
                    email_domain = email.split("@")[1] if "@" in email else ""
                    if main_domain in email_domain:
                        if validate_email_mx(email):
                            log('found', f"General contact email: {email}", 1)
                            return {
                                "email": email,
                                "profile_url": self.page.url,
                                "confidence": 40,
                                "is_general_contact": True
                            }

                # Priority 3: Any valid professional email (not spam/tracking)
                excluded_domains = ['google.com', 'facebook.com', 'twitter.com', 'linkedin.com',
                                   'instagram.com', 'youtube.com', 'mailto.com', 'email.com']
                excluded_prefixes = ['noreply', 'no-reply', 'donotreply', 'unsubscribe', 'bounce', 'mailer']

                for email in all_emails_found:
                    email_domain = email.split("@")[1] if "@" in email else ""
                    email_prefix = email.split("@")[0] if "@" in email else ""

                    # Skip excluded
                    if any(d in email_domain for d in excluded_domains):
                        continue
                    if any(p in email_prefix.lower() for p in excluded_prefixes):
                        continue

                    if validate_email_mx(email):
                        log('found', f"General contact email: {email}", 1)
                        return {
                            "email": email,
                            "profile_url": self.page.url,
                            "confidence": 30,
                            "is_general_contact": True
                        }

            log('warn', "No contact email found on any page", 1)
            return None

        except Exception as e:
            log('warn', f"Contact page fallback failed: {str(e)[:30]}", 1)
            return None

    async def _extract_emails_to_set(self, email_set: set):
        """Extract all emails from current page and add to set"""
        try:
            # Method 1: mailto links
            for link in await self.page.locator("a[href^='mailto:']").all():
                try:
                    href = await link.get_attribute("href")
                    if href:
                        email = href.replace("mailto:", "").split("?")[0].split("&")[0].strip()
                        if "@" in email and "." in email.split("@")[1]:
                            email_set.add(email.lower())
                except:
                    pass

            # Method 2: Regex on page text
            try:
                text = await self.page.inner_text("body")
                for email in EMAIL_PATTERN.findall(text):
                    if "@" in email and "." in email.split("@")[1]:
                        email_set.add(email.lower())
            except:
                pass

            # Method 3: Check HTML for obfuscated emails
            try:
                html = await self.page.content()
                # Decode common obfuscations
                html = html.replace("&#64;", "@").replace("&#46;", ".")
                html = html.replace("&#x40;", "@").replace("&#x2e;", ".")
                html = html.replace("[at]", "@").replace("[dot]", ".")
                html = html.replace(" at ", "@").replace(" dot ", ".")

                for email in EMAIL_PATTERN.findall(html):
                    if "@" in email and "." in email.split("@")[1]:
                        email_set.add(email.lower())
            except:
                pass

            # Method 4: Look for email in data attributes
            try:
                for attr in ["data-email", "data-mail", "data-contact"]:
                    for el in await self.page.locator(f"[{attr}]").all():
                        try:
                            email = await el.get_attribute(attr)
                            if email and "@" in email:
                                email_set.add(email.lower())
                        except:
                            pass
            except:
                pass

        except:
            pass

    def _is_good_contact_email(self, email: str, main_domain: str) -> bool:
        """Check if email is a good contact email"""
        if not email or "@" not in email:
            return False

        # Check if from firm domain
        if main_domain in email:
            return True

        # Check if common contact prefix
        good_prefixes = ['info', 'contact', 'enquir', 'office', 'hello', 'mail', 'general']
        email_prefix = email.split("@")[0].lower()
        return any(p in email_prefix for p in good_prefixes)

    async def run(self) -> Optional[dict]:
        log('start', "Universal Email Agent v5")
        log('info', f"Target: {self.name}")
        log('info', f"URL: {self.start_url}")
        print()

        try:
            if not await self.setup_browser():
                return None

            if not await self.load_page():
                await self.cleanup()
                return None

            await handle_popups(self.page)
            print()

            elements = await extract_page_elements(self.page)
            if not elements:
                log('warn', "No elements, waiting for JS...")
                await self.page.wait_for_timeout(5000)
                elements = await extract_page_elements(self.page)

            # If still no elements, try scrolling to trigger lazy loading
            if not elements:
                log('warn', "Still no elements, trying scroll trigger...")
                try:
                    await self.page.evaluate("window.scrollTo(0, 500)")
                    await self.page.wait_for_timeout(2000)
                    await self.page.evaluate("window.scrollTo(0, 0)")
                    await self.page.wait_for_timeout(2000)
                    elements = await extract_page_elements(self.page)
                except:
                    pass

            # Last resort: check for iframes
            if not elements:
                try:
                    frames = self.page.frames
                    for frame in frames[1:]:  # Skip main frame
                        try:
                            frame_elements = await frame.evaluate("""() => {
                                const results = [];
                                document.querySelectorAll('input').forEach((el, i) => {
                                    results.push({index: i, tag: 'INPUT', visible: true,
                                                 placeholder: el.placeholder || '', type: el.type || ''});
                                });
                                return results;
                            }""")
                            if frame_elements:
                                log('info', f"Found {len(frame_elements)} elements in iframe")
                                elements = frame_elements
                                break
                        except:
                            pass
                except:
                    pass

            if not elements:
                log('fail', "Could not extract page elements")

                # Last resort: Try to construct and check profile URLs directly
                log('info', "Trying direct URL approach...")
                direct_urls = self.construct_profile_urls()
                for url in direct_urls:
                    try:
                        log('check', f"Trying: {url[:60]}...")
                        result = await self.process_candidate(url)
                        if result:
                            log('success', f"Email: {result['email']}")
                            log('info', f"Profile: {result['profile_url']}")
                            await self.cleanup()
                            return result
                    except:
                        continue

                # FALLBACK: Try contact page
                print()
                contact_result = await self.get_contact_page_email()
                if contact_result:
                    print()
                    log('success', f"Email (General Contact): {contact_result['email']}")
                    log('info', f"Source: {contact_result['profile_url']}")
                    log('warn', "Note: This is the firm's general contact email, not personal email")
                    await self.cleanup()
                    return contact_result

                await self.cleanup()
                return None

            log('info', f"Found {len(elements)} elements")

            input_loc = await self.find_search_input(elements)
            if not input_loc:
                log('fail', "No search functionality found")

                # FALLBACK: Try contact page
                print()
                contact_result = await self.get_contact_page_email()
                if contact_result:
                    print()
                    log('success', f"Email (General Contact): {contact_result['email']}")
                    log('info', f"Source: {contact_result['profile_url']}")
                    log('warn', "Note: This is the firm's general contact email, not personal email")
                    await self.cleanup()
                    return contact_result

                await self.cleanup()
                return None

            print()

            if not await self.perform_search(input_loc):
                await self.cleanup()
                return None

            print()

            result = await self.check_direct_profile()
            if result:
                log('success', f"Email: {result['email']}")
                log('info', f"Profile: {result['profile_url']}")
                await self.cleanup()
                return result

            log('search', "Analyzing results...")
            html = await self.page.content()
            candidates = analyze_search_results(html, self.name, self.start_url)

            if not candidates:
                log('info', "Standard parsing found 0, trying AI...", 1)
                candidates = await ai_analyze_search_results(self.page, self.name, self.start_url)

            # Filter out search result pages (URLs with searchstring, query params, #fragments)
            filtered_candidates = []
            for url in candidates:
                url_clean = url.split('#')[0]  # Remove fragment
                # Skip URLs that are clearly search results, not profiles
                if 'searchstring=' in url_clean.lower() or 'search=' in url_clean.lower():
                    log('skip', f"Skipping search URL: {url_clean[:50]}...", 1)
                    continue
                filtered_candidates.append(url_clean)
            candidates = filtered_candidates

            # If still no good candidates, try to construct likely profile URL
            if not candidates:
                log('info', "Trying to construct profile URL...", 1)
                candidates = self.construct_profile_urls()

            log('info', f"Found {len(candidates)} candidates", 1)

            for i, url in enumerate(candidates):
                if should_skip_url(url):
                    continue

                print()
                log('check', f"[{i+1}/{len(candidates)}] {url[:60]}...")

                result = await self.process_candidate(url)
                if result:
                    print()
                    log('success', f"Email: {result['email']}")
                    log('info', f"Profile: {result['profile_url']}")
                    await self.cleanup()
                    return result

            log('fail', "No verified email found")

            # FALLBACK: Try to get general contact email from contact page
            print()
            contact_result = await self.get_contact_page_email()
            if contact_result:
                print()
                log('success', f"Email (General Contact): {contact_result['email']}")
                log('info', f"Source: {contact_result['profile_url']}")
                log('warn', "Note: This is the firm's general contact email, not personal email")
                await self.cleanup()
                return contact_result

            await self.cleanup()
            return None

        except Exception as e:
            log('fail', f"Agent error: {str(e)[:40]}")
            await self.cleanup()
            return None

# ============== ENTRY POINT ==============
async def main():
    if len(sys.argv) < 3:
        print("Universal Email Agent v5")
        print("=" * 50)
        print("\nUsage: python universal_email_agent_v5.py <url> \"Full Name\"")
        sys.exit(1)

    agent = UniversalEmailAgent(sys.argv[1], sys.argv[2])
    result = await agent.run()

    print()
    print("=" * 50)

    if result:
        if result.get('is_general_contact'):
            print(f"[OK] SUCCESS (General Contact)")
            print(f"     Email: {result['email']}")
            print(f"     Source: {result['profile_url']}")
            print(f"     Note: Personal email not found, using firm's general contact")
        else:
            print(f"[OK] SUCCESS")
            print(f"     Email: {result['email']}")
            print(f"     Profile: {result['profile_url']}")
            print(f"     Confidence: {result.get('confidence', 'N/A')}%")
    else:
        print("[FAIL] No email found")

    return result

if __name__ == "__main__":
    asyncio.run(main())
