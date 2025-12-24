

########################################################################almost
#!/usr/bin/env python3
"""
smart_email_finder.py - v2.0 SIMPLIFIED & INTELLIGENT
------------------------------------------------------
Handles ANY website type with intelligent fallback logic.

Flow:
1. Try to find professional profile → Extract email
2. If fails → Go to contact page → Extract general email
3. Done! (No complex site-wide searches)
"""

import os
import re
import sys
import json
import requests
import difflib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
from groq import Groq

# -------------- CONFIG --------------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_KEY:
    print("❌ Missing GROQ_API_KEY in .env")
    sys.exit(1)

MODEL = "llama-3.1-8b-instant"
client = Groq(api_key=GROQ_KEY)
UA = "SmartEmailFinder/2.0"

# ------------------------------------
# CORE UTILITIES
# ------------------------------------

def render_page(url, scroll=False, timeout=30000):
    """Render page with Playwright - handles dynamic content."""
    print(f"🌐 Rendering: {url}")
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=UA)
            page.goto(url, wait_until="networkidle", timeout=timeout)
            
            if scroll:
                for _ in range(3):
                    page.mouse.wheel(0, 2000)
                    page.wait_for_timeout(500)
            
            html = page.content()
            
            # Extract links
            links = []
            for a in page.query_selector_all("a[href]"):
                try:
                    href = a.get_attribute("href")
                    if href:
                        links.append(urljoin(url, href.split("#")[0]))
                except:
                    continue
            
            browser.close()
            return html, list(dict.fromkeys(links))
    except Exception as e:
        print(f"⚠️ Playwright failed: {e}, trying simple fetch")
        return simple_fetch(url)


def simple_fetch(url):
    """Simple HTTP GET - fallback for when Playwright fails."""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        if r.status_code >= 400:
            return "", []
        
        soup = BeautifulSoup(r.text, "html.parser")
        links = [urljoin(url, a.get("href", "")) for a in soup.find_all("a", href=True)]
        return r.text, list(dict.fromkeys(links))
    except:
        return "", []


def extract_emails_from_text(text):
    """Extract all emails from text using regex."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    # Filter out obvious junk
    valid_emails = [
        e.lower() for e in emails 
        if len(e) < 100 
        and not any(x in e.lower() for x in ["noreply", "no-reply", "example", "test"])
    ]
    return list(dict.fromkeys(valid_emails))


def extract_text_from_html(html):
    """Extract clean text from HTML."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
        element.decompose()
    return soup.get_text(" ", strip=True)


# ------------------------------------
# AI HELPERS
# ------------------------------------

def ai_find_directory(home_url, links):
    """AI finds the professionals directory URL."""
    # Filter to likely candidates
    candidates = [
        l for l in links 
        if any(k in l.lower() for k in ["people", "professional", "team", "attorney", "lawyer", "our-people"])
        and not any(x in l.lower() for x in ["news", "blog", "event", "career", "job"])
    ]
    
    if not candidates:
        candidates = links[:20]
    
    if not candidates:
        return None
    
    prompt = f"""
Find the professionals/people directory URL for this law firm website.

Homepage: {home_url}
Candidate URLs: {json.dumps(candidates[:30])}

Return ONLY JSON: {{"directory": "<url>"}}
If none found, return: {{"directory": null}}
"""
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(re.search(r'\{.*\}', text, re.DOTALL).group(0))
        return data.get("directory")
    except:
        return candidates[0] if candidates else None


def ai_pick_profile(person_name, profile_urls):
    """AI picks the most likely profile URL for the person."""
    if not profile_urls:
        return None
    
    # Try fuzzy matching first
    name_parts = person_name.lower().replace(".", "").split()
    matches = [u for u in profile_urls if all(part in u.lower() for part in name_parts[:2])]
    
    if len(matches) == 1:
        return matches[0]
    
    # If multiple or none, ask AI
    prompt = f"""
Pick the profile URL that most likely belongs to: "{person_name}"

URLs: {json.dumps((matches or profile_urls)[:50])}

Return ONLY JSON: {{"profile_url": "<url>"}}
If uncertain, return: {{"profile_url": null}}
"""
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=200
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(re.search(r'\{.*\}', text, re.DOTALL).group(0))
        return data.get("profile_url")
    except:
        return matches[0] if matches else None


def ai_validate_email(email, person_name, page_text):
    """AI validates if email belongs to person."""
    prompt = f"""
Does this email belong to this person based on the page content?

Email: {email}
Person: {person_name}
Page text: {page_text[:1000]}

Consider:
1. Username matches name (e.g., "john.smith" matches "John Smith")
2. Email appears near person's name in text
3. Not a general email (info@, contact@)

Return ONLY JSON: {{"valid": true/false, "confidence": 0.0-1.0}}
"""
    
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150
        )
        text = resp.choices[0].message.content.strip()
        data = json.loads(re.search(r'\{.*\}', text, re.DOTALL).group(0))
        return data
    except:
        return {"valid": True, "confidence": 0.5}


# ------------------------------------
# MAIN EXTRACTION LOGIC
# ------------------------------------

def extract_from_contact_page(home_url):
    """Extract general contact email from contact page."""
    print("📧 Searching contact page for general email...")
    
    # Common contact page URLs
    contact_urls = [
        urljoin(home_url, "/contact"),
        urljoin(home_url, "/contact-us"),
        urljoin(home_url, "/contact-us/"),
        urljoin(home_url, "/en/contact"),
        urljoin(home_url, "/en/contact-us"),
        home_url
    ]
    
    for url in contact_urls:
        try:
            html, _ = simple_fetch(url)
            if not html:
                continue
            
            soup = BeautifulSoup(html, "html.parser")
            
            # Look for mailto links
            for a in soup.find_all("a", href=re.compile(r"^mailto:", re.I)):
                email = a["href"].replace("mailto:", "").replace("MAILTO:", "").split("?")[0].strip()
                if email and "@" in email:
                    print(f"✅ Found contact email: {email}")
                    return email.lower()
            
            # Look for emails in text
            text = soup.get_text()
            emails = extract_emails_from_text(text)
            
            if emails:
                # Prefer contact/info emails
                priority = [e for e in emails if any(x in e for x in ["contact", "info", "general"])]
                email = priority[0] if priority else emails[0]
                print(f"✅ Found contact email: {email}")
                return email.lower()
        
        except Exception as e:
            print(f"⚠️ Error on {url}: {e}")
            continue
    
    print("❌ No contact email found")
    return None


def find_email_from_site(home_url: str, person_name: str):
    """
    Main function - Find email for a person on a law firm website.
    
    Returns list of dicts: [{'email': '...', 'url': '...', 'confidence': 0.95}]
    """
    print(f"\n🔍 SEARCHING: {person_name} @ {home_url}")
    print("=" * 70)
    
    try:
        # STEP 1: Render homepage and get links
        home_html, home_links = render_page(home_url)
        
        # STEP 2: Find professionals directory
        directory = ai_find_directory(home_url, home_links)
        
        if not directory:
            print("⚠️ No professionals directory found")
            print("↪️ Going to contact page for general email")
            general_email = extract_from_contact_page(home_url)
            if general_email:
                return [{"email": general_email, "url": home_url, "confidence": 0.6, "context": "General contact"}]
            return []
        
        print(f"✅ Directory: {directory}")
        
        # STEP 3: Get all profile links from directory
        print("🔎 Collecting profile links...")
        dir_html, dir_links = render_page(directory, scroll=True)
        
        # Filter to actual profile links
        profile_urls = [
            l for l in dir_links
            if any(k in l.lower() for k in ["/people/", "/professional/", "/attorney/", "/team/", "/bio/", "/person/"])
            and l != directory  # Don't include directory itself
            and not any(x in l.lower() for x in ["search", "filter", "sort", "page=", "?"])
        ]
        
        profile_urls = list(dict.fromkeys(profile_urls))[:100]  # Limit to 100
        print(f"✅ Found {len(profile_urls)} profile URLs")
        
        if not profile_urls:
            print("⚠️ No profile links found")
            print("↪️ Going to contact page for general email")
            general_email = extract_from_contact_page(home_url)
            if general_email:
                return [{"email": general_email, "url": home_url, "confidence": 0.6, "context": "General contact"}]
            return []
        
        # STEP 4: Pick the right profile
        profile_url = ai_pick_profile(person_name, profile_urls)
        
        if not profile_url:
            print("⚠️ Could not identify profile")
            print("↪️ Going to contact page for general email")
            general_email = extract_from_contact_page(home_url)
            if general_email:
                return [{"email": general_email, "url": home_url, "confidence": 0.6, "context": "General contact"}]
            return []
        
        print(f"✅ Profile: {profile_url}")
        
        # STEP 5: Extract email from profile
        profile_html, _ = render_page(profile_url, scroll=True)
        profile_text = extract_text_from_html(profile_html)
        emails = extract_emails_from_text(profile_text)
        
        if not emails:
            print("⚠️ No emails on profile page")
            print("↪️ Going to contact page for general email")
            general_email = extract_from_contact_page(home_url)
            if general_email:
                return [{"email": general_email, "url": home_url, "confidence": 0.6, "context": "General contact"}]
            return []
        
        # STEP 6: Validate emails with AI
        results = []
        for email in emails:
            validation = ai_validate_email(email, person_name, profile_text)
            if validation.get("valid", False):
                results.append({
                    "email": email,
                    "url": profile_url,
                    "confidence": validation.get("confidence", 0.8),
                    "context": f"Found on {person_name}'s profile"
                })
        
        if results:
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            print(f"✅ Found {len(results)} valid email(s)")
            for r in results:
                print(f"   📧 {r['email']} (confidence: {r['confidence']:.2f})")
            return results
        
        # No valid emails found - fallback to contact
        print("⚠️ No valid professional emails found")
        print("↪️ Going to contact page for general email")
        general_email = extract_from_contact_page(home_url)
        if general_email:
            return [{"email": general_email, "url": home_url, "confidence": 0.6, "context": "General contact"}]
        
        return []
    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("↪️ Trying contact page as fallback")
        general_email = extract_from_contact_page(home_url)
        if general_email:
            return [{"email": general_email, "url": home_url, "confidence": 0.5, "context": "Fallback contact"}]
        return []


# ------------------------------------
# CLI INTERFACE
# ------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python smart_email_finder.py <website_url> '<person_name>'")
        print("Example: python smart_email_finder.py https://www.finnegan.com 'Anthony J. Lombardi'")
        sys.exit(1)
    
    website = sys.argv[1].rstrip("/")
    person = sys.argv[2].strip()
    
    results = find_email_from_site(website, person)
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    
    if results:
        for r in results:
            print(f"📧 {r['email']}")
            print(f"🔗 {r['url']}")
            print(f"📊 Confidence: {r['confidence']:.2f}")
            print(f"💬 {r.get('context', 'N/A')}")
            print("-" * 70)
    else:
        print("❌ No emails found")
    
    print("=" * 70)


if __name__ == "__main__":
    main()
