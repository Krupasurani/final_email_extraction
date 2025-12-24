

#!/usr/bin/env python3
"""
REVISED AI-DRIVEN EMAIL DISCOVERY SYSTEM
Implements complete workflow from Data_Input_and_Categorization_Logic_Updated.docx
- Purpose Selection (Patent/Trademark)
- Entity Type Detection (Law Firm/Company/Individual)
- Professional Role Matching
- Complete edge case handling from examples
"""

import re
import json
import logging
import os
import asyncio
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from website_finder_ai import find_official_website
from email_extraction_stage import find_email_from_site
import concurrent.futures
import logging

logger = logging.getLogger(__name__)
load_dotenv()

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from groq import Groq
    from ddgs import DDGS
    import requests
    from bs4 import BeautifulSoup
    from email_validator import validate_email, EmailNotValidError
    import pandas as pd
except ImportError as e:
    print(f"Missing package: {e}")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EntityData:
    """Represents parsed entity information"""
    raw_name: str
    address: str
    attorney_name: Optional[str]
    entity_type: str  # 'law_firm', 'company', 'individual', 'unknown'
    resolved_firm_name: Optional[str]  # For individuals, their firm
    full_name: str  # For individuals
    first_name: str
    last_name: str
    raw_data: dict
    entity_confidence: float = 0.0
    entity_reasoning: str = ""

@dataclass  
class SearchContext:
    """Search purpose and parameters"""
    purpose: str  # 'patent' or 'trademark'
    target_practice_areas: List[str]
    preferred_designations: List[str]
    location_from_address: Dict[str, str]  # city, state, country

# LAW FIRM CUES (from document)
LAW_FIRM_CUES = [
    'law', 'legal', 'attorneys', 'lawyers', 'advocates', 'counsels', 'chambers', 
    'associates', 'legal consultants', 'legal services', 'ip', 'patents', 'trademarks',
    'copyright', 'conseil', 'avocat', 'propriété intellectuelle', 'marques', 'brevets',
    'abogados', 'patentes', 'propiedad intelectual', 'rechtsanwälte', 'kanzlei',
    'anwaltskanzlei', 'patentanwalt', 'avvocati', 'studio legale', 'advogados',
    'llp', 'pllc', 'p.c.', 'pc'
]

# COMPANY CUES (basic list - document mentions separate attachment)
COMPANY_CUES = [
    'inc', 'corp', 'corporation', 'ltd', 'limited', 'company', 'co.',
    'technologies', 'systems', 'solutions', 'industries', 'manufacturing',
    'pharma', 'pharmaceutical', 'biotech', 'medical', 'healthcare'
]

class AIEmailDiscoveryPipeline:
    """
    REVISED AI-Driven Email Discovery Pipeline
    Implements complete workflow from requirements document
    """
    
    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            logger.error("GROQ_API_KEY required!")
            exit(1)

        self.groq = Groq(api_key=self.groq_api_key)
        self.ddgs = DDGS()

        browser_config = BrowserConfig(headless=True, verbose=False)
        self.crawler = AsyncWebCrawler(config=browser_config)

        # AI Decision tracking
        self.decisions_log = []

        # Store people directory URL from Stage 3 for Stage 4
        self.people_directory_url = None

        logger.info("🤖 REVISED AI-Driven Email Discovery Pipeline Initialized")
    
    def log_ai_decision(self, stage: str, decision: str, confidence: float, reasoning: str):
        """Track all AI decisions for transparency"""
        self.decisions_log.append({
            'stage': stage,
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': time.time()
        })
        logger.info(f"🧠 AI [{stage}]: {decision} (confidence: {confidence:.2f})")
    
    async def llm_query(self, prompt: str, max_tokens: int = 500, temperature: float = 0.1) -> str:
        """Make LLM query with rate limiting"""
        await asyncio.sleep(1.5)
        
        try:
            response = self.groq.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            await asyncio.sleep(5)
            return ""
    
    # ============================================================================
    # NEW STAGE 0: PURPOSE SELECTION & CONTEXT CREATION
    # ============================================================================
    
    def create_search_context(self, purpose: str, address: str) -> SearchContext:
        """
        Create search context based on purpose (Patent/Trademark)
        Per document specifications
        """
        # Extract location from address
        location = self.parse_location_from_address(address)
        
        if purpose.lower() == 'patent':
            return SearchContext(
                purpose='patent',
                target_practice_areas=[
                    'Patents',
                    'Intellectual Property',
                    'Patents and Innovation',
                    'IP',
                    'IP and Technology'
                ],
                preferred_designations=[
                    'Senior Partner',
                    'Partner',
                    'Senior Principal',
                    'Principal',
                    'Senior Associate',
                    'Senior Counsel',
                    'Senior Patent Agent'
                ],
                location_from_address=location
            )
        else:  # trademark
            return SearchContext(
                purpose='trademark',
                target_practice_areas=[
                    'Trademarks',
                    'Brands',
                    'Trademarks and Brands',
                    'Trademarks and Antitrust Laws',
                    'Intellectual Property',
                    'IP'
                ],
                preferred_designations=[
                    'Senior Partner',
                    'Partner',
                    'Senior Principal',
                    'Principal',
                    'Senior Attorney',
                    'Senior Associate',
                    'Senior Counsel',
                    'Trademark Specialist'
                ],
                location_from_address=location
            )
    
    def parse_location_from_address(self, address: str) -> Dict[str, str]:
        """Extract location components from address"""
        location = {
            'city': '',
            'state': '',
            'country': ''
        }
        
        # US State codes
        us_states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 
                     'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                     'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                     'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                     'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
        
        # Try to find state
        for state in us_states:
            if re.search(rf'\b{state}\b', address, re.IGNORECASE):
                location['state'] = state
                break
        
        # Try to find city (word before state)
        if location['state']:
            match = re.search(rf'([A-Za-z\s]+),?\s+{location["state"]}', address, re.IGNORECASE)
            if match:
                location['city'] = match.group(1).strip()
        
        # Detect country
        if 'US' in address.upper() or any(state in address.upper() for state in us_states):
            location['country'] = 'US'
        
        return location
    
    def generate_email_pattern(self, name: str, firm_url: str) -> str:
        """
        FIX 2: Generate common email patterns from professional name
        Returns most likely email pattern based on firm domain
        """
        try:
            # Extract domain from firm URL
            domain = urlparse(firm_url).netloc.lower().replace('www.', '')
            if not domain:
                return None
            
            # Parse name
            name_clean = name.strip()
            
            # Remove titles and suffixes
            titles = ['dr.', 'dr', 'ph.d.', 'ph.d', 'phd', 'esq.', 'esq', 'jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv']
            for title in titles:
                name_clean = re.sub(rf'\b{re.escape(title)}\b', '', name_clean, flags=re.IGNORECASE)
            
            # Remove special characters but keep spaces
            name_clean = re.sub(r'[^\w\s]', ' ', name_clean)
            name_clean = ' '.join(name_clean.split())  # Normalize spaces
            
            # Split into parts
            parts = name_clean.split()
            if len(parts) < 2:
                return None
            
            first_name = parts[0].lower()
            last_name = parts[-1].lower()
            
            # Generate patterns in order of likelihood
            patterns = [
                f"{first_name}.{last_name}@{domain}",        # john.doe@firm.com (most common)
                f"{first_name}{last_name}@{domain}",          # johndoe@firm.com
                f"{first_name[0]}.{last_name}@{domain}",      # j.doe@firm.com
                f"{first_name[0]}{last_name}@{domain}",       # jdoe@firm.com
                f"{first_name}_{last_name}@{domain}",         # john_doe@firm.com
            ]
            
            # Return first pattern (most common)
            return patterns[0]
            
        except Exception as e:
            logger.warning(f"      Pattern generation failed: {e}")
            return None
    
    # ============================================================================
    # NEW STAGE 1: ENTITY TYPE DETECTION
    # ============================================================================

    async def ai_detect_entity_type(self, raw_name: str, address: str, attorney_name: Optional[str]) -> EntityData:
        """
        STAGE 1: Detect entity type (Law Firm / Company / Individual)
        Enhanced with:
        - Global corporate fallback layer
        - Final default fallback = law_firm
        """
        logger.info(f"\n{'='*80}")
        logger.info("STAGE 1: ENTITY TYPE DETECTION")
        logger.info(f"Raw Name: {raw_name}")
        logger.info(f"Address: {address}")
        logger.info(f"Attorney Name: {attorney_name or 'Not provided'}")
        logger.info(f"{'='*80}")

        raw_name_clean = ' '.join(str(raw_name or '').split())
        attorney_name_clean = ' '.join(attorney_name.split()) if attorney_name else None
        lower_name = raw_name_clean.lower()

        # Original cues
        has_law_cue = any(cue in lower_name for cue in LAW_FIRM_CUES)
        has_company_cue = any(cue in lower_name for cue in COMPANY_CUES)
        looks_like_person = self.looks_like_person_name(raw_name_clean)

        # 🌍 Global company identifiers (fallback layer)
        global_company_identifiers = [
            "pvt ltd", "private limited", "ltd", "limited", "llc", "inc", "incorporated",
            "corp", "corporation", "co.", "company", "plc", "pte ltd", "sdn bhd",
            "ag", "gmbh", "s.a.", "sas", "bv", "nv", "oy", "ab", "as", "kk", "gk",
            "sarl", "sl", "spa", "srl", "sa de cv", "pty ltd", "cie", "lda", "zrt", "rt",
            "d.o.o.", "a.s.", "oü", "sp. z o.o.", "kft", "eood", "ad", "a/s", "eurl",
            "s.p.a.", "ooo", "jsc", "ao", "paо", "zaо", "ug", "compagnie", "cía",
            "compañía", "industries", "technologies", "medical", "devices", "systems",
            "solutions"
        ]

        # -----------------------------
        # CASE 1: Attorney name provided
        # -----------------------------
        if attorney_name_clean and len(attorney_name_clean) > 3:
            logger.info("   Case: Attorney name provided → Raw name likely firm")

            prompt = f"""You are an entity classification expert.

    Given:
    - Name: {raw_name_clean}
    - Address: {address}
    - Attorney: {attorney_name_clean}

    Task: Decide if this is a LAW FIRM or COMPANY.
    Return JSON:
    {{"entity_type": "law_firm" or "company", "confidence": 0.0-1.0, "reasoning": "short"}}"""

            result = await self.llm_query(prompt, max_tokens=300)
            try:
                data = json.loads(re.search(r'\{.*\}', result, re.DOTALL).group())
                entity_type = data.get('entity_type', 'law_firm')
                conf = data.get('confidence', 0.8)
                reason = data.get('reasoning', '')
            except Exception as e:
                logger.warning(f"   AI parse failed: {e}, fallback to law firm")
                entity_type, conf, reason = 'law_firm', 0.7, 'AI parse failed'

            if entity_type == 'company':
                logger.info(f"   ✅ Detected COMPANY ({conf:.0%})")
                return EntityData(raw_name=raw_name_clean, address=address, attorney_name=attorney_name_clean,
                                  entity_type='company', resolved_firm_name=None, full_name='',
                                  first_name='', last_name='', raw_data={}, entity_confidence=conf,
                                  entity_reasoning=reason)

            # Default = law firm
            logger.info(f"   ✅ Detected LAW FIRM ({conf:.0%})")
            parts = attorney_name_clean.split()
            return EntityData(raw_name=raw_name_clean, address=address, attorney_name=attorney_name_clean,
                              entity_type='law_firm', resolved_firm_name=raw_name_clean,
                              full_name=attorney_name_clean, first_name=parts[0] if parts else '',
                              last_name=parts[-1] if len(parts) > 1 else '', raw_data={},
                              entity_confidence=conf, entity_reasoning=reason)

        # ------------------------------------------
        # CASE 2: Structured address (Attorney + Firm)
        # ------------------------------------------
        attorney_from_addr, firm_from_addr = self.extract_attorney_from_structured_address(address)
        if attorney_from_addr and firm_from_addr:
            logger.info("   Case: Structured address → parsed firm/attorney")
            parts = attorney_from_addr.split()
            return EntityData(raw_name=raw_name_clean, address=address, attorney_name=attorney_from_addr,
                              entity_type='law_firm', resolved_firm_name=firm_from_addr,
                              full_name=attorney_from_addr, first_name=parts[0] if parts else '',
                              last_name=parts[-1] if len(parts) > 1 else '', raw_data={},
                              entity_confidence=0.85, entity_reasoning="Structured address parsed")

        # ------------------------------------------
        # CASE 3: Looks like person → reverse lookup
        # ------------------------------------------
        if looks_like_person:
            logger.info("   Case: Looks like individual → reverse lookup")
            return await self.ai_reverse_lookup_individual(raw_name_clean, address)

        # ------------------------------------------
        # CASE 4: Clear cues
        # ------------------------------------------
        if has_law_cue and not has_company_cue:
            logger.info("   Case: Clear LAW FIRM cues")
            return EntityData(raw_name=raw_name_clean, address=address, attorney_name=None,
                              entity_type='law_firm', resolved_firm_name=raw_name_clean,
                              full_name='', first_name='', last_name='', raw_data={},
                              entity_confidence=0.9, entity_reasoning="Law firm cues in name")

        if has_company_cue and not has_law_cue:
            logger.info("   Case: Clear COMPANY cues")
            return EntityData(raw_name=raw_name_clean, address=address, attorney_name=None,
                              entity_type='company', resolved_firm_name=None,
                              full_name='', first_name='', last_name='', raw_data={},
                              entity_confidence=0.9, entity_reasoning="Company cues in name")

        # ------------------------------------------
        # CASE 5: Ambiguous → Global fallback layer
        # ------------------------------------------
        logger.info("   Case: Ambiguous entity → checking global company identifiers")

        if any(cue in lower_name for cue in global_company_identifiers):
            logger.info("🤖 Fallback: Global corporate keyword → COMPANY")
            return EntityData(raw_name=raw_name_clean, address=address, attorney_name=None,
                              entity_type='company', resolved_firm_name=None,
                              full_name='', first_name='', last_name='', raw_data={},
                              entity_confidence=0.8, entity_reasoning="Global corporate identifier fallback")

        # ------------------------------------------
        # CASE 6: Default → Treat as law firm
        # ------------------------------------------
        logger.info("⚖️ Final fallback: Defaulting to LAW FIRM (domain specialization)")
        return EntityData(raw_name=raw_name_clean, address=address, attorney_name=None,
                          entity_type='law_firm', resolved_firm_name=raw_name_clean,
                          full_name='', first_name='', last_name='', raw_data={},
                          entity_confidence=0.6, entity_reasoning="Final fallback to law firm")


    def looks_like_person_name(self, name: str) -> bool:
        """
        Check if name looks like a person (First Last or Last, First format)
        FIX 5 & 8: Improved to handle complex address formats with NAME= prefix
        """
        # FIX 8: Handle complex address formats
        # Example: "De Vries & Metman " with address "NAME=AALBERS, Arnt Reinier..."
        # This is NOT a person name, it's a firm name
        
        # If name contains firm indicators, definitely not a person
        firm_indicators = ['&', 'LLP', 'LLC', 'P.C.', 'PC', 'Ltd', 'Limited']
        if any(indicator in name for indicator in firm_indicators):
            return False
        
        # Check for comma (Last, First format)
        if ',' in name:
            parts = name.split(',')
            if len(parts) == 2:
                # Both parts should be relatively short
                if all(len(p.strip().split()) <= 3 for p in parts):
                    # Additional check: not firm name parts
                    if not any(indicator in name for indicator in ['&', 'and']):
                        return True
        
        # Check for simple First Last format
        words = name.split()
        if 2 <= len(words) <= 4:
            # Check if has any law/company indicators
            name_lower = name.lower()
            if not any(cue in name_lower for cue in LAW_FIRM_CUES + COMPANY_CUES):
                # Additional validation: not multiple surnames with &
                if '&' not in name:
                    return True
        
        return False
    
    def extract_attorney_from_structured_address(self, address: str) -> tuple:
        """
        FIX 5 & 8: Extract attorney name from structured address format
        Example: "NAME=AALBERS, Arnt Reinier De Vries & Metman..."
        Returns: (attorney_name, firm_name or None)
        """
        if 'NAME=' not in address:
            return None, None
        
        try:
            # Extract the NAME= portion
            match = re.search(r'NAME=([^,]+,\s*[^\s]+(?:\s+[^\s]+)?)', address)
            if match:
                attorney_name = match.group(1).strip()
                
                # Try to find firm name after attorney name
                remaining = address[match.end():].strip()
                
                # Look for firm indicators
                firm_match = re.search(r'([A-Z][a-zA-Z\s&]+(?:LLP|LLC|P\.C\.|PC|Ltd))', remaining)
                if firm_match:
                    firm_name = firm_match.group(1).strip()
                    return attorney_name, firm_name
                
                return attorney_name, None
            
        except Exception as e:
            logger.debug(f"      Structured address parsing failed: {e}")
        
        return None, None
    
    async def ai_reverse_lookup_individual(self, person_name: str, address: str) -> EntityData:
        """
        Reverse lookup: Individual → Find their law firm
        FIX: Enhanced with multiple search strategies and better AI prompts
        Example 3 from document: ACHTSAM, Jessica L. → Knobbe Martens
        """
        logger.info(f"   🔍 ENHANCED REVERSE LOOKUP for: {person_name}")
        
        # Normalize name (handle Last, First format)
        if ',' in person_name:
            parts = person_name.split(',', 1)
            normalized_name = f"{parts[1].strip()} {parts[0].strip()}"
        else:
            normalized_name = person_name
        
        # Extract location for better searches
        location = self.parse_location_from_address(address)
        city = location.get('city', '')
        state = location.get('state', '')
        
        # FIX: Multiple search strategies
        search_strategies = [
            f'"{person_name}" "{address}" attorney',
            f'"{normalized_name}" {city} attorney' if city else None,
            f'"{normalized_name}" {city} lawyer law firm' if city else None,
            f'"{normalized_name}" {state} patent attorney' if state and not city else None,
            f'"{normalized_name}" attorney profile',
        ]
        search_strategies = [s for s in search_strategies if s]
        
        all_results = []
        for i, query in enumerate(search_strategies, 1):
            logger.info(f"      Strategy {i}/{len(search_strategies)}: {query[:60]}...")
            try:
                results = list(self.ddgs.text(query, max_results=5))
                if results:
                    all_results.extend(results)
                    logger.info(f"         → {len(results)} results")
                    if len(all_results) >= 10:
                        break
                await asyncio.sleep(2)
            except:
                continue
        
        if not all_results:
            logger.warning("   ❌ No results from any strategy")
            name_parts = normalized_name.split()
            return EntityData(
                raw_name=person_name, address=address, attorney_name=None,
                entity_type='individual', resolved_firm_name=None,
                full_name=normalized_name,
                first_name=name_parts[0] if name_parts else '',
                last_name=name_parts[-1] if len(name_parts) > 1 else '',
                raw_data={}, entity_confidence=0.5,
                entity_reasoning="Individual - no search results"
            )
        
        # Deduplicate
        seen = set()
        unique_results = []
        for r in all_results:
            url = r.get('href', '')
            if url and url not in seen:
                seen.add(url)
                unique_results.append(r)
        
        logger.info(f"   ✅ Collected {len(unique_results)} unique results")
        
        # Enhanced AI prompt
        results_text = "\n".join([
            f"{i+1}. Title: {r.get('title', 'N/A')}\n"
            f"   URL: {r.get('href', 'N/A')}\n"
            f"   Snippet: {r.get('body', '')[:250]}"
            for i, r in enumerate(unique_results[:10])
        ])
        
        prompt = f"""Expert task: Identify law firm for this attorney.

ATTORNEY: {person_name} → {normalized_name}
LOCATION: {city}, {state} ({address})

SEARCH RESULTS:
{results_text}

TASK: Find the LAW FIRM name.

LOOK FOR:
1. Phrases: "Partner at [Firm]", "[Name] - [Firm]", "Attorney at [Firm]"
2. Firm indicators: LLP, PC, P.C., Law Offices, & Associates
3. URL domains: firmname.com (NOT justia/avvo/linkedin)
4. Location match: same city/state - THIS IS CRITICAL!

CRITICAL VALIDATION:
- The firm MUST be in the same city/state as the attorney
- If location doesn't match, confidence should be LOW (0.3-0.5)
- If location matches, confidence should be HIGH (0.8-0.9)

ANALYZE ALL RESULTS CAREFULLY!

Return JSON:
{{
    "firm_found": true/false,
    "firm_name": "EXACT firm name" or null,
    "confidence": 0.0-1.0,
    "reasoning": "specific phrases/URLs showing firm name AND location match",
    "location_match": true/false (does firm location match attorney location?),
    "source_index": which result number (1-10)
}}"""
        
        result = await self.llm_query(prompt, max_tokens=500)
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            data = json.loads(json_match.group() if json_match else result)
            
            if data.get('firm_found') and data.get('firm_name'):
                firm_name = data['firm_name'].strip()
                confidence = data.get('confidence', 0.7)
                location_match = data.get('location_match', False)
                
                # FIX 4: LOCATION VALIDATION
                # Lower confidence if location doesn't match
                if not location_match and confidence > 0.5:
                    logger.warning(f"   ⚠️ Location mismatch detected, lowering confidence")
                    confidence = min(confidence, 0.55)
                
                if len(firm_name) > 3 and firm_name.lower() not in ['none', 'n/a']:
                    logger.info(f"   ✅ FIRM FOUND: {firm_name} (conf: {confidence:.0%})")
                    if location_match:
                        logger.info(f"      ✅ Location match confirmed")
                    else:
                        logger.warning(f"      ⚠️ Location may not match")
                    
                    name_parts = normalized_name.split()
                    return EntityData(
                        raw_name=person_name, address=address,
                        attorney_name=normalized_name,
                        entity_type='law_firm',
                        resolved_firm_name=firm_name,
                        full_name=normalized_name,
                        first_name=name_parts[0] if name_parts else '',
                        last_name=name_parts[-1] if len(name_parts) > 1 else '',
                        raw_data={}, entity_confidence=confidence,
                        entity_reasoning=f"Reversed: {data.get('reasoning', '')[:150]}"
                    )
        except Exception as e:
            logger.error(f"   Parse error: {e}")
        
        logger.warning("   ⚠️ Firm not identified")
        name_parts = normalized_name.split()
        return EntityData(
            raw_name=person_name, address=address, attorney_name=None,
            entity_type='individual', resolved_firm_name=None,
            full_name=normalized_name,
            first_name=name_parts[0] if name_parts else '',
            last_name=name_parts[-1] if len(name_parts) > 1 else '',
            raw_data={}, entity_confidence=0.6,
            entity_reasoning="Individual - firm not identified"
        )

    
    async def ai_web_search_entity_type(self, name: str, address: str) -> EntityData:
        """
        Web search to determine entity type for ambiguous cases
        Handles: LLP verification, ambiguous names (Example 2, 4)
        """
        logger.info(f"   🔍 Web search for entity type: {name}")
        
        search_query = f'"{name}" "{address}"'
        
        try:
            results = list(self.ddgs.text(search_query, max_results=5))
            
            if not results:
                logger.warning("   No web results, defaulting to law firm (per document)")
                return EntityData(
                    raw_name=name,
                    address=address,
                    attorney_name=None,
                    entity_type='law_firm',  # Default per document
                    resolved_firm_name=name,
                    full_name='',
                    first_name='',
                    last_name='',
                    raw_data={},
                    entity_confidence=0.5,
                    entity_reasoning="No web results, defaulted to law firm"
                )
            
            # Get first result URL for deeper check
            first_url = results[0].get('href', '')
            
            results_text = "\n".join([
                f"{i+1}. {r.get('title', '')}\n   URL: {r.get('href', '')}\n   {r.get('body', '')[:200]}"
                for i, r in enumerate(results[:5])
            ])
            
            prompt = f"""You are an entity classification expert.

ENTITY: {name}
ADDRESS: {address}

SEARCH RESULTS:
{results_text}

Task: Determine if this is a LAW FIRM or COMPANY.

CRITICAL RULES (from document):
1. Law Firm indicators: legal services, attorneys, IP services, patents, trademarks
2. Company indicators: products, manufacturing, technology, healthcare (non-legal)
3. LLP: Check website → legal services = Law Firm, otherwise = Company
4. If ambiguous after checks → DEFAULT to Law Firm

Return JSON:
{{
    "entity_type": "law_firm" or "company",
    "confidence": 0.0-1.0,
    "reasoning": "detailed explanation including website analysis"
}}"""
            
            result = await self.llm_query(prompt, max_tokens=400)
            
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            data = json.loads(json_match.group() if json_match else result)
            
            entity_type = data.get('entity_type', 'law_firm')
            confidence = data.get('confidence', 0.7)
            reasoning = data.get('reasoning', '')
            
            logger.info(f"   ✅ Classified as: {entity_type.upper()} (confidence: {confidence:.0%})")
            
            if entity_type == 'company':
                return EntityData(
                    raw_name=name,
                    address=address,
                    attorney_name=None,
                    entity_type='company',
                    resolved_firm_name=None,
                    full_name='',
                    first_name='',
                    last_name='',
                    raw_data={},
                    entity_confidence=confidence,
                    entity_reasoning=reasoning
                )
            else:
                return EntityData(
                    raw_name=name,
                    address=address,
                    attorney_name=None,
                    entity_type='law_firm',
                    resolved_firm_name=name,
                    full_name='',
                    first_name='',
                    last_name='',
                    raw_data={},
                    entity_confidence=confidence,
                    entity_reasoning=reasoning
                )
                
        except Exception as e:
            logger.error(f"   Web search failed: {e}, defaulting to law firm")
            return EntityData(
                raw_name=name,
                address=address,
                attorney_name=None,
                entity_type='law_firm',
                resolved_firm_name=name,
                full_name='',
                first_name='',
                last_name='',
                raw_data={},
                entity_confidence=0.5,
                entity_reasoning="Web search failed, defaulted to law firm"
            )
    # ============================================================================
    # STAGE 2: WEBSITE SEARCH & VALIDATION  (AI Website Finder v5.3 Integration)
    # ============================================================================

    async def ai_search_and_validate_website(self, entity: EntityData, context: SearchContext) -> Optional[str]:
        """
        Integrates the external AI Website Finder (v5.3) to get the firm's official website.
        Called inside process_record() before email crawling.
        """
        firm_name = entity.resolved_firm_name or entity.raw_name
        address   = entity.address or ""

        if not firm_name:
            logger.warning("⚠️ Missing firm name — skipping website search")
            return None

        try:
            logger.info(f"🔍 Finding official website for: {firm_name}")

            # Use the new AI Website Finder
            result = find_official_website(firm_name, address)

            best_url   = result.get("best_url")
            reason     = result.get("reason", "")
            confidence = result.get("confidence", 0.0)

            if not best_url:
                logger.warning(f"⚠️ No official website found for {firm_name}")
                return None

            # Log the reasoning
            self.log_ai_decision(
                "Website Finder (v5.3)",
                f"Selected: {best_url}",
                confidence,
                reason
            )

            # Attach to entity/context
            entity.official_website    = best_url
            context.website_confidence = confidence
            context.website_reason     = reason

            logger.info(f"✅ Verified Website: {best_url}")
            return best_url

        except Exception as e:
            logger.error(f"❌ Website finder failed for {firm_name}: {e}")
            return None



    # ============================================================================
    # NEW STAGE 3: PROFESSIONAL IDENTIFICATION (Per Document Requirements)
    # ============================================================================
    
    # ============================================================================
    # STAGE 3: PROFESSIONAL IDENTIFICATION (Final – Corrected Logic)
    # ============================================================================
    
    async def ai_identify_professionals(
        self, base_url: str, entity: EntityData, context: SearchContext
    ) -> List[Dict]:
        """
        Stage 3 intelligently chooses between two routes:
          A. Attorney name provided  → find exact profile
          B. No attorney name        → identify relevant professional(s)
        """

        logger.info(f"\n{'='*80}")
        logger.info("STAGE 3: PROFESSIONAL IDENTIFICATION")
        logger.info(f"Purpose: {context.purpose.upper()}")
        logger.info(f"{'='*80}")

        # Find and store people directory URL for Stage 4
        self.people_directory_url = await self.find_people_section(base_url)
        if self.people_directory_url:
            logger.info(f"   ✅ Found people directory: {self.people_directory_url}")
        else:
            logger.warning(f"   ⚠️ Could not find people directory, using base URL")
            self.people_directory_url = base_url

        # -------------------------------------------------------------
        # CASE A – Attorney name explicitly provided
        # -------------------------------------------------------------
        if entity.attorney_name and entity.attorney_name.strip():
            name = entity.attorney_name.strip()
            logger.info(f"   Case A: Attorney name provided → {name}")
            logger.info(f"   Searching directly for profile...")
    
            profile_url = await self.search_attorney_profile(base_url, entity)
            if profile_url:
                logger.info(f"   ✅ Found attorney profile: {profile_url}")
                return [{
                    "name": name,
                    "profile_url": profile_url,
                    "designation": "Provided Attorney",
                    "practice_area": context.purpose,
                    "search_method": "direct_profile_search"
                }]
            else:
                logger.warning("   ⚠️ Attorney profile not found; stopping at Stage 3 for this record.")
                # If we really want a fallback, uncomment below line:
                # return await self.find_relevant_professionals(base_url, entity, context)
                return []  # no further guessing when attorney given but not found
    
        # -------------------------------------------------------------
        # CASE B – No attorney name: discover relevant professionals
        # -------------------------------------------------------------
        logger.info(f"   Case B: No attorney name provided → discover by purpose: {context.purpose}")
        logger.info(f"   Target practice areas: {', '.join(context.target_practice_areas)}")
        logger.info(f"   Target designations: {', '.join(context.preferred_designations[:3])}...")
    
        professionals = await self.find_relevant_professionals(base_url, entity, context)
        if professionals:
            logger.info(f"   ✅ Found {len(professionals)} relevant professional(s)")
            for i, prof in enumerate(professionals, 1):
                logger.info(f"      [{i}] {prof.get('name', 'Unknown')} - {prof.get('designation', 'N/A')}")
        else:
            logger.warning("   ⚠️ No relevant professionals identified")
    
        return professionals
    
    
    async def search_attorney_profile(self, base_url: str, entity: EntityData) -> Optional[str]:
        """
        Search for specific attorney's profile on firm website
        Example 1 from document: Find Anthony J. Lombardi at Finnegan
        """
        attorney_name = entity.attorney_name
        firm_name = entity.resolved_firm_name or entity.raw_name
        
        # Try direct web search first
        search_query = f'"{attorney_name}" site:{urlparse(base_url).netloc}'
        logger.info(f"   Search query: {search_query}")
        
        try:
            results = list(self.ddgs.text(search_query, max_results=5))
            
            if results:
                # Filter for profile pages
                for result in results:
                    url = result.get('href', '')
                    title = result.get('title', '').lower()
                    snippet = result.get('body', '').lower()
                    
                    # Check if it's a profile page
                    profile_indicators = ['profile', 'bio', 'attorney', 'people', 'team', 'professional']
                    if any(indicator in url.lower() for indicator in profile_indicators):
                        if attorney_name.lower() in title or attorney_name.lower() in snippet:
                            logger.info(f"   ✅ Found profile via search: {url}")
                            return url
            
            await asyncio.sleep(2)
        except Exception as e:
            logger.warning(f"   Profile search failed: {e}")
        
        # Fallback: Try common profile page patterns
        common_paths = [
            f'/people/{attorney_name.lower().replace(" ", "-")}',
            f'/attorneys/{attorney_name.lower().replace(" ", "-")}',
            f'/professionals/{attorney_name.lower().replace(" ", "-")}',
            f'/team/{attorney_name.lower().replace(" ", "-")}',
        ]
        
        for path in common_paths:
            test_url = urljoin(base_url, path)
            try:
                response = requests.get(test_url, timeout=10, allow_redirects=True)
                if response.status_code == 200 and attorney_name.lower() in response.text.lower():
                    logger.info(f"   ✅ Found profile via pattern: {test_url}")
                    return test_url
            except:
                pass
        
        return None
    
    async def find_relevant_professionals(self, base_url: str, entity: EntityData,
                                         context: SearchContext) -> List[Dict]:
        """
        Find relevant professionals when no specific attorney name provided
        Implements document logic:
        - Navigate to People/Team/Professionals section
        - Filter by practice area (Patent/Trademark)
        - Filter by location
        - Prioritize by designation
        - Return top 2 if multiple at same level
        """
        logger.info(f"   🔍 Analyzing website structure...")
        
        # Step 1: Find the "People" section
        people_page_url = await self.find_people_section(base_url)
        
        if not people_page_url:
            logger.warning("   ⚠️ Could not locate People/Professionals section")
            return []
        
        logger.info(f"   ✅ Found people section: {people_page_url}")
        
        # Step 2: Analyze people page structure
        html = await self.fetch_page(people_page_url)
        if not html:
            logger.warning("   ⚠️ Could not load people page")
            return []
        
        # Step 3: AI analyzes page and identifies professionals
        professionals = await self.ai_extract_professionals_from_page(
            html, people_page_url, entity, context
        )
        
        return professionals
    
    async def find_people_section(self, base_url: str) -> Optional[str]:
        """
        Find the People/Team/Professionals section of the website
        Per document: "People," "Team", "Professionals," "About us" or "Attorneys"
        """
        html = await self.fetch_page(base_url)
        if not html:
            return None
        
        soup = BeautifulSoup(html, 'html.parser')
        
        # Look for navigation links
        people_keywords = [
            'people', 'team', 'professionals', 'attorneys', 'lawyers',
            'our people', 'our team', 'our attorneys', 'our professionals',
            'about us', 'who we are', 'staff', 'members'
        ]
        
        # Check all links
        for link in soup.find_all('a', href=True):
            text = link.get_text(strip=True).lower()
            href = link['href']
            
            if any(keyword in text for keyword in people_keywords):
                full_url = urljoin(base_url, href)
                logger.info(f"   Found potential people section: {text} → {full_url}")
                return full_url
        
        # Try common paths
        common_paths = [
            '/people', '/team', '/professionals', '/attorneys', '/lawyers',
            '/our-people', '/our-team', '/our-attorneys', '/about/people',
            '/about/team', '/en/people', '/en/team', '/en/professionals'
        ]
        
        for path in common_paths:
            test_url = urljoin(base_url, path)
            try:
                response = requests.get(test_url, timeout=10)
                if response.status_code == 200:
                    return test_url
            except:
                pass
        
        return base_url  # Fallback to homepage
    
    async def ai_extract_professionals_from_page(self, html: str, page_url: str,
                                                 entity: EntityData, context: SearchContext) -> List[Dict]:
        """
        AI extracts and filters professionals from people page
        Implements filtering logic from document
        """
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text()[:8000]  # Limit for token management
        
        # Extract any structured data
        people_links = []
        for link in soup.find_all('a', href=True):
            text_content = link.get_text(strip=True)
            if len(text_content) > 5 and len(text_content) < 100:
                # Looks like a person name
                if not any(word in text_content.lower() for word in ['practice', 'office', 'industry', 'service']):
                    people_links.append({
                        'text': text_content,
                        'url': urljoin(page_url, link['href'])
                    })
        
        people_info = "\n".join([f"- {p['text']}: {p['url']}" for p in people_links[:50]])
        
        location_filter = ""
        if context.location_from_address.get('city'):
            location_filter = f"City: {context.location_from_address['city']}"
        elif context.location_from_address.get('state'):
            location_filter = f"State: {context.location_from_address['state']}"
        
        prompt = f"""You are a legal professional identifier. Analyze this law firm's people page.

FIRM: {entity.resolved_firm_name or entity.raw_name}
PURPOSE: {context.purpose.upper()} search
PAGE URL: {page_url}

TARGET PRACTICE AREAS: {', '.join(context.target_practice_areas)}
PREFERRED DESIGNATIONS (priority order): {', '.join(context.preferred_designations)}
{f"LOCATION FILTER: {location_filter}" if location_filter else ""}

PEOPLE/LINKS FOUND:
{people_info if people_info else "No structured links found"}

PAGE EXCERPT:
{text[:3000]}

TASK:
1. Identify professionals working in {context.purpose.upper()} practice
2. Filter by location if multiple offices: {location_filter}
3. Prioritize by designation (prefer Senior Partner > Partner > Principal > etc.)
4. If multiple at highest designation, return TOP 2

CRITICAL: Look for:
- Practice area mentions: {', '.join(context.target_practice_areas[:3])}
- Designations: {', '.join(context.preferred_designations[:5])}
- Person names with relevant credentials

Return JSON:
{{
    "professionals": [
        {{
            "name": "full name",
            "designation": "title/role",
            "practice_area": "relevant practice",
            "profile_url": "URL if found",
            "confidence": 0.0-1.0,
            "reasoning": "why selected"
        }}
    ],
    "method": "how professionals were identified"
}}

Return UP TO 2 professionals. If none found, return empty array."""
        
        result = await self.llm_query(prompt, max_tokens=800, temperature=0.3)
        
        try:
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            data = json.loads(json_match.group() if json_match else result)
            
            professionals = data.get('professionals', [])
            method = data.get('method', '')
            
            if professionals:
                self.log_ai_decision(
                    "Professional Identifier",
                    f"Found {len(professionals)} professional(s)",
                    0.8,
                    method
                )
            
            # Ensure profile URLs are absolute
            for prof in professionals:
                url = prof.get('profile_url', '')
                if url and not url.startswith('http'):
                    prof['profile_url'] = urljoin(page_url, url)
                prof['search_method'] = 'ai_page_analysis'
            
            return professionals[:2]  # Maximum 2
            
        except Exception as e:
            logger.error(f"   Failed to extract professionals: {e}")
            return []
    
    # ---------------------------------------------------------------------------
    # Helper functions for Stage 4 + Stage 5 integration
    # ---------------------------------------------------------------------------
    def _run_email_finder_safely(func, firm_url, person_name):
        """
        Runs the synchronous email_extraction_stage.main() safely and captures
        its printed RESULT (direct) dictionary from stdout.
        """
        import io, sys, re, json
        buf, old = io.StringIO(), sys.stdout
        sys.stdout = buf
        try:
            # Call the provided main() function directly
            func()
        except SystemExit:
            pass
        finally:
            sys.stdout = old

        text = buf.getvalue()
        m = re.search(r"RESULT \(direct\):\s*(\{.*\})", text, re.S)
        if not m:
            return {}
        try:
            raw = m.group(1)
            return json.loads(raw.replace("'", '"'))
        except Exception:
            return {}


    def _make_general_fallback(firm_url):
        """
        Generates a simple fallback general contact email if no personal
        email was found for the firm.
        """
        from urllib.parse import urlparse
        domain = urlparse(firm_url).netloc.replace("www.", "")
        return {
            "email": f"info@{domain}",
            "source_url": firm_url,
            "confidence": 0.5,
            "reasoning": "Generated fallback general firm email",
            "validation_method": "AutoFallback",
            "type": "firm_general",
        }
    # ---------------------------------------------------------------------------
    async def external_email_extractor(self, homepage_url: str, person_name: str) -> dict:
        """
        Run the synchronous email_extraction_stage.find_email_from_site()
        safely inside a background thread to avoid Playwright async conflicts.
        """
        try:
            loop = asyncio.get_running_loop()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = await loop.run_in_executor(
                    pool,
                    lambda: find_email_from_site(homepage_url, person_name)
                )

            if not results:
                return {'email': None}

            best = sorted(results, key=lambda r: r.get('confidence', 0), reverse=True)[0]

            return {
                'email': best.get('email'),
                'source_url': best.get('url', homepage_url),
                'confidence': best.get('confidence', 0.8),
                'context': best.get('context', '')[:250],
                'validation_method': 'external_import',
                'type': 'professional'
            }

        except Exception as e:
            logger.error(f"❌ External extractor failed: {e}")
            return {'email': None}

    # ============================================================================
    # STAGE 4: AI EMAIL EXTRACTION WITH AUTO PEOPLE-PAGE DISCOVERY + SAFE JSON
    # ============================================================================
    # ============================================================================
    # STAGE 4: EMAIL EXTRACTION  –  USE MAIN LOGIC FROM email_extraction_stage.py
    # ============================================================================
  
    # ============================================================================
    # STAGE 5: INTELLIGENT CRAWLING (Priority: Profile → Homepage → Fallback)
    # ============================================================================

    async def ai_intelligent_crawl(self, base_url: str, entity: EntityData,
                                   context: SearchContext, professionals: List[Dict]) -> Dict:
        """
        STAGE 4 & 5: Email extraction using Universal Email Agent v5
        Uses the people directory URL found in Stage 3.
        """
        # Use the directory URL stored from Stage 3, fallback to base_url
        search_url = self.people_directory_url if self.people_directory_url else base_url

        logger.info(f"📬 Stage 4+5: Using Universal Email Agent v5")
        logger.info(f"   Directory URL: {search_url}")

        result = {
            'personal_email': None,
            'general_email': None,
            'professionals_found': professionals
        }

        # Priority 1: If attorney name provided
        if entity.attorney_name:
            logger.info(f"   Searching for: {entity.attorney_name}")
            email_data = await self.external_email_extractor(search_url, entity.attorney_name)
            if email_data.get('email'):
                result['personal_email'] = email_data
                return result

        # Priority 2: Try identified professionals
        if professionals:
            for prof in professionals[:2]:
                name = prof.get('name')
                if not name:
                    continue
                logger.info(f"   Searching for: {name}")
                email_data = await self.external_email_extractor(search_url, name)
                if email_data.get('email'):
                    result['personal_email'] = email_data
                    return result

        # Priority 3: Last fallback - firm-wide search
        logger.info(f"   Fallback: Searching for general firm contact")
        email_data = await self.external_email_extractor(search_url, entity.resolved_firm_name or entity.raw_name)
        if email_data.get('email'):
            result['general_email'] = email_data
        else:
            logger.warning("⚠️ No email found after full fallback search")

        return result


    
    # ============================================================================
    # UTILITIES
    # ============================================================================
    def safe_json_parse(self, text: str):
        """Parse AI JSON response safely, handling malformed replies and partial JSON."""
        if not text:
            return {}

        # Try direct load first
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try to locate JSON block inside text
        try:
            match = re.search(r'\{[\s\S]*\}|\[[\s\S]*\]', text)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

        # Attempt cleanup of typical AI output issues
        try:
            cleaned = text.strip()
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            cleaned = re.sub(r'[\n\r\t]+', ' ', cleaned)
            if cleaned.startswith("{") or cleaned.startswith("["):
                return json.loads(cleaned)
        except Exception:
            pass

        # If all fails, return empty dict
        logger.warning("⚠️ safe_json_parse ultimate fallback: returning empty object")
        return {}

    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content - EXISTING"""
        try:
            response = requests.get(url, timeout=15, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }, allow_redirects=True)
            if response.status_code == 200:
                return response.text
        except:
            pass
        return None
    
    # ============================================================================
    # MAIN PIPELINE
    # ============================================================================
    
    async def process_record(self, record_data: Dict[str, str], purpose: str) -> Dict:
        """
        REVISED: Main pipeline execution per document workflow
        
        Workflow:
        1. Detect entity type (Law Firm / Company / Individual)
        2. If Company → Stop
        3. If Law Firm or Individual → Find website
        4. Identify professionals (specific attorney or relevant professional)
        5. Extract emails
        """
        self.decisions_log = []
        
        logger.info(f"\n{'='*100}")
        logger.info(f"🎯 PROCESSING NEW RECORD")
        logger.info(f"{'='*100}")
        
        # Extract raw data
        raw_name = record_data.get('Name', record_data.get('Representative', ''))
        address = record_data.get('Address', record_data.get('Representative address', ''))
        attorney_name = record_data.get('Attorney Name', record_data.get('Agent  name', ''))
        
        # Clean attorney name
        if pd.isna(attorney_name) or not attorney_name or str(attorney_name).strip().lower() in ['nan', 'none', '']:
            attorney_name = None
        else:
            attorney_name = str(attorney_name).strip()
        
        result = {
            'raw_data': record_data,
            'purpose': purpose,
            'entity': None,
            'firm_website': None,
            'professionals_identified': [],
            'professional_email': None,
            'firm_general_email': None,
            'final_email': None,
            'status': 'started',
            'ai_decisions': []
        }
        
        # STAGE 1: Entity Type Detection
        entity = await self.ai_detect_entity_type(raw_name, address, attorney_name)
        result['entity'] = asdict(entity)
        
        # Check if company
        if entity.entity_type == 'company':
            logger.info(f"\n{'='*80}")
            logger.info(f"✋ COMPANY DETECTED - NO FURTHER ACTION")
            logger.info(f"Entity: {entity.raw_name}")
            logger.info(f"{'='*80}\n")
            
            result['status'] = 'company_skipped'
            result['ai_decisions'] = self.decisions_log.copy()
            return result
        
        # For law firms and individuals (resolved to firms)
        if entity.entity_type not in ['law_firm']:
            logger.warning(f"⚠️ Entity type '{entity.entity_type}' - treating as unresolved")
            result['status'] = 'entity_unresolved'
            result['ai_decisions'] = self.decisions_log.copy()
            return result
        
        # Create search context
        context = self.create_search_context(purpose, address)
        
        # STAGE 2: Find firm website
        logger.info(f"\n{'='*80}")
        logger.info(f"STAGE 2: WEBSITE DISCOVERY")
        logger.info(f"{'='*80}")
        
        firm_url = await self.ai_search_and_validate_website(entity, context)
        
        if not firm_url:
            result['status'] = 'no_firm_website'
            result['ai_decisions'] = self.decisions_log.copy()
            return result
        
        result['firm_website'] = firm_url
        logger.info(f"✅ Verified Website: {firm_url}")
        
        # STAGE 3: Identify professionals
        professionals = await self.ai_identify_professionals(firm_url, entity, context)
        result['professionals_identified'] = professionals
        
        # STAGE 4 & 5: Email extraction via intelligent crawl
        logger.info(f"\n{'='*80}")
        logger.info(f"STAGE 4: EMAIL EXTRACTION")
        logger.info(f"{'='*80}")
        
        emails = await self.ai_intelligent_crawl(firm_url, entity, context, professionals)
        
        if emails['personal_email']:
            result['professional_email'] = emails['personal_email']
            result['final_email'] = emails['personal_email']
            
            if entity.attorney_name:
                result['status'] = 'found_attorney_email'
            else:
                result['status'] = f'found_{context.purpose}_professional'
            
            if emails['general_email']:
                result['firm_general_email'] = emails['general_email']
            
        elif emails['general_email']:
            result['firm_general_email'] = emails['general_email']
            result['final_email'] = emails['general_email']
            result['status'] = 'found_firm_general'
            logger.info(f"✅ Using general email as fallback")
        else:
            result['status'] = 'no_email_found'
        
        result['ai_decisions'] = self.decisions_log.copy()
        
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ RECORD PROCESSING COMPLETE")
        logger.info(f"Status: {result['status']}")
        if result['final_email']:
            logger.info(f"Email: {result['final_email'].get('email', 'N/A')}")
        logger.info(f"{'='*80}\n")
        
        return result
    
    async def process_all(self, records_data: List[Dict[str, str]], purpose: str) -> Dict:
        """Process all records with purpose selection"""
        
        results = {}
        
        logger.info(f"\n{'='*100}")
        logger.info(f"🤖 REVISED AI-DRIVEN EMAIL DISCOVERY PIPELINE")
        logger.info(f"   Purpose: {purpose.upper()}")
        logger.info(f"   Processing {len(records_data)} records")
        logger.info(f"{'='*100}\n")
        
        for i, record_data in enumerate(records_data, 1):
            logger.info(f"\n{'='*100}\n[{i}/{len(records_data)}] PROCESSING RECORD {i}\n{'='*100}")
            
            try:
                result = await self.process_record(record_data, purpose)
                
                # Use name as key
                name_key = record_data.get('Name', record_data.get('Representative', f'Record_{i}'))
                results[str(name_key)] = result
                
                if i < len(records_data):
                    logger.info(f"\n⏸️  Waiting before next record...")
                    await asyncio.sleep(5)
                    
            except KeyboardInterrupt:
                logger.warning("\n⚠️  Interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
                name_key = record_data.get('Name', f'Error_{i}')
                results[str(name_key)] = {'status': 'error', 'error': str(e)}
        
        return results


def print_results(results: Dict, purpose: str):
    """Print results summary"""
    
    print(f"\n{'='*100}")
    print(f"🤖 REVISED AI-DRIVEN EMAIL DISCOVERY - RESULTS")
    print(f"Purpose: {purpose.upper()}")
    print(f"{'='*100}")
    
    companies_skipped = {}
    attorney_emails = {}
    professional_emails = {}
    general_emails = {}
    failed = {}
    
    for name, result in results.items():
        status = result.get('status', '')
        if status == 'company_skipped':
            companies_skipped[name] = result
        elif status == 'found_attorney_email':
            attorney_emails[name] = result
        elif 'professional' in status:
            professional_emails[name] = result
        elif status == 'found_firm_general':
            general_emails[name] = result
        else:
            failed[name] = result
    
    if companies_skipped:
        print(f"\n🏢 COMPANIES SKIPPED ({len(companies_skipped)}):")
        for name in list(companies_skipped.keys())[:10]:
            print(f"   - {name}")
        if len(companies_skipped) > 10:
            print(f"   ... and {len(companies_skipped) - 10} more")
    
    if attorney_emails:
        print(f"\n✅ ATTORNEY EMAILS FOUND ({len(attorney_emails)}):")
        for name, result in attorney_emails.items():
            email_data = result['final_email']
            entity = result.get('entity', {})
            print(f"\n   {name}")
            print(f"   👤 Attorney: {entity.get('full_name', 'N/A')}")
            print(f"   📧 {email_data['email']}")
            print(f"   🎯 Confidence: {email_data['confidence']:.0%}")
            print(f"   🔗 {email_data['source_url']}")
    
    if professional_emails:
        print(f"\n✅ {purpose.upper()} PROFESSIONAL EMAILS ({len(professional_emails)}):")
        for name, result in professional_emails.items():
            email_data = result['final_email']
            print(f"\n   {name}")
            print(f"   📧 {email_data['email']}")
            print(f"   👔 Type: {purpose.title()} Professional")
            print(f"   🎯 Confidence: {email_data['confidence']:.0%}")
    
    if general_emails:
        print(f"\n⚠️  GENERAL EMAILS ({len(general_emails)}):")
        for name, result in general_emails.items():
            email_data = result['final_email']
            print(f"\n   {name}")
            print(f"   📧 {email_data['email']}")
    
    if failed:
        print(f"\n❌ FAILED ({len(failed)}):")
        for name, result in failed.items():
            print(f"   {name}: {result.get('status', 'unknown')}")
    
    total = len(results)
    found = len(attorney_emails) + len(professional_emails) + len(general_emails)
    
    print(f"\n{'='*100}")
    print(f"📊 SUMMARY:")
    print(f"   Total Processed: {total}")
    print(f"   🏢 Companies (Skipped): {len(companies_skipped)}")
    print(f"   ✅ Emails Found: {found}")
    print(f"      - Attorney: {len(attorney_emails)}")
    print(f"      - Professional: {len(professional_emails)}")
    print(f"      - General: {len(general_emails)}")
    print(f"   ❌ Failed: {len(failed)}")
    if total - len(companies_skipped) > 0:
        success_rate = found / (total - len(companies_skipped)) * 100
        print(f"   📈 Success Rate (excluding companies): {success_rate:.1f}%")
    print(f"{'='*100}\n")
    
    # Save JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f'revised_results_{purpose}_{timestamp}.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {filename}\n")


def load_from_excel(file_path: str) -> List[Dict[str, str]]:
    """Load data from Excel - UPDATED for new column names"""
    try:
        df = pd.read_excel(file_path)
        
        # Map columns - support both old and new formats
        records = []
        for _, row in df.iterrows():
            # Try new format first
            name = row.get('Name', row.get('Representative', ''))
            address = row.get('Address', row.get('Representative address', ''))
            attorney = row.get('Attorney Name', row.get('Agent  name', ''))
            
            records.append({
                'Name': str(name),
                'Address': str(address),
                'Attorney Name': str(attorney)
            })
        
        logger.info(f"✅ Loaded {len(records)} records")
        return records
    except Exception as e:
        logger.error(f"Failed to load Excel: {e}")
        return []


def export_to_excel(results: Dict, original_file: str, purpose: str):
    """Export results to Excel - UPDATED"""
    try:
        df = pd.read_excel(original_file)
        
        # Map old to new column names if needed
        column_mapping = {
            'Representative': 'Name',
            'Representative address': 'Address',
            'Agent  name': 'Attorney Name'
        }
        df.rename(columns=column_mapping, inplace=True)
        
        # Ensure output columns exist
        required_cols = ['Salutation', 'Attorney Name', 'Email', 
                        'Email ID type (Personal / General)', 'Webpage link', 'Law Firm / Company']
        for col in required_cols:
            if col not in df.columns:
                df[col] = ''
            df[col] = df[col].astype(str)
        
        for idx, row in df.iterrows():
            name_key = str(row.get('Name', '')).strip()
            
            result = None
            for key, res in results.items():
                if name_key in str(key) or str(key) in name_key:
                    result = res
                    break
            
            if not result or result.get('status') in ['parse_error', 'error', 'company_skipped']:
                if result and result.get('status') == 'company_skipped':
                    df.at[idx, 'Law Firm / Company'] = 'company'
                continue
            
            entity_data = result.get('entity', {})
            final_email = result.get('final_email')
            status = result.get('status', '')
            
            # Set attorney name and salutation
            if entity_data.get('attorney_name'):
                first_name = entity_data.get('first_name', '').lower()
                common_male = ['john', 'william', 'james', 'robert', 'michael', 'david', 
                              'anthony', 'richard', 'paul', 'stephen']
                salutation = 'Mr.' if first_name in common_male else 'Ms.'
                df.at[idx, 'Salutation'] = salutation
                df.at[idx, 'Attorney Name'] = entity_data.get('attorney_name', '')
            
            # Set email info
            if final_email:
                df.at[idx, 'Email'] = final_email.get('email', '')
                df.at[idx, 'Webpage link'] = final_email.get('source_url', '')
                
                if status == 'found_attorney_email':
                    df.at[idx, 'Email ID type (Personal / General)'] = 'Personal - Attorney'
                elif 'professional' in status:
                    df.at[idx, 'Email ID type (Personal / General)'] = f'Personal - {purpose.title()} Professional'
                elif status == 'found_firm_general':
                    df.at[idx, 'Email ID type (Personal / General)'] = 'General'
            
            df.at[idx, 'Law Firm / Company'] = "law firm"
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f'revised_output_{purpose}_{timestamp}.xlsx'
        df.to_excel(output_file, index=False)
        
        logger.info(f"✅ Excel exported: {output_file}")
        print(f"✅ Excel saved: {output_file}\n")
        
    except Exception as e:
        logger.error(f"Excel export failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main execution"""
    
    if not os.getenv('GROQ_API_KEY'):
        print("❌ Set GROQ_API_KEY in .env file")
        return
    
    # PURPOSE SELECTION
    print("\n" + "="*100)
    print("🎯 PURPOSE SELECTION")
    print("="*100)
    print("Select search purpose:")
    print("  1. Patent search")
    print("  2. Trademark search")
    print("="*100)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1':
            purpose = 'patent'
            break
        elif choice == '2':
            purpose = 'trademark'
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    
    print(f"\n✅ Selected: {purpose.upper()} search\n")
    
    pipeline = None
    try:
        file_path = input("📁 Enter Excel file path: ").strip()
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        records_data = load_from_excel(file_path)
        if not records_data:
            return
        
        pipeline = AIEmailDiscoveryPipeline()
        results = await pipeline.process_all(records_data, purpose)
        
        print_results(results, purpose)
        export_to_excel(results, file_path, purpose)
        
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline and hasattr(pipeline, 'crawler'):
            try:
                await pipeline.crawler.aclose()
            except:
                pass


if __name__ == "__main__":
    asyncio.run(main())

# import sys
# sys.path.append(r"D:\invtree\final")
# from email_extraction_stage import find_email_from_site

# # print(find_email_from_site("https://www.finnegan.com", "Anthony J. Lombardi"))
