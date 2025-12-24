#!/usr/bin/env python3
"""
Email Extraction Stage - Powered by Universal Email Agent v5
=============================================================
This module provides email extraction using the new Universal Email Agent v5.
It maintains backward compatibility with the existing main.py interface.

Interface:
    find_email_from_site(home_url: str, person_name: str) -> List[Dict]

Returns:
    [{'email': '...', 'url': '...', 'context': '...', 'confidence': 0.95}]
"""

import os
import sys
import asyncio
from typing import List, Dict
from dotenv import load_dotenv

# Import the standalone Universal Email Agent v5
from universal_email_agent_v5 import UniversalEmailAgent

load_dotenv()

def find_email_from_site(home_url: str, person_name: str) -> List[Dict]:
    """
    Main interface for email extraction - maintains backward compatibility.
    Uses Universal Email Agent v5 for intelligent email extraction.

    Args:
        home_url: The homepage or people directory URL
        person_name: Full name of the person to find

    Returns:
        List of dicts: [{'email': '...', 'url': '...', 'context': '...', 'confidence': 0.95}]
    """
    print(f"\n🔍 Email Extraction Stage (v5)")
    print(f"Target: {person_name} @ {home_url}")
    print("=" * 70)

    # Create and run Universal Email Agent
    agent = UniversalEmailAgent(home_url, person_name)
    result = asyncio.run(agent.run())

    print("=" * 70)

    if not result:
        print("❌ No email found")
        return []

    # Convert to expected format
    is_general = result.get('is_general_contact', False)
    context = "General contact email" if is_general else f"Found on {person_name}'s profile"

    return_data = [{
        'email': result['email'].lower(),
        'url': result.get('profile_url', home_url),
        'context': context,
        'confidence': result.get('confidence', 95) / 100.0  # Convert to 0-1 scale
    }]

    print(f"✅ Found: {result['email']}")
    if is_general:
        print("   Note: General contact email (not personal)")

    return return_data


# CLI interface for standalone testing
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python email_extraction_stage_v5.py <url> \"Full Name\"")
        sys.exit(1)

    website = sys.argv[1].rstrip("/")
    person = sys.argv[2].strip()

    results = find_email_from_site(website, person)

    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)

    if results:
        for r in results:
            print(f"📧 Email: {r['email']}")
            print(f"🔗 URL: {r['url']}")
            print(f"📊 Confidence: {r['confidence']:.2%}")
            print(f"💬 Context: {r.get('context', 'N/A')}")
    else:
        print("❌ No emails found")

    print("=" * 70)
