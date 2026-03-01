"""Test script for SectionWriterAgent with chart creation.

Tests the section writer agent with sample data containing numerical data
suitable for chart visualization.
"""

from pathlib import Path

import dotenv

from src.config import ROOT_DIR
from src.deep_research_agent.agents.section_writer_agent import SectionWriterAgent

dotenv.load_dotenv()

# Output directory
OUTPUT_DIR = ROOT_DIR / "output"


# Sample source content with numerical data suitable for charts
SAMPLE_SOURCE_CONTENT = """
SOURCE [1]:
# Global Electric Vehicle Sales Report 2023

## Annual Sales Data
The global electric vehicle market saw significant growth in 2023:
- Q1 2023: 2.3 million units sold
- Q2 2023: 2.8 million units sold
- Q3 2023: 3.1 million units sold
- Q4 2023: 3.9 million units sold

Total 2023 sales: 12.1 million units (up 35% from 2022's 8.9 million units)

## Regional Market Share
- China: 58% of global EV sales
- Europe: 24% of global EV sales
- North America: 12% of global EV sales
- Rest of World: 6% of global EV sales

## Top Selling Models 2023
1. Tesla Model Y: 1.2 million units
2. BYD Song Plus: 680,000 units
3. Tesla Model 3: 520,000 units
4. BYD Dolphin: 410,000 units
5. Volkswagen ID.4: 320,000 units

---

SOURCE [2]:
# EV Market Projections 2024-2030

## Growth Forecast
Analysts project continued strong growth:
- 2024: 15.5 million units (projected)
- 2025: 19.2 million units (projected)
- 2026: 23.8 million units (projected)
- 2027: 28.1 million units (projected)
- 2028: 32.5 million units (projected)
- 2029: 37.0 million units (projected)
- 2030: 42.0 million units (projected)

## Battery Cost Trends
Average battery pack cost per kWh:
- 2020: $140/kWh
- 2021: $132/kWh
- 2022: $138/kWh (temporary increase due to supply chain)
- 2023: $128/kWh
- 2024: $115/kWh (projected)
- 2025: $100/kWh (projected)

---

SOURCE [3]:
# Charging Infrastructure Development

## Global Public Charging Points
- 2020: 1.3 million public chargers
- 2021: 1.8 million public chargers
- 2022: 2.7 million public chargers
- 2023: 3.9 million public chargers

## Fast Charger Distribution by Region (2023)
- China: 760,000 fast chargers
- Europe: 180,000 fast chargers
- North America: 95,000 fast chargers
- Rest of World: 65,000 fast chargers

Average charging time has decreased from 45 minutes in 2020 to 25 minutes in 2023
for 80% charge on latest fast chargers.

---
"""

SAMPLE_RESEARCH_BRIEF = """
Create a comprehensive analysis of the global electric vehicle market,
including current sales trends, regional distribution, future projections,
and infrastructure development. The report should highlight key data points
and trends that demonstrate the growth trajectory of the EV industry.
"""

SAMPLE_SECTION_NAMES = """1. Introduction
2. Global EV Sales Analysis
3. Regional Market Distribution
4. Future Market Projections
5. Charging Infrastructure
6. Conclusion"""

SAMPLE_PREVIOUS_SECTION = """# Introduction

The electric vehicle industry has undergone a remarkable transformation over the past decade,
evolving from a niche market to a major force in the global automotive sector. This report
examines the current state of the EV market, analyzing sales data, regional trends, and
future projections that illustrate the industry's growth trajectory [1][2].
"""


def test_section_with_chart():
    """Test section writer with data suitable for chart creation."""

    print("=" * 60)
    print("Testing SectionWriterAgent with Chart Creation")
    print("=" * 60)

    # Initialize the section writer agent
    agent = SectionWriterAgent(reasoning="low")

    # Test case: Section that should generate a chart
    section_name = "Global EV Sales Analysis"
    section_description = """
    Write a detailed analysis of global EV sales trends. Include:
    - Quarterly sales breakdown for 2023
    - Year-over-year growth comparison
    - Top selling models and their market performance

    Present the quarterly data in a way that highlights the growth trend throughout the year.
    Use visual elements where appropriate to make the data more digestible.
    """

    print(f"\nWriting section: {section_name}")
    print("-" * 40)

    content = agent.write_section(
        research_brief=SAMPLE_RESEARCH_BRIEF,
        section_names=SAMPLE_SECTION_NAMES,
        cur_section=section_name,
        section_description=section_description,
        previous_section=SAMPLE_PREVIOUS_SECTION,
        source_content=SAMPLE_SOURCE_CONTENT,
    )

    print("\n" + "=" * 60)
    print("GENERATED SECTION CONTENT:")
    print("=" * 60)
    print(content)

    # Check if chart was included
    if "![" in content and "](charts/" in content:
        print("\n" + "=" * 60)
        print("SUCCESS: Chart was created and included in the section!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("NOTE: No chart was included in the section.")
        print("=" * 60)

    return content


def test_section_without_chart():
    """Test section writer with qualitative data (no chart expected)."""

    print("\n" + "=" * 60)
    print("Testing SectionWriterAgent WITHOUT Chart (Qualitative Section)")
    print("=" * 60)

    qualitative_source = """
SOURCE [1]:
# EV Industry Challenges and Opportunities

## Key Challenges
- Range anxiety remains a concern for potential buyers
- Charging infrastructure gaps in rural areas
- Higher upfront costs compared to ICE vehicles
- Battery recycling and sustainability concerns
- Grid capacity limitations in some regions

## Emerging Opportunities
- Government incentives and subsidies expanding globally
- Declining battery costs making EVs more affordable
- New entrants increasing competition and innovation
- Vehicle-to-grid technology enabling new revenue streams
- Autonomous driving integration possibilities

---
"""

    agent = SectionWriterAgent(reasoning="low")

    section_name = "Conclusion"
    section_description = """
    Write a concluding section that summarizes the key findings and provides
    a forward-looking perspective on the EV industry. Discuss both the challenges
    and opportunities facing the market.
    """

    print(f"\nWriting section: {section_name}")
    print("-" * 40)

    content = agent.write_section(
        research_brief=SAMPLE_RESEARCH_BRIEF,
        section_names=SAMPLE_SECTION_NAMES,
        cur_section=section_name,
        section_description=section_description,
        previous_section=SAMPLE_PREVIOUS_SECTION,
        source_content=qualitative_source,
    )

    print("\n" + "=" * 60)
    print("GENERATED SECTION CONTENT:")
    print("=" * 60)
    print(content)

    return content


def save_markdown_report(content_with_chart: str, content_without_chart: str):
    """Save the generated content as a markdown file for viewing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    markdown_content = f"""# Section Writer Test Report

This report tests the SectionWriterAgent's ability to create charts when appropriate.

---

{content_with_chart}

---

{content_without_chart}

---

## Sources

- [1] Global Electric Vehicle Sales Report 2023
- [2] EV Market Projections 2024-2030
- [3] Charging Infrastructure Development
"""

    output_path = OUTPUT_DIR / "test_section_writer_output.md"
    output_path.write_text(markdown_content, encoding="utf-8")
    print(f"\nMarkdown report saved to: {output_path}")

    # Also list any generated chart files
    charts_dir = OUTPUT_DIR / "charts"
    if charts_dir.exists():
        print("\nGenerated chart files:")
        for f in charts_dir.iterdir():
            print(f"  - {f.name}")


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# SECTION WRITER AGENT - CHART CREATION TEST")
    print("#" * 60)

    # Test 1: Section with numerical data (should create chart)
    content_with_chart = test_section_with_chart()

    # Test 2: Section with qualitative data (should not create chart)
    content_without_chart = test_section_without_chart()

    # Save as markdown file
    save_markdown_report(content_with_chart, content_without_chart)

    # Summary
    print("\n" + "#" * 60)
    print("# TEST SUMMARY")
    print("#" * 60)

    chart_created = "![" in content_with_chart and "](charts/" in content_with_chart
    print(f"\nTest 1 (Numerical data): Chart {'CREATED' if chart_created else 'NOT created'}")

    chart_in_qualitative = "![" in content_without_chart and "](charts/" in content_without_chart
    print(f"Test 2 (Qualitative data): Chart {'CREATED' if chart_in_qualitative else 'NOT created'}")

    print(f"\nView the markdown report at: {OUTPUT_DIR / 'test_section_writer_output.md'}")


if __name__ == "__main__":
    main()
