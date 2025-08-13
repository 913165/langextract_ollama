import langextract as lx
import textwrap



# Updated prompt for retail industry (expanded from 3 to 6 extraction categories)
prompt = textwrap.dedent("""\
Extract retail brands, product sentiments, customer feedback, promotional campaigns, pricing strategies, and competitor mentions in order of appearance.
Use exact text for extractions without any modifications, ensuring that the extracted data preserves the full integrity of the original wording.
Do not paraphrase, merge, or overlap entities, and make sure each extracted element stands alone with clear contextual meaning.
Provide deeply meaningful and elaborate attributes for each entity to add richer context, such as category, target market, market positioning, and any unique selling points explicitly mentioned in the text.""")

# Example for retail industry (expanded from 3 to 6 extractions)
examples = [
    lx.data.ExampleData(
        text=(
            "Nike launches a new running shoe, receiving rave reviews for comfort, style, and exceptional durability from athletes and casual users alike. The company also announced a limited-time discount campaign to boost sales, and analysts noted its competitive positioning against Adidas in the premium footwear segment."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="retail_brand",
                extraction_text="Nike",
                attributes={
                    "type": "sportswear",
                    "category": "athletic footwear",
                    "target_market": "professional runners and lifestyle consumers"
                },
            ),
            lx.data.Extraction(
                extraction_class="product_sentiment",
                extraction_text="rave reviews",
                attributes={
                    "direction": "positive",
                    "intensity": "high",
                    "source": "multiple independent reviewers"
                },
            ),
            lx.data.Extraction(
                extraction_class="customer_feedback",
                extraction_text="comfort, style, and exceptional durability",
                attributes={
                    "focus": "product quality",
                    "specific_features": ["ergonomic fit", "modern design", "long-lasting materials"],
                    "audience": "athletes and everyday users"
                },
            ),
            lx.data.Extraction(
                extraction_class="promotional_campaign",
                extraction_text="limited-time discount campaign",
                attributes={
                    "duration": "short-term",
                    "purpose": "boost sales",
                    "discount_type": "percentage off"
                },
            ),
            lx.data.Extraction(
                extraction_class="pricing_strategy",
                extraction_text="discount",
                attributes={
                    "approach": "competitive pricing",
                    "impact": "increased demand"
                },
            ),
            lx.data.Extraction(
                extraction_class="competitor_mention",
                extraction_text="Adidas",
                attributes={
                    "relationship": "market competitor",
                    "segment": "premium footwear"
                },
            ),
        ],
    )
]

# New input text for retail industry (expanded for 6 extraction categories)
input_text = (
    "Apple unveils the latest iPhone model, praised by technology critics and customers alike for its advanced camera system capable of professional-grade photography, its sleek and modern design crafted from aerospace-grade materials, and its seamless integration with other Apple devices for an enhanced ecosystem experience. The company launched an aggressive trade-in promotion to encourage upgrades, adopted a value-based pricing approach, and faced renewed competition from Samsung in the flagship smartphone market."
)

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="llama3.2:1b",  # Automatically selects Ollama provider
    model_url="http://172.27.185.57:11434"
)

# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results_ollama.jsonl")

# Generate the interactive visualization from the file
html_content = lx.visualize("test_output/extraction_results_ollama.jsonl")
with open("visualization_ollama.html", "w", encoding="utf-8") as f:
    f.write(html_content)
