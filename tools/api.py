import os
import json
from openai import OpenAI
class DeepSeekAPI:
    SYSTEM_PROMPT = """The user will provide a biomedical document. Your task is to generate a list of up to 4 relevant natural language search queries that someone might use after reading the document. These queries should be phrased like search engine inputs (not questions), and should reflect the main ideas, findings, or concepts discussed in the document.

Only return a JSON object with a single key "query", whose value is a list of up to 4 natural language search queries derived from the document.

EXAMPLE INPUT:

Document: (--)-alpha-Bisabolol has a primary antipeptic action depending on dosage, which is not caused by an alteration of the pH-value. The proteolytic activity of pepsin is reduced by 50 percent through addition of bisabolol in the ratio of 1/0.5. The antipeptic action of bisabolol only occurs in case of direct contact. In case of a previous contact with the substrate, the inhibiting effect is lost.

EXAMPLE JSON OUTPUT:
{
    "query": [
        "antipeptic mechanism of (--)-alpha-bisabolol",
        "pepsin inhibition by (--)-alpha-bisabolol",
        "effect of (--)-alpha-bisabolol on gastric enzymes",
        "dosage-dependent action of (--)-alpha-bisabolol"
    ]
}"""

    def __init__(
        self,
        api_key,
    ):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    def generate_queries(
        self, 
        document,
    ):
        messages = [
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Document: {document}"}]
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            response_format={
                'type': 'json_object'
            }
        )
        return json.loads(response.choices[0].message.content)