from typing import Optional
import os
from dotenv import load_dotenv
from tavily import TavilyClient

class MarketSearch:
    def __init__(self):
        load_dotenv()
        self.client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    def get_related_markets(self, question: str) -> str:
        try:
            # Buscar información relacionada
            search_result = self.client.search(
                query=question,
                search_depth="advanced",
                max_results=5
            )
            
            # Formatear los resultados
            formatted_results = []
            for result in search_result['results']:
                formatted_results.append(f"- {result['title']}: {result['content'][:200]}...")
            
            return "\n".join(formatted_results)
        except Exception as e:
            print(f"Warning: Could not fetch related information: {e}")
            return "No additional market information available."
