import os
import json
import ast
import re
from typing import List, Dict, Any
import math
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from colorama import Fore, Style
from collections import defaultdict
import spacy  # Para NLP
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # Para vectorización de texto

from agents.polymarket.gamma import GammaMarketClient as Gamma
from agents.connectors.chroma import PolymarketRAG as Chroma
from agents.utils.objects import SimpleEvent, SimpleMarket
from agents.application.prompts import Prompter
from agents.polymarket.polymarket import Polymarket
from agents.connectors.search import MarketSearch

def retain_keys(data, keys_to_retain):
    if isinstance(data, dict):
        return {
            key: retain_keys(value, keys_to_retain)
            for key, value in data.items()
            if key in keys_to_retain
        }
    elif isinstance(data, list):
        return [retain_keys(item, keys_to_retain) for item in data]
    else:
        return data

class Executor:
    def __init__(self, default_model='gpt-3.5-turbo-16k') -> None:
        load_dotenv()
        max_token_model = {'gpt-3.5-turbo-16k':15000, 'gpt-4-1106-preview':95000}
        self.token_limit = max_token_model.get(default_model)
        self.prompter = Prompter()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model=default_model, #gpt-3.5-turbo"
            temperature=0,
        )
        self.gamma = Gamma()
        self.chroma = Chroma()
        self.polymarket = Polymarket()
        self.search = MarketSearch()
        
        # Cargar modelo de spaCy para NLP
        self.nlp = spacy.load("en_core_web_sm")
        
        # Definir categorías y sus palabras clave principales
        self.category_keywords = {
            "sports": {
                "leagues": ["nfl", "nba", "mlb", "nhl", "epl", "uefa", "fifa", "f1"],
                "events": ["super bowl", "world cup", "champions league", "stanley cup", "playoffs"],
                "roles": ["player", "coach", "team", "manager", "mvp", "rookie"],
                "actions": ["win", "score", "lead", "defeat", "qualify", "advance"],
                "metrics": ["points", "goals", "assists", "rebounds", "touchdowns", "yards"],
                "competitions": ["tournament", "championship", "series", "match", "game", "race"]
            },
            "crypto": {
                "assets": ["bitcoin", "ethereum", "solana", "btc", "eth", "sol"],
                "metrics": ["price", "market cap", "volume", "liquidity", "supply"],
                "actions": ["buy", "sell", "trade", "mine", "stake", "yield"],
                "concepts": ["blockchain", "defi", "nft", "dao", "token", "coin"],
                "events": ["halving", "fork", "upgrade", "launch", "listing", "airdrop"]
            },
            "politics": {
                "roles": ["president", "senator", "governor", "minister", "candidate"],
                "events": ["election", "vote", "debate", "campaign", "inauguration"],
                "institutions": ["congress", "senate", "parliament", "fed", "court"],
                "policies": ["bill", "law", "regulation", "reform", "policy"],
                "economic": ["rate", "inflation", "recession", "budget", "debt"]
            },
            "entertainment": {
                "awards": ["oscar", "grammy", "emmy", "golden globe", "tony"],
                "media": ["movie", "film", "song", "album", "show", "series"],
                "roles": ["actor", "actress", "director", "producer", "artist"],
                "events": ["premiere", "release", "concert", "festival", "ceremony"],
                "metrics": ["box office", "ratings", "views", "sales", "streams"]
            },
            "tech": {
                "companies": ["apple", "google", "microsoft", "meta", "openai"],
                "products": ["iphone", "android", "windows", "chatgpt", "tesla"],
                "concepts": ["ai", "cloud", "quantum", "metaverse", "web3"],
                "events": ["launch", "release", "update", "acquisition", "ipo"],
                "metrics": ["revenue", "users", "growth", "valuation", "share"]
            }
        }
        
        # Entrenar vectorizador
        self.vectorizer = self._train_vectorizer()

    def _train_vectorizer(self):
        # Crear corpus de entrenamiento desde keywords
        corpus = []
        labels = []
        for category, keyword_groups in self.category_keywords.items():
            for group in keyword_groups.values():
                corpus.extend(group)
                labels.extend([category] * len(group))
                
        vectorizer = TfidfVectorizer(ngram_range=(1, 3))
        vectorizer.fit(corpus)
        return vectorizer

    def get_llm_response(self, user_input: str) -> str:
        system_message = SystemMessage(content=str(self.prompter.market_analyst()))
        human_message = HumanMessage(content=user_input)
        messages = [system_message, human_message]
        result = self.llm.invoke(messages)
        return result.content

    def get_superforecast(
        self, event_title: str, market_question: str, outcome: str
    ) -> str:
        messages = self.prompter.superforecaster(
            description=event_title, question=market_question, outcome=outcome
        )
        result = self.llm.invoke(messages)
        return result.content


    def estimate_tokens(self, text: str) -> int:
        # This is a rough estimate. For more accurate results, consider using a tokenizer.
        return len(text) // 4  # Assuming average of 4 characters per token

    def process_data_chunk(self, data1: List[Dict[Any, Any]], data2: List[Dict[Any, Any]], user_input: str) -> str:
        system_message = SystemMessage(
            content=str(self.prompter.prompts_polymarket(data1=data1, data2=data2))
        )
        human_message = HumanMessage(content=user_input)
        messages = [system_message, human_message]
        result = self.llm.invoke(messages)
        return result.content


    def divide_list(self, original_list, i):
        # Calculate the size of each sublist
        sublist_size = math.ceil(len(original_list) / i)
        
        # Use list comprehension to create sublists
        return [original_list[j:j+sublist_size] for j in range(0, len(original_list), sublist_size)]
    
    def get_polymarket_llm(self, user_input: str) -> str:
        data1 = self.gamma.get_current_events()
        data2 = self.gamma.get_current_markets()
        
        combined_data = str(self.prompter.prompts_polymarket(data1=data1, data2=data2))
        
        # Estimate total tokens
        total_tokens = self.estimate_tokens(combined_data)
        
        # Set a token limit (adjust as needed, leaving room for system and user messages)
        token_limit = self.token_limit
        if total_tokens <= token_limit:
            # If within limit, process normally
            return self.process_data_chunk(data1, data2, user_input)
        else:
            # If exceeding limit, process in chunks
            chunk_size = len(combined_data) // ((total_tokens // token_limit) + 1)
            print(f'total tokens {total_tokens} exceeding llm capacity, now will split and answer')
            group_size = (total_tokens // token_limit) + 1 # 3 is safe factor
            keys_no_meaning = ['image','pagerDutyNotificationEnabled','resolvedBy','endDate','clobTokenIds','negRiskMarketID','conditionId','updatedAt','startDate']
            useful_keys = ['id','questionID','description','liquidity','clobTokenIds','outcomes','outcomePrices','volume','startDate','endDate','question','questionID','events']
            data1 = retain_keys(data1, useful_keys)
            cut_1 = self.divide_list(data1, group_size)
            cut_2 = self.divide_list(data2, group_size)
            cut_data_12 = zip(cut_1, cut_2)

            results = []

            for cut_data in cut_data_12:
                sub_data1 = cut_data[0]
                sub_data2 = cut_data[1]
                sub_tokens = self.estimate_tokens(str(self.prompter.prompts_polymarket(data1=sub_data1, data2=sub_data2)))

                result = self.process_data_chunk(sub_data1, sub_data2, user_input)
                results.append(result)
            
            combined_result = " ".join(results)
            
        
            
            return combined_result
    def filter_events(self, events: "list[SimpleEvent]") -> str:
        prompt = self.prompter.filter_events(events)
        result = self.llm.invoke(prompt)
        return result.content

    def detect_category(self, question: str) -> str:
        # Preprocesar texto
        doc = self.nlp(question.lower())
        
        # Extraer entidades nombradas y frases clave
        entities = [ent.text for ent in doc.ents]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Calcular scores para cada categoría
        scores = defaultdict(float)
        
        # 1. Coincidencia directa con palabras clave
        for category, keyword_groups in self.category_keywords.items():
            for group_name, keywords in keyword_groups.items():
                weight = 2.0 if group_name == "primary" else 1.0
                for keyword in keywords:
                    if keyword in question.lower():
                        scores[category] += weight
                        
        # 2. Análisis de entidades y frases
        for entity in entities + noun_phrases:
            # Vectorizar y comparar con keywords de cada categoría
            entity_vector = self.vectorizer.transform([entity])
            for category, keyword_groups in self.category_keywords.items():
                for keywords in keyword_groups.values():
                    keyword_vectors = self.vectorizer.transform(keywords)
                    similarity = np.mean(entity_vector.dot(keyword_vectors.T).toarray())
                    scores[category] += similarity
        
        # Normalizar scores
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
            
        # Retornar categoría con mayor score si supera un umbral
        best_category = max(scores.items(), key=lambda x: x[1], default=("other", 0))
        return best_category[0] if best_category[1] > 0.3 else "other"

    def filter_events_with_rag(self, events: "list[tuple[SimpleEvent, float]]") -> "list[tuple[SimpleEvent, float]]":
        """Filtra eventos por categoría"""
        market_category = str(os.getenv("MARKET_CATEGORY", "all")).lower().strip()
        print(f"\nFiltering {len(events)} events for category: {market_category}")
        
        # Si es 'all', retornar todos los eventos
        if market_category == 'all':
            print(f"Returning all {len(events)} events")
            return events
        
        filtered_events = []
        for event_tuple in events:
            event = event_tuple[0]
            question = event.metadata.get("question", event.title)
            event_category = self.detect_category(question)
            
            print(f"Question: {question}")
            print(f"Detected category: {event_category}")
            
            if event_category == market_category:
                filtered_events.append(event_tuple)
        
        print(f"\nFound {len(filtered_events)} {market_category} markets")
        if not filtered_events:
            print(f"\nNo markets found for category: {market_category}")
        
        return filtered_events

    def map_filtered_events_to_markets(
        self, filtered_events: "list[tuple[Dict, float]]"
    ) -> "list[tuple[SimpleMarket, float]]":
        if not filtered_events:
            return []
            
        markets = []
        
        for event_tuple in filtered_events:
            event_dict = event_tuple[0]  # Ahora es un diccionario
            event = event_dict['event']  # Obtener el SimpleEvent del diccionario
            trade_data = event_dict['trade']  # Obtener los datos del trade
            
            if not isinstance(event, SimpleEvent):
                continue
                
            # Usar metadata para obtener los market_ids
            market_ids = event.metadata.get("markets", "").split(",")
            
            for market_id in market_ids:
                if not market_id:
                    continue
                try:
                    # Usar los datos del trade si están disponibles
                    market_data = trade_data.get('market_data', self.gamma.get_market(market_id))
                    
                    # Crear SimpleMarket
                    simple_market = SimpleMarket(
                        id=int(market_data.get("id")),
                        question=market_data.get("question", ""),
                        description=market_data.get("description", ""),
                        end=market_data.get("endDate", ""),
                        active=True,
                        funded=True,
                        rewardsMinSize=0.0,
                        rewardsMaxSpread=0.0,
                        spread=0.0,
                        outcomes=str(market_data.get("outcomes", "[]")),
                        outcome_prices=str(market_data.get("outcomePrices", "[]")),
                        clob_token_ids=str(market_data.get("clobTokenIds", "[]"))
                    )
                    markets.append((simple_market, event_tuple[1]))
                    
                except Exception as e:
                    print(f"{Fore.RED}Error getting market {market_id}: {str(e)}{Style.RESET_ALL}")
                    continue
                    
        return markets

    def filter_markets(self, markets) -> "list[tuple]":
        if not markets:
            print(f"{Fore.YELLOW}No markets to filter{Style.RESET_ALL}")
            return []
            
        prompt = self.prompter.filter_markets()
        print()
        print("... prompting ... ", prompt)
        print()
        return self.chroma.markets(markets, prompt)

    def extract_probability(self, conclusion: str) -> float:
        try:
            # Buscar un número entre 0 y 1 en el texto
            import re
            probability_matches = re.findall(r"likelihood of (\d*\.?\d+)", conclusion)
            if probability_matches:
                return float(probability_matches[0])
            
            # Si no encuentra el formato exacto, buscar cualquier número entre 0 y 1
            number_matches = re.findall(r"(\d*\.?\d+)", conclusion)
            for match in number_matches:
                num = float(match)
                if 0 <= num <= 1:
                    return num
                    
            raise ValueError("No valid probability found in conclusion")
            
        except Exception as e:
            print(f"{Fore.RED}Error extracting probability: {str(e)}{Style.RESET_ALL}")
            return None

    def source_best_trade(self, market_tuple) -> dict:
        try:
            market = market_tuple[0]  # SimpleMarket
            
            # Extraer los datos necesarios directamente del SimpleMarket
            outcome_prices = ast.literal_eval(market.outcome_prices)
            outcomes = ast.literal_eval(market.outcomes)
            question = market.question
            description = market.description

            # Obtener información relacionada
            related_info = self.search.get_related_markets(question)
            context = f"""
            Related markets and information:
            {related_info}
            
            Original question: {question}
            """
            
            # Obtener análisis con contexto ampliado
            prompt = self.prompter.superforecaster(context, description, outcome_prices)
            result = self.llm.invoke(prompt)
            analysis = result.content
            
            print(f"{Fore.YELLOW}AI Analysis:")
            print(analysis)
            
            # Extraer la conclusión y probabilidad de manera más robusta
            conclusion = analysis.split("CONCLUSION:")[1].strip()
            ai_probability = self.extract_probability(conclusion)
            
            if ai_probability is None:
                return None
            
            # Determinar si hay edge vs precio de mercado
            market_yes = float(outcome_prices[0])
            market_no = float(outcome_prices[1])
            
            # Calcular edge y confianza
            edge_yes = abs(ai_probability - market_yes)
            edge_no = abs((1 - ai_probability) - market_no)
            confidence_yes = ai_probability
            confidence_no = 1 - ai_probability
            
            # Crear trade basado en el análisis
            if edge_yes > 0.02 or confidence_yes > 0.75:  # Si hay edge significativo o alta confianza en YES
                trade_dict = {
                    'side': 'BUY',
                    'position': 'YES',
                    'price': market_yes,
                    'edge': edge_yes,
                    'confidence': confidence_yes
                }
            elif edge_no > 0.02 or confidence_no > 0.75:  # Si hay edge significativo o alta confianza en NO
                trade_dict = {
                    'side': 'BUY',
                    'position': 'NO',
                    'price': market_no,
                    'edge': edge_no,
                    'confidence': confidence_no
                }
            else:
                print(f"{Fore.YELLOW}Neither sufficient edge ({max(edge_yes, edge_no):.2%}) nor confidence ({max(confidence_yes, confidence_no):.2%}){Style.RESET_ALL}")
                return None
            
            trade_dict.update({
                'analysis': analysis,
                'prediction': conclusion
            })

            return trade_dict

        except Exception as e:
            print(f"{Fore.RED}Error processing market: {str(e)}{Style.RESET_ALL}")
            return None

    def format_trade_prompt_for_execution(self, best_trade: str) -> float:
        if isinstance(best_trade, str):
            # Si es string, parsearlo
            data = best_trade.split(",")
            size = re.findall("\d+\.\d+", data[1])[0]
            return float(1.0)  # Monto fijo de 1 USDC para pruebas
        elif isinstance(best_trade, dict):
            # Si ya es diccionario, usar el size directamente
            return float(best_trade.get('size', 1.0))
        
        # Por defecto, retornar 1 USDC
        return 1.0

    def source_best_market_to_create(self, filtered_markets) -> str:
        prompt = self.prompter.create_new_market(filtered_markets)
        print()
        print("... prompting ... ", prompt)
        print()
        result = self.llm.invoke(prompt)
        content = result.content
        return content
