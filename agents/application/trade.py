from agents.application.executor import Executor as Agent
from agents.polymarket.gamma import GammaMarketClient as Gamma
from agents.polymarket.polymarket import Polymarket
from colorama import init, Fore, Style

import shutil
import os
import ast
import time

init()  # Inicializar colorama


class Trader:
    def __init__(self):
        self.polymarket = Polymarket()
        self.gamma = Gamma()
        self.agent = Agent()
        self.dry_run = os.getenv("DRY_RUN", "false").lower() == "true"
        # Si está en dry_run, usar delay más corto o 0
        default_delay = "0" if self.dry_run else "300"
        self.analysis_delay = int(os.getenv("ANALYSIS_DELAY_SECONDS", default_delay))
        
        if self.dry_run:
            print(f"\n{Fore.GREEN}🔍 Running in DRY RUN mode - no transactions will be executed")
            print(f"Analysis delay disabled in DRY RUN mode{Style.RESET_ALL}\n")
        else:
            print(f"{Fore.BLUE}Analysis delay set to {self.analysis_delay} seconds{Style.RESET_ALL}")

    def pre_trade_logic(self) -> None:
        self.clear_local_dbs()

    def clear_local_dbs(self) -> None:
        try:
            shutil.rmtree("local_db_events")
        except:
            pass
        try:
            shutil.rmtree("local_db_markets")
        except:
            pass

    def one_best_trade(self) -> None:
        """

        one_best_trade is a strategy that evaluates all events, markets, and orderbooks

        leverages all available information sources accessible to the autonomous agent

        then executes that trade without any human intervention

        """
        try:
            while True:  # Bucle infinito para seguir ejecutando
                self.pre_trade_logic()

                events = self.polymarket.get_all_events()
                print(f"{Fore.LIGHTBLUE_EX}1. FOUND {len(events)} EVENTS{Style.RESET_ALL}")

                # Filtrar primero por volumen y pins
                high_quality_events = []
                for event in events:
                    try:
                        event_data = event.dict()
                        market_ids = event_data.get('markets', '').split(',')
                        
                        for market_id in market_ids:
                            if not market_id:
                                continue
                                
                            market_data = self.gamma.get_market(market_id)
                            volume = float(market_data.get('volume', 0))
                            is_pinned = market_data.get('featured', False)
                            
                            if volume > 10000 or is_pinned:
                                # Crear un diccionario con el evento y sus datos de trade
                                event_with_trade = {
                                    'event': event,
                                    'trade': {
                                        'market_data': market_data
                                    }
                                }
                                high_quality_events.append((event_with_trade, 1.0))
                                print(f"\nHigh quality market found: {market_data.get('question', '')}")
                                print(f"Volume: ${volume:,.2f}")
                                print(f"Featured: {is_pinned}")
                                print("---")
                                break
                                
                    except Exception as e:
                        print(f"Error processing event: {e}")
                        continue

                print(f"{Fore.LIGHTBLUE_EX}2. FOUND {len(high_quality_events)} HIGH QUALITY EVENTS{Style.RESET_ALL}")

                # Continuar con el filtrado RAG solo para eventos de alta calidad
                filtered_events = self.agent.filter_events_with_rag(high_quality_events)
                print(f"{Fore.LIGHTBLUE_EX}3. FILTERED {len(filtered_events)} EVENTS{Style.RESET_ALL}")

                markets = self.agent.map_filtered_events_to_markets(filtered_events)
                print()
                print(f"{Fore.LIGHTBLUE_EX}4. FOUND {len(markets)} MARKETS{Style.RESET_ALL}")

                print()
                filtered_markets = self.agent.filter_markets(markets)
                print(f"{Fore.LIGHTBLUE_EX}5. FILTERED {len(filtered_markets)} MARKETS{Style.RESET_ALL}")

                # Para las respuestas de la IA
                print(f"\n{Fore.YELLOW}AI analyzing markets...{Style.RESET_ALL}")

                for market_tuple in filtered_markets:
                    try:
                        market_data = market_tuple[0]  # SimpleMarket
                        print(f"\n{Fore.YELLOW}=== Analyzing Market ===")
                        print(f"Market: {market_data.question}")
                        print(f"Current Prices:")
                        prices = ast.literal_eval(market_data.outcome_prices)
                        print(f"YES: ${prices[0]} ({Fore.RED}{float(prices[0])*100:.1f}%{Style.RESET_ALL})")
                        print(f"NO: ${prices[1]} ({Fore.RED}{float(prices[1])*100:.1f}%{Style.RESET_ALL})")
                        print(f"Volume: ${float(market_data.volume if hasattr(market_data, 'volume') else 0):,.2f}")

                        if not hasattr(market_data, 'clob_token_ids') or not market_data.clob_token_ids:
                            print(f"Market {market_data.question} does not have token IDs")
                            continue

                        best_trade = self.agent.source_best_trade(market_tuple)
                        
                        if best_trade and isinstance(best_trade, dict):
                            print(f"\nAI Decision:")
                            position = best_trade.get('position', 'UNKNOWN')
                            print(f"Action: BUY {position}")
                            
                            # Asegurar que el precio es float
                            target_price = float(best_trade.get('price', 0))
                            edge = best_trade.get('edge', 0)
                            
                            print(f"Target Price: ${target_price}")
                            print(f"Expected Edge: ${edge:.4f}")
                            print(f"Confidence: High based on market conditions")
                            print(f"Reasoning: {best_trade.get('prediction', 'No prediction available')}")
                            print(f"===================={Style.RESET_ALL}")
                            
                            amount = 1.0
                            best_trade['size'] = amount
                            best_trade['price'] = target_price
                            
                            print(f"\n{Fore.GREEN}6. TRYING TRADE FOR MARKET {market_data.question}")
                            print(f"   Amount: ${amount} USDC")
                            print(f"   Price: {best_trade['price']}")
                            print(f"   Side: BUY {best_trade.get('position')}{Style.RESET_ALL}")

                            if self.dry_run:
                                print("\n🔍 DRY RUN: Trade would be executed with these parameters")
                                print(f"   Token ID: {market_data.clob_token_ids}")
                                print(f"   Market Question: {market_data.question}")
                                print("Skipping actual transaction...")
                                continue

                            amount = self.agent.format_trade_prompt_for_execution(best_trade)
                            trade = self.polymarket.execute_market_order(market_data, amount)
                            
                            if trade:
                                print(f"7. TRADED SUCCESSFULLY {trade}")
                                print(f"\n{Fore.BLUE}Trade successful! Waiting {self.analysis_delay} seconds before next analysis...{Style.RESET_ALL}")
                                time.sleep(self.analysis_delay)
                                break  # Salir del bucle de mercados y empezar nuevo ciclo
                            else:
                                print("Trade failed or skipped, trying next market...")
                                print(f"\n{Fore.BLUE}Waiting {self.analysis_delay} seconds before next analysis...{Style.RESET_ALL}")
                                time.sleep(self.analysis_delay)
                                continue

                    except Exception as e:
                        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
                        print(f"\n{Fore.BLUE}Waiting {self.analysis_delay} seconds before next analysis...{Style.RESET_ALL}")
                        time.sleep(self.analysis_delay)
                        continue

                print("\nNo eligible markets found for trading")
                print(f"\n{Fore.BLUE}No trades found. Waiting {self.analysis_delay} seconds before next analysis...{Style.RESET_ALL}")
                time.sleep(self.analysis_delay)

        except Exception as e:
            print(f"Error {e}")
            print(f"\n{Fore.RED}Critical error occurred. Waiting {self.analysis_delay} seconds before retrying...{Style.RESET_ALL}")
            time.sleep(self.analysis_delay)

    def maintain_positions(self):
        pass

    def incentive_farm(self):
        pass


if __name__ == "__main__":
    t = Trader()
    t.one_best_trade()
