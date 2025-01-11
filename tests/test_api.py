import httpx
import json
from datetime import datetime

def test_gamma_api():
    print("\n" + "="*50)
    print("INICIANDO TEST DE GAMMA API")
    print("="*50)
    
    gamma_url = "https://gamma-api.polymarket.com"
    
    params = {
        "active": "true",
        "closed": "false",
        "archived": "false",
        "limit": "10"
    }
    
    print(f"\nURL base: {gamma_url}")
    print(f"Parámetros: {json.dumps(params, indent=2)}")
    
    try:
        print("\nHaciendo request...")
        response = httpx.get(f"{gamma_url}/markets", params=params)
        print(f"Código de estado: {response.status_code}")
        print(f"URL completa: {response.url}")
        
        if response.status_code == 200:
            markets = response.json()
            print(f"\nNúmero de mercados encontrados: {len(markets)}")
            
            if markets:
                for market in markets:
                    print("\n" + "="*80)
                    print(f"MERCADO: {market.get('question')}")
                    print("="*80)
                    
                    # Información básica
                    print("\nINFORMACIÓN BÁSICA:")
                    print(f"ID: {market.get('id')}")
                    print(f"Categoría: {market.get('category', 'No especificada')}")
                    print(f"Descripción: {market.get('description', 'No disponible')}")
                    
                    # Fechas
                    end_date = datetime.fromisoformat(market.get('endDate').replace('Z', '+00:00'))
                    start_date = None
                    if market.get('startDate'):
                        start_date = datetime.fromisoformat(market.get('startDate').replace('Z', '+00:00'))
                    
                    print(f"\nFECHAS:")
                    if start_date:
                        print(f"Fecha inicio: {start_date.strftime('%d/%m/%Y %H:%M')} UTC")
                    print(f"Fecha fin: {end_date.strftime('%d/%m/%Y %H:%M')} UTC")
                    
                    # Estado
                    print(f"\nESTADO:")
                    print(f"Activo: {'Sí' if market.get('active') else 'No'}")
                    print(f"Cerrado: {'Sí' if market.get('closed') else 'No'}")
                    print(f"Archivado: {'Sí' if market.get('archived') else 'No'}")
                    print(f"Restringido: {'Sí' if market.get('restricted') else 'No'}")
                    
                    # Métricas
                    print(f"\nMÉTRICAS:")
                    print(f"Volumen 24h: {market.get('volume24hr', 0):,.2f}")
                    print(f"Volumen total: {float(market.get('volume', 0)):,.2f}")
                    print(f"Liquidez: {float(market.get('liquidity', 0)):,.2f}")
                    
                    # Trading
                    print(f"\nTRADING:")
                    print(f"Tamaño mínimo orden: {market.get('orderMinSize', 'No especificado')}")
                    print(f"Tick size mínimo: {market.get('orderPriceMinTickSize', 'No especificado')}")
                    print(f"Spread: {market.get('spread', 'No especificado')}")
                    
                    # Precios y resultados
                    if 'outcomePrices' in market and 'outcomes' in market:
                        print(f"\nRESULTADOS Y PRECIOS:")
                        outcomes = eval(market.get('outcomes', '[]'))
                        prices = eval(market.get('outcomePrices', '[]'))
                        for outcome, price in zip(outcomes, prices):
                            print(f"{outcome}: {float(price):,.3f}")
                    
                    # Links
                    if market.get('resolutionSource'):
                        print(f"\nFUENTES:")
                        print(f"Fuente de resolución: {market.get('resolutionSource')}")
                    
                    print("\n" + "-"*80)
            else:
                print("\nNo se encontraron mercados")
                
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        if 'response' in locals():
            print(f"Respuesta: {response.text}")

if __name__ == "__main__":
    # Limpiar la pantalla antes de ejecutar
    print("\033[H\033[J")  # Código ANSI para limpiar la pantalla
    test_gamma_api()