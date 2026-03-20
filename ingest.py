import os
import json
import requests
import pandas as pd
from tqdm import tqdm
import yfinance as yf

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

OUTPUT_FILE = "data/processed_data.json"
os.makedirs("data", exist_ok=True)


def load_earnings_calls():
    data = []

    print("Loading earnings calls...")

    if HF_AVAILABLE:
        try:
            dataset = load_dataset("sohomghosh/MiMIC_Multi-Modal_Indian_Earnings_Calls_Dataset", split="train")

            for item in tqdm(dataset):
                # Print the keys of the first item to verify "transcript" is correct
                if not data: 
                    print(f"Available keys in dataset: {item.keys()}")
                
                # The dataset contains transcript_link and transcript_file_name
                # If transcript key is missing, we use other fields as text content for now
                text = item.get("transcript", "")
                
                if not text:
                    # Fallback: create a summary text from available metadata if transcript is missing
                    company = item.get("company_name", "unknown")
                    ticker = item.get("ticker", "unknown")
                    year = item.get("year", "unknown")
                    sales = item.get("Sales", "N/A")
                    net_profit = item.get("Net Profit", "N/A")
                    
                    text = f"Financial results for {company} ({ticker}) for year {year}. Sales: {sales}, Net Profit: {net_profit}."
                
                if len(text) < 50:
                    print(f"Skipping short content ({len(text)} chars) for: {item.get('company_name')}")
                    continue

                data.append({
                    "text": text,
                    "company": item.get("company_name", "unknown"),
                    "source": "earnings_call",
                    "date": item.get("RESULT DATE", "unknown"),
                    "metadata": {
                        "ticker": item.get("ticker", ""),
                        "year": item.get("year", ""),
                        "sales": item.get("Sales", ""),
                        "net_profit": item.get("Net Profit", ""),
                        "transcript_link": item.get("transcript_link", "")
                    }
                })

        except Exception as e:
            print("HF dataset failed, fallback to CSV:", e)

    return data


def load_news(api_key, query="Indian stock market"):
    data = []

    print(f"Fetching news from News API for query: {query}...")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        articles = response.json().get("articles", [])

        for a in articles:
            text = str(a.get("title", "")) + ". " + str(a.get("description", ""))

            if len(text) < 50:
                continue

            data.append({
                "text": text,
                "company": "unknown",
                "source": "news",
                "date": a.get("publishedAt", "unknown"),
                "metadata": {
                    "url": a.get("url", ""),
                    "author": a.get("author", "")
                }
            })
    except Exception as e:
        print(f"Failed to fetch news: {e}")

    return data


def load_company_fundamentals(companies):
    data = []

    print("Loading company fundamentals from Yahoo Finance...")

    for company in companies:
        try:
            ticker = yf.Ticker(company)

            info = ticker.info

            text = info.get("longBusinessSummary", "")

            if not text or len(text) < 50:
                continue

            data.append({
                "text": text,
                "company": company,
                "source": "company_info",
                "date": "latest",
                "metadata": {
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "marketCap": info.get("marketCap", "")
                }
            })

        except Exception as e:
            print(f"Error loading {company}: {e}")

    return data



def main():
    all_data = []

    # Earnings calls
    earnings_data = load_earnings_calls()
    all_data.extend(earnings_data)

    # News (News API)
    NEWS_API_KEY = "562401740e95408dbe040c9939520af0"  
    if NEWS_API_KEY:
        news_data = load_news(NEWS_API_KEY)
        all_data.extend(news_data)
    else:
        print("Skipping News API: No API key provided.")

    # Company fundamentals (Yahoo Finance)
    companies = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]  
    fundamentals_data = load_company_fundamentals(companies)
    all_data.extend(fundamentals_data)

    print(f"\nTotal documents collected: {len(all_data)}")

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
