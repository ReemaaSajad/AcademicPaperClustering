import requests
import xml.etree.ElementTree as ET
import pandas as pd
import argparse
import sys
import os
import json
import time

def fetch_arxiv_papers(keyword, max_results=100):
    """Fetch papers from arXiv API"""
    base_url = "http://export.arxiv.org/api/query?"
    query = f"search_query=all:{keyword}&start=0&max_results={max_results}"
    response = requests.get(base_url + query)
    
    if response.status_code != 200:
        raise Exception("Failed to fetch from arXiv API")

    root = ET.fromstring(response.text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}

    papers = []
    for entry in root.findall('atom:entry', ns):
        title = entry.find('atom:title', ns).text.strip()
        abstract = entry.find('atom:summary', ns).text.strip().replace("\n", " ")
        authors = [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)]
        link = entry.find('atom:id', ns).text
        papers.append({
            'title': title,
            'abstract': abstract,
            'authors': ", ".join(authors),
            'link': link,
            'source': 'arXiv'
        })

    return papers

def fetch_semantic_scholar_papers(keyword, max_results=50):
    """Fetch papers from Semantic Scholar API"""
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    papers = []
    offset = 0
    limit = min(50, max_results)  

    while len(papers) < max_results:
        params = {
            'query': keyword,
            'offset': offset,
            'limit': min(limit, max_results - len(papers)),
            'fields': 'title,abstract,authors,url,year,publicationDate'
        }
        
        try:
            response = requests.get(base_url, params=params)
            
            if response.status_code != 200:
                if response.status_code == 429:  # Rate limit
                    print("Rate limit reached, waiting...")
                    time.sleep(1)
                    continue
                else:
                    raise Exception(f"Failed to fetch from Semantic Scholar API: {response.status_code}")
            
            data = response.json()
            
            if 'data' not in data or not data['data']:
                break
                
            for paper in data['data']:
                if len(papers) >= max_results:
                    break
                    
                title = paper.get('title', 'No title')
                abstract = paper.get('abstract', 'No abstract available')
                authors = [author.get('name', 'Unknown') for author in paper.get('authors', [])]
                link = paper.get('url', 'No URL available')
                
                papers.append({
                    'title': title,
                    'abstract': abstract if abstract else 'No abstract available',
                    'authors': ", ".join(authors) if authors else 'Unknown authors',
                    'link': link if link else 'No URL available',
                    'source': 'Semantic Scholar'
                })
            
            offset += len(data['data'])
            
            # If we got fewer results than requested, we've reached the end
            if len(data['data']) < limit:
                break
                
            # Add a small delay to be respectful to the API
            time.sleep(0.1)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error when fetching from Semantic Scholar: {e}")
        except json.JSONDecodeError as e:
            raise Exception(f"Error parsing Semantic Scholar response: {e}")
    
    return papers

def fetch_papers(keyword, source, max_results=100):
    """Main function to fetch papers from specified source"""
    if source.lower() == 'arxiv':
        papers = fetch_arxiv_papers(keyword, max_results)
        filename = "arxiv_papers.csv"
    elif source.lower() == 'semantic_scholar' or source.lower() == 'semanticscholar':
        papers = fetch_semantic_scholar_papers(keyword, max_results)
        filename = "semantic_scholar_papers.csv"
    elif source.lower() == 'both':
        # Fetch from both sources
        arxiv_papers = fetch_arxiv_papers(keyword, max_results // 2)
        semantic_papers = fetch_semantic_scholar_papers(keyword, max_results // 2)
        papers = arxiv_papers + semantic_papers
        filename = "combined_papers.csv"
    else:
        raise ValueError(f"Unsupported source: {source}. Choose from 'arxiv', 'semantic_scholar', or 'both'")
    
    df = pd.DataFrame(papers)
    
    # Create directory if it doesn't exist
    os.makedirs("data/raw", exist_ok=True)
    
    filepath = f"data/raw/{filename}"
    df.to_csv(filepath, index=False)
    print(f"Fetched {len(df)} papers on '{keyword}' from {source} and saved to {filepath}")
    
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch papers from arXiv or Semantic Scholar APIs")
    parser.add_argument("keyword", help="Keyword to search for")
    parser.add_argument("--source", 
                       choices=['arxiv', 'semantic_scholar', 'both'], 
                       default='arxiv',
                       help="API source to fetch from (default: arxiv)")
    parser.add_argument("--max-results", "-n", type=int, default=100, 
                       help="Maximum number of results to fetch (default: 100)")
    
    args = parser.parse_args()
    
    try:
        fetch_papers(args.keyword, args.source, args.max_results)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
