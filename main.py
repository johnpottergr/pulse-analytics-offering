from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import subprocess
import json
import os
from openai import OpenAI
import requests

app = FastAPI()

class PerformanceRequest(BaseModel):
    content: str
    period: str = "last_7_days"  # e.g., last_7_days, last_30_days

@app.post("/analyze-performance")
def analyze_performance(request: PerformanceRequest):
    # Fetch trends from ContentStudio
    input_data = {
        "action": "scrape_analyze",
        "urls": ["https://api.contentstudio.io/v1/trends?query=AI"],  # Replace with actual endpoint
        "my_content": request.content,
        "n_clusters": 3
    }
    result = subprocess.run(
        ["python3", "python_script.py"],
        input=json.dumps(input_data),
        text=True,
        capture_output=True
    )
    analysis = json.loads(result.stdout) if result.returncode == 0 else {"error": "Analysis failed"}
    trends = analysis.get("trends", [])

    # Fetch Google Analytics data (placeholder)
    ga_data = requests.get(
        f"https://www.googleapis.com/analytics/v3/data/ga?ids=ga:{os.getenv('GA_PROPERTY_ID', 'your_ga_property_id')}&start-date={request.period}&end-date=today&metrics=ga:pageviews,ga:sessions",
        headers={"Authorization": f"Bearer {os.getenv('GOOGLE_ANALYTICS_API_KEY', 'your_ga_api_key_here')}"
    }).json()
    traffic = ga_data.get("totalsForAllResults", {"ga:pageviews": 0, "ga:sessions": 0})

    # Craft prompt for DeepSeek analysis
    prompt = f"Analyze performance for content: {request.content}. Trends: {', '.join(trends)}. Traffic: {traffic}. Provide insights on engagement drivers and recommendations for the {request.period} period."
    
    # Initialize DeepSeek client
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key_here"),
        base_url="https://api.deepseek.com"
    )
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=1.3
        )
        insights = response.choices[0].message.content.strip()
    except Exception as e:
        insights = f"Error generating insights: {str(e)}"

    return {
        "trends": trends,
        "traffic": traffic,
        "insights": insights
    }

@app.post("/alerts")
def generate_alerts(request: PerformanceRequest):
    # Placeholder for n8n-triggered alerts based on insights
    return {"alert": f"Check performance insights for {request.content} over {request.period}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
