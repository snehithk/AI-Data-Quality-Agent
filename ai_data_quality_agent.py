#!/usr/bin/env python3
"""
AI Data Quality Agent
Author: Big Data Engineer
Description: AI-powered data quality assessment tool using LLMs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

# IMPORTANT: Replace with your own API key from https://openrouter.ai/
OPENROUTER_API_KEY = "YOUR_API_KEY_HERE"  # Get free key at https://openrouter.ai/
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"

# ============================================================================
# STEP 1: CREATE SAMPLE DATA WITH QUALITY ISSUES
# ============================================================================

def create_sample_data():
    """
    Create sample stock market dataset with intentional quality issues
    """
    np.random.seed(42)
    date_range = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    data = {
        'date': date_range,
        'stock_symbol': ['AAPL'] * 100,
        'open_price': np.random.uniform(150, 200, 100),
        'close_price': np.random.uniform(150, 200, 100),
        'volume': np.random.randint(1000000, 10000000, 100),
    }
    
    # Introduce data quality issues
    data['close_price'][10] = None  # Missing value
    data['volume'][20] = -100  # Negative volume (impossible)
    data['open_price'][30] = np.inf  # Infinite value
    data['stock_symbol'][40] = ''  # Empty string
    
    df = pd.DataFrame(data)
    
    print(f"Dataset created with {len(df)} rows")
    print("\nData quality issues introduced:")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Negative volume: {(df['volume'] < 0).sum()}")
    print(f"Infinite values: {np.isinf(df.select_dtypes(include=[float])).sum().sum()}")
    
    return df

# ============================================================================
# STEP 2: DATA QUALITY CHECKS
# ============================================================================

def analyze_data_quality(df):
    """
    Run comprehensive data quality checks and return findings
    """
    findings = []
    
    # Check 1: Missing values
    missing_cols = df.isnull().sum()
    for col in missing_cols[missing_cols > 0].index:
        findings.append({
            'type': 'MISSING_VALUES',
            'column': col,
            'severity': 'HIGH',
            'count': int(missing_cols[col]),
            'description': f"Column '{col}' has {int(missing_cols[col])} missing value(s)"
        })
    
    # Check 2: Negative values in numeric columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if col != 'stock_symbol':
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                findings.append({
                    'type': 'NEGATIVE_VALUES',
                    'column': col,
                    'severity': 'MEDIUM',
                    'count': int(negative_count),
                    'description': f"Column '{col}' has {int(negative_count)} negative value(s)"
                })
    
    # Check 3: Infinite values
    for col in df.select_dtypes(include=['float64']).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            findings.append({
                'type': 'INFINITE_VALUES',
                'column': col,
                'severity': 'HIGH',
                'count': int(inf_count),
                'description': f"Column '{col}' contains {int(inf_count)} infinite value(s)"
            })
    
    # Check 4: Empty strings
    for col in df.select_dtypes(include=['object']).columns:
        empty_count = (df[col] == '').sum()
        if empty_count > 0:
            findings.append({
                'type': 'EMPTY_STRINGS',
                'column': col,
                'severity': 'MEDIUM',
                'count': int(empty_count),
                'description': f"Column '{col}' has {int(empty_count)} empty string(s)"
            })
    
    return findings

# ============================================================================
# STEP 3: AI-POWERED DIAGNOSIS USING LLM
# ============================================================================

def get_ai_diagnosis(quality_findings, api_key=None):
    """
    Call OpenRouter LLM to generate AI-powered data quality diagnosis
    """
    if api_key is None:
        api_key = OPENROUTER_API_KEY
    
    if api_key == "YOUR_API_KEY_HERE":
        return """
        ⚠️  API KEY NOT CONFIGURED
        
        To get AI-powered diagnosis:
        1. Visit https://openrouter.ai/
        2. Sign up for a free account
        3. Get your API key
        4. Replace OPENROUTER_API_KEY in this script
        
        For now, showing mock diagnosis based on findings.
        """
    
    # Format findings for LLM
    findings_text = "\n".join([
        f"- {f['type']} in column '{f['column']}': {f['description']} (Severity: {f['severity']})"
        for f in quality_findings
    ])
    
    prompt = f"""You are a Data Quality Expert. Analyze these data pipeline issues and provide:
    1. Root cause analysis
    2. Impact assessment  
    3. Recommended fixes
    
    Issues detected:
    {findings_text}
    
    Provide a concise, professional diagnostic report:"""
    
    # Call OpenRouter API
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Error calling LLM API: {response.status_code} - {response.text}"
    
    except Exception as e:
        return f"Exception calling LLM: {str(e)}"

# ============================================================================
# STEP 4: GENERATE REPORTS
# ============================================================================

def generate_report(quality_findings, ai_diagnosis):
    """
    Generate comprehensive data quality report
    """
    report = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║          AI DATA QUALITY AGENT - DIAGNOSTIC REPORT                            ║
    ║                     Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}                        ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    📊 ISSUES FOUND: {len(quality_findings)}
    """
    
    for idx, finding in enumerate(quality_findings, 1):
        report += f"""
    {idx}. [{finding['severity']}] {finding['type']}
       Column: {finding['column']}
       Issue: {finding['description']}
    """
    
    report += f"""
    
    🤖 AI DIAGNOSIS:
    ────────────────────────────────────────────────────────────────────────────────
    {ai_diagnosis}
    """
    
    return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("AI DATA QUALITY AGENT")
    print("="*80)
    
    # Step 1: Create sample data
    print("\n[Step 1] Creating sample dataset...")
    df = create_sample_data()
    
    # Step 2: Run quality checks
    print("\n[Step 2] Running data quality checks...")
    quality_findings = analyze_data_quality(df)
    
    print(f"\n✅ Found {len(quality_findings)} issues")
    for finding in quality_findings:
        print(f"  • {finding['type']}: {finding['description']}")
    
    # Step 3: Get AI diagnosis
    print("\n[Step 3] Getting AI diagnosis...")
    ai_diagnosis = get_ai_diagnosis(quality_findings)
    
    # Step 4: Generate report
    print("\n[Step 4] Generating report...")
    report = generate_report(quality_findings, ai_diagnosis)
    
    # Print report
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    # Save findings to JSON
    with open('quality_findings.json', 'w') as f:
        json.dump(quality_findings, f, indent=2, default=str)
    
    print("\n✅ Report saved to quality_findings.json")
    print("\nNext steps:")
    print("1. Get your free API key from https://openrouter.ai/")
    print("2. Replace OPENROUTER_API_KEY in this script")
    print("3. Run again to get real AI-powered diagnosis")
