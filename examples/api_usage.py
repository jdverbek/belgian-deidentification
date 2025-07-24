#!/usr/bin/env python3
"""
API usage example for the Belgian Document Deidentification System.

This example demonstrates how to:
1. Start the API server
2. Make requests to deidentify text
3. Handle responses and errors
"""

import asyncio
import json
import time
from typing import Dict, Any

import httpx

# API configuration
API_BASE_URL = "http://localhost:8000"
API_TIMEOUT = 30.0


async def check_api_health() -> bool:
    """Check if the API is healthy and ready."""
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            # Check health endpoint
            health_response = await client.get(f"{API_BASE_URL}/health")
            health_response.raise_for_status()
            
            # Check readiness endpoint
            ready_response = await client.get(f"{API_BASE_URL}/ready")
            ready_response.raise_for_status()
            
            print("✅ API is healthy and ready")
            return True
            
    except httpx.RequestError as e:
        print(f"❌ API connection error: {e}")
        return False
    except httpx.HTTPStatusError as e:
        print(f"❌ API health check failed: {e}")
        return False


async def deidentify_text(text: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Deidentify text using the API."""
    request_data = {
        "text": text,
        "config": config or {
            "mode": "anonymization",
            "entities": ["PERSON", "ADDRESS", "DATE", "MEDICAL_ID", "PHONE", "EMAIL"],
            "preserve_structure": True
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/v1/deidentify/text",
                json=request_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as e:
        return {"error": f"Request error: {e}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error: {e.response.status_code} - {e.response.text}"}


async def get_api_metrics() -> Dict[str, Any]:
    """Get API metrics."""
    try:
        async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
            response = await client.get(f"{API_BASE_URL}/metrics")
            response.raise_for_status()
            return response.json()
            
    except httpx.RequestError as e:
        return {"error": f"Request error: {e}"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP error: {e.response.status_code}"}


async def main():
    """Main API usage example."""
    print("🏥 Belgian Document Deidentification System - API Usage Example")
    print("=" * 70)
    
    # Step 1: Check API availability
    print("\n🔍 Step 1: Checking API availability...")
    
    if not await check_api_health():
        print("\n❌ API is not available. Please start the API server first:")
        print("   python -m belgian_deidentification.api.main")
        print("   or")
        print("   docker-compose up")
        return 1
    
    # Sample Dutch medical texts
    sample_texts = [
        {
            "name": "Simple patient record",
            "text": """
Patiënt: Jan Vermeulen
Geboortedatum: 15-06-1978
Adres: Marktplein 34, 1000 Brussel
Telefoon: 02-555-1234

Diagnose: Hypertensie
Behandeling: Lisinopril 10mg dagelijks
            """
        },
        {
            "name": "Consultation note",
            "text": """
Datum: 12-11-2023
Patiënt: Maria van den Berg (geb. 22-03-1985)
Adres: Kerkstraat 67, 2000 Antwerpen
Email: maria.vandenberg@email.be

Consult bij Dr. Sophie Claes
Klachten: Hoofdpijn sinds 1 week
Medicatie: Paracetamol 500mg bij pijn
            """
        },
        {
            "name": "Lab results",
            "text": """
Laboratorium resultaten
Patiënt ID: PAT123456789
Naam: Ahmed Hassan
Geboortedatum: 08-09-1970
Datum afname: 10-11-2023

Glucose: 6.2 mmol/L (normaal)
HbA1c: 7.1% (verhoogd)
Cholesterol: 5.8 mmol/L (verhoogd)
            """
        }
    ]
    
    # Step 2: Process each sample text
    print("\n🔄 Step 2: Processing sample texts...")
    
    for i, sample in enumerate(sample_texts, 1):
        print(f"\n📄 Processing sample {i}: {sample['name']}")
        print("-" * 50)
        
        # Show original text (truncated)
        original_preview = sample['text'][:200] + "..." if len(sample['text']) > 200 else sample['text']
        print(f"Original text preview:\n{original_preview}")
        
        # Deidentify the text
        start_time = time.time()
        result = await deidentify_text(sample['text'])
        processing_time = time.time() - start_time
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        # Display results
        print(f"\n✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📊 Entities found: {result.get('entities_found', 0)}")
        print(f"🗑️  Entities removed: {result.get('entities_removed', 0)}")
        print(f"🎯 Confidence: {result.get('confidence', 0):.3f}")
        
        # Show deidentified text
        deidentified_text = result.get('deidentified_text', '')
        if deidentified_text:
            deidentified_preview = deidentified_text[:200] + "..." if len(deidentified_text) > 200 else deidentified_text
            print(f"\nDeidentified text:\n{deidentified_preview}")
        
        # Show detected entities
        entities = result.get('entities', [])
        if entities:
            print(f"\n🔍 Detected entities ({len(entities)}):")
            for entity in entities[:5]:  # Show first 5 entities
                print(f"  - {entity.get('text', '')} ({entity.get('entity_type', '')})")
            if len(entities) > 5:
                print(f"  ... and {len(entities) - 5} more")
        
        # Show warnings if any
        warnings = result.get('warnings', [])
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    
    # Step 3: Test different configurations
    print("\n⚙️  Step 3: Testing different configurations...")
    
    test_text = """
Patiënt: Lisa De Smet
Geboortedatum: 12-04-1990
Adres: Hoofdstraat 123, 9000 Gent
Telefoon: 09-876-5432
    """
    
    configurations = [
        {
            "name": "Anonymization mode",
            "config": {
                "mode": "anonymization",
                "entities": ["PERSON", "ADDRESS", "PHONE"],
                "preserve_structure": True
            }
        },
        {
            "name": "Pseudonymization mode",
            "config": {
                "mode": "pseudonymization", 
                "entities": ["PERSON", "ADDRESS", "PHONE"],
                "preserve_structure": True
            }
        },
        {
            "name": "High confidence threshold",
            "config": {
                "mode": "anonymization",
                "entities": ["PERSON", "ADDRESS", "PHONE"],
                "confidence_threshold": 0.95
            }
        }
    ]
    
    for config_test in configurations:
        print(f"\n🧪 Testing: {config_test['name']}")
        print("-" * 30)
        
        result = await deidentify_text(test_text, config_test['config'])
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        print(f"Entities found: {result.get('entities_found', 0)}")
        print(f"Entities removed: {result.get('entities_removed', 0)}")
        
        # Show a snippet of the result
        deidentified = result.get('deidentified_text', '')
        if deidentified:
            snippet = deidentified.replace('\n', ' ').strip()[:100]
            print(f"Result snippet: {snippet}...")
    
    # Step 4: Get API metrics
    print("\n📊 Step 4: Getting API metrics...")
    
    metrics = await get_api_metrics()
    
    if "error" in metrics:
        print(f"❌ Error getting metrics: {metrics['error']}")
    else:
        print("✅ API Metrics:")
        
        # Show system info
        system_info = metrics.get('system_info', {})
        print(f"  Version: {system_info.get('version', 'unknown')}")
        print(f"  Mode: {system_info.get('config_mode', 'unknown')}")
        print(f"  Debug: {system_info.get('debug', False)}")
        
        # Show pipeline stats
        pipeline_stats = metrics.get('pipeline_stats', {})
        if pipeline_stats:
            print(f"  Documents processed: {pipeline_stats.get('documents_processed', 0)}")
            print(f"  Entities found: {pipeline_stats.get('entities_found', 0)}")
            print(f"  Processing time: {pipeline_stats.get('processing_time', 0):.2f}s")
    
    # Step 5: Performance test
    print("\n⚡ Step 5: Performance test...")
    
    # Test with multiple concurrent requests
    concurrent_requests = 3
    test_text_short = "Patiënt: Jan Janssen, geboren 01-01-1980, woont in Brussel."
    
    print(f"Making {concurrent_requests} concurrent requests...")
    
    start_time = time.time()
    
    # Create concurrent tasks
    tasks = [
        deidentify_text(test_text_short)
        for _ in range(concurrent_requests)
    ]
    
    # Execute concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_time = time.time() - start_time
    
    # Analyze results
    successful_requests = sum(1 for r in results if isinstance(r, dict) and "error" not in r)
    
    print(f"✅ Completed {successful_requests}/{concurrent_requests} requests")
    print(f"⏱️  Total time: {total_time:.2f} seconds")
    print(f"📈 Average time per request: {total_time/concurrent_requests:.2f} seconds")
    print(f"🚀 Requests per second: {concurrent_requests/total_time:.2f}")
    
    print("\n🎉 API usage example completed successfully!")
    print("\n💡 Next steps:")
    print("   - Integrate the API into your application")
    print("   - Set up authentication for production use")
    print("   - Monitor API performance and metrics")
    print("   - Scale the API using Docker Compose or Kubernetes")
    
    return 0


if __name__ == "__main__":
    try:
        exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n👋 Example interrupted by user")
        exit(0)

