#!/usr/bin/env python3
"""
Batch processing example for the Belgian Document Deidentification System.

This example demonstrates how to:
1. Process multiple documents in batch
2. Monitor progress
3. Handle errors and generate reports
"""

import logging
import time
from pathlib import Path

from belgian_deidentification.core.batch_processor import BatchProcessor
from belgian_deidentification.core.config import Config, DeidentificationMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_documents():
    """Create sample documents for batch processing."""
    sample_documents = [
        {
            "filename": "patient_001.txt",
            "content": """
Patiënt: Anna de Vries
Geboortedatum: 22-08-1980
Adres: Hoofdstraat 45, 2000 Antwerpen
Telefoon: 03-456-7890

Medisch dossier: PAT987654321
Opname: 15-10-2023
Behandelend arts: Dr. Peter Janssen

Diagnose: Migraine
Behandeling: Sumatriptan 50mg bij aanval
            """
        },
        {
            "filename": "patient_002.txt", 
            "content": """
Patiënt: Mohamed El Amrani
Geboortedatum: 10-12-1965
Adres: Nieuwstraat 78, 9000 Gent
Email: m.elamrani@email.be

Dossier nummer: MED456789123
Datum consult: 20-10-2023
Arts: Dr. Sarah Van Damme

Klachten: Rugpijn sinds 2 weken
Behandeling: Fysiotherapie, ibuprofen 400mg
            """
        },
        {
            "filename": "patient_003.txt",
            "content": """
Patiënt: Lisa Peeters
Geboortedatum: 05-07-1992
Adres: Kerkplein 12, 3000 Leuven
Telefoon: 016-123-456

Patiënt ID: PAT111222333
Opname datum: 25-10-2023
Afdeling: Gynaecologie

Reden opname: Zwangerschapscontrole (32 weken)
Behandelend arts: Dr. Emma Claes
            """
        }
    ]
    
    # Create input directory and files
    input_dir = Path("examples/batch_input")
    input_dir.mkdir(exist_ok=True)
    
    for doc in sample_documents:
        file_path = input_dir / doc["filename"]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc["content"])
    
    print(f"✅ Created {len(sample_documents)} sample documents in {input_dir}")
    return input_dir


def main():
    """Main batch processing example."""
    print("🏥 Belgian Document Deidentification System - Batch Processing Example")
    print("=" * 75)
    
    try:
        # Step 1: Create sample documents
        print("\n📁 Step 1: Creating sample documents...")
        input_dir = create_sample_documents()
        
        # Step 2: Configure batch processor
        print("\n⚙️  Step 2: Configuring batch processor...")
        
        config = Config(
            debug=True,
            deidentification={
                'mode': DeidentificationMode.ANONYMIZATION,
                'entities': ['PERSON', 'ADDRESS', 'DATE', 'MEDICAL_ID', 'PHONE', 'EMAIL'],
                'preserve_structure': True,
            }
        )
        
        output_dir = Path("examples/batch_output")
        output_dir.mkdir(exist_ok=True)
        
        batch_processor = BatchProcessor(
            config=config,
            input_dir=input_dir,
            output_dir=output_dir,
            max_workers=2,  # Use 2 workers for this example
            use_multiprocessing=True
        )
        
        print(f"✅ Batch processor configured")
        print(f"   Input directory: {input_dir}")
        print(f"   Output directory: {output_dir}")
        print(f"   Max workers: 2")
        
        # Step 3: Estimate processing time
        print("\n⏱️  Step 3: Estimating processing time...")
        
        # Count input files
        input_files = list(input_dir.glob("*.txt"))
        num_files = len(input_files)
        
        time_estimates = batch_processor.estimate_processing_time(
            num_documents=num_files,
            avg_document_size=500  # Estimated average size
        )
        
        print(f"📊 Processing estimates for {num_files} documents:")
        print(f"   Sequential time: {time_estimates['sequential_estimate']:.1f} seconds")
        print(f"   Parallel time: {time_estimates['parallel_estimate']:.1f} seconds")
        print(f"   Speedup factor: {time_estimates['speedup_factor']:.1f}x")
        
        # Step 4: Process all documents
        print("\n🔄 Step 4: Processing documents...")
        start_time = time.time()
        
        # Process with progress monitoring
        batch_result = batch_processor.process_all(
            file_patterns=["*.txt"],
            recursive=False,
            save_results=True
        )
        
        processing_time = time.time() - start_time
        
        # Step 5: Display results
        print("\n📊 Step 5: Batch Processing Results")
        print("-" * 50)
        
        print(f"✅ Processing completed in {processing_time:.2f} seconds")
        print(f"📄 Total documents: {batch_result.total_documents}")
        print(f"✅ Successful: {batch_result.successful_documents}")
        print(f"❌ Failed: {batch_result.failed_documents}")
        print(f"🔍 Total entities found: {batch_result.total_entities_found}")
        print(f"🗑️  Total entities removed: {batch_result.total_entities_removed}")
        print(f"⏱️  Average processing time: {batch_result.average_processing_time:.2f}s per document")
        
        # Show success rate
        if batch_result.total_documents > 0:
            success_rate = (batch_result.successful_documents / batch_result.total_documents) * 100
            print(f"📈 Success rate: {success_rate:.1f}%")
        
        # Step 6: Show individual results
        print("\n📋 Step 6: Individual Document Results")
        print("-" * 50)
        
        for i, result in enumerate(batch_result.results):
            input_file = input_files[i].name if i < len(input_files) else f"document_{i+1}"
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            
            print(f"{input_file}: {status}")
            print(f"  Entities found: {result.entities_found}")
            print(f"  Entities removed: {result.entities_removed}")
            print(f"  Processing time: {result.processing_time:.2f}s")
            
            if result.errors:
                print(f"  Errors: {', '.join(result.errors)}")
            
            if result.warnings:
                print(f"  Warnings: {', '.join(result.warnings)}")
            
            print()
        
        # Step 7: Show errors if any
        if batch_result.errors:
            print("\n🚨 Step 7: Processing Errors")
            print("-" * 50)
            for error in batch_result.errors:
                print(f"❌ {error}")
        
        # Step 8: Verify output files
        print("\n📁 Step 8: Verifying output files...")
        output_files = list(output_dir.glob("deidentified_*.txt"))
        print(f"✅ Created {len(output_files)} output files:")
        
        for output_file in output_files:
            file_size = output_file.stat().st_size
            print(f"  📄 {output_file.name} ({file_size} bytes)")
        
        # Step 9: Show sample output
        if output_files:
            print("\n📄 Step 9: Sample deidentified output")
            print("-" * 50)
            
            sample_file = output_files[0]
            with open(sample_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Show first 300 characters
                preview = content[:300] + "..." if len(content) > 300 else content
                print(f"File: {sample_file.name}")
                print(preview)
        
        # Step 10: Batch statistics
        print("\n📊 Step 10: Batch Processor Statistics")
        print("-" * 50)
        
        stats = batch_processor.get_processing_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Batch processing example failed: {e}", exc_info=True)
        print(f"❌ Batch processing failed: {e}")
        return 1
    
    print("\n🎉 Batch processing example completed successfully!")
    print("\n💡 Next steps:")
    print("   - Review the deidentified documents in examples/batch_output/")
    print("   - Check the batch results in examples/batch_output/batch_results/")
    print("   - Adjust configuration for your specific needs")
    
    return 0


if __name__ == "__main__":
    exit(main())

