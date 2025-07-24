#!/usr/bin/env python3
"""
Basic usage example for the Belgian Document Deidentification System.

This example demonstrates how to:
1. Initialize the deidentification pipeline
2. Process a simple Dutch medical text
3. Handle the results
"""

import logging
from pathlib import Path

from belgian_deidentification.core.pipeline import DeidentificationPipeline
from belgian_deidentification.core.config import Config, DeidentificationMode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    print("🏥 Belgian Document Deidentification System - Basic Usage Example")
    print("=" * 70)
    
    # Sample Dutch medical text with PHI
    sample_text = """
    Patiënt: Jan van der Berg
    Geboortedatum: 15-03-1975
    Adres: Kerkstraat 123, 1000 Brussel
    Telefoon: 02-123-4567
    Email: jan.vandenberg@email.be
    
    Medisch dossier nummer: PAT123456789
    
    Opname datum: 12-11-2023
    Behandelend arts: Dr. Marie Dubois
    Afdeling: Cardiologie, UZ Brussel
    
    Anamnese:
    De 48-jarige patiënt Jan van der Berg presenteert zich met klachten van 
    thoracale pijn sinds 3 dagen. Patiënt heeft een voorgeschiedenis van 
    hypertensie en diabetes mellitus type 2.
    
    Onderzoek:
    Bloeddruk: 160/95 mmHg
    Hartfrequentie: 88 bpm
    ECG: Normaal sinusritme
    
    Diagnose:
    Stabiele angina pectoris
    
    Behandeling:
    - Metoprolol 50mg 2x daags
    - Atorvastatine 20mg 1x daags
    - Controle over 4 weken bij Dr. Dubois
    
    Datum rapport: 12-11-2023
    """
    
    try:
        # Step 1: Create configuration
        print("\n📋 Step 1: Creating configuration...")
        config = Config(
            debug=True,
            deidentification={
                'mode': DeidentificationMode.ANONYMIZATION,
                'entities': ['PERSON', 'ADDRESS', 'DATE', 'MEDICAL_ID', 'PHONE', 'EMAIL'],
                'preserve_structure': True,
            },
            nlp={
                'confidence_threshold': 0.8,
            }
        )
        print(f"✅ Configuration created with mode: {config.deidentification.mode}")
        
        # Step 2: Initialize pipeline
        print("\n🔧 Step 2: Initializing deidentification pipeline...")
        pipeline = DeidentificationPipeline(config)
        print("✅ Pipeline initialized successfully")
        
        # Step 3: Process the text
        print("\n🔍 Step 3: Processing medical text...")
        print(f"Original text length: {len(sample_text)} characters")
        
        result = pipeline.process_text(sample_text)
        
        # Step 4: Display results
        print("\n📊 Step 4: Processing Results")
        print("-" * 40)
        
        if result.success:
            print("✅ Processing completed successfully!")
            print(f"📈 Entities found: {result.entities_found}")
            print(f"🗑️  Entities removed/replaced: {result.entities_removed}")
            print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")
            
            # Show confidence scores
            if result.confidence_scores:
                print("\n🎯 Confidence Scores:")
                for metric, score in result.confidence_scores.items():
                    print(f"  {metric}: {score:.3f}")
            
            # Show warnings if any
            if result.warnings:
                print("\n⚠️  Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")
            
            # Display deidentified text
            print("\n📄 Deidentified Text:")
            print("-" * 40)
            print(result.deidentified_text)
            
        else:
            print("❌ Processing failed!")
            if result.errors:
                print("\n🚨 Errors:")
                for error in result.errors:
                    print(f"  - {error}")
        
        # Step 5: Save results (optional)
        print("\n💾 Step 5: Saving results...")
        output_dir = Path("examples/output")
        output_dir.mkdir(exist_ok=True)
        
        # Save deidentified text
        output_file = output_dir / "deidentified_example.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.deidentified_text)
        
        print(f"✅ Deidentified text saved to: {output_file}")
        
        # Step 6: Pipeline statistics
        print("\n📊 Step 6: Pipeline Statistics")
        print("-" * 40)
        stats = pipeline.get_stats()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"❌ Example failed: {e}")
        return 1
    
    print("\n🎉 Example completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())

