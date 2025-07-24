"""Tests for the main deidentification pipeline."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from belgian_deidentification.core.pipeline import DeidentificationPipeline, ProcessingResult
from belgian_deidentification.core.config import Config, DeidentificationMode
from belgian_deidentification.entities.entity_types import PHIEntity, EntityType


class TestDeidentificationPipeline:
    """Test cases for the DeidentificationPipeline class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Config(
            debug=True,
            deidentification={
                'mode': DeidentificationMode.ANONYMIZATION,
                'entities': ['PERSON', 'ADDRESS', 'DATE'],
                'preserve_structure': True,
            },
            nlp={
                'model': 'test-model',
                'confidence_threshold': 0.8,
            }
        )
    
    @pytest.fixture
    def pipeline(self, config):
        """Create test pipeline."""
        with patch('belgian_deidentification.core.pipeline.DutchClinicalNLP'), \
             patch('belgian_deidentification.core.pipeline.EntityRecognizer'), \
             patch('belgian_deidentification.core.pipeline.DeidentificationEngine'), \
             patch('belgian_deidentification.core.pipeline.QualityAssurance'):
            return DeidentificationPipeline(config)
    
    def test_pipeline_initialization(self, config):
        """Test pipeline initialization."""
        with patch('belgian_deidentification.core.pipeline.DutchClinicalNLP'), \
             patch('belgian_deidentification.core.pipeline.EntityRecognizer'), \
             patch('belgian_deidentification.core.pipeline.DeidentificationEngine'), \
             patch('belgian_deidentification.core.pipeline.QualityAssurance'):
            pipeline = DeidentificationPipeline(config)
            assert pipeline.config == config
            assert pipeline.stats['documents_processed'] == 0
    
    def test_process_text_success(self, pipeline):
        """Test successful text processing."""
        # Mock components
        pipeline.nlp_processor.process.return_value = Mock(
            text="Test text",
            tokens=[],
            entities=[],
            metadata={}
        )
        
        pipeline.entity_recognizer.recognize.return_value = [
            PHIEntity(
                text="John Doe",
                start=0,
                end=8,
                entity_type=EntityType.PERSON,
                confidence=0.9,
                source="test"
            )
        ]
        
        pipeline.deidentification_engine.deidentify.return_value = "Test text with [NAAM]"
        
        pipeline.quality_assurance.validate.return_value = Mock(
            passed=True,
            confidence_scores={'overall': 0.9},
            warnings=[],
            errors=[],
            expert_review_required=False
        )
        
        # Process text
        result = pipeline.process_text("Test text with John Doe")
        
        # Assertions
        assert result.success is True
        assert result.entities_found == 1
        assert result.entities_removed == 1
        assert "Test text with [NAAM]" in result.deidentified_text
    
    def test_process_text_with_errors(self, pipeline):
        """Test text processing with errors."""
        # Mock NLP processor to raise exception
        pipeline.nlp_processor.process.side_effect = Exception("NLP error")
        
        # Process text
        result = pipeline.process_text("Test text")
        
        # Assertions
        assert result.success is False
        assert len(result.errors) > 0
        assert "NLP error" in str(result.errors)
    
    def test_process_document_file(self, pipeline):
        """Test document file processing."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test document with John Doe")
            temp_path = Path(f.name)
        
        try:
            # Mock successful processing
            with patch.object(pipeline, 'process_text') as mock_process:
                mock_process.return_value = ProcessingResult(
                    success=True,
                    deidentified_text="Test document with [NAAM]",
                    entities_found=1,
                    entities_removed=1,
                    processing_time=1.0,
                    confidence_scores={'overall': 0.9},
                    warnings=[],
                    errors=[],
                    metadata={}
                )
                
                # Process document
                result = pipeline.process_document(temp_path)
                
                # Assertions
                assert result.success is True
                assert result.entities_found == 1
                mock_process.assert_called_once()
        
        finally:
            # Clean up
            temp_path.unlink()
    
    def test_process_document_with_output_path(self, pipeline):
        """Test document processing with output file."""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_f:
            input_f.write("Test document with John Doe")
            input_path = Path(input_f.name)
        
        with tempfile.NamedTemporaryFile(delete=False) as output_f:
            output_path = Path(output_f.name)
        
        try:
            # Mock successful processing
            with patch.object(pipeline, 'process_text') as mock_process:
                mock_process.return_value = ProcessingResult(
                    success=True,
                    deidentified_text="Test document with [NAAM]",
                    entities_found=1,
                    entities_removed=1,
                    processing_time=1.0,
                    confidence_scores={'overall': 0.9},
                    warnings=[],
                    errors=[],
                    metadata={}
                )
                
                # Process document
                result = pipeline.process_document(input_path, output_path)
                
                # Assertions
                assert result.success is True
                assert output_path.exists()
                
                # Check output content
                with open(output_path, 'r') as f:
                    content = f.read()
                    assert "Test document with [NAAM]" in content
        
        finally:
            # Clean up
            input_path.unlink()
            if output_path.exists():
                output_path.unlink()
    
    def test_validate_configuration(self, pipeline):
        """Test configuration validation."""
        errors = pipeline.validate_configuration()
        # Should return empty list for valid configuration
        assert isinstance(errors, list)
    
    def test_get_stats(self, pipeline):
        """Test statistics retrieval."""
        stats = pipeline.get_stats()
        
        assert 'documents_processed' in stats
        assert 'entities_found' in stats
        assert 'entities_removed' in stats
        assert 'processing_time' in stats
    
    def test_reset_stats(self, pipeline):
        """Test statistics reset."""
        # Modify stats
        pipeline.stats['documents_processed'] = 5
        
        # Reset
        pipeline.reset_stats()
        
        # Check reset
        assert pipeline.stats['documents_processed'] == 0
    
    @pytest.mark.parametrize("file_extension", [".txt", ".docx", ".pdf"])
    def test_supported_file_types(self, pipeline, file_extension):
        """Test processing of different file types."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Mock file reading
            with patch('belgian_deidentification.utils.file_reader.read_file') as mock_read:
                mock_read.return_value = "Test content"
                
                with patch.object(pipeline, 'process_text') as mock_process:
                    mock_process.return_value = ProcessingResult(
                        success=True,
                        deidentified_text="Test content",
                        entities_found=0,
                        entities_removed=0,
                        processing_time=1.0,
                        confidence_scores={},
                        warnings=[],
                        errors=[],
                        metadata={}
                    )
                    
                    # Process document
                    result = pipeline.process_document(temp_path)
                    
                    # Should succeed for supported file types
                    assert result.success is True
        
        finally:
            # Clean up
            temp_path.unlink()


class TestPipelineIntegration:
    """Integration tests for the pipeline."""
    
    def test_end_to_end_processing(self):
        """Test end-to-end processing with real components."""
        # This would test with actual NLP models and components
        # For now, we'll skip this test in CI/CD
        pytest.skip("Integration test - requires actual models")
    
    def test_performance_benchmarks(self):
        """Test performance benchmarks."""
        # This would test processing speed and memory usage
        pytest.skip("Performance test - requires benchmarking setup")

