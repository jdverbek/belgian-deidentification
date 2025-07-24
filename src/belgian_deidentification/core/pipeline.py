"""Main deidentification pipeline for processing Belgian healthcare documents."""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from ..nlp.dutch_clinical_nlp import DutchClinicalNLP
from ..entities.entity_recognizer import EntityRecognizer
from ..deidentification.deidentification_engine import DeidentificationEngine
from ..quality.quality_assurance import QualityAssurance
from ..utils.document_loader import DocumentLoader
from ..utils.document_saver import DocumentSaver
from .config import Config, get_config


@dataclass
class ProcessingResult:
    """Result of document processing."""
    success: bool
    entities_found: int
    entities_removed: int
    processing_time: float
    confidence_scores: Dict[str, float]
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]


class DeidentificationPipeline:
    """
    Main pipeline for deidentifying Belgian healthcare documents.
    
    This pipeline orchestrates the entire deidentification process:
    1. Document loading and preprocessing
    2. Dutch clinical NLP processing
    3. Entity recognition and classification
    4. Deidentification processing
    5. Quality assurance and validation
    6. Output generation
    """
    
    def __init__(self, config: Optional[Config] = None, **kwargs):
        """
        Initialize the deidentification pipeline.
        
        Args:
            config: Configuration object. If None, loads from default sources.
            **kwargs: Additional configuration overrides.
        """
        self.config = config or get_config()
        
        # Apply any configuration overrides
        if kwargs:
            config_dict = self.config.dict()
            config_dict.update(kwargs)
            self.config = Config(**config_dict)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.stats = {
            'documents_processed': 0,
            'total_entities_found': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
        }
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        self.logger.info("Initializing deidentification pipeline components...")
        
        try:
            # Document handling
            self.document_loader = DocumentLoader(self.config)
            self.document_saver = DocumentSaver(self.config)
            
            # NLP processing
            self.nlp_processor = DutchClinicalNLP(self.config)
            
            # Entity recognition
            self.entity_recognizer = EntityRecognizer(self.config)
            
            # Deidentification
            self.deidentification_engine = DeidentificationEngine(self.config)
            
            # Quality assurance
            self.quality_assurance = QualityAssurance(self.config)
            
            self.logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            raise
    
    def process_document(
        self,
        document_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process a single document for deidentification.
        
        Args:
            document_path: Path to the input document
            output_path: Path for the output document (optional)
            metadata: Additional metadata for processing
            
        Returns:
            ProcessingResult with details about the processing
        """
        start_time = time.time()
        document_path = Path(document_path)
        
        self.logger.info(f"Processing document: {document_path}")
        
        result = ProcessingResult(
            success=False,
            entities_found=0,
            entities_removed=0,
            processing_time=0.0,
            confidence_scores={},
            warnings=[],
            errors=[],
            metadata=metadata or {}
        )
        
        try:
            # Step 1: Load and preprocess document
            self.logger.debug("Loading document...")
            document = self.document_loader.load(document_path)
            
            if not document.text.strip():
                result.errors.append("Document contains no text content")
                return result
            
            # Step 2: NLP processing
            self.logger.debug("Processing with Dutch clinical NLP...")
            nlp_result = self.nlp_processor.process(document.text)
            
            # Step 3: Entity recognition
            self.logger.debug("Recognizing entities...")
            entities = self.entity_recognizer.recognize(nlp_result)
            result.entities_found = len(entities)
            
            if not entities:
                result.warnings.append("No PHI entities detected in document")
            
            # Step 4: Deidentification
            self.logger.debug("Applying deidentification...")
            deidentified_text = self.deidentification_engine.deidentify(
                text=document.text,
                entities=entities,
                nlp_result=nlp_result
            )
            result.entities_removed = len([e for e in entities if e.action_taken])
            
            # Step 5: Quality assurance
            self.logger.debug("Running quality assurance...")
            qa_result = self.quality_assurance.validate(
                original_text=document.text,
                deidentified_text=deidentified_text,
                entities=entities
            )
            
            result.confidence_scores = qa_result.confidence_scores
            result.warnings.extend(qa_result.warnings)
            
            if qa_result.passed:
                # Step 6: Save output
                if output_path:
                    self.logger.debug(f"Saving deidentified document to: {output_path}")
                    self.document_saver.save(
                        text=deidentified_text,
                        output_path=output_path,
                        original_format=document.format,
                        metadata={
                            **result.metadata,
                            'entities_found': result.entities_found,
                            'entities_removed': result.entities_removed,
                            'processing_timestamp': time.time(),
                        }
                    )
                
                result.success = True
                self.logger.info(f"Successfully processed document: {document_path}")
            else:
                result.errors.append("Document failed quality assurance validation")
                self.logger.warning(f"Document failed QA validation: {document_path}")
        
        except Exception as e:
            error_msg = f"Error processing document {document_path}: {str(e)}"
            result.errors.append(error_msg)
            self.logger.error(error_msg, exc_info=True)
        
        finally:
            # Update timing and stats
            result.processing_time = time.time() - start_time
            self._update_stats(result)
        
        return result
    
    def process_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessingResult:
        """
        Process raw text for deidentification.
        
        Args:
            text: Input text to process
            metadata: Additional metadata for processing
            
        Returns:
            ProcessingResult with details about the processing
        """
        start_time = time.time()
        
        self.logger.info("Processing text input")
        
        result = ProcessingResult(
            success=False,
            entities_found=0,
            entities_removed=0,
            processing_time=0.0,
            confidence_scores={},
            warnings=[],
            errors=[],
            metadata=metadata or {}
        )
        
        try:
            if not text.strip():
                result.errors.append("Input text is empty")
                return result
            
            # NLP processing
            nlp_result = self.nlp_processor.process(text)
            
            # Entity recognition
            entities = self.entity_recognizer.recognize(nlp_result)
            result.entities_found = len(entities)
            
            # Deidentification
            deidentified_text = self.deidentification_engine.deidentify(
                text=text,
                entities=entities,
                nlp_result=nlp_result
            )
            result.entities_removed = len([e for e in entities if e.action_taken])
            
            # Quality assurance
            qa_result = self.quality_assurance.validate(
                original_text=text,
                deidentified_text=deidentified_text,
                entities=entities
            )
            
            result.confidence_scores = qa_result.confidence_scores
            result.warnings.extend(qa_result.warnings)
            result.success = qa_result.passed
            
            if not qa_result.passed:
                result.errors.append("Text failed quality assurance validation")
            
            # Store deidentified text in metadata
            result.metadata['deidentified_text'] = deidentified_text
        
        except Exception as e:
            error_msg = f"Error processing text: {str(e)}"
            result.errors.append(error_msg)
            self.logger.error(error_msg, exc_info=True)
        
        finally:
            result.processing_time = time.time() - start_time
            self._update_stats(result)
        
        return result
    
    def _update_stats(self, result: ProcessingResult) -> None:
        """Update pipeline statistics."""
        self.stats['documents_processed'] += 1
        self.stats['total_entities_found'] += result.entities_found
        self.stats['total_processing_time'] += result.processing_time
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['documents_processed']
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset pipeline statistics."""
        self.stats = {
            'documents_processed': 0,
            'total_entities_found': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
        }
    
    def validate_configuration(self) -> List[str]:
        """
        Validate the current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Ensure required directories exist
            self.config.ensure_directories()
            
            # Validate model availability
            model_path = self.config.get_model_path(self.config.nlp.model)
            if not model_path.exists():
                errors.append(f"Model not found: {model_path}")
            
            # Validate entity types
            for entity_type in self.config.deidentification.entities:
                if entity_type not in ['PERSON', 'ADDRESS', 'DATE', 'MEDICAL_ID', 'PHONE', 'EMAIL', 'ORGANIZATION']:
                    errors.append(f"Unknown entity type: {entity_type}")
            
            # Validate confidence thresholds
            if not 0.0 <= self.config.nlp.confidence_threshold <= 1.0:
                errors.append("NLP confidence threshold must be between 0.0 and 1.0")
            
        except Exception as e:
            errors.append(f"Configuration validation error: {str(e)}")
        
        return errors
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup resources if needed
        pass

