"""Main entity recognizer that coordinates different recognition approaches."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .entity_types import Entity, PHIEntity, EntityType, EntityValidator
from .rule_based_recognizer import RuleBasedRecognizer
from .ml_recognizer import MLRecognizer
from .ensemble_recognizer import EnsembleRecognizer
from ..nlp.dutch_clinical_nlp import NLPResult
from ..core.config import Config


@dataclass
class RecognitionResult:
    """Result of entity recognition."""
    entities: List[PHIEntity]
    confidence_scores: Dict[str, float]
    processing_metadata: Dict[str, Any]


class EntityRecognizer:
    """
    Main entity recognizer for Belgian healthcare documents.
    
    This recognizer coordinates multiple approaches:
    1. Rule-based recognition for high-precision patterns
    2. Machine learning recognition for context-aware detection
    3. Ensemble methods for combining results
    4. Validation and filtering of detected entities
    """
    
    def __init__(self, config: Config):
        """
        Initialize the entity recognizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize recognizers
        self._initialize_recognizers()
        
        # Entity validator
        self.validator = EntityValidator()
        
        # Recognition statistics
        self.stats = {
            'total_entities_detected': 0,
            'entities_by_type': {},
            'entities_by_source': {},
            'validation_pass_rate': 0.0,
        }
    
    def _initialize_recognizers(self) -> None:
        """Initialize all recognition components."""
        self.logger.info("Initializing entity recognizers...")
        
        try:
            # Rule-based recognizer
            self.rule_based_recognizer = RuleBasedRecognizer(self.config)
            
            # ML-based recognizer
            self.ml_recognizer = MLRecognizer(self.config)
            
            # Ensemble recognizer
            self.ensemble_recognizer = EnsembleRecognizer(self.config)
            
            self.logger.info("Entity recognizers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize entity recognizers: {e}")
            raise
    
    def recognize(self, nlp_result: NLPResult) -> List[PHIEntity]:
        """
        Recognize entities in processed text.
        
        Args:
            nlp_result: Result from Dutch clinical NLP processing
            
        Returns:
            List of detected PHI entities
        """
        self.logger.debug(f"Recognizing entities in text of length {len(nlp_result.text)}")
        
        try:
            # Step 1: Rule-based recognition
            rule_entities = self.rule_based_recognizer.recognize(nlp_result)
            self.logger.debug(f"Rule-based recognizer found {len(rule_entities)} entities")
            
            # Step 2: ML-based recognition
            ml_entities = self.ml_recognizer.recognize(nlp_result)
            self.logger.debug(f"ML recognizer found {len(ml_entities)} entities")
            
            # Step 3: Ensemble recognition
            ensemble_entities = self.ensemble_recognizer.combine_results(
                rule_entities, ml_entities, nlp_result
            )
            self.logger.debug(f"Ensemble recognizer produced {len(ensemble_entities)} entities")
            
            # Step 4: Validate entities
            validated_entities = self._validate_entities(ensemble_entities)
            self.logger.debug(f"Validation passed {len(validated_entities)} entities")
            
            # Step 5: Filter by configuration
            filtered_entities = self._filter_entities(validated_entities)
            self.logger.debug(f"Configuration filtering kept {len(filtered_entities)} entities")
            
            # Step 6: Convert to PHI entities and assign risk levels
            phi_entities = self._convert_to_phi_entities(filtered_entities)
            
            # Update statistics
            self._update_stats(phi_entities)
            
            self.logger.info(f"Entity recognition completed: {len(phi_entities)} PHI entities found")
            
            return phi_entities
            
        except Exception as e:
            self.logger.error(f"Error in entity recognition: {e}")
            return []
    
    def _validate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Validate detected entities."""
        validated_entities = []
        
        for entity in entities:
            try:
                if self.validator.validate_entity(entity):
                    validated_entities.append(entity)
                else:
                    self.logger.debug(f"Entity validation failed: {entity.text} ({entity.entity_type})")
            except Exception as e:
                self.logger.warning(f"Error validating entity {entity.text}: {e}")
        
        return validated_entities
    
    def _filter_entities(self, entities: List[Entity]) -> List[Entity]:
        """Filter entities based on configuration."""
        # Get enabled entity types from configuration
        enabled_types = set(self.config.deidentification.entities)
        
        filtered_entities = []
        for entity in entities:
            if entity.entity_type in enabled_types:
                # Apply confidence threshold
                if entity.confidence >= self.config.nlp.confidence_threshold:
                    filtered_entities.append(entity)
                else:
                    self.logger.debug(
                        f"Entity filtered by confidence: {entity.text} "
                        f"({entity.confidence:.3f} < {self.config.nlp.confidence_threshold})"
                    )
            else:
                self.logger.debug(f"Entity type not enabled: {entity.entity_type}")
        
        return filtered_entities
    
    def _convert_to_phi_entities(self, entities: List[Entity]) -> List[PHIEntity]:
        """Convert entities to PHI entities with risk assessment."""
        phi_entities = []
        
        for entity in entities:
            # Determine risk level
            risk_level = self._assess_risk_level(entity)
            
            # Create PHI entity
            phi_entity = PHIEntity(
                text=entity.text,
                start=entity.start,
                end=entity.end,
                entity_type=entity.entity_type,
                subtype=entity.subtype,
                confidence=entity.confidence,
                source=entity.source,
                metadata=entity.metadata.copy(),
                risk_level=risk_level,
            )
            
            phi_entities.append(phi_entity)
        
        return phi_entities
    
    def _assess_risk_level(self, entity: Entity) -> str:
        """Assess the privacy risk level of an entity."""
        # Risk assessment based on entity type and characteristics
        high_risk_types = {EntityType.PERSON, EntityType.MEDICAL_ID}
        medium_risk_types = {EntityType.ADDRESS, EntityType.PHONE, EntityType.EMAIL}
        low_risk_types = {EntityType.DATE, EntityType.AGE, EntityType.ORGANIZATION}
        
        if entity.entity_type in high_risk_types:
            # Further assessment for high-risk types
            if entity.entity_type == EntityType.PERSON:
                # Full names are higher risk than single names
                if len(entity.text.split()) > 1:
                    return "critical"
                else:
                    return "high"
            elif entity.entity_type == EntityType.MEDICAL_ID:
                # Longer IDs are typically higher risk
                if len(entity.text) > 8:
                    return "critical"
                else:
                    return "high"
        
        elif entity.entity_type in medium_risk_types:
            # Address with full details is higher risk
            if entity.entity_type == EntityType.ADDRESS and len(entity.text) > 20:
                return "high"
            else:
                return "medium"
        
        elif entity.entity_type in low_risk_types:
            return "low"
        
        # Default to medium risk
        return "medium"
    
    def _update_stats(self, entities: List[PHIEntity]) -> None:
        """Update recognition statistics."""
        self.stats['total_entities_detected'] += len(entities)
        
        # Update by type
        for entity in entities:
            entity_type = entity.entity_type.value
            self.stats['entities_by_type'][entity_type] = (
                self.stats['entities_by_type'].get(entity_type, 0) + 1
            )
            
            # Update by source
            source = entity.source
            self.stats['entities_by_source'][source] = (
                self.stats['entities_by_source'].get(source, 0) + 1
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get recognition statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset recognition statistics."""
        self.stats = {
            'total_entities_detected': 0,
            'entities_by_type': {},
            'entities_by_source': {},
            'validation_pass_rate': 0.0,
        }
    
    def recognize_text(self, text: str) -> List[PHIEntity]:
        """
        Convenience method to recognize entities in raw text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of detected PHI entities
        """
        # This would require importing and using the NLP processor
        # For now, return empty list with a warning
        self.logger.warning("recognize_text requires NLP processing - use recognize() with NLPResult")
        return []
    
    def get_entity_summary(self, entities: List[PHIEntity]) -> Dict[str, Any]:
        """
        Get summary statistics for a list of entities.
        
        Args:
            entities: List of PHI entities
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_entities': len(entities),
            'by_type': {},
            'by_risk_level': {},
            'by_source': {},
            'confidence_stats': {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
            }
        }
        
        if not entities:
            return summary
        
        # Count by type
        for entity in entities:
            entity_type = entity.entity_type.value
            summary['by_type'][entity_type] = summary['by_type'].get(entity_type, 0) + 1
            
            risk_level = entity.risk_level
            summary['by_risk_level'][risk_level] = summary['by_risk_level'].get(risk_level, 0) + 1
            
            source = entity.source
            summary['by_source'][source] = summary['by_source'].get(source, 0) + 1
        
        # Confidence statistics
        confidences = [entity.confidence for entity in entities]
        summary['confidence_stats'] = {
            'min': min(confidences),
            'max': max(confidences),
            'mean': sum(confidences) / len(confidences),
        }
        
        return summary
    
    def filter_entities_by_type(
        self, entities: List[PHIEntity], entity_types: List[EntityType]
    ) -> List[PHIEntity]:
        """
        Filter entities by type.
        
        Args:
            entities: List of PHI entities
            entity_types: List of entity types to keep
            
        Returns:
            Filtered list of entities
        """
        return [entity for entity in entities if entity.entity_type in entity_types]
    
    def filter_entities_by_confidence(
        self, entities: List[PHIEntity], min_confidence: float
    ) -> List[PHIEntity]:
        """
        Filter entities by confidence threshold.
        
        Args:
            entities: List of PHI entities
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered list of entities
        """
        return [entity for entity in entities if entity.confidence >= min_confidence]
    
    def filter_entities_by_risk(
        self, entities: List[PHIEntity], risk_levels: List[str]
    ) -> List[PHIEntity]:
        """
        Filter entities by risk level.
        
        Args:
            entities: List of PHI entities
            risk_levels: List of risk levels to keep
            
        Returns:
            Filtered list of entities
        """
        return [entity for entity in entities if entity.risk_level in risk_levels]
    
    def resolve_overlapping_entities(self, entities: List[PHIEntity]) -> List[PHIEntity]:
        """
        Resolve overlapping entities by keeping the highest confidence ones.
        
        Args:
            entities: List of PHI entities (may contain overlaps)
            
        Returns:
            List of non-overlapping entities
        """
        if not entities:
            return entities
        
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda e: e.start)
        
        resolved_entities = []
        
        for entity in sorted_entities:
            # Check for overlaps with already resolved entities
            overlaps = False
            for resolved_entity in resolved_entities:
                if entity.overlaps_with(resolved_entity):
                    overlaps = True
                    # Keep the entity with higher confidence
                    if entity.confidence > resolved_entity.confidence:
                        resolved_entities.remove(resolved_entity)
                        resolved_entities.append(entity)
                    break
            
            if not overlaps:
                resolved_entities.append(entity)
        
        return resolved_entities

