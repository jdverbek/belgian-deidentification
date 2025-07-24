"""Main deidentification engine for processing detected entities."""

import logging
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..entities.entity_types import PHIEntity, EntityType
from ..nlp.dutch_clinical_nlp import NLPResult
from ..core.config import Config, DeidentificationMode
from .anonymization_strategies import AnonymizationStrategy, RemovalStrategy, ReplacementStrategy
from .pseudonymization_strategies import PseudonymizationStrategy, CryptoStrategy
from .replacement_generators import ReplacementGenerator


@dataclass
class DeidentificationResult:
    """Result of deidentification processing."""
    deidentified_text: str
    entities_processed: int
    entities_removed: int
    entities_replaced: int
    processing_metadata: Dict[str, Any]


class DeidentificationEngine:
    """
    Main deidentification engine for Belgian healthcare documents.
    
    This engine provides:
    1. Flexible deidentification strategies (anonymization/pseudonymization)
    2. Context-aware entity replacement
    3. Document structure preservation
    4. Audit trail generation
    5. Quality assurance integration
    """
    
    def __init__(self, config: Config):
        """
        Initialize the deidentification engine.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Replacement generator
        self.replacement_generator = ReplacementGenerator(config)
        
        # Processing statistics
        self.stats = {
            'documents_processed': 0,
            'entities_processed': 0,
            'entities_by_action': {},
            'processing_time': 0.0,
        }
    
    def _initialize_strategies(self) -> None:
        """Initialize deidentification strategies."""
        self.logger.info("Initializing deidentification strategies...")
        
        try:
            # Anonymization strategies
            self.removal_strategy = RemovalStrategy(self.config)
            self.replacement_strategy = ReplacementStrategy(self.config)
            
            # Pseudonymization strategies
            self.crypto_strategy = CryptoStrategy(self.config)
            
            self.logger.info("Deidentification strategies initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize deidentification strategies: {e}")
            raise
    
    def deidentify(
        self,
        text: str,
        entities: List[PHIEntity],
        nlp_result: Optional[NLPResult] = None
    ) -> str:
        """
        Deidentify text by processing detected entities.
        
        Args:
            text: Original text to deidentify
            entities: List of detected PHI entities
            nlp_result: Optional NLP processing result for context
            
        Returns:
            Deidentified text
        """
        self.logger.debug(f"Deidentifying text with {len(entities)} entities")
        
        if not entities:
            self.logger.info("No entities to deidentify")
            return text
        
        try:
            # Sort entities by position (reverse order for text modification)
            sorted_entities = sorted(entities, key=lambda e: e.start, reverse=True)
            
            # Process each entity
            deidentified_text = text
            entities_processed = 0
            entities_removed = 0
            entities_replaced = 0
            
            for entity in sorted_entities:
                try:
                    # Determine deidentification action
                    action, replacement_text = self._determine_action(entity, nlp_result)
                    
                    # Apply deidentification
                    deidentified_text = self._apply_deidentification(
                        deidentified_text, entity, action, replacement_text
                    )
                    
                    # Update entity with action taken
                    entity.action_taken = action
                    entity.replacement_text = replacement_text
                    
                    # Update counters
                    entities_processed += 1
                    if action == "removed":
                        entities_removed += 1
                    elif action in ["replaced", "pseudonymized"]:
                        entities_replaced += 1
                    
                    self.logger.debug(f"Processed entity: {entity.text} -> {action}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to process entity {entity.text}: {e}")
                    continue
            
            # Update statistics
            self._update_stats(entities_processed, entities_removed, entities_replaced)
            
            self.logger.info(
                f"Deidentification completed: {entities_processed} entities processed, "
                f"{entities_removed} removed, {entities_replaced} replaced"
            )
            
            return deidentified_text
            
        except Exception as e:
            self.logger.error(f"Error in deidentification: {e}")
            return text  # Return original text on error
    
    def _determine_action(
        self, entity: PHIEntity, nlp_result: Optional[NLPResult] = None
    ) -> Tuple[str, Optional[str]]:
        """
        Determine the appropriate deidentification action for an entity.
        
        Args:
            entity: PHI entity to process
            nlp_result: Optional NLP result for context
            
        Returns:
            Tuple of (action, replacement_text)
        """
        mode = self.config.deidentification.mode
        
        # High-risk entities are always processed aggressively
        if entity.risk_level == "critical":
            if mode == DeidentificationMode.ANONYMIZATION:
                return "removed", None
            else:  # pseudonymization
                replacement = self._generate_pseudonym(entity)
                return "pseudonymized", replacement
        
        # Medium/high risk entities
        elif entity.risk_level in ["high", "medium"]:
            if mode == DeidentificationMode.ANONYMIZATION:
                # Generate realistic replacement
                replacement = self._generate_replacement(entity, nlp_result)
                return "replaced", replacement
            else:  # pseudonymization
                replacement = self._generate_pseudonym(entity)
                return "pseudonymized", replacement
        
        # Low risk entities
        else:
            if mode == DeidentificationMode.ANONYMIZATION:
                # For dates, apply date shifting if configured
                if entity.entity_type == EntityType.DATE and self.config.deidentification.date_shift_days:
                    shifted_date = self._shift_date(entity.text, self.config.deidentification.date_shift_days)
                    return "replaced", shifted_date
                else:
                    # Generate replacement
                    replacement = self._generate_replacement(entity, nlp_result)
                    return "replaced", replacement
            else:  # pseudonymization
                replacement = self._generate_pseudonym(entity)
                return "pseudonymized", replacement
    
    def _apply_deidentification(
        self, text: str, entity: PHIEntity, action: str, replacement_text: Optional[str]
    ) -> str:
        """
        Apply deidentification action to text.
        
        Args:
            text: Text to modify
            entity: Entity to process
            action: Action to take
            replacement_text: Replacement text (if applicable)
            
        Returns:
            Modified text
        """
        start, end = entity.start, entity.end
        
        if action == "removed":
            # Remove the entity text
            if self.config.deidentification.preserve_structure:
                # Replace with placeholder of same length
                replacement = "[REMOVED]"
                if len(replacement) < (end - start):
                    replacement += " " * ((end - start) - len(replacement))
                return text[:start] + replacement[:end-start] + text[end:]
            else:
                # Complete removal
                return text[:start] + text[end:]
        
        elif action in ["replaced", "pseudonymized"]:
            # Replace with new text
            if replacement_text:
                return text[:start] + replacement_text + text[end:]
            else:
                # Fallback to removal
                return text[:start] + "[REDACTED]" + text[end:]
        
        # No action
        return text
    
    def _generate_replacement(
        self, entity: PHIEntity, nlp_result: Optional[NLPResult] = None
    ) -> str:
        """Generate realistic replacement for an entity."""
        try:
            return self.replacement_generator.generate_replacement(entity, nlp_result)
        except Exception as e:
            self.logger.warning(f"Failed to generate replacement for {entity.text}: {e}")
            return self._get_default_replacement(entity)
    
    def _generate_pseudonym(self, entity: PHIEntity) -> str:
        """Generate cryptographic pseudonym for an entity."""
        try:
            return self.crypto_strategy.generate_pseudonym(entity)
        except Exception as e:
            self.logger.warning(f"Failed to generate pseudonym for {entity.text}: {e}")
            return self._get_default_replacement(entity)
    
    def _get_default_replacement(self, entity: PHIEntity) -> str:
        """Get default replacement text for an entity type."""
        defaults = {
            EntityType.PERSON: "[NAAM]",
            EntityType.ADDRESS: "[ADRES]",
            EntityType.DATE: "[DATUM]",
            EntityType.MEDICAL_ID: "[ID]",
            EntityType.PHONE: "[TELEFOON]",
            EntityType.EMAIL: "[EMAIL]",
            EntityType.ORGANIZATION: "[ORGANISATIE]",
            EntityType.AGE: "[LEEFTIJD]",
        }
        
        return defaults.get(entity.entity_type, "[GEREDACTEERD]")
    
    def _shift_date(self, date_text: str, shift_days: int) -> str:
        """
        Shift a date by specified number of days.
        
        Args:
            date_text: Original date text
            shift_days: Number of days to shift
            
        Returns:
            Shifted date text
        """
        try:
            from datetime import datetime, timedelta
            
            # Try to parse common Dutch date formats
            date_formats = [
                "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
                "%d-%m-%y", "%d/%m/%y", "%d.%m.%y",
                "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"
            ]
            
            parsed_date = None
            original_format = None
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_text, fmt)
                    original_format = fmt
                    break
                except ValueError:
                    continue
            
            if parsed_date:
                # Apply shift
                shifted_date = parsed_date + timedelta(days=shift_days)
                return shifted_date.strftime(original_format)
            else:
                self.logger.warning(f"Could not parse date: {date_text}")
                return "[DATUM]"
                
        except Exception as e:
            self.logger.warning(f"Date shifting failed for {date_text}: {e}")
            return "[DATUM]"
    
    def _update_stats(self, processed: int, removed: int, replaced: int) -> None:
        """Update processing statistics."""
        self.stats['documents_processed'] += 1
        self.stats['entities_processed'] += processed
        
        # Update action counts
        if removed > 0:
            self.stats['entities_by_action']['removed'] = (
                self.stats['entities_by_action'].get('removed', 0) + removed
            )
        
        if replaced > 0:
            self.stats['entities_by_action']['replaced'] = (
                self.stats['entities_by_action'].get('replaced', 0) + replaced
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self.stats = {
            'documents_processed': 0,
            'entities_processed': 0,
            'entities_by_action': {},
            'processing_time': 0.0,
        }
    
    def validate_deidentification(
        self, original_text: str, deidentified_text: str, entities: List[PHIEntity]
    ) -> Dict[str, Any]:
        """
        Validate that deidentification was successful.
        
        Args:
            original_text: Original text
            deidentified_text: Deidentified text
            entities: List of processed entities
            
        Returns:
            Validation result dictionary
        """
        validation_result = {
            'passed': True,
            'issues': [],
            'entity_check': {},
            'text_similarity': 0.0,
        }
        
        try:
            # Check that processed entities are not present in deidentified text
            for entity in entities:
                if entity.action_taken in ["removed", "replaced", "pseudonymized"]:
                    if entity.text.lower() in deidentified_text.lower():
                        validation_result['passed'] = False
                        validation_result['issues'].append(
                            f"Entity '{entity.text}' still present after {entity.action_taken}"
                        )
                        validation_result['entity_check'][entity.text] = False
                    else:
                        validation_result['entity_check'][entity.text] = True
            
            # Calculate text similarity (should be high for structure preservation)
            similarity = self._calculate_text_similarity(original_text, deidentified_text)
            validation_result['text_similarity'] = similarity
            
            # Check for structure preservation
            if self.config.deidentification.preserve_structure:
                if abs(len(original_text) - len(deidentified_text)) > len(original_text) * 0.1:
                    validation_result['issues'].append("Text length changed significantly")
            
        except Exception as e:
            validation_result['passed'] = False
            validation_result['issues'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        try:
            # Simple character-level similarity
            if not text1 or not text2:
                return 0.0
            
            # Count matching characters
            matches = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
            max_length = max(len(text1), len(text2))
            
            return matches / max_length if max_length > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def create_audit_record(
        self, original_text: str, deidentified_text: str, entities: List[PHIEntity]
    ) -> Dict[str, Any]:
        """
        Create audit record for deidentification process.
        
        Args:
            original_text: Original text
            deidentified_text: Deidentified text
            entities: List of processed entities
            
        Returns:
            Audit record dictionary
        """
        import time
        
        audit_record = {
            'timestamp': time.time(),
            'mode': self.config.deidentification.mode.value,
            'original_text_hash': hashlib.sha256(original_text.encode()).hexdigest(),
            'deidentified_text_hash': hashlib.sha256(deidentified_text.encode()).hexdigest(),
            'entities_processed': len(entities),
            'entities_by_type': {},
            'entities_by_action': {},
            'configuration': {
                'preserve_structure': self.config.deidentification.preserve_structure,
                'date_shift_days': self.config.deidentification.date_shift_days,
                'replacement_strategy': self.config.deidentification.replacement_strategy,
            }
        }
        
        # Count entities by type and action
        for entity in entities:
            entity_type = entity.entity_type.value
            audit_record['entities_by_type'][entity_type] = (
                audit_record['entities_by_type'].get(entity_type, 0) + 1
            )
            
            if entity.action_taken:
                action = entity.action_taken
                audit_record['entities_by_action'][action] = (
                    audit_record['entities_by_action'].get(action, 0) + 1
                )
        
        return audit_record

