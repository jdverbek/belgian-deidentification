"""clinlp processor for Dutch clinical text processing."""

import logging
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc

try:
    import clinlp
except ImportError:
    clinlp = None

from ..core.config import Config


@dataclass
class ClinlpResult:
    """Result of clinlp processing."""
    doc: Doc
    processing_time: float
    metadata: Dict[str, Any]


class ClinlpProcessor:
    """
    Processor for Dutch clinical text using the clinlp framework.
    
    This processor provides:
    1. Clinical text-specific tokenization
    2. Dutch medical abbreviation handling
    3. Clinical sentence boundary detection
    4. Medical entity recognition
    5. Clinical context analysis
    """
    
    def __init__(self, config: Config):
        """
        Initialize the clinlp processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize clinlp pipeline
        self._initialize_pipeline()
    
    def _initialize_pipeline(self) -> None:
        """Initialize the clinlp processing pipeline."""
        self.logger.info("Initializing clinlp pipeline...")
        
        try:
            if clinlp is None:
                # Fallback to basic spaCy if clinlp not available
                self.logger.warning("clinlp not available, using basic Dutch spaCy model")
                self.nlp = spacy.blank("nl")
                self._add_basic_components()
            else:
                # Use clinlp for Dutch clinical text
                self.nlp = spacy.blank("clinlp")
                self._add_clinlp_components()
            
            self.logger.info("clinlp pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize clinlp pipeline: {e}")
            raise
    
    def _add_basic_components(self) -> None:
        """Add basic components when clinlp is not available."""
        # Add basic tokenizer
        if not self.nlp.has_pipe("tokenizer"):
            # Use default tokenizer
            pass
        
        # Add sentence boundary detection
        if not self.nlp.has_pipe("sentencizer"):
            self.nlp.add_pipe("sentencizer")
        
        # Add basic normalizer
        @spacy.Language.component("basic_normalizer")
        def basic_normalizer(doc):
            for token in doc:
                # Basic normalization
                token.norm_ = token.text.lower()
            return doc
        
        self.nlp.add_pipe("basic_normalizer")
    
    def _add_clinlp_components(self) -> None:
        """Add clinlp-specific components."""
        try:
            # Add clinlp normalizer
            self.nlp.add_pipe("clinlp_normalizer", config={
                "lowercase": True,
                "map_non_ascii": True
            })
            
            # Add clinlp sentencizer
            self.nlp.add_pipe("clinlp_sentencizer", config={
                "sent_end_chars": [".", "!", "?", "\n", "\r"],
                "sent_start_punct": ["-", "*", "[", "("]
            })
            
            # Add rule-based entity matcher
            self.nlp.add_pipe("clinlp_rule_based_entity_matcher", config={
                "attr": "NORM",
                "proximity": 1,
                "fuzzy": 1,
                "fuzzy_min_len": 4,
                "resolve_overlap": True,
                "spans_key": "ents"
            })
            
            # Configure entity matcher with Dutch medical terms
            self._configure_entity_matcher()
            
        except Exception as e:
            self.logger.warning(f"Failed to add clinlp components: {e}")
            # Fall back to basic components
            self._add_basic_components()
    
    def _configure_entity_matcher(self) -> None:
        """Configure the entity matcher with Dutch medical terminology."""
        if not self.nlp.has_pipe("clinlp_rule_based_entity_matcher"):
            return
        
        entity_matcher = self.nlp.get_pipe("clinlp_rule_based_entity_matcher")
        
        # Dutch medical terms for entity recognition
        medical_terms = {
            "symptomen": [
                "koorts", "hoofdpijn", "misselijkheid", "braken", "diarree",
                "hoest", "kortademigheid", "pijn", "zwelling", "jeuk"
            ],
            "diagnoses": [
                "diabetes", "hypertensie", "astma", "copd", "pneumonie",
                "bronchitis", "griep", "verkoudheid", "migraine", "depressie"
            ],
            "medicatie": [
                "paracetamol", "ibuprofen", "aspirine", "antibiotica",
                "insuline", "metformine", "salbutamol", "prednison"
            ],
            "anatomie": [
                "hart", "longen", "lever", "nieren", "hersenen", "maag",
                "darmen", "blaas", "botten", "spieren", "huid"
            ],
            "procedures": [
                "operatie", "onderzoek", "scan", "röntgen", "bloedonderzoek",
                "urinonderzoek", "biopsie", "endoscopie", "echo"
            ]
        }
        
        try:
            entity_matcher.add_terms_from_dict(medical_terms)
            self.logger.debug("Added Dutch medical terms to entity matcher")
        except Exception as e:
            self.logger.warning(f"Failed to add medical terms: {e}")
    
    def process(self, text: str) -> ClinlpResult:
        """
        Process text with clinlp.
        
        Args:
            text: Input text to process
            
        Returns:
            ClinlpResult with processed document
        """
        start_time = time.time()
        
        try:
            # Process text with clinlp pipeline
            doc = self.nlp(text)
            
            # Extract metadata
            metadata = {
                'num_tokens': len(doc),
                'num_sentences': len(list(doc.sents)),
                'num_entities': len(doc.ents),
                'has_clinical_entities': len(doc.ents) > 0,
                'pipeline_components': [pipe for pipe in self.nlp.pipe_names],
            }
            
            # Add token-level metadata
            metadata['token_stats'] = {
                'alpha_tokens': sum(1 for token in doc if token.is_alpha),
                'digit_tokens': sum(1 for token in doc if token.is_digit),
                'punct_tokens': sum(1 for token in doc if token.is_punct),
                'space_tokens': sum(1 for token in doc if token.is_space),
            }
            
            # Add sentence-level metadata
            if doc.sents:
                sentence_lengths = [len(sent) for sent in doc.sents]
                metadata['sentence_stats'] = {
                    'avg_length': sum(sentence_lengths) / len(sentence_lengths),
                    'min_length': min(sentence_lengths),
                    'max_length': max(sentence_lengths),
                }
            
            processing_time = time.time() - start_time
            
            return ClinlpResult(
                doc=doc,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in clinlp processing: {e}")
            raise
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline."""
        return {
            'language': self.nlp.lang,
            'components': list(self.nlp.pipe_names),
            'has_clinlp': clinlp is not None,
            'vocab_size': len(self.nlp.vocab),
        }
    
    def add_custom_patterns(self, patterns: Dict[str, Any]) -> None:
        """
        Add custom patterns to the entity matcher.
        
        Args:
            patterns: Dictionary of patterns to add
        """
        if not self.nlp.has_pipe("clinlp_rule_based_entity_matcher"):
            self.logger.warning("Entity matcher not available")
            return
        
        try:
            entity_matcher = self.nlp.get_pipe("clinlp_rule_based_entity_matcher")
            entity_matcher.add_terms_from_dict(patterns)
            self.logger.info(f"Added {len(patterns)} custom pattern groups")
        except Exception as e:
            self.logger.error(f"Failed to add custom patterns: {e}")
    
    def analyze_clinical_content(self, doc: Doc) -> Dict[str, Any]:
        """
        Analyze clinical content characteristics.
        
        Args:
            doc: Processed spaCy document
            
        Returns:
            Dictionary with clinical content analysis
        """
        analysis = {
            'clinical_indicators': {
                'medical_entities': len([ent for ent in doc.ents if self._is_medical_entity(ent)]),
                'clinical_abbreviations': len([token for token in doc if self._is_clinical_abbreviation(token)]),
                'dosage_mentions': len([token for token in doc if self._is_dosage_mention(token)]),
                'temporal_expressions': len([token for token in doc if self._is_temporal_expression(token)]),
            },
            'text_characteristics': {
                'avg_sentence_length': sum(len(sent) for sent in doc.sents) / len(list(doc.sents)) if doc.sents else 0,
                'complex_sentences': len([sent for sent in doc.sents if len(sent) > 20]),
                'short_sentences': len([sent for sent in doc.sents if len(sent) < 5]),
            },
            'dutch_features': {
                'compound_words': len([token for token in doc if self._is_compound_word(token)]),
                'medical_compounds': len([token for token in doc if self._is_medical_compound(token)]),
            }
        }
        
        return analysis
    
    def _is_medical_entity(self, ent) -> bool:
        """Check if entity is medical-related."""
        medical_labels = ['SYMPTOM', 'DIAGNOSIS', 'MEDICATION', 'PROCEDURE', 'ANATOMY']
        return ent.label_ in medical_labels
    
    def _is_clinical_abbreviation(self, token) -> bool:
        """Check if token is a clinical abbreviation."""
        clinical_abbrevs = [
            'mg', 'ml', 'kg', 'cm', 'mm', 'bpm', 'mmhg', 'ecg', 'mri', 'ct',
            'lab', 'ok', 'ic', 'seh', 'ehbo', 'huisarts', 'specialist'
        ]
        return token.text.lower() in clinical_abbrevs
    
    def _is_dosage_mention(self, token) -> bool:
        """Check if token is part of a dosage mention."""
        dosage_units = ['mg', 'ml', 'gram', 'liter', 'tablet', 'capsule', 'druppel']
        return token.text.lower() in dosage_units or (
            token.like_num and any(unit in token.nbor().text.lower() for unit in dosage_units if token.nbor())
        )
    
    def _is_temporal_expression(self, token) -> bool:
        """Check if token is part of a temporal expression."""
        temporal_words = [
            'dag', 'week', 'maand', 'jaar', 'uur', 'minuut', 'seconde',
            'ochtend', 'middag', 'avond', 'nacht', 'vandaag', 'gisteren', 'morgen'
        ]
        return token.text.lower() in temporal_words
    
    def _is_compound_word(self, token) -> bool:
        """Check if token is a Dutch compound word."""
        # Simple heuristic: long words that might be compounds
        return len(token.text) > 10 and token.is_alpha
    
    def _is_medical_compound(self, token) -> bool:
        """Check if token is a medical compound word."""
        medical_parts = [
            'ziek', 'genees', 'heel', 'onder', 'behandel', 'operatie',
            'diagnose', 'therapie', 'medicijn', 'patient'
        ]
        return any(part in token.text.lower() for part in medical_parts)

