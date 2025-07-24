"""Dutch clinical NLP processing using clinlp and RobBERT."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc, Token, Span

from .clinlp_processor import ClinlpProcessor
from .robbert_processor import RobBERTProcessor
from ..core.config import Config


@dataclass
class NLPResult:
    """Result of NLP processing."""
    text: str
    doc: Doc
    tokens: List[Dict[str, Any]]
    sentences: List[Dict[str, Any]]
    entities: List[Dict[str, Any]]
    linguistic_features: Dict[str, Any]
    processing_metadata: Dict[str, Any]


class DutchClinicalNLP:
    """
    Dutch clinical NLP processor combining clinlp and RobBERT.
    
    This class provides comprehensive Dutch clinical text processing by:
    1. Using clinlp for clinical text preprocessing and tokenization
    2. Applying RobBERT for advanced language understanding
    3. Extracting linguistic features relevant for entity recognition
    4. Providing context analysis for deidentification
    """
    
    def __init__(self, config: Config):
        """
        Initialize the Dutch clinical NLP processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self._initialize_processors()
    
    def _initialize_processors(self) -> None:
        """Initialize clinlp and RobBERT processors."""
        self.logger.info("Initializing Dutch clinical NLP processors...")
        
        try:
            # Initialize clinlp processor
            self.clinlp_processor = ClinlpProcessor(self.config)
            
            # Initialize RobBERT processor
            self.robbert_processor = RobBERTProcessor(self.config)
            
            self.logger.info("Dutch clinical NLP processors initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize NLP processors: {e}")
            raise
    
    def process(self, text: str) -> NLPResult:
        """
        Process text with Dutch clinical NLP.
        
        Args:
            text: Input text to process
            
        Returns:
            NLPResult with comprehensive linguistic analysis
        """
        self.logger.debug(f"Processing text of length {len(text)}")
        
        try:
            # Step 1: Process with clinlp
            clinlp_result = self.clinlp_processor.process(text)
            
            # Step 2: Process with RobBERT for enhanced understanding
            robbert_result = self.robbert_processor.process(text, clinlp_result.doc)
            
            # Step 3: Extract tokens with enhanced features
            tokens = self._extract_tokens(clinlp_result.doc, robbert_result)
            
            # Step 4: Extract sentences
            sentences = self._extract_sentences(clinlp_result.doc)
            
            # Step 5: Extract preliminary entities
            entities = self._extract_entities(clinlp_result.doc, robbert_result)
            
            # Step 6: Extract linguistic features
            linguistic_features = self._extract_linguistic_features(
                clinlp_result.doc, robbert_result
            )
            
            # Step 7: Create processing metadata
            processing_metadata = {
                'text_length': len(text),
                'num_tokens': len(tokens),
                'num_sentences': len(sentences),
                'num_entities': len(entities),
                'language_detected': 'nl',
                'clinical_text_detected': True,
                'processing_time': clinlp_result.processing_time + robbert_result.processing_time,
            }
            
            return NLPResult(
                text=text,
                doc=clinlp_result.doc,
                tokens=tokens,
                sentences=sentences,
                entities=entities,
                linguistic_features=linguistic_features,
                processing_metadata=processing_metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in NLP processing: {e}")
            raise
    
    def _extract_tokens(self, doc: Doc, robbert_result: Any) -> List[Dict[str, Any]]:
        """Extract token information with enhanced features."""
        tokens = []
        
        for i, token in enumerate(doc):
            token_info = {
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'is_alpha': token.is_alpha,
                'is_digit': token.is_digit,
                'is_punct': token.is_punct,
                'is_space': token.is_space,
                'is_stop': token.is_stop,
                'start': token.idx,
                'end': token.idx + len(token.text),
                'shape': token.shape_,
                'norm': token.norm_,
                'is_clinical_term': self._is_clinical_term(token),
                'is_potential_phi': self._is_potential_phi(token),
            }
            
            # Add RobBERT embeddings if available
            if hasattr(robbert_result, 'token_embeddings') and i < len(robbert_result.token_embeddings):
                token_info['embedding'] = robbert_result.token_embeddings[i]
                token_info['attention_weights'] = robbert_result.attention_weights[i] if hasattr(robbert_result, 'attention_weights') else None
            
            tokens.append(token_info)
        
        return tokens
    
    def _extract_sentences(self, doc: Doc) -> List[Dict[str, Any]]:
        """Extract sentence information."""
        sentences = []
        
        for sent in doc.sents:
            sentence_info = {
                'text': sent.text,
                'start': sent.start_char,
                'end': sent.end_char,
                'start_token': sent.start,
                'end_token': sent.end,
                'is_clinical_sentence': self._is_clinical_sentence(sent),
                'contains_phi': self._sentence_contains_phi(sent),
            }
            sentences.append(sentence_info)
        
        return sentences
    
    def _extract_entities(self, doc: Doc, robbert_result: Any) -> List[Dict[str, Any]]:
        """Extract preliminary entity information."""
        entities = []
        
        # Extract entities from spaCy NER
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'start_token': ent.start,
                'end_token': ent.end,
                'confidence': getattr(ent, 'confidence', 0.0),
                'source': 'spacy',
            }
            entities.append(entity_info)
        
        # Add RobBERT-detected entities if available
        if hasattr(robbert_result, 'entities'):
            for ent in robbert_result.entities:
                entity_info = {
                    'text': ent.get('text', ''),
                    'label': ent.get('label', ''),
                    'start': ent.get('start', 0),
                    'end': ent.get('end', 0),
                    'confidence': ent.get('confidence', 0.0),
                    'source': 'robbert',
                }
                entities.append(entity_info)
        
        return entities
    
    def _extract_linguistic_features(self, doc: Doc, robbert_result: Any) -> Dict[str, Any]:
        """Extract linguistic features for deidentification."""
        features = {
            'text_statistics': {
                'num_chars': len(doc.text),
                'num_tokens': len(doc),
                'num_sentences': len(list(doc.sents)),
                'avg_sentence_length': sum(len(sent) for sent in doc.sents) / len(list(doc.sents)) if doc.sents else 0,
            },
            'clinical_indicators': {
                'medical_terms_count': sum(1 for token in doc if self._is_clinical_term(token)),
                'abbreviations_count': sum(1 for token in doc if self._is_medical_abbreviation(token)),
                'dosage_patterns': self._find_dosage_patterns(doc),
                'date_patterns': self._find_date_patterns(doc),
            },
            'dutch_language_features': {
                'compound_words': self._find_compound_words(doc),
                'tussenvoegsels': self._find_tussenvoegsels(doc),
                'dutch_names': self._find_dutch_names(doc),
                'belgian_locations': self._find_belgian_locations(doc),
            },
            'phi_indicators': {
                'potential_names': self._find_potential_names(doc),
                'potential_addresses': self._find_potential_addresses(doc),
                'potential_dates': self._find_potential_dates(doc),
                'potential_ids': self._find_potential_ids(doc),
            }
        }
        
        # Add RobBERT-specific features
        if hasattr(robbert_result, 'semantic_features'):
            features['semantic_features'] = robbert_result.semantic_features
        
        return features
    
    def _is_clinical_term(self, token: Token) -> bool:
        """Check if token is a clinical term."""
        # This would use medical terminology databases
        clinical_indicators = [
            'patient', 'patiënt', 'diagnose', 'behandeling', 'medicatie',
            'symptoom', 'ziekte', 'onderzoek', 'therapie', 'operatie'
        ]
        return token.lemma_.lower() in clinical_indicators
    
    def _is_potential_phi(self, token: Token) -> bool:
        """Check if token could be PHI."""
        # Basic heuristics for PHI detection
        if token.is_alpha and token.text[0].isupper():
            return True  # Potential name
        if token.like_num and len(token.text) >= 4:
            return True  # Potential ID or date
        return False
    
    def _is_clinical_sentence(self, sent: Span) -> bool:
        """Check if sentence appears to be clinical text."""
        clinical_keywords = [
            'patient', 'diagnose', 'behandeling', 'medicatie', 'symptoom',
            'onderzoek', 'therapie', 'operatie', 'arts', 'dokter'
        ]
        text_lower = sent.text.lower()
        return any(keyword in text_lower for keyword in clinical_keywords)
    
    def _sentence_contains_phi(self, sent: Span) -> bool:
        """Check if sentence likely contains PHI."""
        return any(self._is_potential_phi(token) for token in sent)
    
    def _is_medical_abbreviation(self, token: Token) -> bool:
        """Check if token is a medical abbreviation."""
        medical_abbrevs = [
            'mg', 'ml', 'kg', 'cm', 'mm', 'bpm', 'mmhg',
            'ecg', 'mri', 'ct', 'lab', 'ok', 'ic', 'seh'
        ]
        return token.text.lower() in medical_abbrevs
    
    def _find_dosage_patterns(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find dosage patterns in text."""
        patterns = []
        # Implementation would use regex patterns for Dutch dosage formats
        return patterns
    
    def _find_date_patterns(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find date patterns in text."""
        patterns = []
        # Implementation would use regex patterns for Dutch date formats
        return patterns
    
    def _find_compound_words(self, doc: Doc) -> List[str]:
        """Find Dutch compound words."""
        compounds = []
        # Implementation would identify Dutch compound word patterns
        return compounds
    
    def _find_tussenvoegsels(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find Dutch name prefixes (tussenvoegsels)."""
        tussenvoegsels = ['van', 'de', 'der', 'den', 'van der', 'van den', 'van de']
        found = []
        
        for token in doc:
            if token.text.lower() in tussenvoegsels:
                found.append({
                    'text': token.text,
                    'start': token.idx,
                    'end': token.idx + len(token.text)
                })
        
        return found
    
    def _find_dutch_names(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find potential Dutch names."""
        names = []
        # Implementation would use Dutch name patterns and databases
        return names
    
    def _find_belgian_locations(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find Belgian location references."""
        locations = []
        # Implementation would use Belgian location databases
        return locations
    
    def _find_potential_names(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find potential person names."""
        names = []
        # Implementation would use name detection patterns
        return names
    
    def _find_potential_addresses(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find potential addresses."""
        addresses = []
        # Implementation would use Belgian address patterns
        return addresses
    
    def _find_potential_dates(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find potential dates."""
        dates = []
        # Implementation would use date detection patterns
        return dates
    
    def _find_potential_ids(self, doc: Doc) -> List[Dict[str, Any]]:
        """Find potential identifiers."""
        ids = []
        # Implementation would use ID pattern detection
        return ids

