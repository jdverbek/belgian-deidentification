"""RobBERT processor for advanced Dutch language understanding."""

import logging
import time
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForTokenClassification,
    pipeline, Pipeline
)
from spacy.tokens import Doc

from ..core.config import Config


@dataclass
class RobBERTResult:
    """Result of RobBERT processing."""
    token_embeddings: Optional[np.ndarray]
    attention_weights: Optional[np.ndarray]
    entities: List[Dict[str, Any]]
    semantic_features: Dict[str, Any]
    processing_time: float
    metadata: Dict[str, Any]


class RobBERTProcessor:
    """
    RobBERT processor for advanced Dutch language understanding.
    
    This processor provides:
    1. Contextual embeddings for Dutch clinical text
    2. Named entity recognition using fine-tuned models
    3. Attention analysis for entity context
    4. Semantic similarity computation
    5. Clinical concept classification
    """
    
    def __init__(self, config: Config):
        """
        Initialize the RobBERT processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.nlp.use_gpu else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize RobBERT models."""
        self.logger.info("Initializing RobBERT models...")
        
        try:
            # Model name based on configuration
            model_name = self.config.nlp.model
            if "robbert" not in model_name.lower():
                model_name = "DTAI-KULeuven/robbert-2023-dutch-large"
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                add_prefix_space=True
            )
            
            # Initialize base model for embeddings
            self.base_model = AutoModel.from_pretrained(model_name)
            self.base_model.to(self.device)
            self.base_model.eval()
            
            # Initialize NER pipeline
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    aggregation_strategy="simple",
                    device=0 if self.device.type == "cuda" else -1
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize NER pipeline: {e}")
                self.ner_pipeline = None
            
            # Initialize classification pipeline for clinical concepts
            try:
                self.classification_pipeline = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=self.tokenizer,
                    device=0 if self.device.type == "cuda" else -1
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize classification pipeline: {e}")
                self.classification_pipeline = None
            
            self.logger.info("RobBERT models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RobBERT models: {e}")
            raise
    
    def process(self, text: str, doc: Optional[Doc] = None) -> RobBERTResult:
        """
        Process text with RobBERT.
        
        Args:
            text: Input text to process
            doc: Optional spaCy document for alignment
            
        Returns:
            RobBERTResult with embeddings and analysis
        """
        start_time = time.time()
        
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.config.nlp.max_length,
                return_attention_mask=True,
                return_offsets_mapping=True
            )
            
            # Move to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Get embeddings and attention weights
            token_embeddings, attention_weights = self._get_embeddings_and_attention(
                input_ids, attention_mask
            )
            
            # Perform named entity recognition
            entities = self._perform_ner(text)
            
            # Extract semantic features
            semantic_features = self._extract_semantic_features(
                text, token_embeddings, attention_weights
            )
            
            # Create metadata
            metadata = {
                'model_name': self.config.nlp.model,
                'num_tokens': len(inputs["input_ids"][0]),
                'max_attention': float(attention_weights.max()) if attention_weights is not None else 0.0,
                'avg_attention': float(attention_weights.mean()) if attention_weights is not None else 0.0,
                'device': str(self.device),
            }
            
            processing_time = time.time() - start_time
            
            return RobBERTResult(
                token_embeddings=token_embeddings,
                attention_weights=attention_weights,
                entities=entities,
                semantic_features=semantic_features,
                processing_time=processing_time,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error in RobBERT processing: {e}")
            # Return empty result on error
            return RobBERTResult(
                token_embeddings=None,
                attention_weights=None,
                entities=[],
                semantic_features={},
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    def _get_embeddings_and_attention(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get token embeddings and attention weights."""
        try:
            with torch.no_grad():
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True
                )
                
                # Get last hidden state (token embeddings)
                last_hidden_state = outputs.last_hidden_state[0]  # Remove batch dimension
                token_embeddings = last_hidden_state.cpu().numpy()
                
                # Get attention weights (average across heads and layers)
                attentions = outputs.attentions
                if attentions:
                    # Average attention across all heads and layers
                    avg_attention = torch.stack(attentions).mean(dim=(0, 2))  # (layers, heads, seq, seq)
                    attention_weights = avg_attention[0].cpu().numpy()  # Remove batch dimension
                else:
                    attention_weights = None
                
                return token_embeddings, attention_weights
                
        except Exception as e:
            self.logger.warning(f"Failed to get embeddings and attention: {e}")
            return None, None
    
    def _perform_ner(self, text: str) -> List[Dict[str, Any]]:
        """Perform named entity recognition."""
        entities = []
        
        if self.ner_pipeline is None:
            return entities
        
        try:
            # Run NER pipeline
            ner_results = self.ner_pipeline(text)
            
            for result in ner_results:
                entity = {
                    'text': result.get('word', ''),
                    'label': result.get('entity_group', result.get('entity', '')),
                    'confidence': result.get('score', 0.0),
                    'start': result.get('start', 0),
                    'end': result.get('end', 0),
                    'source': 'robbert_ner'
                }
                
                # Map labels to our entity types
                entity['mapped_label'] = self._map_entity_label(entity['label'])
                
                entities.append(entity)
            
        except Exception as e:
            self.logger.warning(f"NER processing failed: {e}")
        
        return entities
    
    def _map_entity_label(self, label: str) -> str:
        """Map RobBERT entity labels to our standard labels."""
        label_mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'LOC': 'ADDRESS',
            'LOCATION': 'ADDRESS',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'MISC': 'OTHER',
            'DATE': 'DATE',
            'TIME': 'DATE',
        }
        
        return label_mapping.get(label.upper(), label)
    
    def _extract_semantic_features(
        self,
        text: str,
        token_embeddings: Optional[np.ndarray],
        attention_weights: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Extract semantic features from embeddings and attention."""
        features = {}
        
        if token_embeddings is not None:
            features['embedding_stats'] = {
                'mean_embedding': token_embeddings.mean(axis=0).tolist(),
                'embedding_norm': float(np.linalg.norm(token_embeddings.mean(axis=0))),
                'embedding_variance': float(token_embeddings.var()),
                'num_tokens': token_embeddings.shape[0],
                'embedding_dim': token_embeddings.shape[1],
            }
        
        if attention_weights is not None:
            features['attention_stats'] = {
                'max_attention': float(attention_weights.max()),
                'mean_attention': float(attention_weights.mean()),
                'attention_variance': float(attention_weights.var()),
                'attention_entropy': self._calculate_attention_entropy(attention_weights),
            }
            
            # Find high-attention tokens
            features['high_attention_tokens'] = self._find_high_attention_tokens(
                text, attention_weights
            )
        
        # Clinical concept classification
        if self.classification_pipeline:
            features['clinical_classification'] = self._classify_clinical_content(text)
        
        # Semantic similarity to clinical concepts
        features['clinical_similarity'] = self._calculate_clinical_similarity(
            token_embeddings
        )
        
        return features
    
    def _calculate_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """Calculate entropy of attention distribution."""
        try:
            # Flatten attention weights and normalize
            flat_attention = attention_weights.flatten()
            flat_attention = flat_attention / flat_attention.sum()
            
            # Calculate entropy
            entropy = -np.sum(flat_attention * np.log(flat_attention + 1e-10))
            return float(entropy)
        except:
            return 0.0
    
    def _find_high_attention_tokens(
        self, text: str, attention_weights: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Find tokens with high attention weights."""
        high_attention_tokens = []
        
        try:
            # Get tokens
            tokens = self.tokenizer.tokenize(text)
            
            # Calculate attention scores per token (sum across all positions)
            token_attention_scores = attention_weights.sum(axis=1)
            
            # Find top attention tokens
            threshold = np.percentile(token_attention_scores, 90)  # Top 10%
            
            for i, (token, score) in enumerate(zip(tokens, token_attention_scores)):
                if score > threshold:
                    high_attention_tokens.append({
                        'token': token,
                        'position': i,
                        'attention_score': float(score),
                        'is_potential_phi': self._is_potential_phi_token(token)
                    })
        
        except Exception as e:
            self.logger.warning(f"Failed to find high attention tokens: {e}")
        
        return high_attention_tokens
    
    def _classify_clinical_content(self, text: str) -> Dict[str, Any]:
        """Classify clinical content using RobBERT."""
        classification = {}
        
        try:
            if self.classification_pipeline:
                result = self.classification_pipeline(text)
                classification = {
                    'label': result[0]['label'] if result else 'UNKNOWN',
                    'confidence': result[0]['score'] if result else 0.0,
                    'is_clinical': self._is_clinical_text(text),
                }
        except Exception as e:
            self.logger.warning(f"Clinical classification failed: {e}")
            classification = {
                'label': 'UNKNOWN',
                'confidence': 0.0,
                'is_clinical': False,
            }
        
        return classification
    
    def _calculate_clinical_similarity(
        self, token_embeddings: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """Calculate similarity to clinical concepts."""
        similarities = {}
        
        if token_embeddings is None:
            return similarities
        
        try:
            # Average token embeddings to get document embedding
            doc_embedding = token_embeddings.mean(axis=0)
            
            # Clinical concept embeddings (would be pre-computed in practice)
            clinical_concepts = {
                'medical_terminology': np.random.randn(doc_embedding.shape[0]),  # Placeholder
                'patient_information': np.random.randn(doc_embedding.shape[0]),  # Placeholder
                'clinical_procedures': np.random.randn(doc_embedding.shape[0]),  # Placeholder
            }
            
            # Calculate cosine similarities
            for concept, concept_embedding in clinical_concepts.items():
                similarity = np.dot(doc_embedding, concept_embedding) / (
                    np.linalg.norm(doc_embedding) * np.linalg.norm(concept_embedding)
                )
                similarities[concept] = float(similarity)
        
        except Exception as e:
            self.logger.warning(f"Clinical similarity calculation failed: {e}")
        
        return similarities
    
    def _is_potential_phi_token(self, token: str) -> bool:
        """Check if token could be PHI based on patterns."""
        # Remove special tokens
        if token.startswith('##') or token in ['[CLS]', '[SEP]', '[PAD]']:
            return False
        
        # Check for potential name patterns
        if token.isalpha() and token[0].isupper():
            return True
        
        # Check for potential ID patterns
        if token.isdigit() and len(token) >= 4:
            return True
        
        # Check for potential date patterns
        if any(char.isdigit() for char in token) and any(char in token for char in ['/', '-', '.']):
            return True
        
        return False
    
    def _is_clinical_text(self, text: str) -> bool:
        """Determine if text appears to be clinical."""
        clinical_indicators = [
            'patient', 'patiënt', 'diagnose', 'behandeling', 'medicatie',
            'symptoom', 'ziekte', 'onderzoek', 'therapie', 'operatie',
            'arts', 'dokter', 'ziekenhuis', 'kliniek', 'medisch'
        ]
        
        text_lower = text.lower()
        clinical_count = sum(1 for indicator in clinical_indicators if indicator in text_lower)
        
        # Consider text clinical if it contains multiple clinical indicators
        return clinical_count >= 2
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            'model_name': self.config.nlp.model,
            'device': str(self.device),
            'tokenizer_vocab_size': len(self.tokenizer.vocab),
            'has_ner_pipeline': self.ner_pipeline is not None,
            'has_classification_pipeline': self.classification_pipeline is not None,
            'max_length': self.config.nlp.max_length,
        }
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        try:
            # Get embeddings for both texts
            result1 = self.process(text1)
            result2 = self.process(text2)
            
            if result1.token_embeddings is None or result2.token_embeddings is None:
                return 0.0
            
            # Average token embeddings
            emb1 = result1.token_embeddings.mean(axis=0)
            emb2 = result2.token_embeddings.mean(axis=0)
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Similarity computation failed: {e}")
            return 0.0

