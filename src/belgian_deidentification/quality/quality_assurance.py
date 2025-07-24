"""Main quality assurance system for deidentification validation."""

import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..entities.entity_types import PHIEntity
from ..core.config import Config
from .validators import DeidentificationValidator, EntityValidator, TextValidator
from .metrics import QualityMetrics, PerformanceMetrics
from .expert_review import ExpertReviewSystem


@dataclass
class QualityAssuranceResult:
    """Result of quality assurance validation."""
    passed: bool
    confidence_scores: Dict[str, float]
    warnings: List[str]
    errors: List[str]
    metrics: Dict[str, Any]
    expert_review_required: bool
    validation_metadata: Dict[str, Any]


class QualityAssurance:
    """
    Comprehensive quality assurance system for deidentification.
    
    This system provides:
    1. Automated validation of deidentification results
    2. Statistical performance monitoring
    3. Expert review workflow management
    4. Quality metrics calculation
    5. Compliance verification
    """
    
    def __init__(self, config: Config):
        """
        Initialize the quality assurance system.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize validators
        self._initialize_validators()
        
        # Initialize metrics systems
        self.quality_metrics = QualityMetrics(config)
        self.performance_metrics = PerformanceMetrics(config)
        
        # Initialize expert review system
        self.expert_review = ExpertReviewSystem(config)
        
        # Quality statistics
        self.stats = {
            'validations_performed': 0,
            'validations_passed': 0,
            'expert_reviews_triggered': 0,
            'average_confidence': 0.0,
            'common_issues': {},
        }
    
    def _initialize_validators(self) -> None:
        """Initialize validation components."""
        self.logger.info("Initializing quality assurance validators...")
        
        try:
            self.deidentification_validator = DeidentificationValidator(self.config)
            self.entity_validator = EntityValidator(self.config)
            self.text_validator = TextValidator(self.config)
            
            self.logger.info("Quality assurance validators initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize validators: {e}")
            raise
    
    def validate(
        self,
        original_text: str,
        deidentified_text: str,
        entities: List[PHIEntity]
    ) -> QualityAssuranceResult:
        """
        Perform comprehensive quality assurance validation.
        
        Args:
            original_text: Original text before deidentification
            deidentified_text: Text after deidentification
            entities: List of processed PHI entities
            
        Returns:
            QualityAssuranceResult with validation details
        """
        start_time = time.time()
        
        self.logger.debug(f"Starting QA validation for {len(entities)} entities")
        
        result = QualityAssuranceResult(
            passed=True,
            confidence_scores={},
            warnings=[],
            errors=[],
            metrics={},
            expert_review_required=False,
            validation_metadata={}
        )
        
        try:
            # Step 1: Validate deidentification completeness
            deident_result = self.deidentification_validator.validate(
                original_text, deidentified_text, entities
            )
            result.confidence_scores['deidentification'] = deident_result.confidence
            result.warnings.extend(deident_result.warnings)
            result.errors.extend(deident_result.errors)
            
            if not deident_result.passed:
                result.passed = False
            
            # Step 2: Validate entity processing
            entity_result = self.entity_validator.validate_entities(entities)
            result.confidence_scores['entity_validation'] = entity_result.confidence
            result.warnings.extend(entity_result.warnings)
            result.errors.extend(entity_result.errors)
            
            if not entity_result.passed:
                result.passed = False
            
            # Step 3: Validate text quality
            text_result = self.text_validator.validate(
                original_text, deidentified_text
            )
            result.confidence_scores['text_quality'] = text_result.confidence
            result.warnings.extend(text_result.warnings)
            result.errors.extend(text_result.errors)
            
            if not text_result.passed:
                result.passed = False
            
            # Step 4: Calculate quality metrics
            quality_metrics = self.quality_metrics.calculate_metrics(
                original_text, deidentified_text, entities
            )
            result.metrics.update(quality_metrics)
            
            # Step 5: Calculate performance metrics
            performance_metrics = self.performance_metrics.calculate_metrics(
                entities, time.time() - start_time
            )
            result.metrics.update(performance_metrics)
            
            # Step 6: Determine if expert review is needed
            result.expert_review_required = self._requires_expert_review(
                result, entities
            )
            
            # Step 7: Create validation metadata
            result.validation_metadata = {
                'validation_time': time.time() - start_time,
                'text_length': len(original_text),
                'entities_count': len(entities),
                'overall_confidence': self._calculate_overall_confidence(result.confidence_scores),
                'validation_timestamp': time.time(),
            }
            
            # Update statistics
            self._update_stats(result)
            
            self.logger.info(
                f"QA validation completed: {'PASSED' if result.passed else 'FAILED'} "
                f"(confidence: {result.validation_metadata['overall_confidence']:.3f})"
            )
            
        except Exception as e:
            self.logger.error(f"Error in quality assurance validation: {e}")
            result.passed = False
            result.errors.append(f"QA validation error: {str(e)}")
        
        return result
    
    def _requires_expert_review(
        self, result: QualityAssuranceResult, entities: List[PHIEntity]
    ) -> bool:
        """Determine if expert review is required."""
        if not self.config.quality_assurance.enable_validation:
            return False
        
        # Always require review if validation failed
        if not result.passed:
            return True
        
        # Check confidence thresholds
        overall_confidence = self._calculate_overall_confidence(result.confidence_scores)
        if overall_confidence < self.config.quality_assurance.confidence_threshold:
            return True
        
        # Check for high-risk entities
        high_risk_entities = [e for e in entities if e.risk_level in ['high', 'critical']]
        if len(high_risk_entities) > 5:  # Threshold for high-risk entity count
            return True
        
        # Random sampling for expert review
        sampling_rate = self.config.quality_assurance.expert_review_sampling
        if random.random() < sampling_rate:
            return True
        
        # Check for specific warning patterns
        warning_patterns = [
            'potential missed entity',
            'low confidence detection',
            'unusual entity pattern',
        ]
        
        for warning in result.warnings:
            if any(pattern in warning.lower() for pattern in warning_patterns):
                return True
        
        return False
    
    def _calculate_overall_confidence(self, confidence_scores: Dict[str, float]) -> float:
        """Calculate overall confidence from individual scores."""
        if not confidence_scores:
            return 0.0
        
        # Weighted average of confidence scores
        weights = {
            'deidentification': 0.4,
            'entity_validation': 0.3,
            'text_quality': 0.3,
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, score in confidence_scores.items():
            weight = weights.get(metric, 0.1)  # Default weight for unknown metrics
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _update_stats(self, result: QualityAssuranceResult) -> None:
        """Update quality assurance statistics."""
        self.stats['validations_performed'] += 1
        
        if result.passed:
            self.stats['validations_passed'] += 1
        
        if result.expert_review_required:
            self.stats['expert_reviews_triggered'] += 1
        
        # Update average confidence
        overall_confidence = result.validation_metadata.get('overall_confidence', 0.0)
        current_avg = self.stats['average_confidence']
        n = self.stats['validations_performed']
        self.stats['average_confidence'] = (current_avg * (n - 1) + overall_confidence) / n
        
        # Track common issues
        for error in result.errors:
            self.stats['common_issues'][error] = self.stats['common_issues'].get(error, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quality assurance statistics."""
        stats = self.stats.copy()
        
        # Calculate additional metrics
        if stats['validations_performed'] > 0:
            stats['pass_rate'] = stats['validations_passed'] / stats['validations_performed']
            stats['expert_review_rate'] = stats['expert_reviews_triggered'] / stats['validations_performed']
        else:
            stats['pass_rate'] = 0.0
            stats['expert_review_rate'] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset quality assurance statistics."""
        self.stats = {
            'validations_performed': 0,
            'validations_passed': 0,
            'expert_reviews_triggered': 0,
            'average_confidence': 0.0,
            'common_issues': {},
        }
    
    def submit_for_expert_review(
        self,
        original_text: str,
        deidentified_text: str,
        entities: List[PHIEntity],
        qa_result: QualityAssuranceResult
    ) -> str:
        """
        Submit case for expert review.
        
        Args:
            original_text: Original text
            deidentified_text: Deidentified text
            entities: List of entities
            qa_result: QA validation result
            
        Returns:
            Review case ID
        """
        try:
            case_id = self.expert_review.submit_case(
                original_text=original_text,
                deidentified_text=deidentified_text,
                entities=entities,
                qa_result=qa_result
            )
            
            self.logger.info(f"Submitted case for expert review: {case_id}")
            return case_id
            
        except Exception as e:
            self.logger.error(f"Failed to submit for expert review: {e}")
            return ""
    
    def get_expert_review_status(self, case_id: str) -> Dict[str, Any]:
        """Get status of expert review case."""
        try:
            return self.expert_review.get_case_status(case_id)
        except Exception as e:
            self.logger.error(f"Failed to get expert review status: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def process_expert_feedback(self, case_id: str, feedback: Dict[str, Any]) -> bool:
        """
        Process feedback from expert review.
        
        Args:
            case_id: Review case ID
            feedback: Expert feedback
            
        Returns:
            True if feedback processed successfully
        """
        try:
            success = self.expert_review.process_feedback(case_id, feedback)
            
            if success:
                # Update system based on feedback
                self._incorporate_expert_feedback(feedback)
                self.logger.info(f"Processed expert feedback for case: {case_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to process expert feedback: {e}")
            return False
    
    def _incorporate_expert_feedback(self, feedback: Dict[str, Any]) -> None:
        """Incorporate expert feedback into the system."""
        # This would update recognition patterns, validation rules, etc.
        # based on expert feedback
        
        feedback_type = feedback.get('type', '')
        
        if feedback_type == 'missed_entity':
            # Update entity recognition patterns
            self.logger.info("Incorporating missed entity feedback")
            
        elif feedback_type == 'false_positive':
            # Update validation rules
            self.logger.info("Incorporating false positive feedback")
            
        elif feedback_type == 'quality_issue':
            # Update quality thresholds
            self.logger.info("Incorporating quality issue feedback")
    
    def generate_quality_report(
        self, time_period: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive quality report.
        
        Args:
            time_period: Time period for report (e.g., 'last_week', 'last_month')
            
        Returns:
            Quality report dictionary
        """
        report = {
            'summary': self.get_stats(),
            'quality_metrics': self.quality_metrics.get_summary(),
            'performance_metrics': self.performance_metrics.get_summary(),
            'expert_review_summary': self.expert_review.get_summary(),
            'recommendations': self._generate_recommendations(),
            'timestamp': time.time(),
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on quality statistics."""
        recommendations = []
        stats = self.get_stats()
        
        # Check pass rate
        if stats['pass_rate'] < 0.9:
            recommendations.append(
                "Consider reviewing entity recognition patterns - pass rate is below 90%"
            )
        
        # Check confidence levels
        if stats['average_confidence'] < 0.8:
            recommendations.append(
                "Average confidence is low - consider model retraining or threshold adjustment"
            )
        
        # Check expert review rate
        if stats['expert_review_rate'] > 0.2:
            recommendations.append(
                "High expert review rate - consider improving automated validation"
            )
        
        # Check common issues
        if stats['common_issues']:
            most_common = max(stats['common_issues'].items(), key=lambda x: x[1])
            if most_common[1] > 5:
                recommendations.append(
                    f"Address common issue: {most_common[0]} (occurred {most_common[1]} times)"
                )
        
        return recommendations
    
    def validate_batch_results(
        self, batch_results: List[Tuple[str, str, List[PHIEntity]]]
    ) -> Dict[str, Any]:
        """
        Validate results from batch processing.
        
        Args:
            batch_results: List of (original_text, deidentified_text, entities) tuples
            
        Returns:
            Batch validation summary
        """
        batch_summary = {
            'total_documents': len(batch_results),
            'passed_documents': 0,
            'failed_documents': 0,
            'expert_review_required': 0,
            'average_confidence': 0.0,
            'common_issues': {},
        }
        
        total_confidence = 0.0
        
        for original_text, deidentified_text, entities in batch_results:
            try:
                result = self.validate(original_text, deidentified_text, entities)
                
                if result.passed:
                    batch_summary['passed_documents'] += 1
                else:
                    batch_summary['failed_documents'] += 1
                
                if result.expert_review_required:
                    batch_summary['expert_review_required'] += 1
                
                confidence = result.validation_metadata.get('overall_confidence', 0.0)
                total_confidence += confidence
                
                # Collect issues
                for error in result.errors:
                    batch_summary['common_issues'][error] = (
                        batch_summary['common_issues'].get(error, 0) + 1
                    )
                
            except Exception as e:
                self.logger.error(f"Batch validation error: {e}")
                batch_summary['failed_documents'] += 1
        
        # Calculate average confidence
        if batch_summary['total_documents'] > 0:
            batch_summary['average_confidence'] = total_confidence / batch_summary['total_documents']
        
        return batch_summary

