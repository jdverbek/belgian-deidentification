"""Batch processor for handling multiple documents efficiently."""

import logging
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json

from .pipeline import DeidentificationPipeline, ProcessingResult
from .config import Config, get_config
from ..utils.progress_tracker import ProgressTracker


@dataclass
class BatchResult:
    """Result of batch processing."""
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_entities_found: int
    total_entities_removed: int
    total_processing_time: float
    average_processing_time: float
    results: List[ProcessingResult]
    errors: List[str]
    metadata: Dict[str, Any]


class BatchProcessor:
    """
    Batch processor for deidentifying multiple documents efficiently.
    
    This processor provides:
    1. Parallel processing of multiple documents
    2. Progress tracking and reporting
    3. Error handling and recovery
    4. Batch statistics and reporting
    5. Memory-efficient processing for large datasets
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        input_dir: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize the batch processor.
        
        Args:
            config: Configuration object
            input_dir: Directory containing input documents
            output_dir: Directory for output documents
            **kwargs: Additional configuration overrides
        """
        self.config = config or get_config()
        
        # Apply configuration overrides
        if kwargs:
            config_dict = self.config.dict()
            config_dict.update(kwargs)
            self.config = Config(**config_dict)
        
        # Set directories
        self.input_dir = Path(input_dir) if input_dir else Path(self.config.data_dir) / "input"
        self.output_dir = Path(output_dir) if output_dir else Path(self.config.data_dir) / "output"
        
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Processing settings
        self.max_workers = kwargs.get('max_workers', mp.cpu_count())
        self.chunk_size = kwargs.get('chunk_size', 10)
        self.use_multiprocessing = kwargs.get('use_multiprocessing', True)
        
        # Progress tracking
        self.progress_tracker = ProgressTracker()
        
        self.logger.info(f"Batch processor initialized with {self.max_workers} workers")
    
    def process_all(
        self,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        save_results: bool = True
    ) -> BatchResult:
        """
        Process all documents in the input directory.
        
        Args:
            file_patterns: List of file patterns to match (e.g., ['*.txt', '*.docx'])
            recursive: Whether to search subdirectories
            save_results: Whether to save processing results
            
        Returns:
            BatchResult with processing statistics
        """
        self.logger.info(f"Starting batch processing from {self.input_dir}")
        
        # Find input files
        input_files = self._find_input_files(file_patterns, recursive)
        
        if not input_files:
            self.logger.warning("No input files found")
            return BatchResult(
                total_documents=0,
                successful_documents=0,
                failed_documents=0,
                total_entities_found=0,
                total_entities_removed=0,
                total_processing_time=0.0,
                average_processing_time=0.0,
                results=[],
                errors=["No input files found"],
                metadata={}
            )
        
        self.logger.info(f"Found {len(input_files)} files to process")
        
        # Process files
        results = self._process_files(input_files)
        
        # Create batch result
        batch_result = self._create_batch_result(results)
        
        # Save results if requested
        if save_results:
            self._save_batch_results(batch_result)
        
        self.logger.info(f"Batch processing completed: {batch_result.successful_documents}/{batch_result.total_documents} successful")
        
        return batch_result
    
    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        output_paths: Optional[List[Union[str, Path]]] = None,
        save_results: bool = True
    ) -> BatchResult:
        """
        Process specific files.
        
        Args:
            file_paths: List of input file paths
            output_paths: List of output file paths (optional)
            save_results: Whether to save processing results
            
        Returns:
            BatchResult with processing statistics
        """
        self.logger.info(f"Processing {len(file_paths)} specific files")
        
        # Prepare file pairs
        file_pairs = []
        for i, input_path in enumerate(file_paths):
            input_path = Path(input_path)
            
            if output_paths and i < len(output_paths):
                output_path = Path(output_paths[i])
            else:
                output_path = self.output_dir / f"deidentified_{input_path.name}"
            
            file_pairs.append((input_path, output_path))
        
        # Process files
        results = self._process_file_pairs(file_pairs)
        
        # Create batch result
        batch_result = self._create_batch_result(results)
        
        # Save results if requested
        if save_results:
            self._save_batch_results(batch_result)
        
        return batch_result
    
    def _find_input_files(
        self,
        file_patterns: Optional[List[str]] = None,
        recursive: bool = True
    ) -> List[Path]:
        """Find input files matching patterns."""
        if file_patterns is None:
            file_patterns = ['*.txt', '*.docx', '*.pdf', '*.rtf']
        
        input_files = []
        
        for pattern in file_patterns:
            if recursive:
                files = list(self.input_dir.rglob(pattern))
            else:
                files = list(self.input_dir.glob(pattern))
            
            input_files.extend(files)
        
        # Remove duplicates and sort
        input_files = sorted(list(set(input_files)))
        
        return input_files
    
    def _process_files(self, input_files: List[Path]) -> List[ProcessingResult]:
        """Process a list of input files."""
        # Create file pairs with output paths
        file_pairs = []
        for input_path in input_files:
            # Create output path maintaining directory structure
            relative_path = input_path.relative_to(self.input_dir)
            output_path = self.output_dir / relative_path.parent / f"deidentified_{relative_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_pairs.append((input_path, output_path))
        
        return self._process_file_pairs(file_pairs)
    
    def _process_file_pairs(
        self, file_pairs: List[tuple]
    ) -> List[ProcessingResult]:
        """Process file pairs with parallel execution."""
        results = []
        
        # Initialize progress tracking
        self.progress_tracker.start(len(file_pairs))
        
        try:
            if self.use_multiprocessing and len(file_pairs) > 1:
                results = self._process_parallel(file_pairs)
            else:
                results = self._process_sequential(file_pairs)
        
        finally:
            self.progress_tracker.finish()
        
        return results
    
    def _process_parallel(self, file_pairs: List[tuple]) -> List[ProcessingResult]:
        """Process files in parallel using multiprocessing."""
        results = []
        
        # Split into chunks for better memory management
        chunks = [
            file_pairs[i:i + self.chunk_size]
            for i in range(0, len(file_pairs), self.chunk_size)
        ]
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit chunks
            future_to_chunk = {
                executor.submit(self._process_chunk, chunk): chunk
                for chunk in chunks
            }
            
            # Collect results
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    self.progress_tracker.update(len(chunk_results))
                except Exception as e:
                    self.logger.error(f"Chunk processing failed: {e}")
                    # Create error results for failed chunk
                    for input_path, output_path in chunk:
                        error_result = ProcessingResult(
                            success=False,
                            entities_found=0,
                            entities_removed=0,
                            processing_time=0.0,
                            confidence_scores={},
                            warnings=[],
                            errors=[f"Parallel processing failed: {str(e)}"],
                            metadata={'input_path': str(input_path), 'output_path': str(output_path)}
                        )
                        results.append(error_result)
                    self.progress_tracker.update(len(chunk))
        
        return results
    
    def _process_sequential(self, file_pairs: List[tuple]) -> List[ProcessingResult]:
        """Process files sequentially."""
        results = []
        
        # Create pipeline instance
        pipeline = DeidentificationPipeline(self.config)
        
        for input_path, output_path in file_pairs:
            try:
                result = pipeline.process_document(
                    document_path=input_path,
                    output_path=output_path,
                    metadata={'input_path': str(input_path), 'output_path': str(output_path)}
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process {input_path}: {e}")
                error_result = ProcessingResult(
                    success=False,
                    entities_found=0,
                    entities_removed=0,
                    processing_time=0.0,
                    confidence_scores={},
                    warnings=[],
                    errors=[f"Processing failed: {str(e)}"],
                    metadata={'input_path': str(input_path), 'output_path': str(output_path)}
                )
                results.append(error_result)
            
            finally:
                self.progress_tracker.update(1)
        
        return results
    
    @staticmethod
    def _process_chunk(file_pairs: List[tuple]) -> List[ProcessingResult]:
        """Process a chunk of files (for multiprocessing)."""
        # Create new pipeline instance for this process
        pipeline = DeidentificationPipeline()
        results = []
        
        for input_path, output_path in file_pairs:
            try:
                result = pipeline.process_document(
                    document_path=input_path,
                    output_path=output_path,
                    metadata={'input_path': str(input_path), 'output_path': str(output_path)}
                )
                results.append(result)
                
            except Exception as e:
                error_result = ProcessingResult(
                    success=False,
                    entities_found=0,
                    entities_removed=0,
                    processing_time=0.0,
                    confidence_scores={},
                    warnings=[],
                    errors=[f"Processing failed: {str(e)}"],
                    metadata={'input_path': str(input_path), 'output_path': str(output_path)}
                )
                results.append(error_result)
        
        return results
    
    def _create_batch_result(self, results: List[ProcessingResult]) -> BatchResult:
        """Create batch result from individual results."""
        total_documents = len(results)
        successful_documents = sum(1 for r in results if r.success)
        failed_documents = total_documents - successful_documents
        
        total_entities_found = sum(r.entities_found for r in results)
        total_entities_removed = sum(r.entities_removed for r in results)
        total_processing_time = sum(r.processing_time for r in results)
        
        average_processing_time = (
            total_processing_time / total_documents if total_documents > 0 else 0.0
        )
        
        # Collect all errors
        errors = []
        for result in results:
            errors.extend(result.errors)
        
        # Create metadata
        metadata = {
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'processing_mode': 'parallel' if self.use_multiprocessing else 'sequential',
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'timestamp': time.time(),
        }
        
        return BatchResult(
            total_documents=total_documents,
            successful_documents=successful_documents,
            failed_documents=failed_documents,
            total_entities_found=total_entities_found,
            total_entities_removed=total_entities_removed,
            total_processing_time=total_processing_time,
            average_processing_time=average_processing_time,
            results=results,
            errors=errors,
            metadata=metadata
        )
    
    def _save_batch_results(self, batch_result: BatchResult) -> None:
        """Save batch processing results."""
        try:
            # Create results directory
            results_dir = self.output_dir / "batch_results"
            results_dir.mkdir(exist_ok=True)
            
            # Save summary
            summary_path = results_dir / f"batch_summary_{int(time.time())}.json"
            summary_data = {
                'total_documents': batch_result.total_documents,
                'successful_documents': batch_result.successful_documents,
                'failed_documents': batch_result.failed_documents,
                'total_entities_found': batch_result.total_entities_found,
                'total_entities_removed': batch_result.total_entities_removed,
                'total_processing_time': batch_result.total_processing_time,
                'average_processing_time': batch_result.average_processing_time,
                'errors': batch_result.errors,
                'metadata': batch_result.metadata,
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            # Save detailed results
            detailed_path = results_dir / f"batch_detailed_{int(time.time())}.json"
            detailed_data = []
            
            for result in batch_result.results:
                result_data = {
                    'success': result.success,
                    'entities_found': result.entities_found,
                    'entities_removed': result.entities_removed,
                    'processing_time': result.processing_time,
                    'confidence_scores': result.confidence_scores,
                    'warnings': result.warnings,
                    'errors': result.errors,
                    'metadata': result.metadata,
                }
                detailed_data.append(result_data)
            
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Batch results saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save batch results: {e}")
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return {
            'input_directory': str(self.input_dir),
            'output_directory': str(self.output_dir),
            'max_workers': self.max_workers,
            'chunk_size': self.chunk_size,
            'use_multiprocessing': self.use_multiprocessing,
            'progress': self.progress_tracker.get_progress(),
        }
    
    def estimate_processing_time(
        self,
        num_documents: int,
        avg_document_size: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Estimate processing time for a batch.
        
        Args:
            num_documents: Number of documents to process
            avg_document_size: Average document size in characters
            
        Returns:
            Dictionary with time estimates
        """
        # Base processing time per document (seconds)
        base_time_per_doc = 2.0
        
        # Adjust for document size
        if avg_document_size:
            size_factor = max(1.0, avg_document_size / 1000)  # 1000 chars baseline
            base_time_per_doc *= size_factor
        
        # Sequential processing time
        sequential_time = num_documents * base_time_per_doc
        
        # Parallel processing time (with overhead)
        parallel_overhead = 0.1  # 10% overhead
        parallel_time = (sequential_time / self.max_workers) * (1 + parallel_overhead)
        
        return {
            'sequential_estimate': sequential_time,
            'parallel_estimate': parallel_time,
            'speedup_factor': sequential_time / parallel_time,
            'time_per_document': base_time_per_doc,
        }

