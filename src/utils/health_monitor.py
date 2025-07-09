"""
Health monitoring utilities for tracking system performance and metrics.
Provides monitoring capabilities for the RAG system.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


class HealthMonitor:
    """Monitor system health and performance metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._lock = threading.Lock()
        
        # Metrics storage
        self.request_metrics = deque(maxlen=max_history)
        self.question_metrics = deque(maxlen=max_history)
        self.batch_metrics = deque(maxlen=max_history)
        self.error_counts = defaultdict(int)
        self.performance_metrics = {}
        
        # Start time
        self.start_time = time.time()
        
        logger.info("📊 Health monitor initialized")
    
    def record_question_metric(self, request_id: str, question_length: int, 
                             chunks_found: int, processing_time_ms: float):
        """Record metrics for a question processing."""
        with self._lock:
            metric = {
                'timestamp': time.time(),
                'request_id': request_id,
                'question_length': question_length,
                'chunks_found': chunks_found,
                'processing_time_ms': processing_time_ms,
                'success': chunks_found > 0
            }
            self.question_metrics.append(metric)
    
    def record_batch_metric(self, batch_id: str, total_questions: int, successful_questions: int):
        """Record metrics for batch processing."""
        with self._lock:
            metric = {
                'timestamp': time.time(),
                'batch_id': batch_id,
                'total_questions': total_questions,
                'successful_questions': successful_questions,
                'success_rate': successful_questions / total_questions if total_questions > 0 else 0
            }
            self.batch_metrics.append(metric)
    
    def record_error(self, error_type: str, error_message: str = None):
        """Record an error occurrence."""
        with self._lock:
            self.error_counts[error_type] += 1
            
            # Also record as a metric for time-based analysis
            error_metric = {
                'timestamp': time.time(),
                'error_type': error_type,
                'error_message': error_message
            }
            # Could store in separate error_metrics deque if needed
    
    def record_performance_metric(self, metric_name: str, value: float, unit: str = None):
        """Record a performance metric."""
        with self._lock:
            self.performance_metrics[metric_name] = {
                'value': value,
                'unit': unit,
                'timestamp': time.time()
            }
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics."""
        with self._lock:
            current_time = time.time()
            
            # Calculate uptime
            uptime_seconds = current_time - self.start_time
            
            # Question processing stats
            question_stats = self._calculate_question_stats()
            
            # Batch processing stats
            batch_stats = self._calculate_batch_stats()
            
            # Error statistics
            error_stats = dict(self.error_counts)
            
            # Performance metrics
            perf_metrics = dict(self.performance_metrics)
            
            return {
                'uptime_seconds': uptime_seconds,
                'uptime_human': self._format_uptime(uptime_seconds),
                'total_questions_processed': len(self.question_metrics),
                'total_batches_processed': len(self.batch_metrics),
                'question_processing': question_stats,
                'batch_processing': batch_stats,
                'errors': error_stats,
                'performance': perf_metrics,
                'last_updated': current_time
            }
    
    def _calculate_question_stats(self) -> Dict[str, Any]:
        """Calculate question processing statistics."""
        if not self.question_metrics:
            return {
                'total_processed': 0,
                'success_rate': 0.0,
                'avg_processing_time_ms': 0.0,
                'avg_chunks_found': 0.0,
                'recent_activity': []
            }
        
        # Convert to list for easier processing
        metrics = list(self.question_metrics)
        
        # Overall stats
        total_processed = len(metrics)
        successful = sum(1 for m in metrics if m['success'])
        success_rate = successful / total_processed if total_processed > 0 else 0.0
        
        # Timing stats
        processing_times = [m['processing_time_ms'] for m in metrics]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
        
        # Chunks stats
        chunks_found = [m['chunks_found'] for m in metrics]
        avg_chunks_found = sum(chunks_found) / len(chunks_found) if chunks_found else 0.0
        
        # Recent activity (last hour)
        one_hour_ago = time.time() - 3600
        recent_metrics = [m for m in metrics if m['timestamp'] > one_hour_ago]
        
        recent_activity = {
            'last_hour_count': len(recent_metrics),
            'last_hour_success_rate': sum(1 for m in recent_metrics if m['success']) / len(recent_metrics) if recent_metrics else 0.0
        }
        
        return {
            'total_processed': total_processed,
            'success_rate': round(success_rate, 3),
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'avg_chunks_found': round(avg_chunks_found, 2),
            'recent_activity': recent_activity
        }
    
    def _calculate_batch_stats(self) -> Dict[str, Any]:
        """Calculate batch processing statistics."""
        if not self.batch_metrics:
            return {
                'total_batches': 0,
                'avg_batch_size': 0.0,
                'avg_success_rate': 0.0
            }
        
        metrics = list(self.batch_metrics)
        
        total_batches = len(metrics)
        total_questions = sum(m['total_questions'] for m in metrics)
        avg_batch_size = total_questions / total_batches if total_batches > 0 else 0.0
        
        success_rates = [m['success_rate'] for m in metrics]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0
        
        return {
            'total_batches': total_batches,
            'total_questions_in_batches': total_questions,
            'avg_batch_size': round(avg_batch_size, 1),
            'avg_success_rate': round(avg_success_rate, 3)
        }
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Format uptime in human-readable format."""
        uptime_delta = timedelta(seconds=int(uptime_seconds))
        days = uptime_delta.days
        hours, remainder = divmod(uptime_delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m {seconds}s"
        elif hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    def get_recent_questions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent question metrics."""
        with self._lock:
            recent = list(self.question_metrics)[-limit:]
            return [
                {
                    'request_id': m['request_id'],
                    'timestamp': datetime.fromtimestamp(m['timestamp']).isoformat(),
                    'question_length': m['question_length'],
                    'chunks_found': m['chunks_found'],
                    'processing_time_ms': m['processing_time_ms'],
                    'success': m['success']
                }
                for m in recent
            ]
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary with counts and rates."""
        with self._lock:
            total_requests = len(self.question_metrics)
            total_errors = sum(self.error_counts.values())
            
            error_rate = total_errors / total_requests if total_requests > 0 else 0.0
            
            return {
                'total_errors': total_errors,
                'error_rate': round(error_rate, 4),
                'error_breakdown': dict(self.error_counts),
                'most_common_error': max(self.error_counts.items(), key=lambda x: x[1])[0] if self.error_counts else None
            }
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing or maintenance)."""
        with self._lock:
            self.request_metrics.clear()
            self.question_metrics.clear()
            self.batch_metrics.clear()
            self.error_counts.clear()
            self.performance_metrics.clear()
            self.start_time = time.time()
            
            logger.info("📊 Health metrics reset")
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external monitoring systems."""
        with self._lock:
            return {
                'timestamp': time.time(),
                'start_time': self.start_time,
                'question_metrics': list(self.question_metrics),
                'batch_metrics': list(self.batch_metrics),
                'error_counts': dict(self.error_counts),
                'performance_metrics': dict(self.performance_metrics)
            }