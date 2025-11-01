"""
Profiling utilities for performance analysis.
Similar to MATLAB's profiling tool, tracks time consumption of major functions.
"""

import time
import functools
import contextlib
from collections import defaultdict
from typing import Dict, List, Optional
import torch
from datetime import datetime
import json
import os


class Profiler:
    """
    Performance profiler for tracking function execution times.
    Similar to MATLAB's profiler, provides detailed timing statistics.
    """

    def __init__(self, enabled: bool = True, use_cuda: bool = True):
        self.enabled = enabled
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.call_counts: Dict[str, int] = defaultdict(int)
        self.total_times: Dict[str, float] = defaultdict(float)
        self.cuda_timings: Dict[str, List[float]] = defaultdict(list)
        self.stack: List[tuple] = []  # (name, start_time, start_cuda_time)

    def clear(self):
        """Clear all profiling data"""
        self.timings.clear()
        self.call_counts.clear()
        self.total_times.clear()
        self.cuda_timings.clear()
        self.stack.clear()

    @contextlib.contextmanager
    def profile(self, name: str):
        """Context manager for profiling a code block"""
        if not self.enabled:
            yield
            return

        # Record start times
        start_time = time.perf_counter()
        if self.use_cuda:
            torch.cuda.synchronize()
            start_cuda = torch.cuda.Event(enable_timing=True)
            start_cuda.record()
        else:
            start_cuda = None

        try:
            self.stack.append((name, start_time, start_cuda))
            yield
        finally:
            # Record end times
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time

            self.timings[name].append(elapsed_time)
            self.call_counts[name] += 1
            self.total_times[name] += elapsed_time

            if self.use_cuda and start_cuda is not None:
                end_cuda = torch.cuda.Event(enable_timing=True)
                end_cuda.record()
                torch.cuda.synchronize()
                cuda_elapsed = start_cuda.elapsed_time(end_cuda) / 1000.0  # Convert ms to seconds
                self.cuda_timings[name].append(cuda_elapsed)

            self.stack.pop()

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all profiled functions"""
        stats = {}
        for name in self.timings.keys():
            times = self.timings[name]
            if times:
                stats[name] = {
                    'calls': self.call_counts[name],
                    'total_time': sum(times),
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'std_time': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 if len(times) > 1 else 0.0,
                }

                if name in self.cuda_timings and self.cuda_timings[name]:
                    cuda_times = self.cuda_timings[name]
                    stats[name]['cuda_total'] = sum(cuda_times)
                    stats[name]['cuda_avg'] = sum(cuda_times) / len(cuda_times)

        return stats

    def print_report(self, sort_by: str = 'total_time', top_n: Optional[int] = None, logger=None):
        """
        Print profiling report sorted by specified metric.

        Args:
            sort_by: Metric to sort by (default: 'total_time')
            top_n: Limit number of results (None for all)
            logger: Optional logger to also write to (in addition to print)
        """
        stats = self.get_stats()

        if not stats:
            msg = "No profiling data available"
            print(msg)
            if logger:
                logger.info(msg)
            return

        # Sort by specified metric
        sorted_stats = sorted(
            stats.items(),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )

        if top_n:
            sorted_stats = sorted_stats[:top_n]

        # Build report lines
        lines = []
        lines.append("\n" + "="*80)
        lines.append(f"Profiling Report (sorted by {sort_by})")
        lines.append("="*80)
        lines.append(f"{'Function':<40} {'Calls':<8} {'Total(s)':<12} {'Avg(s)':<12} {'Min(s)':<12} {'Max(s)':<12}")
        lines.append("-"*80)

        for name, stat in sorted_stats:
            calls = stat['calls']
            total = stat['total_time']
            avg = stat['avg_time']
            min_time = stat['min_time']
            max_time = stat['max_time']

            line = f"{name[:39]:<40} {calls:<8} {total:<12.4f} {avg:<12.4f} {min_time:<12.4f} {max_time:<12.4f}"
            lines.append(line)
            print(line)

            if 'cuda_avg' in stat:
                cuda_line = f"  └─ CUDA avg: {stat['cuda_avg']:.4f}s"
                lines.append(cuda_line)
                print(cuda_line)

        lines.append("="*80)

        # Print summary
        total_time = sum(s['total_time'] for s in stats.values())
        summary_line1 = f"\nTotal profiled time: {total_time:.4f}s"
        summary_line2 = f"Total function calls: {sum(s['calls'] for s in stats.values())}"
        lines.append(summary_line1)
        lines.append(summary_line2)

        # Print to console
        print("\n" + "="*80)
        print(f"Profiling Report (sorted by {sort_by})")
        print("="*80)
        print(f"{'Function':<40} {'Calls':<8} {'Total(s)':<12} {'Avg(s)':<12} {'Min(s)':<12} {'Max(s)':<12}")
        print("-"*80)

        for name, stat in sorted_stats:
            calls = stat['calls']
            total = stat['total_time']
            avg = stat['avg_time']
            min_time = stat['min_time']
            max_time = stat['max_time']
            print(f"{name[:39]:<40} {calls:<8} {total:<12.4f} {avg:<12.4f} {min_time:<12.4f} {max_time:<12.4f}")
            if 'cuda_avg' in stat:
                print(f"  └─ CUDA avg: {stat['cuda_avg']:.4f}s")

        print("="*80)
        print(summary_line1)
        print(summary_line2)

        # Also write to logger if provided (all lines including formatting)
        if logger:
            for line in lines:
                logger.info(line)
            logger.info(summary_line1)
            logger.info(summary_line2)

    def save_report(self, filepath: str, sort_by: str = 'total_time'):
        """Save profiling report to JSON file"""
        stats = self.get_stats()

        report = {
            'timestamp': datetime.now().isoformat(),
            'total_time': sum(s['total_time'] for s in stats.values()),
            'functions': stats
        }

        # Sort functions by specified metric
        sorted_functions = sorted(
            list(stats.items()),
            key=lambda x: x[1].get(sort_by, 0),
            reverse=True
        )
        report['sorted_functions'] = [name for name, _ in sorted_functions]

        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Profiling report saved to: {filepath}")


# Global profiler instance
_global_profiler = Profiler()


def get_profiler() -> Profiler:
    """Get global profiler instance"""
    return _global_profiler


def profile_function(name: Optional[str] = None, profiler: Optional[Profiler] = None):
    """
    Decorator for profiling function execution time.

    Usage:
        @profile_function("my_function")
        def my_function():
            ...
    """
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__qualname__}"
        prof = profiler or _global_profiler

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not prof.enabled:
                return func(*args, **kwargs)

            with prof.profile(func_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


def profile_cuda(name: Optional[str] = None):
    """
    Decorator for profiling CUDA operations.
    Automatically synchronizes CUDA before timing.
    """
    def decorator(func):
        func_name = name or f"{func.__module__}.{func.__qualname__}"

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _global_profiler.enabled:
                return func(*args, **kwargs)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start_time = time.perf_counter()
            if torch.cuda.is_available():
                start_cuda = torch.cuda.Event(enable_timing=True)
                start_cuda.record()

            result = func(*args, **kwargs)

            if torch.cuda.is_available():
                end_cuda = torch.cuda.Event(enable_timing=True)
                end_cuda.record()
                torch.cuda.synchronize()
                cuda_time = start_cuda.elapsed_time(end_cuda) / 1000.0

                _global_profiler.timings[func_name].append(time.perf_counter() - start_time)
                _global_profiler.cuda_timings[func_name].append(cuda_time)
                _global_profiler.call_counts[func_name] += 1
                _global_profiler.total_times[func_name] += cuda_time
            else:
                elapsed = time.perf_counter() - start_time
                _global_profiler.timings[func_name].append(elapsed)
                _global_profiler.call_counts[func_name] += 1
                _global_profiler.total_times[func_name] += elapsed

            return result

        return wrapper
    return decorator


@contextlib.contextmanager
def profile(name: str, profiler: Optional[Profiler] = None):
    """
    Context manager for profiling a code block.

    Usage:
        with profile("my_code_block"):
            # code to profile
            ...
    """
    prof = profiler or _global_profiler
    with prof.profile(name):
        yield
