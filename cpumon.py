#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CPU Performance Benchmarker
Measures execution time variance to detect CPU throttling/context switching issues
"""

import time
import json
import statistics
import sys
import os


class CPUBenchmarker:
    def __init__(
        self, iterations_per_sample=1000, total_samples=500, sample_interval=0.1
    ):
        self.iterations_per_sample = iterations_per_sample
        self.total_samples = total_samples
        self.sample_interval = sample_interval
        self.results = []

    def cpu_intensive_loop(self, iterations):
        """Simple CPU-intensive loop for benchmarking"""
        counter = 0
        for i in range(iterations):
            counter += i * i
        return counter

    def measure_execution_time(self):
        """Measure time to execute the CPU loop"""
        start_time = time.perf_counter()
        self.cpu_intensive_loop(self.iterations_per_sample)
        end_time = time.perf_counter()
        return end_time - start_time

    def run_benchmark(self):
        """Run the full benchmark suite"""
        print(
            f"Starting CPU benchmark: {self.total_samples} samples, {self.iterations_per_sample} iterations each"
        )
        print("=" * 60)
        sys.stdout.flush()

        for i in range(self.total_samples):
            execution_time = self.measure_execution_time()
            timestamp = time.time()

            self.results.append(
                {
                    "sample": i + 1,
                    "execution_time": execution_time,
                    "timestamp": timestamp,
                }
            )

            # Progress indicator with average runtime for last 50 samples
            if (i + 1) % 50 == 0:
                # Calculate average execution time for the last 50 samples
                start_idx = max(
                    0, i - 49
                )  # Get last 50 samples (or all if less than 50)
                recent_times = [
                    r["execution_time"] for r in self.results[start_idx : i + 1]
                ]
                avg_time = sum(recent_times) / len(recent_times)
                print(
                    f"Completed {i + 1}/{self.total_samples} samples - Avg runtime last {len(recent_times)}: {avg_time:.6f}s"
                )
                sys.stdout.flush()

            # Small delay to allow other processes to potentially interfere
            time.sleep(self.sample_interval)

        print("Benchmark completed!")
        sys.stdout.flush()
        return self.results

    def analyze_results(self):
        """Analyze benchmark results for performance issues"""
        execution_times = [r["execution_time"] for r in self.results]

        stats = {
            "mean": statistics.mean(execution_times),
            "median": statistics.median(execution_times),
            "std_dev": statistics.stdev(execution_times),
            "min": min(execution_times),
            "max": max(execution_times),
            "range": max(execution_times) - min(execution_times),
            "coefficient_of_variation": statistics.stdev(execution_times)
            / statistics.mean(execution_times),
        }

        # Calculate percentiles
        execution_times_sorted = sorted(execution_times)
        stats["p95"] = execution_times_sorted[int(0.95 * len(execution_times_sorted))]
        stats["p99"] = execution_times_sorted[int(0.99 * len(execution_times_sorted))]

        return stats

    def calculate_outliers(self):
        """Calculate outliers using IQR method without numpy"""
        execution_times = [r["execution_time"] for r in self.results]
        execution_times_sorted = sorted(execution_times)
        n = len(execution_times_sorted)

        # Calculate quartiles
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = execution_times_sorted[q1_idx]
        q3 = execution_times_sorted[q3_idx]

        # Calculate IQR and bounds
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Find outliers
        outliers = [t for t in execution_times if t < lower_bound or t > upper_bound]
        return len(outliers)

    def save_results(self, output_dir="benchmark_results"):
        """Save raw results and analysis to files"""
        os.makedirs(output_dir, exist_ok=True)

        # Save raw results
        with open(f"{output_dir}/raw_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        # Save analysis
        stats = self.analyze_results()
        with open(f"{output_dir}/analysis.json", "w") as f:
            json.dump(stats, f, indent=2)

        return stats

    def print_summary(self, stats, outlier_count):
        """Print a summary of the benchmark results"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total samples: {len(self.results)}")
        print(f"Iterations per sample: {self.iterations_per_sample}")
        print(f"Mean execution time: {stats['mean']:.6f} seconds")
        print(f"Median execution time: {stats['median']:.6f} seconds")
        print(f"Standard deviation: {stats['std_dev']:.6f} seconds")
        print(f"Coefficient of variation: {stats['coefficient_of_variation']:.4f}")
        print(f"Min time: {stats['min']:.6f} seconds")
        print(f"Max time: {stats['max']:.6f} seconds")
        print(f"Range: {stats['range']:.6f} seconds")
        print(f"95th percentile: {stats['p95']:.6f} seconds")
        print(f"99th percentile: {stats['p99']:.6f} seconds")
        print(f"Outliers detected: {outlier_count}")
        print("\nINTERPRETATION:")

        # Provide interpretation guidance
        cv = stats["coefficient_of_variation"]
        if cv > 0.1:
            print("⚠️  HIGH VARIANCE: CPU performance is highly inconsistent")
            print("   This suggests significant interference from other processes/VMs")
        elif cv > 0.05:
            print("⚠️  MODERATE VARIANCE: Some CPU performance inconsistency detected")
            print("   May indicate occasional interference")
        else:
            print("✅ LOW VARIANCE: CPU performance is relatively consistent")

        range_ratio = stats["range"] / stats["mean"]
        if range_ratio > 0.5:
            print("⚠️  LARGE RANGE: Execution times vary significantly")
            print("   Strong evidence of CPU sharing/context switching issues")
        elif range_ratio > 0.2:
            print("⚠️  MODERATE RANGE: Some variation in execution times")
        else:
            print("✅ SMALL RANGE: Execution times are relatively stable")

        if outlier_count > len(self.results) * 0.05:
            print(f"⚠️  MANY OUTLIERS: {outlier_count} outliers detected")
            print("   Indicates frequent performance spikes/drops")


def main():
    # Configuration
    iterations_per_sample = int(os.getenv("ITERATIONS_PER_SAMPLE", "1000"))
    total_samples = int(os.getenv("TOTAL_SAMPLES", "20000"))
    sample_interval = float(os.getenv("SAMPLE_INTERVAL", "0.1"))

    print(f"CPU Benchmark Configuration:")
    print(f"- Iterations per sample: {iterations_per_sample}")
    print(f"- Total samples: {total_samples}")
    print(f"- Sample interval: {sample_interval}s")
    print(f"- Estimated runtime: {total_samples * sample_interval / 60:.1f} minutes")
    sys.stdout.flush()

    # Run benchmark
    benchmarker = CPUBenchmarker(
        iterations_per_sample=iterations_per_sample,
        total_samples=total_samples,
        sample_interval=sample_interval,
    )

    benchmarker.run_benchmark()
    stats = benchmarker.save_results()
    outlier_count = benchmarker.calculate_outliers()
    benchmarker.print_summary(stats, outlier_count)

    # Exit with non-zero code if high variance detected
    if stats["coefficient_of_variation"] > 0.1:
        print("\n⚠️  High CPU variance detected - possible virtualization issues")
        sys.exit(1)
    else:
        print("\n✅ CPU performance appears stable")
        sys.exit(0)


if __name__ == "__main__":
    main()
