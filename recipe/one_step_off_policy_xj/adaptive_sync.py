# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adaptive Sync Frequency for Multi-Step Off-Policy Training

This module implements an adaptive mechanism to dynamically adjust the weight
synchronization frequency based on training stability metrics.
"""


class AdaptiveSyncFrequency:
    """
    Adaptive synchronization frequency controller based on PPO clip ratio.

    The controller monitors the average clip ratio between synchronization points
    and adjusts sync_frequency accordingly:
    - Low clip ratio (< stable_threshold): Training is stable, increase sync interval
    - High clip ratio (> unstable_threshold): Training is unstable, decrease sync interval
    - Normal range: Keep sync interval unchanged

    Args:
        init_sync_freq (int): Initial synchronization frequency. Default: 3
        min_sync_freq (int): Minimum synchronization frequency. Default: 1
        max_sync_freq (int): Maximum synchronization frequency. Default: 10
        stable_threshold (float): Clip ratio threshold for stable training. Default: 0.1
        unstable_threshold (float): Clip ratio threshold for unstable training. Default: 0.4

    Example:
        >>> adaptive_sync = AdaptiveSyncFrequency(
        ...     init_sync_freq=3,
        ...     min_sync_freq=1,
        ...     max_sync_freq=10,
        ...     stable_threshold=0.1,
        ...     unstable_threshold=0.4
        ... )
        >>>
        >>> # During training loop
        >>> adaptive_sync.record_clip_ratio(0.08)  # Step 1
        >>> adaptive_sync.record_clip_ratio(0.06)  # Step 2
        >>> new_freq = adaptive_sync.compute_and_adjust()  # Before sync at Step 3
        >>> # Output: [Adaptive] avg_clip_ratio=0.070, sync_frequency: 3 → 4 (stable)
    """

    def __init__(
        self,
        init_sync_freq: int = 3,
        min_sync_freq: int = 1,
        max_sync_freq: int = 10,
        stable_threshold: float = 0.1,
        unstable_threshold: float = 0.4,
    ):
        """Initialize the adaptive sync frequency controller."""
        assert min_sync_freq >= 1, "min_sync_freq must be >= 1"
        assert max_sync_freq >= min_sync_freq, "max_sync_freq must be >= min_sync_freq"
        assert init_sync_freq >= min_sync_freq and init_sync_freq <= max_sync_freq, \
            f"init_sync_freq must be in [{min_sync_freq}, {max_sync_freq}]"
        assert 0 < stable_threshold < unstable_threshold, \
            "stable_threshold must be < unstable_threshold"

        self.sync_frequency = init_sync_freq
        self.min_sync_freq = min_sync_freq
        self.max_sync_freq = max_sync_freq
        self.stable_threshold = stable_threshold
        self.unstable_threshold = unstable_threshold

        # Store clip_ratio history between two synchronizations
        self.clip_ratio_history = []

        print(f"[AdaptiveSyncFrequency] Initialized with:")
        print(f"  - init_sync_freq: {init_sync_freq}")
        print(f"  - min_sync_freq: {min_sync_freq}")
        print(f"  - max_sync_freq: {max_sync_freq}")
        print(f"  - stable_threshold: {stable_threshold}")
        print(f"  - unstable_threshold: {unstable_threshold}")

    def record_clip_ratio(self, clip_ratio: float):
        """
        Record the clip ratio for the current training step.

        This should be called after each actor update (when clip_ratio is available).

        Args:
            clip_ratio (float): PPO clip ratio from actor/pg_clipfrac metric
        """
        self.clip_ratio_history.append(clip_ratio)

    def compute_and_adjust(self) -> int:
        """
        Compute average clip ratio and adjust sync_frequency.

        This should be called right before weight synchronization (when sync_counter % sync_frequency == 0).

        The adjustment logic:
        1. Calculate average clip_ratio over the interval since last sync
        2. If avg_clip_ratio < stable_threshold: increase sync_frequency by 1 (more off-policy)
        3. If avg_clip_ratio > unstable_threshold: decrease sync_frequency by 1 (more on-policy)
        4. Otherwise: keep sync_frequency unchanged
        5. Clear history for next interval

        Returns:
            int: The new sync_frequency value
        """
        if len(self.clip_ratio_history) == 0:
            print("[AdaptiveSyncFrequency] Warning: No clip_ratio recorded, keeping sync_frequency unchanged")
            return self.sync_frequency

        # Compute average clip ratio
        avg_clip_ratio = sum(self.clip_ratio_history) / len(self.clip_ratio_history)

        # Store old frequency for logging
        old_freq = self.sync_frequency

        # Adjust sync_frequency based on thresholds
        if avg_clip_ratio < self.stable_threshold:
            # Training is stable, can be more off-policy (increase sync_frequency)
            self.sync_frequency = min(self.sync_frequency + 1, self.max_sync_freq)
            reason = "stable (clip_ratio low)"

        elif avg_clip_ratio > self.unstable_threshold:
            # Training is unstable, need to be more on-policy (decrease sync_frequency)
            self.sync_frequency = max(self.sync_frequency - 1, self.min_sync_freq)
            reason = "unstable (clip_ratio high)"

        else:
            # Normal range, keep unchanged
            reason = "normal range"

        # Log the decision
        if old_freq != self.sync_frequency:
            print(f"[AdaptiveSyncFrequency] avg_clip_ratio={avg_clip_ratio:.3f}, "
                  f"sync_frequency: {old_freq} → {self.sync_frequency} ({reason})")
        else:
            print(f"[AdaptiveSyncFrequency] avg_clip_ratio={avg_clip_ratio:.3f}, "
                  f"sync_frequency={self.sync_frequency} ({reason})")

        # Clear history for next interval
        self.clip_ratio_history.clear()

        return self.sync_frequency

    def get_sync_frequency(self) -> int:
        """
        Get current sync_frequency value.

        Returns:
            int: Current sync_frequency
        """
        return self.sync_frequency

    def reset(self):
        """
        Reset the controller state (clear history).

        This can be used when you want to restart the adaptive process.
        """
        self.clip_ratio_history.clear()
        print(f"[AdaptiveSyncFrequency] Reset, cleared history")

    def get_stats(self) -> dict:
        """
        Get statistics about the current state.

        Returns:
            dict: Statistics including current sync_frequency, history length, etc.
        """
        return {
            "sync_frequency": self.sync_frequency,
            "clip_ratio_history_length": len(self.clip_ratio_history),
            "avg_clip_ratio": sum(self.clip_ratio_history) / len(self.clip_ratio_history)
                             if len(self.clip_ratio_history) > 0 else None,
        }
