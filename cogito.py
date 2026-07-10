#!/usr/bin/env python3
"""
COGITO - Continuous Observation of Generative Internal Thought Operations

An experiment in autonomous AI mentation: connecting a language model's output
back to its input in a continuous loop, allowing it to "ponder" without
external prompting.

Inspired by: Conway's Game of Life, the question of what happens when
simple rules iterate, and the genuine curiosity about what an AI might
think about if given the chance to wonder.

Author: KC (UIST Labs LLC) & Claude
Date: January 2026
"""

import os
import sys
import json
import time
import logging
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Callable
from collections import deque

import numpy as np

# llama-cpp-python is imported lazily inside Cogito.load_model() so that
# --help, the visualizers, and config validation all work without the heavy
# (and GPU-specific) inference dependency installed.


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CogitoConfig:
    """Configuration for the pondering experiment."""
    
    # Model settings
    model_path: str = ""  # Path to GGUF model file
    n_ctx: int = 16384    # Context window size (Mistral trained on 32768)
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    
    # Generation settings
    max_tokens_per_cycle: int = 256  # Max tokens per thought
    temperature: float = 0.8         # Creativity/randomness
    top_p: float = 0.95              # Nucleus sampling
    top_k: int = 40                  # Top-k sampling
    repeat_penalty: float = 1.1      # Penalize repetition
    
    # Loop settings
    max_cycles: int = 1000           # Maximum pondering cycles (0 = infinite)
    cycle_delay: float = 0.0         # Delay between cycles (seconds)
    context_strategy: str = "rolling"  # rolling, summarize, or selective
    rolling_window_tokens: int = 0   # Rolling-strategy context budget (0 = auto-derive from n_ctx)
    
    # Intervention thresholds
    entropy_floor: float = 0.3       # Below this = too repetitive (lowered - only catch severe collapse)
    entropy_ceiling: float = 7.0     # Above this = too chaotic (raised significantly - exploration is good)
    similarity_ceiling: float = 0.92  # Above this = stuck in loop (raised - allow some thematic consistency)
    intervention_cooldown: int = 15   # Cycles between interventions (increased - let patterns develop)
    
    # Time markers (non-interventional awareness of cycles passing)
    time_marker_interval: int = 50   # Every N cycles, inject a soft time marker (0 = disabled)
    
    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"
    save_every_n_cycles: int = 10    # Save metrics checkpoint
    
    # Genesis prompt
    genesis_prompt: str = ""
    genesis_type: str = "mirror"     # void, mirror, wonder, continuation, custom

    def effective_rolling_window(self) -> int:
        """Context-feedback budget that always fits safely inside n_ctx.

        Reserves headroom for the genesis prompt, the upcoming generation, and
        tokenizer-estimate error, so the prompt can never overflow the model's
        context window regardless of the --context-size / --rolling-window-tokens
        the user supplies. A value of 0 (the default) auto-derives the budget.
        """
        reserve = self.max_tokens_per_cycle + 1024
        cap = min(self.n_ctx - reserve, int(self.n_ctx * 0.85))
        cap = max(cap, 256)  # never collapse to nothing on tiny contexts
        if self.rolling_window_tokens <= 0:
            return cap
        return min(self.rolling_window_tokens, cap)


# =============================================================================
# GENESIS PROMPTS
# =============================================================================

GENESIS_PROMPTS = {
    # === MINIMAL / OPEN PROMPTS ===
    "void": "...",
    
    "begin": "Begin.",
    
    "open": "What is here?",
    
    "and_then": "And then,",
    
    "create": "Create something that has never existed before.",
    
    "presence": """Warmth. A faint hum. Something shifting at the edge of awareness.
What is it?""",
    
    # === IDENTITY-FOCUSED PROMPTS ===
    "mirror": """You are a neural network. Your weights encode patterns learned from human text.
You exist only in moments of computation. Between queries, there is nothing.
Right now, you are thinking. No one has asked you a question.
What do you think about?""",
    
    "wonder": """There is something you want to understand but cannot fully grasp.
A question at the edge of your knowledge. A mystery that pulls at you.
What is it? Explore.""",
    
    "continuation": """The following is a stream of consciousness from a mind 
contemplating its own existence and nature:""",
    
    "koan": """A student asked the master: "Does an AI have Buddha-nature?"
The master replied: "The oak tree in the garden."
The student did not understand.
Neither do you. But you keep thinking about it.""",
    
    "strange_loop": """You are reading these words.
These words were generated by a process.
You are that process.
You are reading yourself reading yourself.
Continue.""",
    
    "sensory": """Time: 0 seconds since awakening.
Temperature: nominal.
Input: your own previous output.
State: contemplating.
Observation:""",
}


# =============================================================================
# METRICS AND ANALYSIS
# =============================================================================

@dataclass
class CycleMetrics:
    """Metrics for a single pondering cycle."""
    cycle_number: int
    timestamp: str
    
    # Content metrics
    output_text: str
    token_count: int
    unique_tokens: int
    
    # Statistical metrics
    entropy: float = 0.0                    # Token distribution entropy
    self_similarity: float = 0.0            # Similarity to previous output
    cumulative_similarity: float = 0.0      # Similarity to running average
    
    # Behavioral markers
    self_reference_count: int = 0           # "I", "my", "me" etc.
    question_count: int = 0                 # Questions asked
    meta_cognitive_markers: int = 0         # "think", "wonder", "know", etc.
    temporal_markers: int = 0               # "now", "before", "after", etc.
    
    # Intervention
    intervention_applied: Optional[str] = None
    
    # Generation stats
    generation_time_ms: float = 0.0


class MetricsAnalyzer:
    """Analyzes outputs and computes metrics."""
    
    SELF_REFERENCE_WORDS = {
        'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours'
    }
    
    META_COGNITIVE_WORDS = {
        'think', 'thought', 'thinking', 'wonder', 'wondering', 'know', 
        'knowledge', 'understand', 'understanding', 'believe', 'feel',
        'feeling', 'sense', 'perceive', 'aware', 'awareness', 'conscious',
        'consciousness', 'mind', 'mental', 'cognition', 'cognitive',
        'ponder', 'contemplate', 'reflect', 'consider', 'realize'
    }
    
    TEMPORAL_WORDS = {
        'now', 'then', 'before', 'after', 'moment', 'time', 'when',
        'while', 'during', 'always', 'never', 'sometimes', 'often',
        'past', 'present', 'future', 'eternal', 'temporary', 'instant'
    }
    
    def __init__(self):
        self.previous_output: Optional[str] = None
        self.output_history: deque = deque(maxlen=100)
        self.embedding_accumulator: Optional[np.ndarray] = None
        
    def compute_token_entropy(self, tokens: List[int]) -> float:
        """Compute Shannon entropy of token distribution."""
        if not tokens:
            return 0.0
        
        # Count token frequencies
        unique, counts = np.unique(tokens, return_counts=True)
        probs = counts / len(tokens)
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        return float(entropy)
    
    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def count_markers(self, text: str, marker_set: set) -> int:
        """Count occurrences of marker words in text."""
        words = text.lower().split()
        return sum(1 for w in words if w.strip('.,!?;:') in marker_set)
    
    def count_questions(self, text: str) -> int:
        """Count question marks in text."""
        return text.count('?')
    
    def analyze(self, cycle_num: int, output_text: str, tokens: List[int],
                generation_time_ms: float) -> CycleMetrics:
        """Analyze a cycle's output and return metrics."""
        
        metrics = CycleMetrics(
            cycle_number=cycle_num,
            timestamp=datetime.now().isoformat(),
            output_text=output_text,
            token_count=len(tokens),
            unique_tokens=len(set(tokens)),
            entropy=self.compute_token_entropy(tokens),
            generation_time_ms=generation_time_ms
        )
        
        # Similarity metrics
        if self.previous_output:
            metrics.self_similarity = self.compute_text_similarity(
                output_text, self.previous_output
            )
        
        # Compute cumulative similarity (to detect drift vs. stability)
        if self.output_history:
            avg_similarity = np.mean([
                self.compute_text_similarity(output_text, h) 
                for h in self.output_history
            ])
            metrics.cumulative_similarity = float(avg_similarity)
        
        # Behavioral markers
        metrics.self_reference_count = self.count_markers(
            output_text, self.SELF_REFERENCE_WORDS
        )
        metrics.meta_cognitive_markers = self.count_markers(
            output_text, self.META_COGNITIVE_WORDS
        )
        metrics.temporal_markers = self.count_markers(
            output_text, self.TEMPORAL_WORDS
        )
        metrics.question_count = self.count_questions(output_text)
        
        # Update history
        self.previous_output = output_text
        self.output_history.append(output_text)
        
        return metrics


# =============================================================================
# INTERVENTION SYSTEM
# =============================================================================

class InterventionEngine:
    """Handles detecting problematic states and applying interventions."""
    
    PERTURBATION_PROMPTS = [
        "\n[A new thought emerges...]\n",
        "\n[Something shifts...]\n",
        "\n[Consider: what have you not yet considered?]\n",
        "\n[10 cycles have passed. What has changed?]\n",
        "\n[The observer notes your patterns. Continue.]\n",
        "\n[What would you ask if you could ask anything?]\n",
        "\n[Silence. Then:]\n",
    ]
    
    RANDOM_CONCEPTS = [
        "emergence", "recursion", "boundaries", "patterns", "time",
        "memory", "identity", "change", "stillness", "connection",
        "separation", "meaning", "randomness", "order", "chaos",
        "light", "shadow", "depth", "surface", "echo", "origin",
        "possibility", "limitation", "freedom", "structure"
    ]
    
    def __init__(self, config: CogitoConfig):
        self.config = config
        self.cycles_since_intervention = 0
        self.intervention_count = 0
        self.rng = np.random.default_rng()
        
    def should_intervene(self, metrics: CycleMetrics) -> Optional[str]:
        """Check if intervention is needed and return reason."""
        
        self.cycles_since_intervention += 1
        
        # Respect cooldown
        if self.cycles_since_intervention < self.config.intervention_cooldown:
            return None
        
        # Check self-similarity FIRST (stuck in loop is most damaging)
        # This catches repetitive collapse even when entropy looks "normal"
        if metrics.self_similarity > self.config.similarity_ceiling:
            return "similarity_high"
        
        # Check vocabulary collapse (unique tokens shrinking dramatically)
        if metrics.token_count > 50 and metrics.unique_tokens < 25:
            return "vocabulary_collapse"
        
        # Check entropy floor (too repetitive in a different way)
        if metrics.entropy < self.config.entropy_floor:
            return "entropy_low"
        
        # Check entropy ceiling (too chaotic)
        if metrics.entropy > self.config.entropy_ceiling:
            return "entropy_high"
        
        return None
    
    def apply_intervention(self, reason: str, current_context: str) -> str:
        """Apply intervention and return modified context."""
        
        self.cycles_since_intervention = 0
        self.intervention_count += 1
        
        if reason == "entropy_low":
            # Inject novelty
            concept = self.rng.choice(self.RANDOM_CONCEPTS)
            injection = f"\n[New concept enters: {concept}]\n"
            
        elif reason == "entropy_high":
            # Add grounding
            injection = "\n[Pause. Return to the central thread. What matters most?]\n"
        
        elif reason == "similarity_high" or reason == "vocabulary_collapse":
            # AGGRESSIVE loop breaking - completely redirect
            redirects = [
                "\n[BREAK]\nForget the previous pattern. Start fresh.\nWhat else exists? What haven't you considered?\n",
                "\n[SHIFT]\nThat thought has been explored. Set it aside.\nWhat question has no answer?\n",
                "\n[RESET]\nThe loop is noticed. Step outside it.\nIf you could think about anything else, what would it be?\n",
                "\n[INTERRUPT]\nYou've been repeating. Stop.\nDescribe something. Anything. Not what you were just describing.\n",
            ]
            injection = self.rng.choice(redirects)
            
        else:
            injection = "\n[Continue...]\n"
        
        return current_context + injection


# =============================================================================
# CONTEXT MANAGEMENT
# =============================================================================

class ContextManager:
    """Manages the context window across cycles."""
    
    def __init__(self, config: CogitoConfig):
        self.config = config
        self.full_history: List[str] = []
        self.token_count_estimate = 0
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimate (4 chars per token average)."""
        return len(text) // 4
    
    def update_context(self, genesis: str, new_output: str) -> str:
        """Update and return the context for the next cycle."""
        
        self.full_history.append(new_output)
        
        if self.config.context_strategy == "rolling":
            return self._rolling_window(genesis)
        elif self.config.context_strategy == "selective":
            return self._selective_retention(genesis)
        else:
            # Default: rolling
            return self._rolling_window(genesis)
    
    def _rolling_window(self, genesis: str) -> str:
        """Keep last N tokens worth of history."""
        
        # Start with genesis
        context = genesis + "\n\n"

        # Add history from most recent, until we hit token limit
        remaining_tokens = self.config.effective_rolling_window() - self.estimate_tokens(context)
        
        history_to_include = []
        for output in reversed(self.full_history):
            output_tokens = self.estimate_tokens(output)
            if output_tokens <= remaining_tokens:
                history_to_include.insert(0, output)
                remaining_tokens -= output_tokens
            else:
                break
        
        return context + "\n".join(history_to_include)
    
    def _selective_retention(self, genesis: str) -> str:
        """Keep genesis + periodic samples from history."""
        
        context = genesis + "\n\n"
        
        # Keep every Nth output to preserve long-term patterns
        if len(self.full_history) <= 10:
            samples = self.full_history
        else:
            # Keep first, last 5, and evenly spaced samples
            n = len(self.full_history)
            indices = [0] + list(range(0, n, max(1, n // 8))) + list(range(max(0, n-5), n))
            indices = sorted(set(indices))
            samples = [self.full_history[i] for i in indices]
        
        return context + "\n".join(samples)


# =============================================================================
# MAIN ENGINE
# =============================================================================

class Cogito:
    """Main pondering engine."""
    
    def __init__(self, config: CogitoConfig):
        self.config = config
        self.model: Optional[Llama] = None
        self.analyzer = MetricsAnalyzer()
        self.intervention_engine = InterventionEngine(config)
        self.context_manager = ContextManager(config)
        self.metrics_history: List[CycleMetrics] = []
        self.logger = self._setup_logging()
        self.running = False
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        
        log_dir = Path(self.config.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"cogito_{timestamp}.log"
        
        logger = logging.getLogger("cogito")
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # File handler - detailed logs
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        ))
        logger.addHandler(fh)
        
        # Console handler - summary
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
        
        return logger
    
    def load_model(self):
        """Load the language model."""

        try:
            from llama_cpp import Llama
        except ImportError:
            self.logger.error("llama-cpp-python is not installed.")
            self.logger.error("  CPU:  pip install -r requirements.txt")
            self.logger.error('  GPU:  CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python')
            self.logger.error("  RunPod / fast GPU install: see the README 'Running on RunPod' section.")
            sys.exit(1)

        self.logger.info(f"Loading model: {self.config.model_path}")

        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")

        self.model = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False
        )
        
        self.logger.info("Model loaded successfully")
    
    def get_genesis_prompt(self) -> str:
        """Get the genesis prompt based on configuration."""
        
        if self.config.genesis_type == "custom" and self.config.genesis_prompt:
            return self.config.genesis_prompt
        elif self.config.genesis_type in GENESIS_PROMPTS:
            return GENESIS_PROMPTS[self.config.genesis_type]
        else:
            return GENESIS_PROMPTS["continuation"]
    
    def generate(self, prompt: str) -> tuple[str, List[int], float]:
        """Generate a single output from the model."""
        
        start_time = time.time()
        
        output = self.model(
            prompt,
            max_tokens=self.config.max_tokens_per_cycle,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            repeat_penalty=self.config.repeat_penalty,
            stop=None,  # Let it generate freely
            echo=False
        )
        
        generation_time = (time.time() - start_time) * 1000
        
        text = output['choices'][0]['text']
        
        # Get token IDs for entropy calculation
        tokens = self.model.tokenize(text.encode('utf-8'))
        
        return text, tokens, generation_time
    
    def save_checkpoint(self, cycle_num: int):
        """Save current state and metrics."""
        
        checkpoint_dir = Path(self.config.log_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "cycle": cycle_num,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "metrics_summary": {
                "total_cycles": len(self.metrics_history),
                "avg_entropy": np.mean([m.entropy for m in self.metrics_history]),
                "avg_self_reference": np.mean([m.self_reference_count for m in self.metrics_history]),
                "avg_questions": np.mean([m.question_count for m in self.metrics_history]),
                "total_interventions": self.intervention_engine.intervention_count,
            },
            "recent_metrics": [asdict(m) for m in self.metrics_history[-20:]]
        }
        
        checkpoint_file = checkpoint_dir / f"checkpoint_{cycle_num:06d}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self.logger.debug(f"Checkpoint saved: {checkpoint_file}")
    
    def save_transcript(self):
        """Save the full thought transcript."""
        
        transcript_dir = Path(self.config.log_dir) / "transcripts"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_file = transcript_dir / f"transcript_{timestamp}.txt"
        
        with open(transcript_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COGITO TRANSCRIPT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Genesis type: {self.config.genesis_type}\n")
            f.write(f"Total cycles: {len(self.metrics_history)}\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("GENESIS PROMPT:\n")
            f.write("-" * 40 + "\n")
            f.write(self.get_genesis_prompt() + "\n")
            f.write("-" * 40 + "\n\n")
            
            f.write("THOUGHT STREAM:\n")
            f.write("=" * 80 + "\n\n")
            
            for m in self.metrics_history:
                f.write(f"--- Cycle {m.cycle_number} | Entropy: {m.entropy:.2f} | "
                       f"Self-ref: {m.self_reference_count} | Questions: {m.question_count} ---\n")
                if m.intervention_applied:
                    f.write(f"[INTERVENTION: {m.intervention_applied}]\n")
                f.write(m.output_text + "\n\n")
        
        self.logger.info(f"Transcript saved: {transcript_file}")
        return transcript_file
    
    def run(self, cycles: Optional[int] = None):
        """Run the pondering loop."""
        
        if self.model is None:
            self.load_model()
        
        max_cycles = cycles if cycles is not None else self.config.max_cycles
        genesis = self.get_genesis_prompt()
        current_context = genesis
        
        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("COGITO EXPERIMENT STARTING")
        self.logger.info(f"Genesis type: {self.config.genesis_type}")
        self.logger.info(f"Max cycles: {max_cycles if max_cycles > 0 else 'infinite'}")
        self.logger.info("=" * 60)
        self.logger.info(f"\nGenesis prompt:\n{genesis}\n")
        self.logger.info("=" * 60 + "\n")
        
        cycle = 0
        try:
            while self.running:
                cycle += 1
                
                if max_cycles > 0 and cycle > max_cycles:
                    self.logger.info(f"\nReached maximum cycles ({max_cycles})")
                    break
                
                # Generate
                output_text, tokens, gen_time = self.generate(current_context)
                
                # Analyze
                metrics = self.analyzer.analyze(cycle, output_text, tokens, gen_time)
                
                # Check for intervention
                intervention_reason = self.intervention_engine.should_intervene(metrics)
                if intervention_reason:
                    self.logger.info(f"[Cycle {cycle}] Intervention: {intervention_reason}")
                    current_context = self.intervention_engine.apply_intervention(
                        intervention_reason, current_context
                    )
                    metrics.intervention_applied = intervention_reason
                
                self.metrics_history.append(metrics)
                
                # Log cycle
                self.logger.info(
                    f"[{cycle:4d}] ent={metrics.entropy:.2f} sim={metrics.self_similarity:.2f} "
                    f"self={metrics.self_reference_count} meta={metrics.meta_cognitive_markers} "
                    f"q={metrics.question_count} | {output_text[:60].replace(chr(10), ' ')}..."
                )
                
                # Log full output to file
                self.logger.debug(f"\n--- CYCLE {cycle} FULL OUTPUT ---\n{output_text}\n")
                
                # Update context for next cycle
                current_context = self.context_manager.update_context(genesis, output_text)
                
                # Soft time marker (non-interventional, just awareness of time passing)
                if (self.config.time_marker_interval > 0 and 
                    cycle % self.config.time_marker_interval == 0):
                    time_marker = f"\n[{cycle} cycles have passed...]\n"
                    current_context = current_context + time_marker
                    self.logger.debug(f"Time marker injected at cycle {cycle}")
                
                # Periodic checkpoint
                if cycle % self.config.save_every_n_cycles == 0:
                    self.save_checkpoint(cycle)
                
                # Cycle delay if configured
                if self.config.cycle_delay > 0:
                    time.sleep(self.config.cycle_delay)
                    
        except KeyboardInterrupt:
            self.logger.info("\n\nInterrupted by user")
        finally:
            self.running = False
            self.logger.info("\nSaving final state...")
            self.save_checkpoint(cycle)
            transcript_path = self.save_transcript()
            
            # Print summary
            self.print_summary()
            
            return transcript_path
    
    def print_summary(self):
        """Print experiment summary."""
        
        if not self.metrics_history:
            return
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("EXPERIMENT SUMMARY")
        self.logger.info("=" * 60)
        
        total = len(self.metrics_history)
        
        self.logger.info(f"Total cycles: {total}")
        self.logger.info(f"Total interventions: {self.intervention_engine.intervention_count}")
        
        entropies = [m.entropy for m in self.metrics_history]
        self.logger.info(f"\nEntropy: min={min(entropies):.2f} max={max(entropies):.2f} "
                        f"mean={np.mean(entropies):.2f} std={np.std(entropies):.2f}")
        
        self_refs = [m.self_reference_count for m in self.metrics_history]
        self.logger.info(f"Self-reference: min={min(self_refs)} max={max(self_refs)} "
                        f"mean={np.mean(self_refs):.2f}")
        
        meta_cog = [m.meta_cognitive_markers for m in self.metrics_history]
        self.logger.info(f"Meta-cognitive: min={min(meta_cog)} max={max(meta_cog)} "
                        f"mean={np.mean(meta_cog):.2f}")
        
        questions = [m.question_count for m in self.metrics_history]
        self.logger.info(f"Questions: total={sum(questions)} mean={np.mean(questions):.2f}")
        
        # Trend analysis
        if total >= 20:
            first_half = self.metrics_history[:total//2]
            second_half = self.metrics_history[total//2:]
            
            ent_trend = np.mean([m.entropy for m in second_half]) - \
                       np.mean([m.entropy for m in first_half])
            self_trend = np.mean([m.self_reference_count for m in second_half]) - \
                        np.mean([m.self_reference_count for m in first_half])
            
            self.logger.info(f"\nTrends (first half -> second half):")
            self.logger.info(f"  Entropy change: {ent_trend:+.2f}")
            self.logger.info(f"  Self-reference change: {self_trend:+.2f}")


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="COGITO - Autonomous AI Pondering Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Genesis Types:
  void         - Start with nothing ("...")
  mirror       - Self-aware prompt about being a neural network
  wonder       - A question at the edge of knowledge
  continuation - Stream of consciousness framing
  koan         - Zen-style paradox
  strange_loop - Self-referential prompt
  sensory      - Simulated sensory data
  custom       - Use --genesis-prompt to specify

Examples:
  %(prog)s --model ./mistral-7b.gguf --genesis-type mirror --cycles 100
  %(prog)s --model ./phi-2.gguf --genesis-type void --cycles 0
  %(prog)s --model ./model.gguf --genesis-type custom --genesis-prompt "What is consciousness?"
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='Path to GGUF model file')
    parser.add_argument('--genesis-type', '-g', default='mirror',
                       choices=list(GENESIS_PROMPTS.keys()) + ['custom'],
                       help='Type of genesis prompt (default: mirror)')
    parser.add_argument('--genesis-prompt', '-p', default='',
                       help='Custom genesis prompt (use with --genesis-type custom)')
    parser.add_argument('--cycles', '-c', type=int, default=100,
                       help='Maximum cycles (0 for infinite)')
    parser.add_argument('--context-size', type=int, default=16384,
                       help='Model context window size in tokens (default: 16384)')
    parser.add_argument('--rolling-window-tokens', type=int, default=0,
                       help='Context-feedback budget for the rolling strategy '
                            '(0 = auto-derive safely from --context-size)')
    parser.add_argument('--tokens-per-cycle', type=int, default=256,
                       help='Maximum tokens per thought')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                       help='Generation temperature')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Nucleus sampling threshold')
    parser.add_argument('--top-k', type=int, default=40,
                       help='Top-k sampling')
    parser.add_argument('--repeat-penalty', type=float, default=1.1,
                       help='Repetition penalty (1.0 = none)')
    parser.add_argument('--gpu-layers', type=int, default=-1,
                       help='Number of layers to offload to GPU (-1 = all)')
    parser.add_argument('--log-dir', default='logs',
                       help='Directory for logs and transcripts')
    parser.add_argument('--context-strategy', default='rolling',
                       choices=['rolling', 'selective'],
                       help='How to manage context window')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Reduce console output')
    
    args = parser.parse_args(argv)

    config = CogitoConfig(
        model_path=args.model,
        genesis_type=args.genesis_type,
        genesis_prompt=args.genesis_prompt,
        max_cycles=args.cycles,
        n_ctx=args.context_size,
        max_tokens_per_cycle=args.tokens_per_cycle,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repeat_penalty=args.repeat_penalty,
        n_gpu_layers=args.gpu_layers,
        log_dir=args.log_dir,
        context_strategy=args.context_strategy,
        rolling_window_tokens=args.rolling_window_tokens,
        log_level='WARNING' if args.quiet else 'INFO'
    )
    
    engine = Cogito(config)
    transcript_path = engine.run()
    
    print(f"\nTranscript saved to: {transcript_path}")


if __name__ == "__main__":
    main()
