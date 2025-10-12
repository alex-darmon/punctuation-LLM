"""
Main experiment runner for generating text samples across different LLMs and author styles.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv

from config import (
    AUTHORS,
    LLM_PROVIDERS,
    EXPERIMENT_CONFIG,
    API_SETTINGS
)
from llm_clients import get_client
from prompt_generator import PromptGenerator


class ExperimentRunner:
    """Orchestrates the generation of text samples across multiple configurations."""

    def __init__(self, output_base_dir: str = 'outputs'):
        """
        Initialize the experiment runner.

        Args:
            output_base_dir: Base directory for storing outputs
        """
        load_dotenv()  

        self.output_base_dir = Path(output_base_dir)
        self.prompt_generator = PromptGenerator()

        self._setup_directories()

        self.results = []
        self.errors = []

    def _setup_directories(self):
        """Create necessary output directories."""
        self.output_base_dir.mkdir(exist_ok=True)
        (self.output_base_dir / 'samples').mkdir(exist_ok=True)
        (self.output_base_dir / 'metadata').mkdir(exist_ok=True)
        (self.output_base_dir / 'logs').mkdir(exist_ok=True)

    def run_experiment(
        self,
        providers: Optional[List[str]] = None,
        authors: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        samples_per_config: int = 10,
        models: Optional[Dict[str, str]] = None,
        include_author_sample: bool = False
    ) -> pd.DataFrame:
        """
        Run the full experiment across specified configurations.

        Args:
            providers: List of LLM providers ('anthropic', 'openai')
            authors: List of author keys to target
            difficulties: List of difficulty levels
            samples_per_config: Number of samples per configuration
            models: Optional dict mapping provider to specific model

        Returns:
            DataFrame with experiment results
        """
        if providers is None:
            providers = ['anthropic', 'openai']

        if authors is None:
            authors = list(AUTHORS.keys())

        if difficulties is None:
            difficulties = EXPERIMENT_CONFIG['difficulty_levels']

        if models is None:
            models = {
                'anthropic': LLM_PROVIDERS['anthropic']['default_model'],
                'openai': LLM_PROVIDERS['openai']['default_model']
            }

        print(f"Starting experiment at {datetime.now()}")
        print(f"Providers: {providers}")
        print(f"Authors: {authors}")
        print(f"Difficulties: {difficulties}")
        print(f"Samples per config: {samples_per_config}")
        print("-" * 60)

        # Generate all prompt configurations
        prompt_configs = self.prompt_generator.generate_batch_prompts(
            target_author_keys=authors,
            difficulties=difficulties,
            samples_per_config=samples_per_config,
            include_author_sample=include_author_sample
        )

        total_samples = len(prompt_configs) * len(providers)
        current_sample = 0

        for provider in providers:
            model = models.get(provider)
            print(f"\n{'='*60}")
            print(f"Running experiments with {provider.upper()} ({model})")
            print(f"{'='*60}\n")

            try:
                client = get_client(provider, model)
            except ValueError as e:
                print(f"Error initializing {provider} client: {e}")
                self.errors.append({
                    'provider': provider,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                continue

            for config in prompt_configs:
                current_sample += 1
                print(f"[{current_sample}/{total_samples}] Generating: "
                      f"{config['target_author_name']} | "
                      f"{config['difficulty']} | "
                      f"Sample {config['sample_id']}")

                result = self._generate_sample(
                    client=client,
                    provider=provider,
                    model=model,
                    config=config
                )

                if result:
                    self.results.append(result)
                    self._save_sample(result)

                # Rate limiting
                time.sleep(API_SETTINGS['rate_limit_delay'])

        # Save final results
        results_df = self._save_results()

        print(f"\n{'='*60}")
        print(f"Experiment completed at {datetime.now()}")
        print(f"Total samples generated: {len(self.results)}")
        print(f"Total errors: {len(self.errors)}")
        print(f"{'='*60}\n")

        return results_df

    def _generate_sample(
        self,
        client,
        provider: str,
        model: str,
        config: Dict
    ) -> Optional[Dict]:
        """Generate a single text sample."""
        try:
            start_time = time.time()

            # Use generate_to_length to reach target word count
            target_words = EXPERIMENT_CONFIG['word_count']
            generated_text = client.generate_to_length(
                prompt=config['prompt'],
                target_word_count=target_words,
                temperature=0.7,
                max_tokens=2000,
                max_iterations=5
            )

            end_time = time.time()
            generation_time = end_time - start_time

            # Count words
            word_count = len(generated_text.split())

            result = {
                'provider': provider,
                'model': model,
                'target_author': config['target_author'],
                'target_author_name': config['target_author_name'],
                'difficulty': config['difficulty'],
                'contemporary_author': config.get('contemporary_author'),
                'sample_id': config['sample_id'],
                'config_id': config['config_id'],
                'generated_text': generated_text,
                'word_count': word_count,
                'generation_time': generation_time,
                'timestamp': datetime.now().isoformat(),
                'prompt': config['prompt']
            }

            return result

        except Exception as e:
            error_info = {
                'provider': provider,
                'model': model,
                'config_id': config['config_id'],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.errors.append(error_info)
            print(f"  ERROR: {e}")
            return None

    def _save_sample(self, result: Dict):
        """Save individual sample to file."""
        # Create filename
        filename = f"{result['provider']}_{result['config_id']}.txt"
        filepath = self.output_base_dir / 'samples' / filename

        # Save generated text
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(result['generated_text'])

        # Save metadata
        metadata_filename = f"{result['provider']}_{result['config_id']}_metadata.json"
        metadata_filepath = self.output_base_dir / 'metadata' / metadata_filename

        metadata = {k: v for k, v in result.items() if k != 'generated_text'}

        with open(metadata_filepath, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

    def _save_results(self) -> pd.DataFrame:
        """Save aggregated results to CSV."""
        if not self.results:
            print("No results to save.")
            return pd.DataFrame()

        # Create DataFrame
        results_df = pd.DataFrame(self.results)

        # Save to CSV (without full text)
        summary_df = results_df.drop(columns=['generated_text', 'prompt'])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.output_base_dir / f'results_summary_{timestamp}.csv'
        summary_df.to_csv(csv_path, index=False)

        print(f"Results saved to: {csv_path}")

        # Save errors if any
        if self.errors:
            errors_df = pd.DataFrame(self.errors)
            errors_path = self.output_base_dir / f'errors_{timestamp}.csv'
            errors_df.to_csv(errors_path, index=False)
            print(f"Errors saved to: {errors_path}")

        return results_df

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """Generate summary statistics from results."""
        if results_df.empty:
            return {}

        analysis = {
            'total_samples': len(results_df),
            'by_provider': results_df.groupby('provider').size().to_dict(),
            'by_author': results_df.groupby('target_author_name').size().to_dict(),
            'by_difficulty': results_df.groupby('difficulty').size().to_dict(),
            'avg_word_count': results_df['word_count'].mean(),
            'avg_generation_time': results_df['generation_time'].mean(),
            'word_count_by_author': results_df.groupby('target_author_name')['word_count'].mean().to_dict(),
            'generation_time_by_provider': results_df.groupby('provider')['generation_time'].mean().to_dict()
        }

        return analysis


def main():
    """Main entry point for running experiments."""
    # Initialize runner
    runner = ExperimentRunner(output_base_dir='outputs')

    # Run experiment with default settings
    # You can customize these parameters
    results_df = runner.run_experiment(
        providers=['anthropic', 'openai'],
        authors=['jane_austen', 'charles_dickens', 'hg_wells'],
        difficulties=['easy', 'medium', 'hard'],
        samples_per_config=10
    )

    # Analyze results
    if not results_df.empty:
        analysis = runner.analyze_results(results_df)
        print("\nExperiment Analysis:")
        print(json.dumps(analysis, indent=2))


if __name__ == '__main__':
    main()
