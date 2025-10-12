"""
Prompt generation utilities for the punctuation style experiment.
"""

import random
from pathlib import Path
from typing import Dict, List, Optional
from config import (
    AUTHORS,
    CONTEMPORARY_AUTHORS,
    PROMPT_TEMPLATES,
    TRANSFORMATION_PROMPT_TEMPLATE,
    EXPERIMENT_CONFIG
)


class PromptGenerator:

    def __init__(self, authors_sample_dir: str = 'authors_sample'):
        self.authors = AUTHORS
        self.contemporary_authors = CONTEMPORARY_AUTHORS
        self.templates = PROMPT_TEMPLATES
        self.word_count = EXPERIMENT_CONFIG['word_count']
        self.authors_sample_dir = Path(authors_sample_dir)

    def load_author_sample(self, author_key: str) -> Optional[str]:

        sample_file = self.authors_sample_dir / f"{author_key}.txt"
        if sample_file.exists():
            with open(sample_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def generate_prompt(
        self,
        target_author_key: str,
        difficulty: str = 'easy',
        contemporary_author: Optional[str] = None,
        source_text: Optional[str] = None,
        include_author_sample: bool = False
    ) -> str:

        if target_author_key not in self.authors:
            raise ValueError(f"Unknown author key: {target_author_key}")

        if difficulty not in ['easy', 'medium', 'hard']:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        author_info = self.authors[target_author_key]

        if difficulty == 'hard' and source_text:
            return self._generate_transformation_prompt(
                author_info, source_text
            )

        template = self.templates[difficulty]

        if difficulty in ['medium', 'hard'] and not contemporary_author:
            contemporary_author = random.choice(self.contemporary_authors)

        sample_text = ""
        if include_author_sample:
            loaded_sample = self.load_author_sample(target_author_key)
            if loaded_sample:
                sample_text = f"\n\nHere is an example excerpt from {author_info['name']}'s writing for reference:\n\n---\n{loaded_sample}\n---\n\nNow, write your original story following the same WRITING STYLE (sentence structure, punctuation patterns, prose rhythm) but with completely different content:"
            else:
                print(f"Warning: No sample file found for {target_author_key}")

        prompt = template.format(
            author_name=author_info['name'],
            reference_book=author_info['reference_book'],
            style_notes=author_info['style_notes'],
            contemporary_author=contemporary_author if difficulty in ['medium', 'hard'] else ''
        )

        prompt = prompt + sample_text

        return prompt

    def _generate_transformation_prompt(
        self,
        author_info: Dict,
        source_text: str
    ) -> str:
        return TRANSFORMATION_PROMPT_TEMPLATE.format(
            author_name=author_info['name'],
            reference_book=author_info['reference_book'],
            style_notes=author_info['style_notes'],
            source_text=source_text,
            word_count=self.word_count
        )

    def generate_batch_prompts(
        self,
        target_author_keys: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None,
        samples_per_config: int = 10,
        include_author_sample: bool = False
    ) -> List[Dict]:

        if target_author_keys is None:
            target_author_keys = list(self.authors.keys())

        if difficulties is None:
            difficulties = EXPERIMENT_CONFIG['difficulty_levels']

        prompts = []

        for author_key in target_author_keys:
            for difficulty in difficulties:
                for i in range(samples_per_config):
                    # Select a contemporary author for medium/hard
                    contemporary = None
                    if difficulty in ['medium', 'hard']:
                        contemporary = random.choice(self.contemporary_authors)

                    prompt = self.generate_prompt(
                        target_author_key=author_key,
                        difficulty=difficulty,
                        contemporary_author=contemporary,
                        include_author_sample=include_author_sample
                    )

                    prompts.append({
                        'prompt': prompt,
                        'target_author': author_key,
                        'target_author_name': self.authors[author_key]['name'],
                        'difficulty': difficulty,
                        'contemporary_author': contemporary,
                        'sample_id': i,
                        'config_id': f"{author_key}_{difficulty}_{i}"
                    })

        return prompts

    def get_author_info(self, author_key: str) -> Dict:
        return self.authors.get(author_key)

    def list_authors(self) -> List[str]:
        return list(self.authors.keys())

    def list_contemporary_authors(self) -> List[str]:
        return self.contemporary_authors


# Convenience function for quick prompt generation
def create_prompt(author: str, difficulty: str = 'easy') -> str:

    generator = PromptGenerator()
    return generator.generate_prompt(author, difficulty)
