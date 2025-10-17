#!/usr/bin/env python3
"""
Multi-Agent Author Style Transfer System
Analyzes an author's writing style and rewrites modern text to match it.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import anthropic


@dataclass
class StyleProfile:
    """Represents an author's writing style profile."""
    author_name: str
    vocabulary_patterns: Dict[str, any]
    sentence_structure: Dict[str, any]
    tone_characteristics: List[str]
    common_phrases: List[str]
    literary_devices: List[str]
    punctuation_style: Dict[str, any]
    paragraph_structure: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_prompt(self) -> str:
        """Convert style profile to a prompt format."""
        return f"""
Author: {self.author_name}

Vocabulary Patterns:
{json.dumps(self.vocabulary_patterns, indent=2)}

Sentence Structure:
{json.dumps(self.sentence_structure, indent=2)}

Tone Characteristics:
{', '.join(self.tone_characteristics)}

Common Phrases:
{', '.join(self.common_phrases)}

Literary Devices:
{', '.join(self.literary_devices)}

Punctuation Style:
{json.dumps(self.punctuation_style, indent=2)}

Paragraph Structure:
{self.paragraph_structure}
"""


class StyleAnalyzerAgent:
    """Agent responsible for analyzing author's writing samples."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.model = "claude-sonnet-4-5-20250929"
    
    def analyze(self, author_name: str, sample_text: str) -> Dict:
        """Analyze writing sample to extract style characteristics."""
        
        prompt = f"""You are a literary style analysis expert. Analyze the following text by {author_name} and extract detailed writing style characteristics.

TEXT TO ANALYZE:
{sample_text}

Provide a comprehensive analysis in JSON format with these categories:

1. vocabulary_patterns: {{
   "word_complexity": "simple/moderate/complex/varied",
   "archaic_words": ["list any archaic or period-specific words"],
   "favorite_adjectives": ["most common descriptive words"],
   "favorite_verbs": ["most common action words"],
   "lexical_diversity": "low/medium/high"
}}

2. sentence_structure: {{
   "average_length": "short/medium/long/varied",
   "complexity": "simple/compound/complex/compound-complex",
   "common_patterns": ["list sentence construction patterns"],
   "use_of_fragments": "rare/occasional/frequent",
   "clause_preferences": "independent/dependent/balanced"
}}

3. tone_characteristics: ["list 5-7 key tone descriptors like formal, whimsical, sardonic, etc."]

4. common_phrases: ["list 5-10 characteristic phrases or expressions this author uses"]

5. literary_devices: ["list devices used: metaphor, simile, alliteration, repetition, etc."]

6. punctuation_style: {{
   "comma_usage": "sparse/moderate/heavy",
   "dash_preference": "em-dash/en-dash/hyphen/none",
   "semicolon_usage": "rare/occasional/frequent",
   "exclamation_usage": "rare/occasional/frequent",
   "ellipsis_usage": "rare/occasional/frequent"
}}

7. paragraph_structure: "brief description of how paragraphs are typically structured"

Respond with ONLY valid JSON, no additional text."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        # Parse JSON response
        try:
            analysis = json.loads(response_text)
            return analysis
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                analysis = json.loads(response_text[start:end])
                return analysis
            raise ValueError("Failed to parse style analysis")


class StyleProfilerAgent:
    """Agent that creates a comprehensive style profile from analysis."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
    
    def create_profile(self, author_name: str, analysis: Dict) -> StyleProfile:
        """Convert analysis into a structured style profile."""
        
        return StyleProfile(
            author_name=author_name,
            vocabulary_patterns=analysis.get("vocabulary_patterns", {}),
            sentence_structure=analysis.get("sentence_structure", {}),
            tone_characteristics=analysis.get("tone_characteristics", []),
            common_phrases=analysis.get("common_phrases", []),
            literary_devices=analysis.get("literary_devices", []),
            punctuation_style=analysis.get("punctuation_style", {}),
            paragraph_structure=analysis.get("paragraph_structure", "")
        )


class RewriterAgent:
    """Agent that rewrites text in the author's style."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.model = "claude-sonnet-4-5-20250929"
    
    def rewrite(
        self, 
        modern_text: str, 
        style_profile: StyleProfile, 
        previous_attempt: Optional[str] = None,
        feedback: Optional[Dict] = None
    ) -> str:
        """Rewrite modern text using the author's style."""
        
        base_prompt = f"""You are a master literary style imitator. Your task is to rewrite the following modern text to match the writing style of {style_profile.author_name}.

STYLE PROFILE:
{style_profile.to_prompt()}

MODERN TEXT TO REWRITE:
{modern_text}"""

        if previous_attempt and feedback:
            # Add feedback for revision
            improvements = feedback.get('improvements', [])
            strengths = feedback.get('strengths', [])
            
            prompt = base_prompt + f"""

PREVIOUS ATTEMPT:
{previous_attempt}

QUALITY ASSESSMENT OF PREVIOUS ATTEMPT:
Overall Score: {feedback.get('overall_score', 0)}/100

What worked well:
{chr(10).join(f'- {s}' for s in strengths)}

What needs improvement:
{chr(10).join(f'- {i}' for i in improvements)}

INSTRUCTIONS FOR REVISION:
1. Build on the strengths identified above
2. Address each improvement area specifically
3. Maintain the core meaning and information from the modern text
4. Better match {style_profile.author_name}'s characteristics
5. Pay special attention to the feedback about vocabulary, structure, tone, and authenticity

Provide ONLY the revised rewritten text, no explanations or commentary."""
        else:
            # First attempt
            prompt = base_prompt + f"""

INSTRUCTIONS:
1. Maintain the core meaning and information from the modern text
2. Transform the writing style to match {style_profile.author_name}'s characteristics
3. Apply the vocabulary patterns, sentence structures, tone, and literary devices from the style profile
4. Use similar punctuation and paragraph structuring
5. Incorporate characteristic phrases where appropriate
6. Ensure the result feels authentic to {style_profile.author_name}'s voice

Provide ONLY the rewritten text, no explanations or commentary."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return message.content[0].text.strip()


class QualityCheckerAgent:
    """Agent that validates the rewrite quality."""
    
    def __init__(self, client: anthropic.Anthropic):
        self.client = client
        self.model = "claude-sonnet-4-5-20250929"
    
    def check_quality(
        self, 
        original_text: str, 
        rewritten_text: str, 
        style_profile: StyleProfile
    ) -> Dict[str, any]:
        """Evaluate how well the rewrite matches the style profile."""
        
        prompt = f"""You are a literary quality assessment expert. Evaluate how well the rewritten text matches {style_profile.author_name}'s writing style.

STYLE PROFILE:
{style_profile.to_prompt()}

ORIGINAL MODERN TEXT:
{original_text}

REWRITTEN TEXT:
{rewritten_text}

Provide an assessment in JSON format:
{{
    "overall_score": 0-100,
    "vocabulary_match": 0-100,
    "structure_match": 0-100,
    "tone_match": 0-100,
    "authenticity": 0-100,
    "strengths": ["list 3-5 things done well"],
    "improvements": ["list 3-5 areas that could be improved"],
    "recommendation": "accept/revise"
}}

Respond with ONLY valid JSON."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        
        try:
            assessment = json.loads(response_text)
            return assessment
        except json.JSONDecodeError:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                assessment = json.loads(response_text[start:end])
                return assessment
            raise ValueError("Failed to parse quality assessment")


class AuthorStyleTransferSystem:
    """Main orchestrator for the multi-agent system."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the system with Claude API."""
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY must be set")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize agents
        self.analyzer = StyleAnalyzerAgent(self.client)
        self.profiler = StyleProfilerAgent(self.client)
        self.rewriter = RewriterAgent(self.client)
        self.checker = QualityCheckerAgent(self.client)
        
        self.style_profile: Optional[StyleProfile] = None
    
    def load_sample_file(self, file_path: str) -> str:
        """Load the author's sample text from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Sample file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def analyze_author_style(self, author_name: str, sample_file_path: str) -> StyleProfile:
        """Analyze author's style from sample file."""
        print(f"\n📚 Loading sample text for {author_name}...")
        sample_text = self.load_sample_file(sample_file_path)
        
        print(f"✨ Analyzing writing style...")
        analysis = self.analyzer.analyze(author_name, sample_text)
        
        print(f"📝 Creating style profile...")
        self.style_profile = self.profiler.create_profile(author_name, analysis)
        
        print(f"✅ Style profile created for {author_name}!")
        return self.style_profile
    
    def transform_text(
        self, 
        modern_text: str, 
        max_iterations: int = 3
    ) -> tuple[str, Dict]:
        """Transform modern text to author's style with quality checking."""
        
        if not self.style_profile:
            raise ValueError("No style profile loaded. Run analyze_author_style first.")
        
        print(f"\n🔄 Rewriting text in {self.style_profile.author_name}'s style...")
        
        best_rewrite = None
        best_score = 0
        best_assessment = None
        previous_attempt = None
        previous_feedback = None
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n  Iteration {iteration}/{max_iterations}...")
            
            # Rewrite the text (with feedback from previous iteration if available)
            rewritten = self.rewriter.rewrite(
                modern_text, 
                self.style_profile,
                previous_attempt=previous_attempt,
                feedback=previous_feedback
            )
            
            # Check quality
            assessment = self.checker.check_quality(
                modern_text, 
                rewritten, 
                self.style_profile
            )
            
            current_score = assessment.get("overall_score", 0)
            print(f"  Quality score: {current_score}/100")
            
            if current_score > best_score:
                best_score = current_score
                best_rewrite = rewritten
                best_assessment = assessment
            
            # Store this attempt and feedback for next iteration
            previous_attempt = rewritten
            previous_feedback = assessment
            
            # If we got a high score or recommendation to accept, stop
            if current_score >= 85 or assessment.get("recommendation") == "accept":
                print(f"  ✅ Achieved satisfactory quality!")
                break
        
        print(f"\n🎉 Final quality score: {best_score}/100")
        return best_rewrite, best_assessment
    
    def save_results(
        self, 
        output_dir: str, 
        rewritten_text: str, 
        assessment: Dict
    ):
        """Save the rewritten text and assessment to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create author-specific filename prefix
        author_slug = self.style_profile.author_name.lower().replace(' ', '_').replace('.', '')
        
        # Save rewritten text
        text_file = output_path / f"{author_slug}_rewritten_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(rewritten_text)
        
        # Save assessment
        assessment_file = output_path / f"{author_slug}_quality_assessment.json"
        with open(assessment_file, 'w', encoding='utf-8') as f:
            json.dump(assessment, f, indent=2)
        
        # Save style profile
        if self.style_profile:
            profile_file = output_path / f"{author_slug}_style_profile.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(self.style_profile.to_dict(), f, indent=2)
        
        print(f"\n💾 Results saved to {output_dir}/")
        print(f"  - {author_slug}_rewritten_text.txt")
        print(f"  - {author_slug}_quality_assessment.json")
        print(f"  - {author_slug}_style_profile.json")


def main():
    """Example usage of the system."""
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python author_style_transfer.py <author_name> <sample_file> <modern_text_file> [output_dir]")
        print("\nExample:")
        print("  python author_style_transfer.py 'Ernest Hemingway' samples/hemingway.txt modern_text.txt output/")
        sys.exit(1)
    
    author_name = sys.argv[1]
    sample_file = sys.argv[2]
    modern_text_file = sys.argv[3]
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "output"
    
    # Initialize system
    print("🚀 Initializing Author Style Transfer System...")
    system = AuthorStyleTransferSystem()
    
    # Analyze author's style
    system.analyze_author_style(author_name, sample_file)
    
    # Load modern text
    print(f"\n📖 Loading modern text from {modern_text_file}...")
    with open(modern_text_file, 'r', encoding='utf-8') as f:
        modern_text = f.read()
    
    # Transform text
    rewritten_text, assessment = system.transform_text(modern_text)
    
    # Display results
    print("\n" + "="*60)
    print("REWRITTEN TEXT")
    print("="*60)
    print(rewritten_text)
    print("\n" + "="*60)
    print("QUALITY ASSESSMENT")
    print("="*60)
    print(json.dumps(assessment, indent=2))
    
    # Save results
    system.save_results(output_dir, rewritten_text, assessment)


if __name__ == "__main__":
    main()