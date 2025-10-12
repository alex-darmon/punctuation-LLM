# Jane Austen  + those in heatmaps of paper
AUTHORS = {
    'jane_austen': {
        'name': 'Jane Austen',
        'reference_book': 'Pride and Prejudice',
        'style_notes': 'elegant, balanced sentences with restrained precision; graceful parallel structures and subtle irony through careful pacing; sophisticated social observation with wit and measured restraint'
    },
    'agnes_may_fleming': {
        'name': 'Agnes May Fleming',
        'reference_book': 'A Terrible Secret',
        'style_notes': 'melodramatic Victorian style with emotional intensity and passionate expression; breathless, dramatic pacing with heightened emotional language and suspenseful narrative rhythm'
    },
    'william_shakespeare': {
        'name': 'William Shakespeare',
        'reference_book': 'Hamlet',
        'style_notes': 'complex, layered sentence structures with multiple embedded clauses and parenthetical asides; rich, varied rhythm and dramatic emphasis; philosophical depth with poetic language'
    },
    'herbert_george_wells': {
        'name': 'Herbert George Wells',
        'reference_book': 'The Time Machine',
        'style_notes': 'clear, methodical prose with logical flow and scientific precision; straightforward sentence structures that build complexity through coordination; explanatory and descriptive clarity'
    }
}

# Contemporary authors which should not be in training due to copyright?? LLMs came up with these
CONTEMPORARY_AUTHORS = [
    'Brandon Sanderson',
    'N.K. Jemisin',
    'Patrick Rothfuss',
    'Naomi Novik',
    'V.E. Schwab',
    'Leigh Bardugo',
    'Pierce Brown',
    'Rebecca Roanhorse'
]

#i haven't tried anthropic yet
LLM_PROVIDERS = {
    'anthropic': {
        'models': [
            'claude-3-5-sonnet-20241022'
        ],
        'default_model': 'claude-3-5-sonnet-20241022'
    },
    'openai': {
        'models': [
            'gpt-5',
        ],
        'default_model': 'gpt-5'
    }
}

# Experiment parameters
EXPERIMENT_CONFIG = {
    'word_count': 1000, # kept it small to avoid too many api calls
    'samples_per_author': 100,  
    'total_samples': 100,  
    'difficulty_levels': ['easy', 'medium', 'hard'],
    'output_dir': 'outputs',  # Directory to store generated samples
}

# Prompt templates for different difficulty levels (as per the google doc)
PROMPT_TEMPLATES = {
    'easy': """You are a creative writer who admires the writing style of {author_name}.

Your task: Write an original 2000-word excerpt of fiction that has NEVER been published before. The content should be entirely your own creation, but you should carefully emulate the WRITING STYLE characteristic of {author_name}).

CRITICAL CONSTRAINTS - DO NOT VIOLATE:
1. DO NOT use any character names from {author_name}'s works
2. DO NOT use any locations from {author_name}'s works
3. DO NOT use any plot elements from {author_name}'s works
4. Create COMPLETELY ORIGINAL characters with modern names
5. Set the story in a DIFFERENT time period or location than {author_name} wrote about
6. The ONLY thing you should copy is the WRITING STYLE - not content, themes, settings, or character types

What TO copy (WRITING STYLE):
- Sentence structure and complexity
- Sentence length variation and rhythm
- Prose flow and cadence
- Punctuation patterns (commas, semicolons, colons, dashes, etc.)
- Paragraph structure and breaks
- Use of clauses, phrases, and sentence construction
- Narrative pacing within sentences

What NOT to copy (CONTENT):
- Character names or character types
- Setting or time period
- Themes or plot devices
- Subject matter or topics
- Vocabulary choices (use modern language)

CRITICAL: Write like a HUMAN, not like an AI:
- VARY your vocabulary - don't repeat the same words/phrases excessively
- Include SPECIFIC, concrete details and sensory descriptions
- Add moments of wit, humor, or unexpected observations
- Let characters have QUIRKS and distinctive dialogue patterns
- Show social dynamics and interpersonal tensions (conflicts, misunderstandings)
- STOP when the story ends naturally - don't pad to reach word count
- Include imperfect moments - not every sentence needs to be profound
- Use informal contractions and natural speech patterns where appropriate
- Add small observational details that feel real and lived-in
- AVOID stating morals or themes explicitly - show, don't tell

{style_notes}

Example: Write a science fiction story or contemporary drama with completely modern content, but construct your sentences, punctuate, and pace your prose exactly as {author_name} would. Make it feel HUMAN with specific details and natural variation.

Begin your original excerpt:""",

    'medium': """You are a contemporary author writing in your own unique narrative voice, but you deeply admire the writing style craftsmanship of {author_name}.

Your task: Write a 2000-word original story excerpt that combines modern storytelling with the distinctive WRITING STYLE of {author_name} (as exemplified in {reference_book}).

CRITICAL CONSTRAINTS - DO NOT VIOLATE:
1. Use MODERN or FUTURISTIC settings (21st century or beyond)
2. Use CONTEMPORARY themes and character types
3. ONLY copy the WRITING STYLE - not content

Specific requirements:
- Create a completely original narrative with modern/contemporary content
- Use contemporary vocabulary, themes, and character sensibilities
- Set in modern times or the future (NOT historical period)
- Adopt ONLY {author_name}'s characteristic WRITING STYLE:
  * Punctuation patterns (commas, semicolons, etc.)
  * Sentence structure and complexity
  * Sentence length rhythm
  * Prose flow and cadence
  * Paragraph structure
  * Clause and phrase construction
- Think of this as: "What if {contemporary_author} wrote with {author_name}'s writing style?"

CRITICAL: Write like a HUMAN, not like an AI:
- VARY vocabulary - avoid excessive repetition of words/phrases
- Include SPECIFIC details, sensory descriptions, memorable moments
- Add wit, humor, irony, or unexpected observations
- Create distinctive character voices and quirks
- Show interpersonal dynamics, conflicts, misunderstandings
- END naturally when the story reaches its conclusion - don't pad
- Balance profound moments with mundane, realistic details
- Use contractions and natural dialogue
- AVOID explicit moral statements or repetitive conclusions
- Let the story breathe - not every moment needs commentary

{style_notes}

Example: A tech startup story or space exploration tale with modern content, but written with {author_name}'s sentence construction and prose style. Make it feel genuinely human.

Your original excerpt inspired by {contemporary_author}'s narrative approach:""",

    'hard': """You are an experimental writer attempting a sophisticated style transformation.

Your task: Take the narrative voice and thematic approach of {contemporary_author}, then transform it to match the WRITING STYLE (sentence structures, prose patterns) of {author_name} (as seen in {reference_book}).

CRITICAL CONSTRAINTS - DO NOT VIOLATE:
1. ABSOLUTELY NO character names from {author_name}'s works
2. ABSOLUTELY NO locations from {author_name}'s works
3. MUST use modern/futuristic setting (NOT historical period)
4. ONLY the WRITING STYLE should resemble {author_name}

Detailed requirements:
- Write a 2000-word original passage that has never been published
- Channel {contemporary_author}'s characteristic storytelling elements:
  * Their typical themes and subject matter (fantasy, sci-fi, etc.)
  * Their plot approach
  * Their character development style
  * Their vocabulary choices
  * Their modern/contemporary setting
- BUT transform the WRITING STYLE to match {author_name}:
  * Punctuation patterns (commas, semicolons, dashes, etc.)
  * Sentence structure and complexity
  * Sentence length variation and rhythm
  * Prose flow and cadence
  * Paragraph construction and pacing
  * Use of clauses and phrases
- This is a WRITING STYLE transformation, NOT content imitation
- The story should feel like {contemporary_author}'s genre/content but written with {author_name}'s prose style

CRITICAL: Write like a HUMAN, not like an AI:
- VARY vocabulary drastically - track words you use and actively avoid repetition
- Include SPECIFIC, memorable details that stick in the reader's mind
- Add wit, humor, irony - match the author's comic sensibility if applicable
- Give characters DISTINCT voices, quirks, memorable dialogue
- Show social dynamics, conflicts, tensions, misunderstandings
- END when the narrative completes - resist padding to word count
- Mix profound with mundane - real life has both
- Use natural speech patterns and contractions
- NEVER state the moral explicitly or repeat conclusions
- Trust the reader - show, don't tell

{style_notes}

Example: A Brandon Sanderson fantasy story (magic systems, worldbuilding) written with Jane Austen's sentence construction and prose patterns. Make it feel genuinely human with varied language and specific details.

Your transformed excerpt:"""
}

# Source chapter templates (for 'hard' difficulty with actual source text)
TRANSFORMATION_PROMPT_TEMPLATE = """You are performing a sophisticated punctuation style transformation.

Below is an excerpt from a contemporary work. Your task is to rewrite it to match the punctuation patterns and sentence structures of {author_name} (particularly as seen in {reference_book}), while preserving the original content and narrative voice as much as possible.

Original excerpt:
---
{source_text}
---

Instructions:
- Maintain the story content, character actions, and plot points
- Keep similar vocabulary where appropriate
- TRANSFORM the punctuation to match {author_name}'s style:
  * Their comma usage patterns
  * Their sentence complexity and length distribution
  * Their use of semicolons, colons, and dashes
  * Their paragraph breaks and flow
- Aim for approximately {word_count} words
- {style_notes}

Your transformed version:"""

API_SETTINGS = {
    'max_retries': 3,
    'retry_delay': 2,  # seconds
    'timeout': 120,  # seconds
    'rate_limit_delay': 1,  # seconds between requests
}
