

import os
from dotenv import load_dotenv
from llm_clients import get_client
from prompt_generator import PromptGenerator

def test_single_run(provider='anthropic', author='jane_austen', difficulty='easy', include_sample=False):

    # Load environment variables
    load_dotenv()

    # Check API key
    if provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not found in .env file")
            return
        print(f"Anthropic API key found: {api_key[:8]}...")
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("ERROR: OPENAI_API_KEY not found in .env file")
            return
        print(f"OpenAI API key found: {api_key[:8]}...")

    print(f"\nTest Configuration:")
    print(f"  Provider: {provider}")
    print(f"  Author: {author}")
    print(f"  Difficulty: {difficulty}")
    print("-" * 60)

    # Generate prompt
    print("\nGenerating prompt...")
    generator = PromptGenerator()
    prompt = generator.generate_prompt(author, difficulty, include_author_sample=include_sample)

    if include_sample:
        print("Including author sample from file (if available)")

    print(f"\nPrompt preview (first 200 chars):")
    print(prompt[:200] + "...")
    print("-" * 60)

    # Get client
    print(f"\nInitializing {provider} client...")
    try:
        client = get_client(provider)
        print("Client initialized successfully!")
    except Exception as e:
        print(f"ERROR initializing client: {e}")
        return

    # Get target word count from config
    from config import EXPERIMENT_CONFIG
    target_words = EXPERIMENT_CONFIG['word_count']

    # Generate text
    print(f"\nGenerating text to reach {target_words} words (may take multiple iterations)...")
    try:
        generated_text = client.generate_to_length(
            prompt=prompt,
            target_word_count=target_words,
            temperature=0.7,
            max_tokens=2000,
            max_iterations=5
        )

        word_count = len(generated_text.split())

        print("\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Generated {word_count} words")
        print(f"\nFirst 500 characters of generated text:")
        print("-" * 60)
        print(generated_text[:500] + "...")
        print("-" * 60)

        # Save to file
        output_file = f"test_output_{provider}_{author}_{difficulty}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(generated_text)

        print(f"\nFull text saved to: {output_file}")

    except Exception as e:
        print(f"\nERROR generating text: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys

    # Parse command line arguments
    provider = sys.argv[1] if len(sys.argv) > 1 else 'anthropic'
    author = sys.argv[2] if len(sys.argv) > 2 else 'jane_austen'
    difficulty = sys.argv[3] if len(sys.argv) > 3 else 'easy'
    include_sample = sys.argv[4].lower() == 'true' if len(sys.argv) > 4 else False

    # Run test
    test_single_run(provider, author, difficulty, include_sample)
