import os
import argparse
from typing import Iterator
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast

def get_training_corpus(sample_limit: int = 30000) -> Iterator[str]:
    """Streams a mixture of English text, Hindi text, and Code to train BPE."""
    print("Streaming text mixture for tokenizer training...")
    
    # 1. English text (WikiText-103)
    try:
        en_ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        en_iter = iter(en_ds)
    except Exception as e:
        print(f"Warning: Failed to load WikiText: {e}")
        en_iter = iter([])

    # 2. Hindi text (portion of cc100 or wikipedia hi)
    try:
        hi_ds = load_dataset("wikipedia", "20220301.hi", split="train", streaming=True)
        hi_iter = iter(hi_ds)
    except Exception as e:
        print(f"Warning: Failed to load Hindi Wikipedia: {e}")
        hi_iter = iter([])

    # 3. Code (python, javascript, etc. from code_search_net)
    try:
        code_ds = load_dataset("code_search_net", "python", split="train", streaming=True)
        code_iter = iter(code_ds)
    except Exception as e:
        print(f"Warning: Failed to load CodeSearchNet: {e}")
        code_iter = iter([])

    count = 0
    while count < sample_limit:
        # Alternating stream
        # English
        try:
            item = next(en_iter)
            text = item.get("text", "").strip()
            if len(text) > 50:
                yield text
                count += 1
        except StopIteration:
            pass

        # Hindi
        try:
            item = next(hi_iter)
            text = item.get("text", "").strip()
            if len(text) > 50:
                yield text
                count += 1
        except StopIteration:
            pass

        # Code
        try:
            item = next(code_iter)
            text = item.get("whole_funcstring", "").strip() or item.get("code", "").strip()
            if len(text) > 30:
                yield text
                count += 1
        except StopIteration:
            pass

def train_bpe_tokenizer(vocab_size: int = 65536, sample_limit: int = 30000, push_to_hub: bool = False, repo_id: str = None):
    print(f"Starting Byte-Level BPE Tokenizer training. Target vocab size: {vocab_size}")
    
    # 1. Initialize tokenizer with BPE model
    # ByteLevel pre-tokenizer handles whitespace and splits into bytes.
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # 2. Setup Trainer
    # Define our special tokens
    special_tokens = ["<unk>", "<s>", "</s>", "<pad>", "<|im_start|>", "<|im_end|>"]
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True
    )

    # 3. Train
    corpus = get_training_corpus(sample_limit=sample_limit)
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    
    # 4. Wrap with Hugging Face PreTrainedTokenizerFast
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_prefix_space=False,
    )
    
    # Ensure correct special token mappings
    # Set ChatML or custom templates
    fast_tokenizer.add_special_tokens({
        "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
    })

    # Save locally
    output_dir = "./tokenizer_output"
    os.makedirs(output_dir, exist_ok=True)
    fast_tokenizer.save_pretrained(output_dir)
    print(f"✅ Tokenizer trained and saved locally at {output_dir}")

    # 5. Verification tests
    print("\n--- Verification tests ---")
    test_texts = [
        "Hello, this is a test of the Ember tokenizer. Hope it works beautifully!",
        "नमस्ते, यह केस्ट्रेल टोकनाइज़र का एक परीक्षण है। आशा है कि यह बहुत अच्छा काम करेगा।",
        "def quick_sort(arr):\n    if len(arr) <= 1: return arr\n    pivot = arr[len(arr) // 2]\n    return quick_sort([x for x in arr if x < pivot]) + [pivot]",
        "<|im_start|>user\nWrite a hello world program in python.<|im_end|>\n<|im_start|>assistant\n```python\nprint('Hello World')\n```<|im_end|>"
    ]
    
    for i, t in enumerate(test_texts):
        encoded = fast_tokenizer.encode(t)
        decoded = fast_tokenizer.decode(encoded, skip_special_tokens=False)
        print(f"\nTest {i+1}:")
        print(f"Original length: {len(t)} chars | Encoded tokens: {len(encoded)}")
        print(f"Tokens: {encoded}")
        print(f"Decoded: {decoded}")
        assert decoded == t, f"Round-trip decoding failed for: {t}"
    print("✅ All round-trip verification tests passed!")

    # 6. Push to HF Hub if requested
    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id is required to push to HF Hub")
        print(f"\nPushing tokenizer to HF Hub: {repo_id}...")
        fast_tokenizer.push_to_hub(repo_id, private=True)
        print("✅ Pushed tokenizer to Hugging Face Hub successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ember Custom BPE Tokenizer")
    parser.add_argument("--vocab_size", type=int, default=65536, help="Target vocabulary size")
    parser.add_argument("--sample_limit", type=int, default=30000, help="Number of text samples to stream for training")
    parser.add_argument("--push", action="store_true", help="Push to HF Hub after training")
    parser.add_argument("--repo_id", type=str, default="Kush26/ember-tokenizer", help="HF Hub repository ID")
    
    args = parser.parse_args()
    
    # We can execute training
    train_bpe_tokenizer(
        vocab_size=args.vocab_size,
        sample_limit=args.sample_limit,
        push_to_hub=args.push,
        repo_id=args.repo_id
    )
