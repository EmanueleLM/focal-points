import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
import gc, shutil
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

tok = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
if tok:
    os.environ["HUGGINGFACE_HUB_TOKEN"] = tok


class LLM:
    def __init__(
        self,
        model_id: str,
        top_p: float | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int | None = None,
        quantization: str | None = None,  # None, "8bit", or "4bit"
    ):
        self.model_id = model_id
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.quantization = quantization

        # Load the tokenizer (same for all cases)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
        )

        # Build a BitsAndBytesConfig only if quantization is requested
        bnb_config = None

        if quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        # Prepare kwargs for from_pretrained()
        model_kwargs: dict = {
            "token": os.getenv("HF_TOKEN"),
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["dtype"] = torch.bfloat16

        # Load the model once
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        self._log_device_allocation()

        # Build the generation kwargs dictionary
        generation_kwargs: dict = {
            "do_sample": True,
            "return_full_text": False,
        }
        if self.top_p is not None:
            generation_kwargs["top_p"] = self.top_p
        if self.temperature is not None:
            generation_kwargs["temperature"] = self.temperature
        if self.max_new_tokens is not None:
            generation_kwargs["max_new_tokens"] = self.max_new_tokens
        if self.num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = self.num_return_sequences

        # Create a text-generation pipeline that accepts a batch of prompts
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            disable_compile=(True if self.model_id == "google/gemma-3-4b-it" else None),
            # this fixes this bug with Gemma: https://github.com/huggingface/transformers/issues/38333
            **generation_kwargs,
        )

        self.chat_template = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": None},
        ]

    def _log_device_allocation(self) -> None:
        if not torch.cuda.is_available():
            print("[INFO] CUDA not available; skipping device allocation check.")
            return

        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            print("[INFO] Model device map:")
            for module, device in device_map.items():
                print(f"  - {module}: {device}")
        else:
            print(
                "[INFO] Model did not expose a device map; assuming single-device placement."
            )

        for idx in range(torch.cuda.device_count()):
            allocated_gib = torch.cuda.memory_allocated(idx) / (1024**3)
            reserved_gib = torch.cuda.memory_reserved(idx) / (1024**3)
            print(
                f"[INFO] CUDA:{idx} memory allocated: {allocated_gib:.2f} GiB "
                f"(reserved {reserved_gib:.2f} GiB)"
            )

    def clear_cache(self):
        # free VRAM / system RAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del self.generator, self.model
        gc.collect()

    def generate_batch(self, prompts: list[str]) -> list[list[str]]:
        formatted_chats = []
        for prompt in prompts:
            chat = [
                {"role": "system", "content": self.chat_template[0]["content"]},
                {"role": "user", "content": prompt},
            ]
            formatted_chats.append(chat)

        results = self.generator(formatted_chats)
        all_responses: list[list[str]] = []

        for out in results:
            if isinstance(out, list):
                texts = [
                    "".join(
                        c
                        for c in entry["generated_text"].strip().lower()
                        if c.isalnum() or c.isspace() or c in "<>/"
                    )
                    for entry in out
                ]
            else:
                texts = [
                    "".join(
                        c
                        for c in out["generated_text"].strip().lower()
                        if c.isalnum() or c.isspace() or c in "<>/"
                    )
                ]
            all_responses.append(texts)

        return all_responses

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0][0]
