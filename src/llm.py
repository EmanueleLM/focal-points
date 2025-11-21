from abc import ABC, abstractmethod
import os
import torch
import gc
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)


class LLM(ABC):
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a single response for the provided prompt."""
        raise NotImplementedError

    @abstractmethod
    def generate_batch(self, prompts: list[str]) -> list[list[str]]:
        """Generate responses for a list of prompts."""
        raise NotImplementedError


class LocalLLM(LLM):
    def __init__(
        self,
        model_id: str,
        top_p: float | None = None,
        temperature: float | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int | None = None,
        quantization: str | None = None,  # None, "8bit", or "4bit"
        reasoning: str | None = None,
    ):
        super().__init__(model_id=model_id)
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.quantization = quantization

        if reasoning:
            if reasoning.lower() in {"low", "medium", "high"}:
                self.reasoning = reasoning.lower()
                print(f"[INFO] Local model reasoning level: {self.reasoning}")
            else:
                self.reasoning = None
                print(
                    "[WARNING] Invalid reasoning level provided; ignoring reasoning parameter and using default."
                )

        # Load the tokenizer (same for all cases)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
        )

        # Build a BitsAndBytesConfig only if quantization is requested
        bnb_config = None

        if quantization == "8bit" and self.model_id != "openai/gpt-oss-120b":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
        elif quantization == "4bit" and self.model_id != "openai/gpt-oss-120b":
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

        system_prompt = "You are a helpful assistant."
        if self.model_id == "openai/gpt-oss-120b" and self.reasoning:
            system_prompt = f"{system_prompt}\nReasoning: {self.reasoning}"

        self.chat_template = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": None},
        ]

        # Print system prompt for verification
        print(f"[INFO] System prompt set to: \n{system_prompt}")

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


class APILLM(LLM):
    def __init__(
        self,
        model_id: str,
        reasoning: dict | None = None,
        num_return_sequences: int = 1,
    ) -> None:
        super().__init__(model_id=model_id)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.reasoning = reasoning
        self.num_return_sequences = num_return_sequences

        # Add print statement to confirm initialization
        print(f"[INFO] Initialized APILLM with model_id: {self.model_id}")
        print(f"[INFO] API reasoning parameters: {self.reasoning}")
        print(f"[INFO] API sequences per prompt: {self.num_return_sequences}")

    def _extract_output_text(self, response) -> str:
        text_chunks: list[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for content in getattr(item, "content", []):
                    if getattr(content, "type", None) == "output_text":
                        text_chunks.append(content.text)
            elif getattr(item, "type", None) == "output_text":
                text_chunks.append(item.text)
        if not text_chunks and getattr(response, "output_text", None):
            text_chunks.append(response.output_text)
        return "".join(text_chunks).strip()

    def generate_batch(self, prompts: list[str]) -> list[list[str]]:
        all_outputs: list[list[str]] = []
        for prompt in prompts:
            prompt_outputs: list[str] = []
            for _ in range(self.num_return_sequences):
                response = self.client.responses.create(
                    model=self.model_id,
                    input=prompt,
                    reasoning=self.reasoning,
                )
                raw_text = self._extract_output_text(response)
                normalized = "".join(
                    c
                    for c in raw_text.strip().lower()
                    if c.isalnum() or c.isspace() or c in "<>/"
                )
                prompt_outputs.append(normalized)
            all_outputs.append(prompt_outputs)
        return all_outputs

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0][0]


def load_model(
    model_id: str,
    top_p: float | None = None,
    temperature: float | None = None,
    max_new_tokens: int | None = None,
    num_return_sequences: int = 1,
    quantization: str | None = None,
    reasoning: str | None = None,
) -> LLM:
    """Load either an API or locally hosted LLM implementation"""

    API_MODELS = {"gpt-5", "gpt-5.1"}

    LOCAL_MODELS = {
        "meta-llama/Llama-3.3-70B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2-72B-Instruct",
        "Qwen/Qwen2.5-72B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct-1M",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct-1M",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "Qwen/Qwen2-0.5B-Instruct",
        "google/gemma-3-1b-it",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
        "microsoft/Phi-4-mini-instruct",
        "openai/gpt-oss-120b",
    }

    if model_id in API_MODELS:
        reasoning_payload = None
        if reasoning:
            reasoning_payload = {"effort": reasoning}  # OPENAI style

        return APILLM(
            model_id=model_id,
            reasoning=reasoning_payload,
            num_return_sequences=num_return_sequences,
        )
    elif model_id in LOCAL_MODELS:
        return LocalLLM(
            model_id=model_id,
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            quantization=quantization,
            reasoning=reasoning,
        )

    raise ValueError(f"Model ID '{model_id}' is not recognized as a valid model.")
