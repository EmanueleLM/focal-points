import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

load_dotenv()
login(token=os.environ["HF_TOKEN"])


class LLM:
    def __init__(
            self,
            model_id: str,
            top_p: float = None,
            temperature: float = None,
            max_new_tokens: int = None,
            num_return_sequences: int = None,
            quantization: str | None = None,  # None, "8bit", or "4bit"
    ):
        self.model_id = model_id
        self.top_p = top_p
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.quantization = quantization

        # Load the tokenizer (same for all cases)
        # Load the tokenizer (same for all cases)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
        )

        # Build a BitsAndBytesConfig only if quantization is requested
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
                bnb_4bit_compute_dtype="float16",
            )

        # Prepare kwargs for from_pretrained()
        # Prepare kwargs for from_pretrained()
        model_kwargs: dict = {
            "use_auth_token": os.getenv("HF_TOKEN"),
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            # No quantization scenario
            model_kwargs["torch_dtype"] = "auto"

        # Load the model once
        # Load the model once
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        # Build the generation kwargs dictionary
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
            "chat-completion",
            model=self.model,
            tokenizer=self.tokenizer,
            **generation_kwargs,
        )

        # Fixed system instruction
        self.system_message = (
            "You are an assistant that only replies with a single answer "
            "wrapped in <answer> tags, and nothing else."
        )

    def generate_batch(self, prompts: list[str]) -> list[list[str]]:
        results = self.generator(prompts)
        all_responses: list[list[str]] = []

        batch_inputs = []
        for user_prompt in prompts:
            batch_inputs.append([
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_prompt}
            ])

        results = self.generator(batch_inputs)

        # Extract the "generated_text" for each completion
        for out in results:
            if isinstance(out, list):
                texts = [entry["generated_text"] for entry in out]
            else:
                texts = [out["generated_text"]]
            all_responses.append(texts)

        return all_responses

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0][0]