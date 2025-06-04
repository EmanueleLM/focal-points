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
            max_length: int = None,
            num_return_sequences: int = None,
            quantization: str | None = None,  # None, "8bit", or "4bit"
    ):
        self.model_id = model_id
        self.top_p = top_p
        self.temperature = temperature
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.quantization = quantization

        # 1) Load the tokenizer (same for all cases)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
        )

        # 2) Build a BitsAndBytesConfig only if quantization is requested
        bnb_config = None
        load_in_8bit = False
        load_in_4bit = False

        if quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            load_in_8bit = True

        elif quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
            )
            load_in_4bit = True

        # 3) Prepare kwargs for from_pretrained()
        model_kwargs: dict = {
            "use_auth_token": os.getenv("HF_TOKEN"),
            "trust_remote_code": True,
            "device_map": "auto",
        }

        # If we have a quantization config, inject it
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
            model_kwargs["load_in_8bit"] = load_in_8bit
            model_kwargs["load_in_4bit"] = load_in_4bit
        else:
            # No quantization → let HF pick an appropriate dtype (e.g. float16 on A100/H200)
            model_kwargs["torch_dtype"] = "auto"

        # 4) Load the model once
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        # 5) Build the generation kwargs dictionary
        generation_kwargs: dict = {}
        if self.top_p is not None:
            generation_kwargs["top_p"] = self.top_p
        if self.temperature is not None:
            generation_kwargs["temperature"] = self.temperature
        if self.max_length is not None:
            generation_kwargs["max_length"] = self.max_length
        if self.num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = self.num_return_sequences

        # 6) Create a text-generation pipeline that accepts a batch of prompts
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            **generation_kwargs,
        )

    def generate_batch(self, prompts: list[str]) -> list[list[str]]:
        """
        Generate responses for each prompt in one call.
        Returns a list (length = len(prompts)) of lists of strings.
        """
        results = self.generator(prompts)
        all_responses: list[list[str]] = []

        for out in results:
            if isinstance(out, list):
                # multiple sequences returned for this prompt
                texts = [entry["generated_text"] for entry in out]
            else:
                texts = [out["generated_text"]]
            all_responses.append(texts)

        return all_responses

    def generate(self, prompt: str) -> str:
        return self.generate_batch([prompt])[0][0]
