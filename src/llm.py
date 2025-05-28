from huggingface_hub import login
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

load_dotenv()
login(token=os.environ["HF_TOKEN"])


class LLM:
    def __init__(
            self,
            model_name: str,
            model_id: str,
            top_p: float = None,
            temperature: float = None,
            max_length: int = None,
            num_return_sequences: int = None,
    ):

        self.name = model_name
        self.model_id = model_id
        self.top_p = top_p
        self.temperature = temperature
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN"),
            device_map="auto",
            trust_remote_code=True
        )

        generation_kwargs = {}
        if self.top_p is not None:
            generation_kwargs["top_p"] = self.top_p
        if self.temperature is not None:
            generation_kwargs["temperature"] = self.temperature
        if self.max_length is not None:
            generation_kwargs["max_length"] = self.max_length
        if self.num_return_sequences is not None:
            generation_kwargs["num_return_sequences"] = self.num_return_sequences

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            device="auto",
            **generation_kwargs
        )

    def generate_batch(self, prompts: list[str]) -> list[list[str]]:
        """
        Generate responses for each prompt.

        Args:
            prompts (list[str]): A list of input prompt strings.

        Returns:
            list[list[str]]: For each prompt, a list of generated responses.
        """
        results = self.generator(prompts)

        all_responses = []
        for out in results:
            if isinstance(out, list):
                # multiple sequences returned for this prompt
                texts = [entry["generated_text"] for entry in out]
            else:
                # a single sequence returned
                texts = [out["generated_text"]]
            all_responses.append(texts)

        return all_responses
