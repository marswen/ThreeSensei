import os
import torch
from typing import List
from typing import Optional
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.chat_models import ChatOpenAI


if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device('cpu')


class Llama2Chat(LLM):
    tokenizer: object = None
    model: object = None
    temperature: float = 0
    top_p: float = 0

    def __init__(self, model_name_or_path, dtype_precision, temperature, top_p):
        super().__init__()
        self.load_model(model_name_or_path, dtype_precision)
        self.temperature = temperature
        self.top_p = top_p

    @property
    def _llm_type(self) -> str:
        return "Llama-2-chat"

    def load_model(self, model_name_or_path, dtype_precision):
        from transformers import LlamaTokenizer, LlamaForCausalLM
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
        if dtype_precision == 'float32':
            self.model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                                          device_map='auto')
        elif dtype_precision == 'float16':
            self.model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=torch.float16,
                                                          device_map='auto')
        elif dtype_precision == 'int8':
            self.model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=torch.float16,
                                                          load_in_8bit=True,
                                                          device_map='auto')
        else:
            self.model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                                          device_map='auto')

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=self.temperature,
            top_p=self.top_p
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if stop is None:
            content = response.strip()[len(prompt):]
        else:
            content = enforce_stop_tokens(response.strip()[len(prompt):], stop)
        return content

    def chat(self, prompt: str, temperature: float = 0.5, top_p: float = 0.75, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=temperature,
            top_p=top_p
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if stop is None:
            content = response
        else:
            content = enforce_stop_tokens(response, stop)
        return content


class ChatGLM2(LLM):
    tokenizer: object = None
    model: object = None
    temperature: float = 0
    top_p: float = 0

    def __init__(self, model_name_or_path, dtype_precision, temperature, top_p):
        super().__init__()
        self.load_model(model_name_or_path, dtype_precision)
        self.temperature = temperature
        self.top_p = top_p

    @property
    def _llm_type(self) -> str:
        return "ChatGLM-2"

    def load_model(self, model_name_or_path, dtype_precision):
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       trust_remote_code=True)
        if dtype_precision == 'float32':
            self.model = AutoModel.from_pretrained(model_name_or_path,
                                                   trust_remote_code=True,
                                                   device_map='auto')
        elif dtype_precision == 'float16':
            self.model = AutoModel.from_pretrained(model_name_or_path,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float16,
                                                   device_map='auto')
        elif dtype_precision == 'int8':
            self.model = AutoModel.from_pretrained(model_name_or_path,
                                                   trust_remote_code=True,
                                                   torch_dtype=torch.float16,
                                                   load_in_8bit=True,
                                                   device_map='auto')
        else:
            self.model = AutoModel.from_pretrained(model_name_or_path,
                                                   trust_remote_code=True,
                                                   device_map='auto')

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(self.tokenizer,
                                      prompt,
                                      temperature=self.temperature,
                                      top_p=self.top_p
                                      )
        if stop is None:
            content = response
        else:
            content = enforce_stop_tokens(response, stop)
        return content

    def chat(self, prompt: str, temperature: float = 0.5, top_p: float = 0.75, stop: Optional[List[str]] = None) -> str:
        response, _ = self.model.chat(self.tokenizer,
                                      prompt,
                                      temperature=temperature,
                                      top_p=top_p
                                      )
        if stop is None:
            content = response
        else:
            content = enforce_stop_tokens(response, stop)
        return content


class BaichuanChat(LLM):
    tokenizer: object = None
    model: object = None
    temperature: float = 0
    top_p: float = 0

    def __init__(self, model_name_or_path, dtype_precision, temperature, top_p):
        super().__init__()
        self.load_model(model_name_or_path, dtype_precision)
        self.temperature = temperature
        self.top_p = top_p

    @property
    def _llm_type(self) -> str:
        return "Baichuan-chat"

    def load_model(self, model_name_or_path, dtype_precision):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from transformers.generation.utils import GenerationConfig
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       trust_remote_code=True)

        if dtype_precision == 'float32':
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              trust_remote_code=True)
        elif dtype_precision == 'float16':
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              trust_remote_code=True,
                                                              torch_dtype=torch.float16)
        elif dtype_precision == 'int8':
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              trust_remote_code=True,
                                                              torch_dtype=torch.float16)
            self.model = self.model.quantize(8).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              trust_remote_code=True)
        self.model = self.model.to(device)
        self.model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        # TODO load params from .env
        output = self.model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=self.temperature,
            top_p=self.top_p
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if stop is None:
            content = response.strip()[len(prompt):]
        else:
            content = enforce_stop_tokens(response.strip()[len(prompt):], stop)
        return content

    def chat(self, prompt: str, temperature: float = 0.5, top_p: float = 0.75, stop: Optional[List[str]] = None) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        output = self.model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=temperature,
            top_p=top_p
        )
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        if stop is None:
            content = response
        else:
            content = enforce_stop_tokens(response, stop)
        return content


def load_llm():
    load_dotenv()
    model_path = os.getenv('MODEL_PATH')
    temperature = float(os.getenv('TEMPERATURE', 0.5))
    top_p = float(os.getenv('TOP_P', 0.75))
    dtype_precision = os.getenv('DTYPE_PRECISION')
    openai_key = os.getenv('OPENAI_API_KEY')
    openai_model_name = os.getenv('OPENAI_MODEL_NAME')
    if model_path is not None:
        if 'llama' in model_path.lower():
            llm_model = Llama2Chat(model_path, dtype_precision, temperature, top_p)
        elif 'chatglm' in model_path.lower():
            llm_model = ChatGLM2(model_path, dtype_precision, temperature, top_p)
        elif 'baichuan' in model_path.lower():
            llm_model = BaichuanChat(model_path, dtype_precision, temperature, top_p)
        else:
            raise ValueError('Model was not correctly specified.')
    else:
        if openai_key is not None and openai_model_name is not None:
            llm_model = ChatOpenAI(model_name=openai_model_name)
        else:
            raise ValueError('Model was not correctly specified.')
    return llm_model
