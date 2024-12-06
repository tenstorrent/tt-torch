import torch
from transformers import PhiForCausalLM, PhiConfig, AutoTokenizer, StaticCache

from tt_torch.dynamo.backend import backend
from tt_torch.tools.utils import CompilerConfig, CompileDepth
from tt_torch.tools.verify import verify_module


def test_cache():
    config = PhiConfig()
    config.use_cache=True
    config.num_hidden_layers = 1

    class Basic(torch.nn.Module):
        def __init__(self):
            super().__init__()
        
        def forward(self, key_states, value_states, cache_position, static_cache):
            cache_kwargs = {"cache_position" : cache_position}
            k_out, v_out = static_cache.update(key_states, value_states, 0, cache_kwargs)
            return k_out + 0, v_out + 0

    
    state_shape = (1, 32, 1, 64)
    cache_positon = torch.tensor([0]).reshape(1).to(torch.int64)
    key_states = torch.randn(state_shape).to(torch.bfloat16)
    value_states = torch.randn(state_shape).to(torch.bfloat16)

    static_cache = StaticCache(config, batch_size=1, max_cache_len=64, dtype=torch.bfloat16)
    model = Basic()
    
    inputs = (key_states, value_states, cache_positon, static_cache)
    ret = model(*inputs)
    verify_module(model, inputs=inputs)

def test_phi():
    config = PhiConfig()
    config.use_cache=True
    config.num_hidden_layers = 1

    model = PhiForCausalLM(config).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.bfloat16)

    prompt = "I like to"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    

    max_new_tokens = 50
    inputs['position_ids'] = torch.arange(inputs.input_ids.shape[1]).reshape(1, -1)
    inputs['past_key_values'] = StaticCache(config, batch_size=1, max_cache_len=(inputs.input_ids.shape[1] + max_new_tokens), dtype=torch.bfloat16)
    inputs['cache_position'] = torch.arange(inputs.input_ids.shape[1])
    model.generate(**inputs)
    cc = CompilerConfig()
    cc.compile_depth = CompileDepth.TORCH_MLIR

    full_sentence = inputs.input_ids
    
    for i in range(max_new_tokens):
        tt_mod = torch.compile(model, backend=backend, options=cc)
        results = tt_mod(**inputs)
        
        if i == 0:
            torch._dynamo.reset_code_caches()

        next_token = results["logits"].detach()[:, -1].argmax(-1).reshape(1, 1)
        full_sentence = torch.cat((full_sentence, next_token), dim=-1)

        inputs['input_ids'] = next_token
        inputs['position_ids'] = inputs['position_ids'][0, -1].reshape(1, 1) + 1
        inputs['cache_position'] = inputs['cache_position'][-1].reshape(1) + 1
        inputs['past_key_values'] = results['past_key_values']
        inputs['attention_mask'] = torch.cat((inputs['attention_mask'], torch.tensor(1).reshape(1, 1)), dim=-1)

        print(tokenizer.decode(full_sentence[0]))

