from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tt_torch.tools.utils import CompilerConfig
from tt_torch.dynamo.backend import BackendOptions
import torch
import torch_xla

config = AutoConfig.from_pretrained('xai-org/grok-2')
config.num_hidden_layers = 1
model = AutoModelForCausalLM.from_config(config)
model.eval()
batch_size = 1
max_length = 128
input_ids = torch.randint(
    0, model.config.vocab_size, (batch_size, max_length), dtype=torch.int32
)

cc = CompilerConfig()
cc.enable_consteval = True
cc.consteval_parameters = True

options = BackendOptions()
options.compiler_config = cc

tt_model = torch.compile(model, backend="tt-experimental", dynamic=False, options=options)
output = tt_model(input_ids=input_ids).last_hidden_state    
breakpoint()