from .agent import Agent
from .copyisallyouneed import Copyisallyouneed
from .copyisallyouneed_pretrain import CopyisallyouneedPretrain
from .copyisallyouneed_prebatch import CopyisallyouneedNegPrebatch
from .copyisallyouneed_queryside import CopyisallyouneedQueryside
from .copyisallyouneed_prebatch_all_ref import CopyisallyouneedAllRef
from .copyisallyouneed_prebatch_all_ref_v2 import CopyisallyouneedAllRefV2
from .copyisallyouneed_prebatch_all_ref_v2_asyn import CopyisallyouneedAllRefV2Asyn
from .copyisallyouneed_prebatch_all_ref_v2_pipeline import CopyisallyouneedAllRefV2Pipeline
from .copyisallyouneed_prefix_only import CopyisallyouneedPrefixOnly
from .gpt2 import GPT2Baseline
# from .knnlm import KNNLMBaseline
# import ipdb

def load_model(args):
    model_name = args['models'][args['model']]['model_name'][args['mode']]
    model = globals()[model_name](**args)
    agent = Agent(model, args)
    return agent
