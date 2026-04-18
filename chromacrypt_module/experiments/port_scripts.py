import os
import shutil

source_lpips = r'c:\Users\marti\Documents\Projects\research_lab\working_scripts\Experiments\experiment_lpips_threshold_sweep.py'
dest_lpips = r'c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\benchmark_lpips_sweep.py'

source_abl = r'c:\Users\marti\Documents\Projects\research_lab\working_scripts\Benchmarks\benchmark_full_channel_ablation_sweep.py'
dest_abl = r'c:\Users\marti\Documents\Projects\research_lab\chromacrypt_module\experiments\benchmark_whitebox_ablation.py'

def process_file(src, dst, replacements):
    with open(src, 'r') as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(dst, 'w') as f:
        f.write(content)

process_file(source_lpips, dest_lpips, [
    ('from oklch_defense.differentiable_color_ops import DifferentiableColorOps',
     'from chromacrypt_module.differentiable_color_ops import DifferentiableColorOps\nfrom chromacrypt_module import utils as core_utils'),
    ('out_dir = os.path.join(project_root, "working_scripts", "results")',
     'out_dir = os.path.join(os.path.dirname(__file__), "results")'),
    ('import core_utils\n    data_dir = core_utils.find_imagenet_dir()',
     'data_dir = core_utils.find_imagenet_dir()'),
    ('working_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',
     'working_scripts = os.path.dirname(os.path.abspath(__file__))')
])

process_file(source_abl, dest_abl, [
    ('from oklch_defense.differentiable_color_ops import DifferentiableColorOps',
     'from chromacrypt_module.differentiable_color_ops import DifferentiableColorOps'),
    ('from core_utils import load_imagenet_val_batch, OKLCHModelWrapper, DEVICE',
     'from chromacrypt_module.utils import load_imagenet_val_batch, OKLCHModelWrapper\nDEVICE = "cuda" if torch.cuda.is_available() else "cpu"'),
    ('working_scripts = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))',
     'working_scripts = os.path.dirname(os.path.abspath(__file__))')
])

print('Ported missing benchmark scripts to chromacrypt_module successfully!')
