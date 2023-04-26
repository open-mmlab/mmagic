# Directly inherit the entire recipe you want to use.
_base_ = 'mmagic::srcnn/srcnn_x4k915_1xb16-1000k_div2k.py'

# This line is to import your own modules.
custom_imports = dict(imports='models')

# Set your model, training, testing configurations.
