"""author@qsun1"""

def mmagic_to_official_name_mapping_g_ema():
    name_map = {}
    
    # 1st term: 16 
    for i in range(0, 8):
        mmagic_name = f'generator_ema.module.style_mapping.{i+1}.bias'
        offical_name = f'mapping.fc{i}.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.style_mapping.{i+1}.linear.weight_orig'
        offical_name = f'mapping.fc{i}.weight'
        name_map[mmagic_name] = offical_name
       
    # 2nd term: 10
    mmagic_name = 'generator_ema.module.constant_input.input'
    offical_name = 'synthesis.b4.const'
    name_map[mmagic_name] = offical_name
    
    mmagic_name = 'generator_ema.module.conv1.conv.weight_orig' # torch.Size([1, 512, 512, 3, 3])
    offical_name = 'synthesis.b4.conv1.weight' # torch.Size([512, 512, 3, 3])
    name_map[mmagic_name] = offical_name
    
    mmagic_name = 'generator_ema.module.conv1.conv.style_modulation.bias'
    offical_name = 'synthesis.b4.conv1.affine.bias'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator_ema.module.conv1.conv.style_modulation.linear.weight_orig'
    offical_name = 'synthesis.b4.conv1.affine.weight'
    name_map[mmagic_name] = offical_name
    
    mmagic_name = 'generator_ema.module.conv1.noise_injector.weight' # torch.Size([1])
    offical_name = 'synthesis.b4.conv1.noise_strength' # torch.Size([])
    name_map[mmagic_name] = offical_name
 
    mmagic_name = 'generator_ema.module.conv1.activate.bias'
    offical_name = 'synthesis.b4.conv1.bias'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator_ema.module.to_rgb1.bias'
    offical_name = 'synthesis.b4.torgb.bias'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator_ema.module.to_rgb1.conv.weight_orig'
    offical_name = 'synthesis.b4.torgb.weight'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator_ema.module.to_rgb1.conv.style_modulation.bias'
    offical_name = 'synthesis.b4.torgb.affine.bias'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator_ema.module.to_rgb1.conv.style_modulation.linear.weight_orig'
    offical_name = 'synthesis.b4.torgb.affine.weight'
    name_map[mmagic_name] = offical_name

    # 3nd term: 7 x 14 = 98
    for i in range(0, 7):
        j = 2**(i+3)
        mmagic_name = f'generator_ema.module.convs.{2*i}.conv.weight_orig'
        offical_name = f'synthesis.b{j}.conv0.weight'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator_ema.module.convs.{2*i}.conv.style_modulation.bias'
        offical_name = f'synthesis.b{j}.conv0.affine.bias'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator_ema.module.convs.{2*i}.conv.style_modulation.linear.weight_orig'
        offical_name = f'synthesis.b{j}.conv0.affine.weight'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator_ema.module.convs.{2*i}.noise_injector.weight'
        offical_name = f'synthesis.b{j}.conv0.noise_strength'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator_ema.module.convs.{2*i}.activate.bias'
        offical_name = f'synthesis.b{j}.conv0.bias'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator_ema.module.convs.{2*i+1}.conv.weight_orig'
        offical_name = f'synthesis.b{j}.conv1.weight'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.convs.{2*i+1}.conv.style_modulation.bias'
        offical_name = f'synthesis.b{j}.conv1.affine.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.convs.{2*i+1}.conv.style_modulation.linear.weight_orig'
        offical_name = f'synthesis.b{j}.conv1.affine.weight'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.convs.{2*i+1}.noise_injector.weight'
        offical_name = f'synthesis.b{j}.conv1.noise_strength'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.convs.{2*i+1}.activate.bias'
        offical_name = f'synthesis.b{j}.conv1.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.to_rgbs.{i}.bias'
        offical_name = f'synthesis.b{j}.torgb.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.to_rgbs.{i}.conv.weight_orig'
        offical_name = f'synthesis.b{j}.torgb.weight'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.to_rgbs.{i}.conv.style_modulation.bias'
        offical_name = f'synthesis.b{j}.torgb.affine.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator_ema.module.to_rgbs.{i}.conv.style_modulation.linear.weight_orig'
        offical_name = f'synthesis.b{j}.torgb.affine.weight'
        name_map[mmagic_name] = offical_name
    
    print(len(name_map.keys()))
        
    return name_map
