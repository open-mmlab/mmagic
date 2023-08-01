"""
    convert official stylegan2 model ckpt to mmagic model pth

"""
import torch
from tmp import mmagic_to_official_name_mapping_g_ema
import sys
sys.path.append(".")

stylegan_pth = './path_files/stylegan2_lions_512_pytorch.pth'
model = torch.load(stylegan_pth)
# print(model['G_ema'].state_dict()['synthesis.b64.conv1.resample_filter'])
G_state_dict = model['G'].state_dict()
G_ema_state_dict = model['G_ema'].state_dict()
# print((G_ema_state_dict.keys()))
# print(model.keys())
# print('----------------------')

n = 0

# for name, parameters in model["G"].named_parameters():  # 124 
#     if "mapping" in name:
#         n += 1
#         print(name, ';', parameters.size())
# print(n)
# print('----------------------')

# for name, parameters in model["G_ema"].named_parameters(): 
#     n += 1 
#     print(name, ';', parameters.size())
# print(n)

# for name, parameters in model["D"].named_parameters():  
#     n += 1 
#     print(name, ';', parameters.size())
# print(n)

# print('======================================')

def mmagic_to_official_name_mapping():
    name_map = {}
    
    # 1st term: 16 
    for i in range(0, 8):
        mmagic_name = f'generator.style_mapping.{i+1}.bias'
        offical_name = f'mapping.fc{i}.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.style_mapping.{i+1}.linear.weight_orig'
        offical_name = f'mapping.fc{i}.weight'
        name_map[mmagic_name] = offical_name
       
    # 2nd term: 10
    mmagic_name = 'generator.constant_input.input'
    offical_name = 'synthesis.b4.const'
    name_map[mmagic_name] = offical_name
    
    mmagic_name = 'generator.conv1.conv.weight_orig' # torch.Size([1, 512, 512, 3, 3])
    offical_name = 'synthesis.b4.conv1.weight' # torch.Size([512, 512, 3, 3])
    name_map[mmagic_name] = offical_name
    
    mmagic_name = 'generator.conv1.conv.style_modulation.bias'
    offical_name = 'synthesis.b4.conv1.affine.bias'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator.conv1.conv.style_modulation.linear.weight_orig'
    offical_name = 'synthesis.b4.conv1.affine.weight'
    name_map[mmagic_name] = offical_name
    
    mmagic_name = 'generator.conv1.noise_injector.weight' # torch.Size([1])
    offical_name = 'synthesis.b4.conv1.noise_strength' # torch.Size([])
    name_map[mmagic_name] = offical_name
 
    mmagic_name = 'generator.conv1.activate.bias'
    offical_name = 'synthesis.b4.conv1.bias'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator.to_rgb1.bias'
    offical_name = 'synthesis.b4.torgb.bias'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator.to_rgb1.conv.weight_orig'
    offical_name = 'synthesis.b4.torgb.weight'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator.to_rgb1.conv.style_modulation.bias'
    offical_name = 'synthesis.b4.torgb.affine.bias'
    name_map[mmagic_name] = offical_name

    mmagic_name = 'generator.to_rgb1.conv.style_modulation.linear.weight_orig'
    offical_name = 'synthesis.b4.torgb.affine.weight'
    name_map[mmagic_name] = offical_name

    # 3nd term: 7 x 14 = 98
    for i in range(0, 7):
        j = 2**(i+3)
        mmagic_name = f'generator.convs.{2*i}.conv.weight_orig'
        offical_name = f'synthesis.b{j}.conv0.weight'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator.convs.{2*i}.conv.style_modulation.bias'
        offical_name = f'synthesis.b{j}.conv0.affine.bias'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator.convs.{2*i}.conv.style_modulation.linear.weight_orig'
        offical_name = f'synthesis.b{j}.conv0.affine.weight'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator.convs.{2*i}.noise_injector.weight'
        offical_name = f'synthesis.b{j}.conv0.noise_strength'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator.convs.{2*i}.activate.bias'
        offical_name = f'synthesis.b{j}.conv0.bias'
        name_map[mmagic_name] = offical_name
        
        mmagic_name = f'generator.convs.{2*i+1}.conv.weight_orig'
        offical_name = f'synthesis.b{j}.conv1.weight'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.convs.{2*i+1}.conv.style_modulation.bias'
        offical_name = f'synthesis.b{j}.conv1.affine.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.convs.{2*i+1}.conv.style_modulation.linear.weight_orig'
        offical_name = f'synthesis.b{j}.conv1.affine.weight'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.convs.{2*i+1}.noise_injector.weight'
        offical_name = f'synthesis.b{j}.conv1.noise_strength'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.convs.{2*i+1}.activate.bias'
        offical_name = f'synthesis.b{j}.conv1.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.to_rgbs.{i}.bias'
        offical_name = f'synthesis.b{j}.torgb.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.to_rgbs.{i}.conv.weight_orig'
        offical_name = f'synthesis.b{j}.torgb.weight'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.to_rgbs.{i}.conv.style_modulation.bias'
        offical_name = f'synthesis.b{j}.torgb.affine.bias'
        name_map[mmagic_name] = offical_name

        mmagic_name = f'generator.to_rgbs.{i}.conv.style_modulation.linear.weight_orig'
        offical_name = f'synthesis.b{j}.torgb.affine.weight'
        name_map[mmagic_name] = offical_name
    
    print(len(name_map.keys()))
        
    return name_map

if __name__ == '__main__':  
    mmagic_pth='./512.pth'
    model_mmagic = torch.load(mmagic_pth)
    # import ipdb; ipdb.set_trace();
    model_mmagic_state_dict = model_mmagic.state_dict()
    name_mapping = mmagic_to_official_name_mapping()
    name_mapping_g_ema = mmagic_to_official_name_mapping_g_ema()

    # n = 0
    for name, parameters in model_mmagic.named_parameters():  
        if 'generator.' in name:
            official_name = name_mapping[name]
            shape_mmagic = model_mmagic_state_dict[name].shape
            model_mmagic_state_dict[name] = G_state_dict[official_name].reshape(shape_mmagic)
        
        elif 'generator_ema.module.' in name:
            official_name = name_mapping_g_ema[name]
            shape_mmagic = model_mmagic_state_dict[name].shape
            model_mmagic_state_dict[name] = G_ema_state_dict[official_name].reshape(shape_mmagic)
            
    
    
    # supplement
    
    model_mmagic_state_dict['generator.w_avg'] = G_state_dict['mapping.w_avg']
    model_mmagic_state_dict['generator_ema.module.w_avg'] = G_ema_state_dict['mapping.w_avg']
 

    torch.save( model_mmagic_state_dict, './new_ckpts/stylegan2_lions_512_pytorch_mmagic.pth',)

# print(n)
# print('======================================')

# for k, v in model_mmagic.state_dict().items():
#     print(k, ' :', v.shape)

# print(n)


"""
Generator parameters name mapping:
mmagic --> official stylegan2

# in total: 124
generator.style_mapping.{i}.bias --> mapping.fc{i}.bias; size: [512, 512]
generator.style_mapping.{i}.linear.weight_orig --> mapping.fc{i}.weight; size: [512]

# in total: 16

generator.constant_input.input; torch.Size([1, 512, 4, 4])
-->
synthesis.b4.const; torch.Size([512, 4, 4])

generator.conv1.conv.weight_orig ; torch.Size([1, 512, 512, 3, 3])
generator.conv1.conv.style_modulation.bias ; torch.Size([512])
generator.conv1.conv.style_modulation.linear.weight_orig ; torch.Size([512, 512])
generator.conv1.noise_injector.weight ; torch.Size([1])
generator.conv1.activate.bias ; torch.Size([512])
--->
synthesis.b4.conv1.weight ; torch.Size([512, 512, 3, 3])
synthesis.b4.conv1.noise_strength ; torch.Size([])
synthesis.b4.conv1.bias ; torch.Size([512])
synthesis.b4.conv1.affine.weight ; torch.Size([512, 512])
synthesis.b4.conv1.affine.bias ; torch.Size([512])

# in total: 16+6

generator.to_rgb1.bias ; torch.Size([1, 3, 1, 1])
generator.to_rgb1.conv.weight_orig ; torch.Size([1, 3, 512, 1, 1])
generator.to_rgb1.conv.style_modulation.bias ; torch.Size([512])
generator.to_rgb1.conv.style_modulation.linear.weight_orig ; torch.Size([512, 512])
-->
synthesis.b4.torgb.weight ; torch.Size([3, 512, 1, 1])
synthesis.b4.torgb.bias ; torch.Size([3])
synthesis.b4.torgb.affine.weight ; torch.Size([512, 512])
synthesis.b4.torgb.affine.bias ; torch.Size([512])

# in total: 16+6+4


# i: 0-6: in total: 14 x 5 = 70
generator.convs.{2*i}.conv.weight_orig ; torch.Size([1, 512, 512, 3, 3])
generator.convs.{2*i}.conv.style_modulation.bias ; torch.Size([512])
generator.convs.{2*i}.conv.style_modulation.linear.weight_orig ; torch.Size([512, 512])
generator.convs.{2*i}.noise_injector.weight ; torch.Size([1])
generator.convs.{2*i}.activate.bias ; torch.Size([512])
generator.convs.{2*i+1}.conv.weight_orig ; torch.Size([1, 512, 512, 3, 3])
generator.convs.{2*i+1}.conv.style_modulation.bias ; torch.Size([512])
generator.convs.{2*i+1}.conv.style_modulation.linear.weight_orig ; torch.Size([512, 512])
generator.convs.{2*i+1}.noise_injector.weight ; torch.Size([1])
generator.convs.{2*i+1}.activate.bias ; torch.Size([512])

j = 2**(i+3)
synthesis.b{j}.conv0.weight ; torch.Size([512, 512, 3, 3])
synthesis.b{j}.conv0.noise_strength ; torch.Size([])
synthesis.b{j}.conv0.bias ; torch.Size([512])
synthesis.b{j}.conv0.affine.weight ; torch.Size([512, 512])
synthesis.b{j}.conv0.affine.bias ; torch.Size([512])
synthesis.b{j}.conv1.weight ; torch.Size([512, 512, 3, 3])
synthesis.b{j}.conv1.noise_strength ; torch.Size([])
synthesis.b{j}.conv1.bias ; torch.Size([512])
synthesis.b{j}.conv1.affine.weight ; torch.Size([512, 512])
synthesis.b{j}.conv1.affine.bias ; torch.Size([512])


# i: 0-6; in total: 4 x 7 = 28
generator.to_rgbs.{i}.bias ; torch.Size([1, 3, 1, 1])
generator.to_rgbs.{i}.conv.weight_orig ; torch.Size([1, 3, 512, 1, 1])
generator.to_rgbs.{i}.conv.style_modulation.bias ; torch.Size([512])
generator.to_rgbs.{i}.conv.style_modulation.linear.weight_orig ; torch.Size([512, 512])

-->
j = 8,16,32,64,128,256,512; j=2**(i+3)
synthesis.b{j}.torgb.weight ; torch.Size([3, 512, 1, 1])
synthesis.b{j}.torgb.bias ; torch.Size([3])
synthesis.b{j}.torgb.affine.weight ; torch.Size([512, 512])
synthesis.b{j}.torgb.affine.bias ; torch.Size([512])

# in total: 70+28+16+6+4=124
"""