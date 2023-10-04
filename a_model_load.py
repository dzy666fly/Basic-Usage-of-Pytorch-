# ckpt_path = 'model_epoch_360_NutAssemblySquare_success_0.84.pth'
# torch.float32: 4B
# 57.37M --> 229.48MB --> 229.8MB

model = torch.load(ckpt_path, device)

print(type(model))

for k in model.keys():
    print(k)

model_keys = model['model'].keys()
model_values = model['model'].values()

for each in model_keys:
    print(each)
    break
    
for each in model_values:
    print(each.shape, each.dtype)
    break
