# Semix

Official implementation of "Semantic equivariant mixup"



# Use Semix in other Mixup-based method

`semix` can be easily applied to other Mixup-based method by the following traing function

```python
def mixdata(x , y , alpha):
    #use other Mixup method here

for batch_idx, data in enumerate(train_loader, 0):

	inputs_a, inputs_b, mixed_inputs, labels_a, labels_b, lam = mixdata(inputs, labels, alpha) # get mixup samples

	outputs_a = net(inputs_a)
	feature_a = net.feature
	outputs_b = net(inputs_b)
	feature_b = net.feature
	
    outputs = net(lam * inputs_a + (1 - lam) * inputs_b)
	feature = net.feature

	loss = lam * loss_func(outputs, labels_a) + (1 - lam) * loss_func(outputs, labels_b) \
             + zeta * loss_func_reg(feature, lam * feature_a + (1 - lam) * feature_b) # loss_func_reg here is MSE loss

```





# Train

use the following command to train

```cmake
python train.py --dataset cifar10 --arch PreResNet18 --alpha 0.5 --zeta 0.5 --base Mixup
```

where $\alpha$ means sampling $\lambda$ from $beta(\alpha,\alpha)$ and $\zeta$ means intensity of `SEMIX`.

`--base` means choose a base 'Mixup' like Mixup and CutMix.

Specially, to use Early-stop in the training, add following options:

`--ES True --ES_epoch 200`

where `--ES_epoch` means when to turn to Early-stop training stage.

# Evaluate

use the following command to evaluate

```cmake
python test.py --weight_path [Your weight path here] --data_root [Your data root here] --arch PreResNet18
```

We evaluate performance on CIFAR10-C/CIFAR100-C,CIFAR10.1,CIFAR10.2; 

OOD detection with OOD dataset SVHN and LSUN-C with MSP score and shows AUROC value.

