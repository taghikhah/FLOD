
python train.py -id cifar10 -cm wideresnet40 -cw 'weights/cifar10_wideresnet40_last.pt'
python train.py -id cifar100 -cm wideresnet40 -cw 'weights/cifar100_wideresnet40_last.pt'

python test.py -em FLOD -id cifar10 -ood all -cm wideresnet40 -cw 'weights/cifar10_wideresnet40_last.pt' -fw 'weights/cifar10_glow_best.pt'
python test.py -em FLOD -id cifar100 -ood all -cm wideresnet40  -cw 'weights/cifar100_wideresnet40_last.pt' -fw 'weights/cifar100_glow_best.pt'

# python test.py -em FLOD -id cifar10 -ood all -cw 'weights/cifar10_wideresnet40_last.pt' -fw 'weights/cifar10_glow_last.pt'
# python test.py -em FLOD -id cifar100 -ood all -cw 'weights/cifar100_wideresnet40_last.pt' -fw 'weights/cifar100_glow_last.pt'