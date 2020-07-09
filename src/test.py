from models import FusionSideNetFc

model = FusionSideNetFc(300, num_classes=10, alphas=[.3, .3, .4], dropout_prob=.5, side_fc=1024)
model = FusionSideResNet(300, num_classes=10, alphas=[.3, .3, .4], dropout_prob=.5, side_fc=1024)
print(sum(p.numel() for p in model.parameters()))