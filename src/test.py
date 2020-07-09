from models import FusionSideNetFc, FusionSideNetFcResNet, MobileNet, ResNet

# model = FusionSideNetFc(300, num_classes=10, alphas=[.3, .3, .4], dropout_prob=.5, side_fc=512)
# print(sum(p.numel() for p in model.parameters()))
# model = FusionSideNetFc(300, num_classes=10, alphas=[.3, .3, .4], dropout_prob=.5, side_fc=1024)
# print(sum(p.numel() for p in model.parameters()))
# model = FusionSideNetFcResNet(300, num_classes=10, alphas=[.3, .3, .4], dropout_prob=.5, side_fc=512)
# print(sum(p.numel() for p in model.parameters()))
# model = FusionSideNetFcResNet(300, num_classes=10, alphas=[.3, .3, .4], dropout_prob=.5, side_fc=1024)
# print(sum(p.numel() for p in model.parameters()))
# model = MobileNet(10)
# print(sum(p.numel() for p in model.parameters()))
model = ResNet(10)
print(sum(p.numel() for p in model.parameters()))
