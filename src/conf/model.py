from torch.optim import lr_scheduler, Adam

def optimizer(model, lr):
    return adam_optimizer(model=model, lr=lr)

def adam_optimizer(model, lr=0.001):
    return Adam(model.parameters(), lr=lr, weight_decay=0.0)

def expr_lr_scheduler(optimizer, step_size=7):
    # Decay LR by a factor of 0.1 every 7 epochs
    return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
