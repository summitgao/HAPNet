from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./logs')
for i in range(100):
    loss = -i
    writer.add_scalar("loss",loss,i)
writer.close()
