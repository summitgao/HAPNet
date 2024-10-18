from train import *
from test import *

import parameter

parameter._init()

def myTask(lr, epoch_nums, datasetType):
    parameter.set_value('epoch_nums', epoch_nums)
    parameter.set_value('lr', lr)
    parameter.set_value('cuda', cuda)
    parameter.set_value('visualization', visualization)
    myTrain(datasetType, net)
    myTest(datasetType)

cuda = 'cuda0'
net = 'MyNet'
visualization = False

#myTask(0.0001, 100, 3)
#myTask(0.0001, 300, 3)

#myTask(0.0001, 200, 4)
myTask(0.0001, 2, 4)



