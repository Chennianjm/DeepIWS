from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        with torch.no_grad():
            BaseModel.initialize(self, opt)
            self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
            self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)

            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                          opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids, False,
                                          opt.learn_residual, opt.layers)
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        with torch.no_grad():
            input_A = input['A']
            self.input_A = self.input_A.clone().resize_(input_A.size()).copy_(input_A)
            input_B = input['B']
            self.input_B = self.input_B.clone().resize_(input_B.size()).copy_(input_B)
            self.image_paths = input['A_paths']

    def test(self):
        with torch.no_grad():
            self.real_A = self.input_A
            self.mask = self.netG.forward(self.real_A).detach()
            self.fake_B = util.generate_sr(self.real_A, self.mask)
            self.real_B = self.input_B

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        try:
            real_A_1 = real_A[:, :, 0].reshape(512, 512, 1)
            real_A_2 = real_A[:, :, 1].reshape(512, 512, 1)
        except:
            real_A_1 = real_A[:, :, 0].reshape(1024, 1024, 1)
            real_A_2 = real_A[:, :, 1].reshape(1024, 1024, 1)
        real_B = util.tensor2im(self.real_B.data)
        fake_B = util.tensor2im(self.fake_B.data)
        mask = util.tensor2im(self.mask.data)
        del self.real_A, self.fake_B, self.mask
        return OrderedDict([('real_A_1', real_A_1), ('real_A_2', real_A_2), ('fake_B', fake_B), ('mask', mask), ('real_B', real_B)])
