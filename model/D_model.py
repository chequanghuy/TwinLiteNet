import os
import sys
import  torch.nn as nn
import torch
if os.path.isdir('/kaggle/working/efficientvit')== True:
    sys.path.insert(1, os.path.join(sys.path[0], '/kaggle/working/efficientvit'))
    from efficientvit.seg_model_zoo import create_seg_model
else:
    sys.path.insert(1, os.path.join(sys.path[0], '/content/efficientvit'))
    from efficientvit.seg_model_zoo import create_seg_model
class disc_modek(nn.Module):
    def __init__(self):
        super().__init__()
        model = create_seg_model('b0','bdd',pretrained=None)
        self.backbone = model.backbone
        self.head1 = model.head1
        self.head2 = model.head2
    def forward(self,x, Discriminator, domain='source'):
        feed_dict = self.backbone
        feed_dict_da = feed_dict.copy()
        feed_dict_ll = feed_dict.copy()

        segout_da, feat_da = self.head1(feed_dict_da)
        segout_ll, feat_ll = self.head2(feed_dict_ll)

        if domain == 'source':
            pass
        else:
            a4 = Discriminator[0](x4)
            a4 = nn.Tanh(a4)
            a4 = torch.abs(a4)
            a4_big = a4.expand(x4.size())
            x4_a4 = a4_big * x4 + x4