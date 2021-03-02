import re
import torch.nn as nn

from pretrainedmodels.models.xception import pretrained_settings
from pretrainedmodels.models.xception import Xception

from ._base import EncoderMixin


class XceptionEncoder(Xception, EncoderMixin):

    def __init__(self, out_channels, *args, depth=5, **kwargs):
        super().__init__(*args, **kwargs)

        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3

        # modify padding to maintain output shape
        self.conv1.padding = (1, 1)
        self.conv2.padding = (1, 1)

        del self.fc

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("Xception encoder does not support dilated mode "
                         "due to pooling operation for downsampling!")

    def get_stages(self):
        stages = [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu, self.conv2, self.bn2, self.relu),
            self.block1,
            self.block2
        ]
        if self._depth > 3:
            stages.append(nn.Sequential(self.block3, self.block4, self.block5, self.block6, self.block7,
                          self.block8, self.block9, self.block10, self.block11))
        if self._depth > 4:
            stages.append(nn.Sequential(self.block12, self.conv3, self.bn3, self.relu, self.conv4, self.bn4))
        return stages

    def remove_useless_stages(self):
        if self._depth < 5:
            del self.block12
            del self.conv3
            del self.bn3
            del self.relu
            del self.conv4
            del self.bn4
        if self._depth < 4:
            del self.block3
            del self.block4
            del self.block5
            del self.block6
            del self.block7
            del self.block8
            del self.block9
            del self.block10
            del self.block11

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict):
        # remove linear
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')

        super().load_state_dict(state_dict)


xception_encoders = {
    'xception': {
        'encoder': XceptionEncoder,
        'pretrained_settings': pretrained_settings['xception'],
        'params': {
            'out_channels': (3, 64, 128, 256, 728, 2048),
        }
    },
}
