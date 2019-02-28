from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target= targets
        # mel_target.requires_grad = True

        mel_out, mel_out_postnet, _ = model_output
        mel_loss = nn.L1Loss()(mel_out, mel_target) + nn.L1Loss()(mel_out_postnet, mel_target)
        return mel_loss
