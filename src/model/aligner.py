import torch
from torch import nn


class AudioAligner(nn.Module):
    """
    The class for Aligner model that match translated audio to original audio
    without a large loss of information in translated audio.
    The model architecture is based on idea of double lossing as we have 2 goals:\n
    1. Align translated audio from Vocoder (Text-To-Speech model) to original video.
    Takes slightly the style of speaking and all pauses, silence and intonation \n
    2. Keep information from audio as in translated text (great recognizable speech)
    """

    def __init__(self, fc_input_lenght: int = 1048576, fc_hidden=None):
        """
        Args:
            input_length: number of input features.
            fc_hidden: number of hidden features,
            if None then fc_hidden = fc_input_length // 4
        """
        super().__init__()
        self.fc_input_lenght = fc_input_lenght
        self.fc_hidden = fc_hidden
        self.fc_hidden = self.fc_input_lenght // 4 if fc_hidden is None else fc_hidden
        self.mlp = nn.Sequential(
            nn.Linear(in_features=self.fc_input_lenght, out_features=self.fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=self.fc_hidden, out_features=self.fc_hidden // 2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.fc_hidden // 2),
            nn.Linear(in_features=self.fc_hidden // 2, out_features=self.fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=self.fc_hidden, out_features=self.fc_input_lenght),
        )

    def forward(self, aduio: torch.Tensor, **batch):
        """
        Model forward method.

        Args:
            aduio (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """

        return {"output_audio": self.nlp(aduio)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
