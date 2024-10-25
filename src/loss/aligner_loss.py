import torch
import whisper
from torch import nn


class AlignerLoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, alpha: float = 0.5, transcriber_size="tiny"):
        """
        Args:
            alpha: weight of similarity loss in the total loss
            transcriber: Speech-To-Text model that transcriber output audio to text
        """
        super().__init__()
        self.alpha = alpha
        self.transcriber = whisper.load_model(transcriber_size)
        self.similarity_loss = nn.MSELoss()
        self.correctness_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        output_audio: torch.Tensor,
        original_audio: torch.Tensor,
        translated_text: list,
        **batch,
    ):
        """
        Loss function calculation logic.

        Args:
            output_audio (Tensor): audio returned by model.
            original_audio (Tensor): audio before tranlsation.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        # TODO 2. I need to take vocabulary for Speech-to-Text model and softmax vector
        # for each predicted token to be able to apply CrossEntropyLoss
        try:
            print("Transcribing translated audio")
            with torch.no_grad():
                trans = self.model.transcribe(
                    output_audio,
                    language="en",
                    verbose=False,
                    word_timestamps=True,
                )
        except Exception as e:
            print(f"Error transcribing audio: {e}")
        similarity_loss = self.similarity_loss(output_audio, original_audio)
        correctness_loss = self.correctness_loss(trans, translated_text)
        weighted_loss = self.alpha * similarity_loss + correctness_loss
        return {
            "loss": weighted_loss,
            "similarity_loss": similarity_loss,
            "correctness_loss": correctness_loss,
        }
