import torch

from src.datasets.data_utils import split_and_pad_audio


def collate_fn_wrapper(max_lenth: int):
    def collate_fn(dataset_items: list[dict[str, torch.Tensor]]):
        """
        Collate and pad fields in the dataset items.
        Converts individual items into a batch.

        Args:
            dataset_items (list[dict]): list of objects from
                dataset.__getitem__.
        Returns:
            result_batch (dict[Tensor]): dict, containing batch-version
                of the tensors.
        """
        result_batch = {}
        result_batch["data_object"] = torch.vstack(
            sum(
                [
                    split_and_pad_audio(elem["data_object"][0], max_lenth)
                    for elem in dataset_items
                ],
                [],
            )
        )
        result_batch["target"] = torch.vstack(
            sum(
                [
                    split_and_pad_audio(elem["target"][0], max_lenth)
                    for elem in dataset_items
                ],
                [],
            )
        )
        return result_batch

    return collate_fn
