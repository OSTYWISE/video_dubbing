from itertools import repeat

import torch
from hydra.utils import instantiate

from src.datasets.collate import collate_fn_wrapper
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(
    config, device, index_dict: dict[str, list[dict[str, torch.Tensor]]]
):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
        index_dict: dict with keys in ["train", "val", "test"].
            Values are lists that contains dict with "path" and "target_path" for training objects
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets, index_dict=index_dict)  # TODO

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )
        collate_fn = collate_fn_wrapper(config.models.fc_input_lenght)

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            drop_last=(dataset_partition == "train"),
            shuffle=config.dataloader.shuffle
            if config.dataloader.shuffle is not None
            else (dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms


def split_and_pad_audio(audio: torch.Tensor, max_length: int):
    """
    Split audio into pieces of same length
    and pad the last one to make all element of the same shape

    Args:
        audio: 1-dimensional Tensor from audio
        max_length: spet of cutting/splitting the audio
    Returns:
        output: tuple of audio pieces (Tensors) shape of (max_length, )
    """
    total_length = audio.shape[0]
    audio_pieces = torch.split(audio, max_length)
    if total_length % max_length != 0:
        remaining_part = audio_pieces[-1]
        padded_chunk = torch.nn.functional.pad(
            remaining_part, (0, max_length - remaining_part.shape[0])
        )
        audio_pieces = audio_pieces[:-1] + (padded_chunk,)

    return list(audio_pieces)
