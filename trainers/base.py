from abc import abstractmethod

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.adpt_data import DataLoaderAdapter
from utils.checkpoint import save_ckpt
from utils.tools import DEVICE


class BaseTrainer:
    def __init__(self, args, exp_dir, writer, data_info, data_adptr):
        r"""

        Args:
            args (argparse.Namespace): Command line arguments.
            exp_dir (str): Directory to save checkpoints and logs.
            writer (SummaryWriter): Tensorboard summary writer.
            data_info (dict): Data Meta Information Dictionary.
            data_adptr (DataLoaderAdapter):

        """
        self.args = args
        self.exp_dir = exp_dir
        self.writer = writer
        self.device = DEVICE

        self.data_info = data_info
        self.data_adptr = data_adptr
        self.model = None
        self.model_params = None

        self._reset = args.reset
        self._epochs = args.epochs

        self.l_indices = None
        self.u_indices = None

    @abstractmethod
    def _build_net(self):
        r"""Build the model.

        """
        pass

    @abstractmethod
    def _train_epoch(self, epoch, ldr_l, ldr_u=None):
        r"""Abstract method to perform training for a single epoch.

        Args:
            epoch: The current epoch number.
            ldr_l (DataLoader): A PyTorch DataLoader object for the labeled dataset.
            ldr_u (Optional[DataLoader]): A PyTorch DataLoader object for the unlabeled dataset.

        Returns:
            dict: A dictionary containing relevant training statistics for the current epoch.
            The keys and values in this dictionary can vary depending on the specific implementation
            but might include metrics such as loss values, accuracy scores, etc.

        """
        pass

    @abstractmethod
    def _test_epoch(self, cycle, ldr_te):
        r"""Abstract method to perform testing for a single cycle.

        Args:
            cycle (int): Current active learning cycle.
            ldr_te (DataLoader): A PyTorch DataLoader object for the test dataset.

        Returns:
            dict: A dictionary containing relevant testing statistics for the current cycle.
            The keys and values in this dictionary can vary depending on the specific implementation
            but might include metrics such as loss values, accuracy scores, etc.

        """
        pass

    def get_model(self):
        r"""Get the trained model.

        The return value of the method is the trained main model and auxiliary models.

        """
        return self.model

    def train(self, cycle, l_indices, u_indices=None, v_indices=None, p_indices=None):
        r"""Train the model.

        Args:
            cycle (int): Current active learning cycle.
            l_indices (np.ndarray): The index of the labeled data.
            u_indices (Optional[np.ndarray]): The index of the unlabeled data.
            v_indices (Optional[np.ndarray]): The index of the validation data.
            p_indices (Optional[np.ndarray]): The index of the pre-sampling unlabeled data.

        """
        self.l_indices = l_indices
        self.u_indices = u_indices

        ldr_l = self.data_adptr.get_loader_train_labeled(l_indices)
        ldr_v = self.data_adptr.get_loader_valid(v_indices) if v_indices is not None else None

        self._build_net()

        for epoch in range(1, 1 + self._epochs):
            tr_tag_scalar_dict = self._train_epoch(epoch - 1, ldr_l)
            if self.writer is not None:
                for tag, value in tr_tag_scalar_dict.items():
                    self.writer.add_scalar(f'{tag}_epoch/train_cyc{cycle}', value, epoch)

    def test(self, cycle):
        r"""Test the model.

        Args:
            cycle (int): Current active learning cycle.

        """
        ldr_te = self.data_adptr.get_loader_test()

        tag_scalar_dict = self._test_epoch(cycle, ldr_te)
        if self.writer is not None:
            for tag, value in tag_scalar_dict.items():
                self.writer.add_scalar(f'{tag}_cycle/test', value, cycle)

        return tag_scalar_dict
