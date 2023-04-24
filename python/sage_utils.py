
import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical

import torch
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data import RandomSampler, BatchSampler
from copy import deepcopy
from tqdm.auto import tqdm

class MaskLayer1d(nn.Module):
    '''
    Masking for 1d inputs.

    Args:
      value: replacement value(s) for held out features.
      append: whether to append the mask along feature dimension.
    '''

    def __init__(self, value, append):
        super().__init__()
        self.value = value
        self.append = append

    def forward(self, input_tuple):
        x, S = input_tuple
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x


class MaskLayer2d(nn.Module):
    '''
    Masking for 2d inputs.

    Args:
      value: replacement value(s) for held out features.
      append: whether to append the mask along channels dimension.
    '''

    def __init__(self, value, append):
        super().__init__()
        self.value = value
        self.append = append

    def forward(self, input_tuple):
        x, S = input_tuple
        if len(S.shape) == 3:
            S = S.unsqueeze(1)
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x


class KLDivLoss(nn.Module):
    '''
    KL divergence loss that applies log softmax operation to predictions.

    Args:
      reduction: how to reduce loss value (e.g., 'batchmean').
      log_target: whether the target is expected as a log probabilities (or as
        probabilities).
    '''

    def __init__(self, reduction='batchmean', log_target=False):
        super().__init__()
        self.kld = nn.KLDivLoss(reduction=reduction, log_target=log_target)

    def forward(self, pred, target):
        '''
        Evaluate loss.

        Args:
          pred:
          target:
        '''
        return self.kld(pred.log_softmax(dim=1), target)


class DatasetRepeat(Dataset):
    '''
    A wrapper around multiple datasets that allows repeated elements when the
    dataset sizes don't match. The number of elements is the maximum dataset
    size, and all datasets must be broadcastable to the same size.

    Args:
      datasets: list of dataset objects.
    '''

    def __init__(self, datasets):
        # Get maximum number of elements.
        assert np.all([isinstance(dset, Dataset) for dset in datasets])
        items = [len(dset) for dset in datasets]
        num_items = np.max(items)

        # Ensure all datasets align.
        # assert np.all([num_items % num == 0 for num in items])
        self.dsets = datasets
        self.num_items = num_items
        self.items = items

    def __getitem__(self, index):
        assert 0 <= index < self.num_items
        return_items = [dset[index % num] for dset, num in
                        zip(self.dsets, self.items)]
        return tuple(itertools.chain(*return_items))

    def __len__(self):
        return self.num_items


class DatasetInputOnly(Dataset):
    '''
    A wrapper around a dataset object to ensure that only the first element is
    returned.

    Args:
      dataset: dataset object.
    '''

    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        return (self.dataset[index][0],)

    def __len__(self):
        return len(self.dataset)


class UniformSampler:
    '''
    For sampling player subsets with cardinality chosen uniformly at random.

    Args:
      num_players: number of players.
    '''

    def __init__(self, num_players):
        self.num_players = num_players

    def sample(self, batch_size):
        '''
        Generate sample.

        Args:
          batch_size:
        '''
        S = torch.ones(batch_size, self.num_players, dtype=torch.float32)
        num_included = (torch.rand(batch_size) * (self.num_players + 1)).int()
        # TODO ideally avoid for loops
        # TODO ideally pass buffer to assign samples in place
        for i in range(batch_size):
            S[i, num_included[i]:] = 0
            S[i] = S[i, torch.randperm(self.num_players)]

        return S


class ShapleySampler:
    '''
    For sampling player subsets from the Shapley distribution.

    Args:
      num_players: number of players.
    '''

    def __init__(self, num_players):
        arange = torch.arange(1, num_players)
        w = 1 / (arange * (num_players - arange))
        w = w / torch.sum(w)
        self.categorical = Categorical(probs=w)
        self.num_players = num_players
        self.tril = torch.tril(
            torch.ones(num_players - 1, num_players, dtype=torch.float32),
            diagonal=0)

    def sample(self, batch_size, paired_sampling):
        '''
        Generate sample.

        Args:
          batch_size: number of samples.
          paired_sampling: whether to use paired sampling.
        '''
        num_included = 1 + self.categorical.sample([batch_size])
        S = self.tril[num_included - 1]
        # TODO ideally avoid for loops
        for i in range(batch_size):
            if paired_sampling and i % 2 == 1:
                S[i] = 1 - S[i - 1]
            else:
                S[i] = S[i, torch.randperm(self.num_players)]
        return S

def validate(surrogate, loss_fn, data_loader):
    '''
    Calculate mean validation loss.

    Args:
      loss_fn: loss function.
      data_loader: data loader.
    '''
    with torch.no_grad():
        # Setup.
        device = next(surrogate.surrogate.parameters()).device
        mean_loss = 0
        N = 0

        for x, y, S in data_loader:
            x = x.to(device)
            y = y.to(device)
            S = S.to(device)
            pred = surrogate(x, S)
            loss = loss_fn(pred, y)
            N += len(x)
            mean_loss += len(x) * (loss - mean_loss) / N

    return mean_loss


def generate_labels(dataset, model, batch_size):
    '''
    Generate prediction labels for a set of inputs.

    Args:
      dataset: dataset object.
      model: predictive model.
      batch_size: minibatch size.
    '''
    with torch.no_grad():
        # Setup.
        preds = []
        if isinstance(model, torch.nn.Module):
            device = next(model.parameters()).device
        else:
            device = torch.device('cpu')
        loader = DataLoader(dataset, batch_size=batch_size)

        for (x,) in loader:
            pred = model(x.to(device)).cpu()
            preds.append(pred)

    return torch.cat(preds)


class Surrogate:
    '''
    Wrapper around surrogate model.

    Args:
      surrogate: surrogate model.
      num_features: number of features.
      groups: (optional) feature groups, represented by a list of lists.
    '''

    def __init__(self, surrogate, num_features, groups=None):
        # Store surrogate model.
        self.surrogate = surrogate

        # Store feature groups.
        if groups is None:
            self.num_players = num_features
            self.groups_matrix = None
        else:
            # Verify groups.
            inds_list = []
            for group in groups:
                inds_list += list(group)
            assert np.all(np.sort(inds_list) == np.arange(num_features))

            # Map groups to features.
            self.num_players = len(groups)
            device = next(surrogate.parameters()).device
            self.groups_matrix = torch.zeros(
                len(groups), num_features, dtype=torch.float32, device=device)
            for i, group in enumerate(groups):
                self.groups_matrix[i, group] = 1

    def train(self,
              train_data,
              val_data,
              batch_size,
              max_epochs,
              loss_fn,
              validation_samples=1,
              validation_batch_size=None,
              lr=1e-3,
              min_lr=1e-5,
              lr_factor=0.5,
              lookback=5,
              training_seed=None,
              validation_seed=None,
              bar=False,
              verbose=False):
        '''
        Train surrogate model.

        Args:
          train_data: training data with inputs and the original model's
            predictions (np.ndarray tuple, torch.Tensor tuple,
            torch.utils.data.Dataset).
          val_data: validation data with inputs and the original model's
            predictions (np.ndarray tuple, torch.Tensor tuple,
            torch.utils.data.Dataset).
          batch_size: minibatch size.
          max_epochs: maximum training epochs.
          loss_fn: loss function (e.g., fastshap.KLDivLoss).
          validation_samples: number of samples per validation example.
          validation_batch_size: validation minibatch size.
          lr: initial learning rate.
          min_lr: minimum learning rate.
          lr_factor: learning rate decrease factor.
          lookback: lookback window for early stopping.
          training_seed: random seed for training.
          validation_seed: random seed for generating validation data.
          verbose: verbosity.
        '''
        # Set up train dataset.
        if isinstance(train_data, tuple):
            x_train, y_train = train_data
            if isinstance(x_train, np.ndarray):
                x_train = torch.tensor(x_train, dtype=torch.float32)
                y_train = torch.tensor(y_train, dtype=torch.float32)
            train_set = TensorDataset(x_train, y_train)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be either tuple of tensors or a '
                             'PyTorch Dataset')

        # Set up train data loader.
        random_sampler = RandomSampler(
            train_set, replacement=True,
            num_samples=int(np.ceil(len(train_set) / batch_size))*batch_size)
        batch_sampler = BatchSampler(
            random_sampler, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler)

        # Set up validation dataset.
        sampler = UniformSampler(self.num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        # S_val = sampler.sample(len(val_data) * validation_samples)

        if isinstance(val_data, tuple):
            x_val, y_val = val_data
            S_val = sampler.sample(len(y_val) * validation_samples)
            if isinstance(x_val, np.ndarray):
                x_val = torch.tensor(x_val, dtype=torch.float32)
                y_val = torch.tensor(y_val, dtype=torch.float32)
            x_val_repeat = x_val.repeat(validation_samples, 1)
            y_val_repeat = y_val.repeat(validation_samples, 1)
            val_set = TensorDataset(
                x_val_repeat, y_val_repeat, S_val)
        elif isinstance(val_data, Dataset):
            S_val = sampler.sample(len(val_data) * validation_samples)
            val_set = DatasetRepeat([val_data, TensorDataset(S_val)])
        else:
            raise ValueError('val_data must be either tuple of tensors or a '
                             'PyTorch Dataset')

        if validation_batch_size is None:
            validation_batch_size = batch_size
        val_loader = DataLoader(val_set, batch_size=validation_batch_size)

        # Setup for training.
        surrogate = self.surrogate
        device = next(surrogate.parameters()).device
        optimizer = optim.Adam(surrogate.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)
        best_loss = validate(self, loss_fn, val_loader).item()
        best_epoch = 0
        best_model = deepcopy(surrogate)
        loss_list = [best_loss]
        if training_seed is not None:
            torch.manual_seed(training_seed)

        for epoch in range(max_epochs):
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader

            for x, y in batch_iter:
                # Prepare data.
                x = x.to(device)
                y = y.to(device)

                # Generate subsets.
                S = sampler.sample(batch_size).to(device=device)

                # Make predictions.
                pred = self.__call__(x, S)
                loss = loss_fn(pred, y)

                # Optimizer step.
                loss.backward()
                optimizer.step()
                surrogate.zero_grad()

            # Evaluate validation loss.
            self.surrogate.eval()
            val_loss = validate(self, loss_fn, val_loader).item()
            self.surrogate.train()

            # Print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Val loss = {:.4f}'.format(val_loss))
                print('')
            scheduler.step(val_loss)
            loss_list.append(val_loss)

            # Check if best model.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(surrogate)
                best_epoch = epoch
                if verbose:
                    print('New best epoch, loss = {:.4f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early')
                break

        # Clean up.
        for param, best_param in zip(surrogate.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        self.loss_list = loss_list
        self.surrogate.eval()

    def train_original_model(self,
                             train_data,
                             val_data,
                             original_model,
                             batch_size,
                             max_epochs,
                             loss_fn,
                             validation_samples=1,
                             validation_batch_size=None,
                             lr=1e-3,
                             min_lr=1e-5,
                             lr_factor=0.5,
                             lookback=5,
                             training_seed=None,
                             validation_seed=None,
                             bar=False,
                             verbose=False):
        '''
        Train surrogate model with labels provided by the original model.

        Args:
          train_data: training data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          val_data: validation data with inputs only (np.ndarray, torch.Tensor,
            torch.utils.data.Dataset).
          original_model: original predictive model (e.g., torch.nn.Module).
          batch_size: minibatch size.
          max_epochs: maximum training epochs.
          loss_fn: loss function (e.g., fastshap.KLDivLoss).
          validation_samples: number of samples per validation example.
          validation_batch_size: validation minibatch size.
          lr: initial learning rate.
          min_lr: minimum learning rate.
          lr_factor: learning rate decrease factor.
          lookback: lookback window for early stopping.
          training_seed: random seed for training.
          validation_seed: random seed for generating validation data.
          verbose: verbosity.
        '''
        # Set up train dataset.
        if isinstance(train_data, np.ndarray):
            train_data = torch.tensor(train_data, dtype=torch.float32)

        if isinstance(train_data, torch.Tensor):
            train_set = TensorDataset(train_data)
        elif isinstance(train_data, Dataset):
            train_set = train_data
        else:
            raise ValueError('train_data must be either tensor or a '
                             'PyTorch Dataset')

        # Set up train data loader.
        random_sampler = RandomSampler(
            train_set, replacement=True,
            num_samples=int(np.ceil(len(train_set) / batch_size))*batch_size)
        batch_sampler = BatchSampler(
            random_sampler, batch_size=batch_size, drop_last=True)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler)

        # Set up validation dataset.
        sampler = UniformSampler(self.num_players)
        if validation_seed is not None:
            torch.manual_seed(validation_seed)
        S_val = sampler.sample(len(val_data) * validation_samples)
        if validation_batch_size is None:
            validation_batch_size = batch_size

        if isinstance(val_data, np.ndarray):
            val_data = torch.tensor(val_data, dtype=torch.float32)

        if isinstance(val_data, torch.Tensor):
            # Generate validation labels.
            y_val = generate_labels(TensorDataset(val_data), original_model,
                                    validation_batch_size)
            y_val_repeat = y_val.repeat(
                validation_samples, *[1 for _ in y_val.shape[1:]])

            # Create dataset.
            val_data_repeat = val_data.repeat(validation_samples, 1)
            val_set = TensorDataset(val_data_repeat, y_val_repeat, S_val)
        elif isinstance(val_data, Dataset):
            # Generate validation labels.
            y_val = generate_labels(val_data, original_model,
                                    validation_batch_size)
            y_val_repeat = y_val.repeat(
                validation_samples, *[1 for _ in y_val.shape[1:]])

            # Create dataset.
            val_set = DatasetRepeat(
                [val_data, TensorDataset(y_val_repeat, S_val)])
        else:
            raise ValueError('val_data must be either tuple of tensors or a '
                             'PyTorch Dataset')

        val_loader = DataLoader(val_set, batch_size=validation_batch_size)

        # Setup for training.
        surrogate = self.surrogate
        device = next(surrogate.parameters()).device
        optimizer = optim.Adam(surrogate.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_factor, patience=lookback // 2, min_lr=min_lr,
            verbose=verbose)
        best_loss = validate(self, loss_fn, val_loader).item()
        best_epoch = 0
        best_model = deepcopy(surrogate)
        loss_list = [best_loss]
        if training_seed is not None:
            torch.manual_seed(training_seed)

        for epoch in range(max_epochs):
            # Batch iterable.
            if bar:
                batch_iter = tqdm(train_loader, desc='Training epoch')
            else:
                batch_iter = train_loader

            for (x,) in batch_iter:
                # Prepare data.
                x = x.to(device)

                # Get original model prediction.
                with torch.no_grad():
                    y = original_model(x)

                # Generate subsets.
                S = sampler.sample(batch_size).to(device=device)

                # Make predictions.
                pred = self.__call__(x, S)
                loss = loss_fn(pred, y)

                # Optimizer step.
                loss.backward()
                optimizer.step()
                surrogate.zero_grad()

            # Evaluate validation loss.
            self.surrogate.eval()
            val_loss = validate(self, loss_fn, val_loader).item()
            self.surrogate.train()

            # Print progress.
            if verbose:
                print('----- Epoch = {} -----'.format(epoch + 1))
                print('Val loss = {:.4f}'.format(val_loss))
                print('')
            scheduler.step(val_loss)
            loss_list.append(val_loss)

            # Check if best model.
            if val_loss < best_loss:
                best_loss = val_loss
                best_model = deepcopy(surrogate)
                best_epoch = epoch
                if verbose:
                    print('New best epoch, loss = {:.4f}'.format(val_loss))
                    print('')
            elif epoch - best_epoch == lookback:
                if verbose:
                    print('Stopping early')
                break

        # Clean up.
        for param, best_param in zip(surrogate.parameters(),
                                     best_model.parameters()):
            param.data = best_param.data
        self.loss_list = loss_list
        self.surrogate.eval()

    def __call__(self, x, S):
        '''
        Evaluate surrogate model.

        Args:
          x: input examples.
          S: coalitions.
        '''
        if self.groups_matrix is not None:
            S = torch.mm(S, self.groups_matrix)

        return self.surrogate((x, S))