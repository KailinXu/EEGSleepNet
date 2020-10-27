import os
import sys
import numpy as np
# from imblearn.under_sampling import RandomUnderSampler # 欠抽样处理库RandomUnderSampler
# from imblearn.combine import SMOTETomek
# from imblearn.over_sampling import SMOTE
import re

class_dict = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM"
}


# model_RandomUnderSampler = RandomUnderSampler()  # 建立RandomUnderSampler模型对象
# smote_enn = SMOTETomek(random_state=0)
# smote_enn = SMOTE(random_state=0)

def get_balance_class_oversample(x, y):
    """
    Balance the number of samples of all classes by (oversampling):
        1. Find the class that has the largest number of samples
        2. Randomly select samples in each class equal to that largest number
    """
    class_labels = np.unique(y)
    n_max_classes = -1
    for c in class_labels:
        n_samples = len(np.where(y == c)[0])
        if n_max_classes < n_samples:
            n_max_classes = n_samples

    balance_x = []
    balance_y = []
    for c in class_labels:
        idx = np.where(y == c)[0]
        n_samples = len(idx)
        n_repeats = int(n_max_classes / n_samples)
        tmp_x = np.repeat(x[idx], n_repeats, axis=0)
        tmp_y = np.repeat(y[idx], n_repeats, axis=0)
        n_remains = n_max_classes - len(tmp_x)
        if n_remains > 0:
            sub_idx = np.random.permutation(idx)[:n_remains]
            tmp_x = np.vstack([tmp_x, x[sub_idx]])
            tmp_y = np.hstack([tmp_y, y[sub_idx]])
        balance_x.append(tmp_x)
        balance_y.append(tmp_y)
    balance_x = np.vstack(balance_x)
    balance_y = np.hstack(balance_y)

    return balance_x, balance_y


def print_n_samples_each_class(labels):
    import numpy as np
    unique_labels = np.unique(labels)
    for c in unique_labels:
        n_samples = len(np.where(labels == c)[0])
        print("{}: {}".format(class_dict[c], n_samples))


class NonSeqDataLoader(object):

    def __init__(self, data_dir, n_folds, fold_idx):
        self.data_dir = data_dir
        self.n_folds = n_folds
        self.fold_idx = fold_idx

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            print(("Loading {} ...".format(npz_f)))
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")
            data.append(tmp_data)
            labels.append(tmp_labels)
        data = np.vstack(data)
        labels = np.hstack(labels)
        return data, labels

    def _load_cv_data(self, list_files):
        """Load training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print(" ")

        # Reshape the data to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_train, label_train, data_val, label_val

    def load_train_data(self, n_files=None, balance=True):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(self.data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(self.data_dir, f))
        npzfiles.sort()

        if n_files is not None:
            npzfiles = npzfiles[:n_files]

        # subject_files = npzfiles[(self.fold_idx * int(len(npzfiles) / self.n_folds)):((self.fold_idx + 1) * int(len(npzfiles) / self.n_folds))]
        if self.fold_idx < 13:
            subject_files = npzfiles[(self.fold_idx * int(len(npzfiles) / self.n_folds)):(
                    (self.fold_idx + 1) * int(len(npzfiles) / self.n_folds))]
        elif self.fold_idx == 13:
            subject_files = npzfiles[(self.fold_idx * int(len(npzfiles) / self.n_folds)):(
                    self.fold_idx * int(len(npzfiles) / self.n_folds) + 1)]
        else:
            subject_files = npzfiles[(self.fold_idx * int(len(npzfiles) / self.n_folds) - 1):(
                    (self.fold_idx + 1) * int(len(npzfiles) / self.n_folds) - 1)]

        # for idx, f in enumerate(allfiles):
        #     if self.fold_idx < 10:
        #         pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
        #     else:
        #         pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
        #     if pattern.match(f):
        #         subject_files.append(os.path.join(self.data_dir, f))
        #
        # if len(subject_files) == 0:
        #     for idx, f in enumerate(allfiles):
        #         if self.fold_idx < 10:
        #             pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]J0\.npz$".format(self.fold_idx))
        #         else:
        #             pattern = re.compile("[a-zA-Z0-9]*{}[1-9]J0\.npz$".format(self.fold_idx))
        #         if pattern.match(f):
        #             subject_files.append(os.path.join(self.data_dir, f))

        train_files = list(set(npzfiles) - set(subject_files))
        train_files.sort()
        subject_files.sort()

        # Load training and validation sets
        print(("\n========== [Fold-{}] ==========\n".format(self.fold_idx)))
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(npz_files=train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(npz_files=subject_files)
        print(" ")

        # Reshape the data to match the input of the model - conv2d
        data_train = np.squeeze(data_train)
        data_val = np.squeeze(data_val)
        data_train = data_train[:, :, np.newaxis, np.newaxis]
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_train = data_train.astype(np.float32)
        label_train = label_train.astype(np.int32)
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        print(("Training set: {}, {}".format(data_train.shape, label_train.shape)))
        print_n_samples_each_class(label_train)
        print(" ")
        print(("Validation set: {}, {}".format(data_val.shape, label_val.shape)))
        print_n_samples_each_class(label_val)
        print(" ")

        if balance:
            # Use balanced-class, oversample training set
            x_train, y_train = get_balance_class_oversample(
                x=data_train, y=label_train
            )
            # # 采样
            # data_train = data_train.reshape(len(data_train), 3000)
            # x_train, y_train = smote_enn.fit_sample(
            #     data_train,
            #     label_train)  # 输入数据并作过采样处理

        else:
            x_train = data_train
            y_train = label_train
        print(("Oversampled training set: {}, {}".format(
            x_train.shape, y_train.shape
        )))
        print_n_samples_each_class(y_train)
        print(" ")

        return x_train, y_train, data_val, label_val

    def load_test_data(self, data_dir, num=0.1):
        # Remove non-mat files, and perform ascending sort
        allfiles = os.listdir(data_dir)
        npzfiles = []
        for idx, f in enumerate(allfiles):
            if ".npz" in f:
                npzfiles.append(os.path.join(data_dir, f))
        npzfiles.sort()
        if (num == 0.1):
            subject_files = npzfiles[:]
        else:
            subject_files = npzfiles[num:num + 1]
        # for idx, f in enumerate(allfiles):
        #     if self.fold_idx < 10:
        #         pattern = re.compile("[a-zA-Z0-9]*0{}[1-9]E0\.npz$".format(self.fold_idx))
        #     else:
        #         pattern = re.compile("[a-zA-Z0-9]*{}[1-9]E0\.npz$".format(self.fold_idx))
        #     if pattern.match(f):
        #         subject_files.append(os.path.join(data_dir, f))
        subject_files.sort()

        print(("\n========== [Fold-{}] ==========\n".format(self.fold_idx)))

        print("Load test set:")
        data_val, label_val = self._load_npz_list_files(subject_files)
        print(("Test set: {}, {}".format(data_val.shape, label_val.shape)))
        print_n_samples_each_class(label_val)
        print(" ")
        # Reshape the data to match the input of the model
        data_val = np.squeeze(data_val)
        data_val = data_val[:, :, np.newaxis, np.newaxis]

        # Casting
        data_val = data_val.astype(np.float32)
        label_val = label_val.astype(np.int32)

        return data_val, label_val
