import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sleep_scoring.helper.plots import plot_learning_curves, plot_confusion_matrix, print_metrics
from sleep_scoring.helper.utils import train, evaluate
from sleep_scoring.helper.feedbackloss import feedbackloss
from sleep_scoring.helper.dataloder import NonSeqDataLoader
from sleep_scoring.model.cnn import SimpleCNN
from optparse import OptionParser

# 针对网络结构固定且输入大小不变时，加快网络运行速度
torch.backends.cudnn.benchmark = True
# 固定网络随机初始化权重值
torch.manual_seed(777)
if torch.cuda.is_available():
    torch.cuda.manual_seed(777)

print('Cuda can be used:', torch.cuda.is_available())

path_output = "./output/bestmodels/"
os.makedirs(path_output, exist_ok=True)

n_below = 8  # n_below
n_above = 6  # 大循环，对应n
training_time = 105  # 修改
model_type = 'SimpleCNN'
batch_size = 128
learning_rate = 0.0002
if_balance = True  # 是否进行简单过采样

# 贝叶斯优化所获得的最佳dropout参数
alpha1 = 0.7322825703178968
alpha2 = 0.45379809990203457

CLASS_MAP = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,  # 深度睡眠I期和II期合并为深度睡眠期
    'Sleep stage R': 4,
}

class_names = ['W', '1', '2', '3', 'R']

if __name__ == "__main__":
    parser = OptionParser()
    # parser.add_option('-m', '--best_model', dest='best_model')
    parser.add_option('-m', '--best_model', dest='best_model', default=model_type)
    parser.add_option('-t', '--train', dest='train_model', default=model_type)
    parser.add_option('-p', '--plot', action="store_true", dest='plot', default=True)
    options, args = parser.parse_args(sys.argv)

    if options.train_model:
        model_type = options.train_model

    if options.best_model:
        model_type = options.best_model
        print("Running best model: ", options.best_model)

    data_loader = NonSeqDataLoader(
        data_dir='./data/eeg_fpz_cz',
        n_folds=14,
        fold_idx=6  # 此处选择第六折作为验证
    )
    x_train, y_train, x_valid, y_valid = data_loader.load_train_data(balance=if_balance)
    x_test, y_test = data_loader.load_test_data('./data/eeg_fpz_cz_test')

    if model_type == 'SimpleCNN':
        model = SimpleCNN(alpha1=alpha1, alpha2=alpha2)

    print(model)

    # 加载预训练模型
    model = torch.load(os.path.join(path_output, '{}.pth'.format(model_type)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 设置feedbackloss对应的初始权重
    weight = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)
    print(weight)
    FeedbackLoss = nn.CrossEntropyLoss(weight=weight)  # 可以给样本加权重，相当于惩罚机制
    FeedbackLoss.to(device)
    # Train
    if not options.best_model:
        train_losses, train_accuracies = [], []
        valid_losses, valid_accuracies = [], []
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        best_val_acc = 0.0

        for try_time in range(n_above):
            # Evaluate best model
            print("Evaluating model ...")
            if torch.cuda.is_available():
                best_model = torch.load(os.path.join(path_output, '{}.pth'.format(model_type)))
            else:
                best_model = torch.load(os.path.join(path_output, '{}.pth'.format(model_type)), map_location='cpu')
            valid_loss, valid_accuracy, valid_results = evaluate(best_model, device, x_valid, y_valid, FeedbackLoss,
                                                                 print_freq=0)
            FeedbackLoss = feedbackloss(valid_results)
            FeedbackLoss.to(device)
            print(
                "Training (model={}, epochs={}, batch_size={})".format(model_type, try_time, batch_size))
            for training_epoch in range(n_below):
                model.train()
                train_loss, train_accuracy = train(model, device, x_train, y_train, FeedbackLoss, optimizer,
                                                   training_epoch, print_freq=100)
                model.eval()
                valid_loss, valid_accuracy, valid_results = evaluate(model, device, x_valid, y_valid, FeedbackLoss,
                                                                     print_freq=0)
                print("valid_accuracy:", valid_accuracy)

                train_losses.append(train_loss)
                valid_losses.append(valid_loss)

                train_accuracies.append(train_accuracy)
                valid_accuracies.append(valid_accuracy)

                is_best = valid_accuracy > best_val_acc
                if is_best:
                    print("save:", training_epoch)
                    best_val_acc = valid_accuracy
                    torch.save(model, os.path.join(path_output, '{}.pth'.format(model_type)))

        if options.plot:
            plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, model_type)

    else:
        # Test best model
        print("Testing model ...")
        test_loss, test_accuracy, test_results = evaluate(model, device, x_test, y_test, FeedbackLoss, print_freq=0)

        y_true, y_pred = zip(*test_results)
        print_metrics(model_type + "_test", y_pred, y_true, save=options.plot)
