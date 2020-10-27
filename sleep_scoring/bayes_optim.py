import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sleep_scoring.helper.plots import plot_learning_curves_new, plot_confusion_matrix, print_metrics
from sleep_scoring.helper.utils import train, evaluate
from sleep_scoring.model.cnn import SimpleCNN
from optparse import OptionParser
from bayes_opt import BayesianOptimization
from sleep_scoring.helper.dataloder import NonSeqDataLoader


torch.manual_seed(777)
if torch.cuda.is_available():
    torch.cuda.manual_seed(777)

print('Cuda can be used:', torch.cuda.is_available())


# Path for saving model
path_output = "./output/bestmodels/"
os.makedirs(path_output, exist_ok=True)

num_training_epoch = 2
model_type = 'SimpleCNN'
batch_size = 128
lr=0.001
num_workers = 0

CLASS_MAP = {
    'Sleep stage W' : 0,
    'Sleep stage 1' : 1,
    'Sleep stage 2' : 2,
    'Sleep stage 3' : 3,
    'Sleep stage 4' : 3, # Map stage 4 to 3
    'Sleep stage R' : 4,
}

def black_box_function(alpha2,alpha3):
    MODEL_TYPE = 'SimpleCNN'

    if MODEL_TYPE == 'SimpleCNN':
        model = SimpleCNN(alpha1=alpha2, alpha2=alpha3)
    else:
        raise AssertionError('Model type does not exist')

        print(model)
    MODEL_TYPE = MODEL_TYPE + str(alpha2)+ "_"+ str(alpha3)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    criterion.to(device)

    # Train
    if not options.best_model:
        print("Training (model={}, epochs={}, batch_size={})".format(MODEL_TYPE, num_training_epoch, batch_size))

        best_val_acc = 0.0
        train_losses, train_accuracies = [], []
        valid_losses, valid_accuracies = [], []
        for training_epoch in range(num_training_epoch):
            model.train()
            train_loss, train_accuracy = train(model, device, x_train, y_train, criterion, optimizer, training_epoch, print_freq=100)
            model.eval()
            valid_loss, valid_accuracy, valid_results = evaluate(model, device, x_valid, y_valid, criterion, print_freq=0)

            print("valid_accuracy:", valid_accuracy)

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            train_accuracies.append(train_accuracy)
            valid_accuracies.append(valid_accuracy)

            is_best = valid_accuracy > best_val_acc
            if is_best:
                print("save:",training_epoch)
                best_val_acc = valid_accuracy
                torch.save(model, os.path.join(path_output, '{}.pth'.format(MODEL_TYPE)))
        torch.save(model, os.path.join(path_output, '{}.pth'.format(MODEL_TYPE + "last")))
        if options.plot:
            plot_learning_curves_new(train_losses, valid_losses, test_losses, train_accuracies, valid_accuracies, test_accuracies, MODEL_TYPE)
    class_names = ['W', '1', '2', '3','R']

    if torch.cuda.is_available():
        best_model = torch.load(os.path.join(path_output, '{}.pth'.format(MODEL_TYPE)))
    else:
        best_model = torch.load(os.path.join(path_output, '{}.pth'.format(MODEL_TYPE)), map_location='cpu')

    # Test best model
    print("Testing model ...")
    test_loss, test_accuracy, test_results = evaluate(best_model, device, x_test, y_test, criterion, print_freq=0)
    if options.plot:
        plot_confusion_matrix(test_results, class_names, MODEL_TYPE + "_test")

    y_true, y_pred = zip(*test_results)
    print_metrics(MODEL_TYPE + "_test", y_pred, y_true, save=options.plot)

    return valid_accuracy


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = OptionParser()
    parser.add_option('-m', '--best_model', dest='best_model')
    # parser.add_option('-m', '--best_model', dest='best_model', default=MODEL_TYPE)
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
        n_folds= 14,
        fold_idx= 10
    )
    x_train, y_train, x_valid, y_valid = data_loader.load_train_data()
    # x_test, y_test = data_loader.load_test_data('./data/eeg_fpz_cz_test')
    x_test=x_valid
    y_test=y_valid

    # Bounded region of parameter space
    pbounds = {'alpha2': (0.3, 0.8),'alpha3': (0.3, 0.8)}

    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds
    )

    optimizer.maximize(
        init_points=5,
        n_iter=16
    )
    print(optimizer.max)



