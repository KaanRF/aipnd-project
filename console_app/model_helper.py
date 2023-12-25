import torch
from torch import nn
from torch import optim
from torchvision import models
import logging
import time


class ModelHelper:
    def __init__(self, arch, hidden_units):
        super().__init__()
        self._model = None
        self.arch = arch
        self.hidden_units = hidden_units

        if self.arch == 'vgg19':
            self._model = models.vgg19(pretrained=True)
        elif self.arch == 'alexnet':
            self._model = models.alexnet(pretrained=True)
        elif self.arch == 'resnet':
            self._model = models.resnet50(pretrained=True)

        # Build a feed-forward network
        self._model.classifier = nn.Sequential(nn.Linear(25088, self.hidden_units),
                                               nn.ReLU(),
                                               nn.Dropout(p=0.2),
                                               nn.Linear(self.hidden_units, 256),
                                               nn.ReLU(),
                                               nn.Dropout(p=0.2),
                                               nn.Linear(256, 128),
                                               nn.LogSoftmax(dim=1))

    def _freezer_paratmeter(self):
        for param in self._model.parameters():
            param.requires_grad = False

    def _get_available_device(self, is_gpu):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_gpu and device == "cpu":
            logging.warning("GPU is requested to be used, but it is not available on your system.")

        logging.warning("Using device is {}".format(device))

        return device

    def _get_criterion(self):
        return nn.NLLLoss()

    def _get_optimizer(self, learn_rate):
        return optim.Adam(self._model.classifier.parameters(), lr=learn_rate)

    def _run_train_model(self, train_loader, valid_loader, num_of_epochs, learn_rate, is_gpu):
        steps = 0
        total_train_loss = 0
        print_every = 15
        train_start_time = time.time()
        device = self._get_available_device(is_gpu)
        optimizer = self._get_optimizer(learn_rate)
        criterion = self._get_criterion()

        for epoch in range(num_of_epochs):
            for inputs, labels in train_loader:
                steps += 1
                # Move input and label tensors to the default device
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                logps = self._model.forward(inputs)
                loss = criterion(logps, labels)

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

                if steps % print_every == 0:
                    total_valid_loss = 0
                    valid_accuracy = 0
                    self._model.eval()  # Turn off dropout during testing and validation !.
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = self._model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            total_valid_loss += batch_loss.item()

                            # Calculate valid accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch + 1}/{num_of_epochs}.. "
                          f"Total train loss: {total_train_loss / print_every:.3f}.. "
                          f"Total validation loss: {total_valid_loss / len(valid_loader):.3f}.. "
                          f"Validation accuracy: {valid_accuracy / len(valid_loader):.3f}.. "
                          f"Device = {device}; Passed time since start : {(time.time() - train_start_time):.2f} seconds")
                    total_train_loss = 0
                    self._model.train()

    def get_model(self):
        return self._model

    def train_model(self, train_loader, valid_loader, is_gpu, num_of_epochs, learn_rate=0.003):
        self._freezer_paratmeter()

        self._model.to(self._get_available_device(is_gpu))

        self._run_train_model(train_loader, valid_loader, num_of_epochs, learn_rate, is_gpu)

    def test_model(self, test_loader, is_gpu):
        total_test_loss = 0
        test_accuracy = 0
        device = self._get_available_device(is_gpu)
        criterion = self._get_criterion()
        criterion = nn.NLLLoss()

        test_start_time = time.time()

        with torch.no_grad():
            self._model.eval()  # Turn off dropout during testing and validation !.
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                log_ps = self._model(images)
                total_test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                test_accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Set back model to train mode.
        self._model.train()

        test_accuracy_result = test_accuracy / len(test_loader)
        print("Total Test Loss: {:.3f}.. ".format(total_test_loss / len(test_loader)),
              "Test Accuracy: {:.3f}".format(test_accuracy_result))
        print(f"Device = {device}; Passed time since start testing : {(time.time() - test_start_time):.2f} seconds")

        if test_accuracy_result > 0.7:
            print("Accuracy higher than %70, your model is well trained...")

    def save_checkpoint(self, checkpoint_save_path, train_dataset, num_of_epochs, learn_rate):

        # See the actual status
        print("The state dict keys: \n\n", self._model.state_dict().keys())

        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'class_to_idx': train_dataset.class_to_idx,
            'epochs': num_of_epochs,
            'optimizer_state_dict': self._get_optimizer().state_dict(),
            'architecture': self.arch,
            'hidden_units': self.hidden_units,
        }

        checkpoint_file = self.arch + '_' + 'checkpoint.pth'
        torch.save(checkpoint, checkpoint_save_path + '/' + checkpoint_file)
