from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        "*** YOUR CODE HERE ***"

        # debug print the instance of data, limit to 5
        # count = 0
        # for data in dataloader:
        #     if count <= 5:
        #         count+=1
        #         print(data)
        #     else:
        #         break

        # inter the loop and update the weight
        # train the model until the coverage is pass (here is 100% accuracy)
        while True:
            mistake = 0

            for data in dataloader:
                pred = model.get_prediction(data['x'])
                learning_rate = 0.5

                error = data['label'] - pred

                if error != 0:
                    mistake += 1
                    # must *data['x'] because, x give the gradient
                    model.w += error * data['x'] * learning_rate

                    # debug 
                    # print("y_pred=", pred)
                    # print("y-y_pred=", error)
                    # print("weight=", model.w)

            if mistake == 0:
                break
            else:
                print("model_acc= ", 1-(mistake/len(dataloader)))

            



def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"

    # print(len(dataset)) 200

    # this is the best hyper params that reach the loss of 0.02 fastest (least iters)
    learning_rate = 0.01 # 0.00001 0.0005 0.0015 0.004
    batch_size = 10 # 100, 5
    epochs = 500 # 100, 1000

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

    # debug print the instance of batch, limit to 5
    # count = 0
    # for batch in dataloader:
    #     if count <= 5:
    #         count+=1
    #         print("batch=", batch)
    #     else:
    #         break

    optimizer = optim.SGD(model.parameters(), learning_rate)

    for _ in range (1, epochs): 
        for batch in dataloader:
            y_pred = model(batch['x'])
            loss = regression_loss(y_pred, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("y_pred, y= {} {}", y_pred, batch['label'])
        
        # debug print loss fl
        # print(loss)


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """
    # target accuracy at least 97%
    learning_rate = 0.01 #
    batch_size = 5 # 
    epochs = 50 # 500, 20

    train_set = DataLoader(dataset=dataset, batch_size=batch_size)

    # debug print the instance of batch, limit to 5
    # count = 0
    # for batch in train_set:
    #     if count <= 5:
    #         count+=1
    #         print("batch=", batch)
    #     else:
    #         break

    optimizer = optim.SGD(model.parameters(), learning_rate)

    for epoch in range (1, epochs): 
        for batch in train_set:
            y_pred = model(batch['x'])
            # using mse in regression loss also work but it take much longer to train, weird???
            # adding this to documents
            # loss = regression_loss(y_pred, batch['label'])
            loss = digitclassifier_loss(y_pred, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("y_pred, y= {} {}", y_pred, batch['label'])
        
        # debug print loss fl
        print("loss=", loss)
        print("accuracy= ", dataset.get_validation_accuracy())

        # implement early stopping if accuracy surpass 97%
        if dataset.get_validation_accuracy() >= 0.97:
            print("train stop at epoch=", epoch)
            break



def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"
    import random   # có thể đưa lên đầu file train.py

    learning_rate = 0.001
    batch_size = 64
    max_epochs = 30
    target_accuracy = 0.86

    train_set = DataLoader(dataset=dataset, batch_size=batch_size)
    # Mỗi batch cùng độ dài từ -> giữ nguyên batch, chỉ xáo thứ tự các batch
    batches = [(movedim(b['x'], 0, 1), b['label']) for b in train_set]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        random.shuffle(batches)
        total_loss = 0.0
        for x, y in batches:
            y_pred = model(x)
            loss = languageid_loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        accuracy = dataset.get_validation_accuracy()
        print("epoch", epoch, "avg loss=", total_loss / len(batches), "val acc=", accuracy)

        # Lưu lại trọng số tốt nhất trên validation
        if accuracy > best_acc:
            best_acc = accuracy
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if accuracy >= target_accuracy:
            break

    # Khôi phục mô hình tốt nhất (tránh trường hợp epoch cuối kém hơn)
    if best_state is not None:
        model.load_state_dict(best_state)



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
    model.train()
    # target accuracy at least 97%
    learning_rate = 0.01 #
    batch_size = 5 # 
    epochs = 50 # 

    train_set = DataLoader(dataset=dataset, batch_size=batch_size)

    # debug print the instance of batch, limit to 5
    # count = 0
    # for batch in train_set:
    #     if count <= 5:
    #         count+=1
    #         print("batch=", batch)
    #     else:
    #         break

    optimizer = optim.SGD(model.parameters(), learning_rate)

    for epoch in range (1, epochs): 
        for batch in train_set:
            y_pred = model(batch['x'])
            loss = digitconvolution_Loss(y_pred, batch['label'])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print("y_pred, y= {} {}", y_pred, batch['label'])
        
        # debug print loss fl
        print("loss=", loss)
        print("accuracy= ", dataset.get_validation_accuracy())

        # implement early stopping if accuracy surpass 97%
        if dataset.get_validation_accuracy() >= 0.97:
            print("train stop at epoch=", epoch)
            break
