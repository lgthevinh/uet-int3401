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
    with no_grad():
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        while True:
            converged = True  # Giả sử mô hình đã hội tụ
            
            for batch in dataloader:
                x = batch['x']     # Lấy vector đặc trưng
                y = batch['label'] # Lấy nhãn thực tế (+1 hoặc -1)
                
                # Dự đoán nhãn của mô hình
                prediction = model.get_prediction(x)
                
                # Nếu dự đoán sai so với nhãn thực tế
                if prediction != y.item():
                    # Cập nhật trọng số theo quy tắc Perceptron: w <- w + y * x
                    # y có dạng tensor/scalar, x có dạng tensor đặc trưng
                    model.w.data += y * x
                    converged = False  # Vẫn còn mẫu dự đoán sai -> chưa hội tụ
            
            # Nếu trong 1 epoch toàn bộ dữ liệu mà không có mẫu nào sai -> hội tụ, dừng huấn luyện
            if converged:
                break


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
    # 1. Chuyển mô hình sang chế độ training
    model.train()
    
    # 2. Khởi tạo DataLoader để chia dữ liệu thành các mini-batch (batch_size=32, trộn ngẫu nhiên dữ liệu mỗi epoch)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. Sử dụng bộ tối ưu hóa Adam với learning rate phù hợp (ví dụ: lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4. Vòng lặp huấn luyện cho đến khi đạt loss thỏa mãn yêu cầu (<= 0.02)
    while True:
        for batch in dataloader:
            x = batch['x']     # Tensor đặc trưng (batch_size x 1)
            y = batch['label'] # Tensor nhãn thực tế (batch_size x 1)

            # Xóa gradient tích lũy từ bước trước
            optimizer.zero_grad()
            
            # Lan truyền tiến (Forward pass) để dự đoán giá trị
            y_pred = model(x)
            
            # Tính toán hàm mất mát (Square Loss / MSE Loss)
            loss = regression_loss(y_pred, y)
            
            # Lan truyền ngược (Backward pass) để tính gradient
            loss.backward()
            
            # Cập nhật trọng số của mạng nơ-ron
            optimizer.step()

        # 5. Kiểm tra giá trị mất mát trên toàn bộ tập dữ liệu sau mỗi epoch
        with no_grad():
            full_x = tensor(dataset.x).float()
            full_y = tensor(dataset.y).float()
            current_loss = regression_loss(model(full_x), full_y).item()
            
            # Nếu loss đạt tiêu chuẩn (<= 0.02), dừng huấn luyện
            if current_loss <= 0.02:
                break


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """


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



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
