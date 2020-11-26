import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.activation = activation
        self.hidden_size = hidden_size
        self.m = nn.Linear(input_size, hidden_size)
        self.n = nn.Linear(hidden_size,hidden_size)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        if h==None:
            h=torch.zeros(self.hidden_size)
        h_seq = []

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   
        for i in range(x.shape[0]):
            if self.activation=='tanh':
                h_seq.append(torch.tanh(self.n(h)+self.m(x[i][0])))
            else:
                h_seq.append(torch.relu(self.n(h) + self.m(x[i][0])))
            h=h_seq[-1]
        h_seq=torch.stack(h_seq).reshape(x.shape)
        #h=h.reshape(1,)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.f_h=nn.Linear(hidden_size,hidden_size)
        self.f_x=nn.Linear(input_size,hidden_size)
        self.o_h=nn.Linear(hidden_size,hidden_size)
        self.o_x=nn.Linear(input_size,hidden_size)
        self.i_h=nn.Linear(hidden_size,hidden_size)
        self.i_x=nn.Linear(input_size,hidden_size)
        self.c_h=nn.Linear(hidden_size,hidden_size)
        self.c_x=nn.Linear(input_size,hidden_size)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################       


    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []


        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   
        if h==None:
            h=torch.zeros(self.hidden_size)
        if c==None:
            c=torch.zeros(self.hidden_size)

        for i in range(x.shape[0]):
            forget_gate = torch.sigmoid(self.f_x(x[i]) + self.f_h(h))
            input_gate = torch.sigmoid(self.f_x(x[i]) + self.f_h(h))
            output_gate = torch.sigmoid(self.f_x(x[i]) + self.f_h(h))
            c = forget_gate*c+input_gate*torch.tanh(self.c_x(x)+self.c_h(h))
            h=output_gate*torch.tanh(c)
            h_seq.append(h)

        h_seq=torch.stack(h_seq)#.reshape(x.shape)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
        return h_seq , (h, c)

