class Module(object):
    def __init__(self):
        super(Module, self).__init__()

    def forward(self, *args):
        raise NotImplementedError("Needs to implement a forward function")

    def __call__(self, *args):
        return self.forward(*args)
