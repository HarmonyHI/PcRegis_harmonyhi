class IOStream:
    def __init__(self, path):
        self.f =open(path, 'a')

    def cprint(self, text):
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

    def print(self, text):
        self.f.write(text + " ")
        self.f.flush()

    def ln(self):
        self.f.write('\n')
        self.f.flush()

    def println(self, text):
        self.f.write(text + '\n')
        self.f.flush()