class LinearRegresssion:

    def __init__(self, vars, outcome, learning_rate, gen):

        #learning rate coefficient
        self.a = learning_rate
        #define generations
        self.gen = gen

        #Adds pseudo-row for calculating the intercept
        vars.insert(0, [1 for _ in range(len(vars[0]))])

        #Matrices representing training set
        self.x = [*zip(*vars)]
        self.y = outcome
        #Initialize weights
        self.theta = [0.5 for _ in range(len(vars))]

        self.train_stochastic_gradient()

    def train_stochastic_gradient (self):
        for k in range(self.gen):
            for i in range(len(self.y)):
                self.theta = [self.theta[j] + self.x[i][j]*self.a*(self.y[i]-self.h(self.x[i])) for j in range(len(self.theta))]

    def C(self):
        return sum([(self.h(self.x[i]) - self.y[i])**2 for i in range(len(self.y))])/2

    def h(self, x):
        return sum([self.theta[i]*x[i] for i in range(len(x))])
