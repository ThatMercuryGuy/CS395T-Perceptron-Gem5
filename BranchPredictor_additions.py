class PerceptronBP(ConditionalPredictor):
    type = "PerceptronBP"
    cxx_class = "gem5::branch_prediction::PerceptronBP"
    cxx_header = "cpu/pred/perceptron_pred.hh"

    globalPredictorSize = Param.Unsigned(128, "N")
    globalHistoryLength = Param.Unsigned(28, "h")
    threshold = Param.Int(68, "theta")
    weightWidth = Param.Unsigned(8, "bits")

class Perceptron8KB(PerceptronBP):
    globalPredictorSize = 208
    globalHistoryLength = 34
    threshold = 79 # 1.93 * 34 + 14
    weightWidth = 9

class Perceptron16KB(PerceptronBP):
    globalPredictorSize = 393
    globalHistoryLength = 36
    threshold = 83 # 1.93 * 36 + 14
    weightWidth = 9

class Perceptron32KB(PerceptronBP):
    globalPredictorSize = 485
    globalHistoryLength = 59
    threshold = 127 # 1.93 * 59 + 14
    weightWidth = 9

# Baseline: The optimal theta recommended by the paper
class Perceptron4KB_Theta68(PerceptronBP):
    globalPredictorSize = 128
    globalHistoryLength = 28
    threshold = 68
    weightWidth = 8

# Low Theta: Perceptrons stop training very early
class Perceptron4KB_Theta20(PerceptronBP):
    globalPredictorSize = 128
    globalHistoryLength = 28
    threshold = 20
    weightWidth = 8

# High Theta: Perceptrons continue training for a very long time
class Perceptron4KB_Theta150(PerceptronBP):
    globalPredictorSize = 128
    globalHistoryLength = 28
    threshold = 150
    weightWidth = 8
