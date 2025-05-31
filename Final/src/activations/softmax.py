from utils.e_cal import return_e

class Softmax:
    def forward(self, input_data):
        output = []
        
        e = return_e()
        
        for sample in input_data:    
            max_val = max(sample)
            exp_vals = [(e ** (x - max_val)) for x in sample]
            sum_exp = sum(exp_vals)
            softmax_vals = [exp_val / sum_exp for exp_val in exp_vals]
            output.append(softmax_vals)
        
        return output