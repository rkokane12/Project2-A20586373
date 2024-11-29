Rudraksha Ravindra Kokane - A20586373 - rkokane@hawk.iit.edu

Find the implementation at https://colab.research.google.com/drive/15-QXYFkyuX9VAvN2051hYCnZK1meVT94?usp=sharing

The Project implemented a custom Gradient Boosting algorithm for regression task of Mahcine Learning. It use sklearn Decision Tree Regression as weak learners. The algorithm is based on Element of Statistical Learning 2nd Edition (Section 10.1 and 10.2) that is additive model. 

About Model:

The Model:
The pseudocode for Element of Statistical Learning Edition 2 Section 10.2
Input: Training data {(x_i, y_i)}^N_i=1, number of iterations M, learning rate ν

1. Initialize f_0(x) = argmin_γ Σ_i^N L(y_i, γ)
   
2. For m = 1 to M:
   a. Compute pseudo-residuals:
      r_im = -[∂L(y_i, f(x_i))/∂f(x_i)]_{f=f_{m-1}} for i = 1, ..., N
      
   b. Fit a base learner (regression tree) h_m(x) to pseudo-residuals {(x_i, r_im)}^N_i=1
   
   c. Compute multiplier γ_m:
      γ_m = argmin_γ Σ_i^N L(y_i, f_{m-1}(x_i) + γh_m(x_i))
      
   d. Update the model:
      f_m(x) = f_{m-1}(x) + ν * γ_m * h_m(x)

3. Output final model:
   f_M(x) = f_0(x) + ν * Σ_{m=1}^M γ_m * h_m(x)



What it does?
The model is example of Gradient Boost Regression, that is ensemble approach that builds series of weak learning decision trees. It is additive model, that is it adds current prediction to ensemble of previous trees.
Using many weak learners results into correction of error of previous tree, by the new one.
It combines all the predictions from each tree, to make final prediction.

What to use it for?
It is regression model, thus should be used for regression tasks in Machine learning.
It is time consuming than Linear Models and thus should be used for data with complex relation between data.
It should be used only when prediction accuracy of weak learners is not high without any overfitting.


How did I ensure it worked correctly?
I used Decision tree as weak Learner from Scikit Learn. 
Also, I compared it against Decision tree regression on California Housing Dataset, and the model gave better accuracy. 
The notebook also shows the plots for true values against predicted ones.
I also implemented Hyper Parameter Tuning in models
Source for its study: https://github.com/scikit-learn/scikit-learn/blob/6e9039160/sklearn/base.py#L227 

Parameters Exposed to user for Tuning?
The fit and predict methods have obvious parameters of Input data required to training and testing.
For fit, it is training_data_values and training_target_values
For predict, it is just the testing_data_values

Other than these, for tuning the model to fit the data correctly, the parameters are:
num_trees: default value 100
		To achieve better results, its values should be in range from 10 to 1000
		Value 100 is choosen as it avoids overfitting, and also gives better accuracy
learn_rate: it is learningrate for gradient descent. default values is 0.1. 
		The values should be in range 0.001 to 0.5 
tree_depth: It gives the maximum depth an individual tree should have. Default values is 3
		The maximum values would be better considered for number of feature in data, but range from 2 to 7 is what I maintain to avoid oversimplification and also 		case of overfitting
L2reg: It is regularisation parameter. The algorithm follows l2 regularization. Default Values: 0.01
	Values should range from 0.0001 to 0.1


Troubles with current implementation:
The algorithm is inefficient with larger dataset, like Covid-19 case dataset, which has 58000+ records. The algorithm worked well with dataset with around 30000 records.
Use of efficient data-structures and numpy methods and better hardware may provide better results. 
The limitations can be addressed by batch processing, or introducing High Performance Computing features.
We can also subsampling in fit method to overcome the limitations. With sufficient time, these issues can be addressed  
