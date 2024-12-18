## Accounting 

This directory adds the ability to account for the RDP Optimized mechanism under composition. 
Requires the dp_accounting library from google, which can be installed via 
```
pip install dp_accounting
```
You will also need to add ```Discrete_RDP_Mechanism.py``` to your python path, since it contains a child of the ```AdditiveNoisePrivacyLoss``` 
class, which is the only thing Connect the Dots needs in order to compose. 

## Common Gotchas
The ```sensitivity``` parameter in the code is an integer shift that corresponds to shifting the RDP Optimized mechanism by ```sensitivity``` bins. 
So, in order to compare to a Gaussian with sensitivity 1, you need to set the ```sensitivity``` parameter to $k$. In general, to compare to a typical 
DP mechanism with sensitivity $s$, you need to set the ```sensitivity``` parameter to $s * k$. 
## How does this work?

```Discrete_RDP_Mechanism.py``` contains a function called ```get_delta_for_epsilon_shift```, which takes as input a list of epsilons
and an integer bin shift, and outputs the corresponding deltas. The function ```get_delta_for_epsilon``` is just a for loop which takes as
input a list of epsilons, and takes the maximum over all deltas corresponding to shifts less than or equal to ```sensitivity```. Connect the 
dots uses this function, i.e. ```get_delta_for_epsilon```, to construct a dominating PLD. 

The only other part left to explain in the code is how we tell Connect the Dots which region of epsilons to construct the PLD over. This is handeled
by two functions ```privacy_loss_tail``` and ```connect_dots_bounds```. I will add an explanation on how these functions work later.

## Subsampling
Technically this code can also account for the RDP Optimized mechanism under composition  with Poisson subsampling via the function ```_pld_for_subsampled_mechanism```
, but I (Felipe) have not tested it. For now, please do not subsample. If you simply set the subsampling paraemter to 1, the function
```_pld_for_subsampled_mechanism``` does not do anything (i.e. it just returns the input pld). 
