Code and data for the paper:

Convolutional-Recursive Deep Learning for 3D Object Classification
Richard Socher, Brody Huval, Bharath Bhat, Christopher D. Manning and Andrew Y. Ng
Advances in Neural Information Processing Systems (NIPS 2012)

This code is provided as is. It is free for academic, non-commercial purposes. 
For questions, please contact richard @ socher .org


Please cite the paper when you use this code:
@incollection{SocherEtAl2012:CRNN,
 title = {{Convolutional-Recursive Deep Learning for 3D Object Classification}},
 author = {{Richard Socher and Brody Huval and Bharath Bhat and Christopher D. Manning and Andrew Y. Ng}},
 booktitle = {{Advances in Neural Information Processing Systems 25}},
 year = {2012}
}


-------------------------------------------
Code
-------------------------------------------

Use the function

runCRNN 

to run the code. Parameters for the model, as well as a debugging option, 
can be found within initParams() along with their explanations.

The code also includes a copy of minFunc, a Matlab function for 
unconstrained optimization, created by Mark Schmidt. 
More information at http://www.di.ens.fr/~mschmidt/Software/minFunc.html.
See dataset to repeat the experiments of the paper.







-------------------------------------------
Data 
-------------------------------------------
The University of Washington RGB-D Object Dataset is required and can be 
downloaded from http://www.cs.washington.edu/rgbd-dataset/dataset/rgbd-dataset/ . 
Once downloaded, extract the rgbd-dataset into the 'data' folder. 

The code takes every 5th frame from the dataset. It will ignore any 
images which do not contain the crop, depthcrop, 
and maskcrop PNG files in the dataset.




