# Vision_Transformers_for_Secular_Resonances
A repository of a Python code based on the Vision Transformer architecture (Dosovitskiy et al. 2020) used to classify images of resonant asteroids interacting with secular resonances.  The algorithm is described in Carruba et al. (2024), Vision Transformers for identifying asteroids interacting with secular resonances, Icarus, under review.  
You will need to install these Python libraries: numpy, pandas, tensorflow, PIL, sys, copy, time, datetime, tracemalloc.
To run the code, you will also need a sample of training, test, and validation images.  We provide a dataset for the nu6 secular resonance available at the link:

https://drive.google.com/file/d/1Tzd7k2VNH39FXeeMZMyQJg3Ir_ccWefj/view?usp=sharing

After unzip the file nu6_Vit.zip, please put the directories TEST, TRAINING, and VALIDATION in the same directory as the ViT code. You can then simply run the code using the command: 

python3 Vit_res_arg.py 

Outputs of the codes are a file nu6_pred_data.csv with the actual labels and predictions, a png file with the history of loss and accuracy of the model, and a file predicted_data.png with images of 50 resonant arguments for asteroids near the nu6 and their classifications. 
Questions about this code can be addressed at the MASB email address: mlasb2021@gmail.com.
