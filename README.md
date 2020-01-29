# FER-global-local-network
Facial expression recognition
![global-local-Network](https://github.com/yummy2108/FER-global-local-network/blob/master/pipeline.png)

## Download data
Download CK+ and Oulu-CASIA databases from network   

CK+ : http://www.pitt.edu/~emotion/ck-spread.htm  

Oulu : https://www.oulu.fi/cmvs/node/41316

## Preprocess datasets
The main codes are in 'Data_preprocess' directory.
1. We use Dlib tools to detect landmarks and extract the eyes, nose, mouth regions. 
Download the dat file from 
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 
and unzip the shape_predictor_68_face_landmarks.dat to 'Data_preprocess' directory. 

2. We give the grouping lists in 'Data_preprocess/data_ck+/CK+_Ten_group.txt' and 'Data_preprocess/data_oulu/Oulu_CASIA_Ten_group.txt'

3. Run 'productPKLforCKplus.py' to preprocess CK+ dataset and save in a '.pkl' file.

4. Run 'productPKLforOuluCasIA.py' to preprocess Oulu-CASIA dataset and save in a '.pkl' file.
