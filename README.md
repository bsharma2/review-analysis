# review-analysis
Summer 2020 Project with Dr. Meng Jiang


### Prepration
```
cd review-analysis
git clone https://github.com/jessevig/bertviz bertviz_repo
wget https://raw.githubusercontent.com/huggingface/transformers/v1.0.0/examples/utils_glue.py

pip install seaborn
pip install regex
pip install pytorch-transformers
pip install tensorboardX
pip install -qq transformers
pip install ipython
pip install flask
pip install flask_cors
pip install sklearn
```

Model weights are in the drive link:

https://drive.google.com/drive/folders/1Uh1O_evUNevM_dnOUI48kS42OvR-9NuM?usp=sharing

Add them to the my_BERT folder.

### Working

The first screen, where the user can enter the review text -

<p align="center"> <img src="first-screen.jpg" width="500" height="400"/> </p>

The second screen is the model output which displays the predicted review rating, sentiment analysis and its visualization - 

<p align="center"> <img src="second-screen.jpg" width="500" height="400"/> </p>
