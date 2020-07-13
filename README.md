# review-analysis
Summer 2020 Project with Dr. Meng Jiang


### Prepration
```
git clone https://github.com/jessevig/bertviz bertviz_repo
wget https://raw.githubusercontent.com/huggingface/transformers/v1.0.0/examples/utils_glue.py

pip install regex
pip install pytorch-transformers
pip install tensorboardX
pip install -qq transformers
```

Model weights are in the drive link:

https://drive.google.com/drive/folders/1Uh1O_evUNevM_dnOUI48kS42OvR-9NuM?usp=sharing

Add them to the my_BERT folder.

The project uses flask to run on web, to install flask:
(source - https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)

```
python3 -m venv venv
virtualenv venv
source venv/bin/activate
pip install flask
export FLASK_APP=microblog.py
flask run
```
