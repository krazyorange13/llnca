%env AWS_ACCESS_KEY_ID=FILL_ME_IN
%env AWS_SECRET_ACCESS_KEY=FILL_ME_IN
%env AWS_DEFAULT_REGION=rpi5-1

!mkdir -p runs/logs
!mkdir -p runs/checkpts

%env DATASET=harv
!mkdir -p data/$DATASET/
!wget https://files.krazyorange.dev/data/$DATASET/$DATASET.corptok.txt -O data/$DATASET/$DATASET.corptok.txt
!wget https://files.krazyorange.dev/data/$DATASET/$DATASET.tokenizer.json -O data/$DATASET/$DATASET.tokenizer.json

!mkdir -p models
!wget https://files.krazyorange.dev/models/harv-rebeta-md-tk6-3300.pth -O models/harv-rebeta-md-tk6-3300.pth

!pip install bpeasy
!pip install tensorboard
# !pip install torch_tpu
