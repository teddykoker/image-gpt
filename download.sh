# Download large files from google drive while confirming virus scan warning
# https://gist.github.com/guysmoilov/ff68ef3416f99bd74a3c431b4f4c739a
function gdrive_download () { 
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=$1" -O- | sed -En 's/.*confirm=([0-9A-Za-z_]+).*/\1/p')
  wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -f /tmp/cookies.txt
}

gdrive_download 1Goci9y1zpY47zF7k0DUtiM1KiD15SgXG models/cifar10_gen.ckpt
