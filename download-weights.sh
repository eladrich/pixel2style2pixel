#!/bin/sh
mkdir pretrained_models
cd pretrained_models

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lB7wk7MwtdxL-LL4Z_T76DuCfk00aSXA' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lB7wk7MwtdxL-LL4Z_T76DuCfk00aSXA" -O psp_celebs_sketch_to_face.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1_S4THAzXb-97DbpXmanjHtXRyKxqjARv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1_S4THAzXb-97DbpXmanjHtXRyKxqjARv" -O psp_ffhq_frontalization.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ZpmSXBpJ9pFEov6-jjQstAlfYbkebECu" -O psp_celebs_super_resolution.pt && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1YKoiVuFaqdvzDP5CZaqa3k5phL-VDmyz" -O psp_ffhq_toonify.pt && rm -rf /tmp/cookies.txt

cd ..
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
