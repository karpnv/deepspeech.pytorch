run_server:
    python server.py --model-path /workspace/data/wwdatasets/deepspeech_44.pth.tar --decoder beam --lm-path /workspace/data/wwdatasets/lm.binary


run_docker:
    nvidia-docker run -d -v /home/ubuntu/karpov:/workspace/data -p 8002:8888 --entrypoint=/bin/bash --ipc=host --name deepspeech2-news deepspeech2.docker:7 -c 'python ./deepspeech.pytorch/server.py --model-path /workspace/data/wwdatasets/model/deepspeech/deepspeech_42.pth --lm-path /workspace/data/wwdatasets/model/deepspeech/russian_news_3gram.binary'

    nvidia-docker run -ti -v /home/ubuntu/karpov:/workspace/data -p 8002:8888 --entrypoint=/bin/bash --ipc=host --name deepspeech2-news deepspeech2.docker:7
    python ./deepspeech.pytorch/server.py --model-path /workspace/data/wwdatasets/model/deepspeech/deepspeech_42.pth --decoder beam --lm-path /workspace/data/wwdatasets/model/deepspeech/russian_news_3gram.binary

    nvidia-docker run -ti -v /home/ubuntu/karpov:/workspace/data -p 8001:8888 --entrypoint=/bin/bash --ipc=host --name deepspeech2 deepspeech2.docker:7
    python ./deepspeech.pytorch/server.py --model-path /workspace/data/wwdatasets/model/deepspeech/deepspeech_42.pth --decoder beam --lm-path /workspace/data/wwdatasets/model/deepspeech/lm.binary


convert_wav:
    ffmpeg -i "robway.wav" -acodec pcm_s16le -ar 16000 -ac 1 -f wav "robway16.wav"
