from transformers import Wav2Vec2Processor, Data2VecAudioForCTC
import torch

if __name__ == '__main__':
    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained("/root/kyzhang/download/data2vec-audio-base-960h")
    model = Data2VecAudioForCTC.from_pretrained("/root/kyzhang/download/data2vec-audio-base-960h")

    print(model)

    x = torch.randn(3, 100000)
    output = model.data2vec_audio.forward(x)
    print(output.last_hidden_state.shape)