import numpy as np
import torch
import torch.nn as nn
from torch import hub

from . import vggish_input, vggish_params


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self.make_layers()
        self.embeddings = nn.Sequential(
            nn.Linear(512 * 4 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 128),
            nn.ReLU(True)
        )

    def make_layers(self):
        layers = []
        in_channels = 1
        for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.cuda()
        x = self.features(x) #! [12, 512, 6, 4] 512: channels, h, w (feature map)
        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2) #! [12, 6, 4, 512]
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x) #! [12, 12288]


class Postprocessor(nn.Module):
    """Post-processes VGGish embeddings. Returns a torch.Tensor instead of a
    numpy array in order to preserve the gradient.

    "The initial release of AudioSet included 128-D VGGish embeddings for each
    segment of AudioSet. These released embeddings were produced by applying
    a PCA transformation (technically, a whitening transform is included as well)
    and 8-bit quantization to the raw embedding output from VGGish, in order to
    stay compatible with the YouTube-8M project which provides visual embeddings
    in the same format for a large set of YouTube videos. This class implements
    the same PCA (with whitening) and quantization transformations."
    """

    def __init__(self):
        """Constructs a postprocessor."""
        super(Postprocessor, self).__init__()
        # Create empty matrix, for user's state_dict to load
        self.pca_eigen_vectors = torch.empty(
            (vggish_params.EMBEDDING_SIZE, vggish_params.EMBEDDING_SIZE,),
            dtype=torch.float,
        ) #! [128, 128]
        self.pca_means = torch.empty(
            (vggish_params.EMBEDDING_SIZE, 1), dtype=torch.float
        ) #! [128, 1]

        self.pca_eigen_vectors = nn.Parameter(self.pca_eigen_vectors, requires_grad=False)
        self.pca_means = nn.Parameter(self.pca_means, requires_grad=False)

    def postprocess(self, embeddings_batch):
        """Applies tensor postprocessing to a batch of embeddings.

        Args:
          embeddings_batch: An tensor of shape [batch_size, embedding_size]
            containing output from the embedding layer of VGGish.

        Returns:
          A tensor of the same shape as the input, containing the PCA-transformed,
          quantized, and clipped version of the input.
        """
        assert len(embeddings_batch.shape) == 2, "Expected 2-d batch, got %r" % (
            embeddings_batch.shape,
        )
        assert (
            embeddings_batch.shape[1] == vggish_params.EMBEDDING_SIZE
        ), "Bad batch shape: %r" % (embeddings_batch.shape,) #! [12, 128]

        # Apply PCA.
        # - Embeddings come in as [batch_size, embedding_size].
        # - Transpose to [embedding_size, batch_size].
        # - Subtract pca_means column vector from each column.
        # - Premultiply by PCA matrix of shape [output_dims, input_dims]
        #   where both are are equal to embedding_size in our case.
        # - Transpose result back to [batch_size, embedding_size].
        pca_applied = torch.mm(self.pca_eigen_vectors, (embeddings_batch.t() - self.pca_means)).t() 

        # Quantize by:
        # - clipping to [min, max] range
        clipped_embeddings = torch.clamp(
            pca_applied, vggish_params.QUANTIZE_MIN_VAL, vggish_params.QUANTIZE_MAX_VAL
        )
        # - convert to 8-bit in range [0.0, 255.0]
        quantized_embeddings = torch.round(
            (clipped_embeddings - vggish_params.QUANTIZE_MIN_VAL)
            * (
                255.0
                / (vggish_params.QUANTIZE_MAX_VAL - vggish_params.QUANTIZE_MIN_VAL)
            )
        )
        return torch.squeeze(quantized_embeddings) #! [12, 128]

    def forward(self, x):
        return self.postprocess(x)


def _vgg():
    return VGG()

# def _spectrogram():
#     config = dict(
#         sr=16000,
#         n_fft=400,
#         n_mels=64,
#         hop_length=160,
#         window="hann",
#         center=False,
#         pad_mode="reflect",
#         htk=True,
#         fmin=125,
#         fmax=7500,
#         output_format='Magnitude',
#         #             device=device,
#     )
#     return Spectrogram.MelSpectrogram(**config)

urls = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

class VGGish(VGG):
    def __init__(self, ckpt = None, device='cpu', pretrained=True, preprocess=True, postprocess=True):
        super().__init__()
        state_dict = hub.load_state_dict_from_url(urls['vggish'])
        # state_dict = torch.load(ckpt['vggish'])
        super().load_state_dict(state_dict)
        print('The checkpoint for vggish is loaded!')

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess
        if self.postprocess:
            self.pproc = Postprocessor()
            state_dict = hub.load_state_dict_from_url(urls['pca'])
            # state_dict = torch.load(ckpt['pca'])
            # TODO: Convert the state_dict to torch
            state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
                state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
            )
            state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(
                state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
            )
            self.pproc.load_state_dict(state_dict)
            print('The checkpoint for pca is loaded!')
        self.to(self.device)

    def forward(self, x, fs = None):
        if self.preprocess:
            x = self._preprocess(x, fs)
        x = x.to(self.device) #! [12, 1, 96, 64]
        x = VGG.forward(self, x) #! [12, 512, 6, 4]
        if self.postprocess:
            x = self._postprocess(x)
        return x

    def _preprocess(self, x, fs):
        x = vggish_input.waveform_to_examples(x, fs)
        return x

    def _postprocess(self, x):
        return self.pproc(x)
