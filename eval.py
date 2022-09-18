import torchaudio
import torch
import evaluation_utils as eval_utils
import torch.nn.functional as F
import torch.utils.data as data

from torch import nn
from data_utils import TextTransform
from asr_model import SpeechRecognitionModel



def data_processing(train_audio_transforms, valid_audio_transforms,text_transform, data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    for (waveform, _, utterance, _, _, _) in data:
        if data_type == 'train':
            spec = train_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        else:
            spec = valid_audio_transforms(waveform).squeeze(0).transpose(0, 1)
        spectrograms.append(spec)
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        # print(label)
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    spectrograms = nn.utils.rnn.pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def GreedyDecoder(output, labels, label_lengths, text_transform, blank_label=28, collapse_repeated=True):
    arg_maxes = torch.argmax(output, dim=2)
    decodes = []
    targets = []
    # print(arg_maxes.shape)
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets
    
def test(model, device, test_loader, text_transform):
    print('\nevaluating...')
    model.eval()
    test_cer, test_wer = [], []
    with torch.no_grad():
        for i, _data in enumerate(test_loader):
            spectrograms, labels, input_lengths, label_lengths = _data 
            spectrograms, labels = spectrograms.to(device), labels.to(device)

            output = model(spectrograms)  # (batch, time, n_class)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # (time, batch, n_class)

            decoded_preds, decoded_targets = GreedyDecoder(output.transpose(0, 1), labels, label_lengths, text_transform)
            # print(decoded_preds,decoded_targets)
            for j in range(len(decoded_preds)):
                test_cer.append(eval_utils.cer(decoded_targets[j], decoded_preds[j]))
                test_wer.append(eval_utils.wer(decoded_targets[j], decoded_preds[j]))
    avg_cer = sum(test_cer)/len(test_cer)
    avg_wer = sum(test_wer)/len(test_wer)

    print('Average CER: {:4f} Average WER: {:.4f}\n'.format(avg_cer, avg_wer))

def main(train_audio_transforms, valid_audio_transforms, text_transform):

    hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "batch_size": 20
    }


    use_cuda = torch.cuda.is_available()
    torch.manual_seed(7)
    device = torch.device("cuda" if use_cuda else "cpu")
    test_dataset = torchaudio.datasets.LIBRISPEECH("./", url="test-clean", download=True)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    test_loader = data.DataLoader(dataset=test_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=False,
                                collate_fn=lambda x: data_processing(train_audio_transforms, valid_audio_transforms, text_transform, x, 'valid'),
                                **kwargs)
    
    # for i, _data in enumerate(test_loader):
    #     print(_data)
    model = SpeechRecognitionModel(
        hparams['n_cnn_layers'], hparams['n_rnn_layers'], hparams['rnn_dim'],
        hparams['n_class'], hparams['n_feats'], hparams['stride'], hparams['dropout']
        ).to(device)
    model.load_state_dict(torch.load("./saved_model/best_model.pt"))
    model.eval()
    print(model)
    test(model, device, test_loader, text_transform)
if __name__ == "__main__":

    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
        torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
        torchaudio.transforms.TimeMasking(time_mask_param=35)
    )
    
    valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

    text_transform = TextTransform()

    main(train_audio_transforms, valid_audio_transforms, text_transform)
    
                                                                                                             