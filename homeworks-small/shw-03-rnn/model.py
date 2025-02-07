import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(
        self,
        dataset: TextDataset,
        embed_size: int = 256,
        hidden_size: int = 256,
        rnn_type: Type = nn.RNN,
        rnn_layers: int = 1,
    ):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=embed_size
        )

        self.rnn = rnn_type(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
        )

        self.linear = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        embedded = self.embedding(indices)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        output_packed, _ = self.rnn(packed)

        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        logits = self.linear(output)
        return logits

    @property
    def device(self):
        return next(self.parameters()).device

    @torch.inference_mode()
    def inference(self, prefix: str = "", temp: float = 1.0) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        self.eval()
        generated_ids = [self.dataset.bos_id]

        if prefix:
            prefix_ids = self.dataset.text2ids(prefix)
            generated_ids.extend(prefix_ids)

        generated_ids = generated_ids[: self.max_length - 1]

        input_tensor = torch.tensor(generated_ids, device=self.device).unsqueeze(0)

        embedded = self.embedding(input_tensor)
        output, hn = self.rnn(embedded)

        for _ in range(self.max_length - len(generated_ids)):
            probas = torch.softmax(self.linear(output[:, -1, :]) / temp, dim=-1)

            next_token = torch.multinomial(probas, num_samples=1).item()
            generated_ids.append(next_token)

            if next_token == self.dataset.eos_id:
                break

            next_input = torch.tensor([[next_token]], device=self.device)
            embedded_next = self.embedding(next_input)

            output_next, hn = self.rnn(embedded_next, hn)
            output = torch.cat([output, output_next], dim=1)

        processed_ids = [
            token
            for token in generated_ids
            if token not in {self.dataset.bos_id, self.dataset.eos_id}
        ]

        output_text = self.dataset.ids2text(processed_ids)
        return output_text
