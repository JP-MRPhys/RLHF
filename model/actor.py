from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch
import tensorflow as tf
from einops.layers.torch import Rearrange


#config model, path model.folder

class actor(torch.nn.Module):


    def __init__(self, config):

        super().__init___()

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')

    
        self.max_model_tokens = 1024
        # save config
        self.config = config

    def parameters(self, **kwargs):
        """Return the parameters of the model
        Args:
            **kwargs:
        """
        return self.model.parameters()
    
    def forward(
        self, sequences: torch.Tensor, sequences_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate logits to have probability distribution over the vocabulary
            of the actions
        Args:
            sequences (torch.Tensor): Sequences of states and actions used to
                    compute token logits for the whole list of sequences
            attention_mask (torch.Tensor): Mask for the sequences attention
        Returns:
            logits (torch.Tensor): Logits for the actions taken
        """
        model_output = self.model.forward(
            sequences, attention_mask=sequences_mask
        )
        if self.config.debug:
            print("ActorModel.forward")
            print("model_output_logits shape", model_output.logits.shape)
            print("model_output logits", model_output.logits)
        return model_output.logits


    @torch.no_grad()
    def generate(
        self, states: torch.Tensor, state_mask: torch.Tensor
    ) -> Tuple:
        """Generate actions and sequences=[states, actions] from state
            (i.e. input of the prompt generator model)
        Args:
            state (torch.Tensor): the input of the user
            state_mask (torch.Tensor): Mask for the state input (for padding)
        Returns:
            actions (torch.Tensor): Actions generated from the state
            sequences (torch.Tensor): Sequences generated from the
                state as [states, actions]
        """
        max_sequence = states.shape[1]
        max_tokens = self.config.max_tokens + max_sequence
        temperature = self.config.temperature
        # What if the states + completion are longer than the max context of
        # the model?
        sequences = self.model.generate(
            inputs=states,
            attention_mask=state_mask,
            max_length=max_tokens,
            temperature=temperature,
        )
        actions = sequences[:, states.shape[1] :]  # noqa E203
        if self.config.debug:
            print("ActorModel.generate")
            print("state", states)
            print("state shape", states.shape)
            print("sequence shape", sequences.shape)
            print("sequence", sequences)
            print("actions shape", actions.shape)
            print("actions", actions)
        return actions, sequences
    
    def load(self, path: Optional[str] = None) -> None:
        """Load the model from the path
        Args:
            path (str): Path to the model
        """
        if path is None:
            path = self.config.model_folder + "/" + self.config.model + ".pt"
            if os.path.exists(self.config.model_folder) is False:
                os.mkdir(self.config.model_folder)
                print(
                    f"Impossible to load the model: {path}"
                    f"The path doesn't exist."
                )
                return
        # load the model
        if os.path.exists(path) is False:
            print(
                f"Impossible to load the model: {path}"
                f"The path doesn't exist."
            )
            return
        model_dict = torch.load(path)
        self.model.load_state_dict(model_dict["model"])

    
    def save(self, path: Optional[str] = None) -> None:
        """Save the model to the path
        Args:
            path (Optional[str], optional): Path to store the model.
                Defaults to None.
        """
        if path is None:
            path = self.config.model_folder + "/" + self.config.model + ".pt"
            if os.path.exists(self.config.model_folder) is False:
                os.mkdir(self.config.model_folder)
        torch.save({"model": self.model.state_dict()}, path)

