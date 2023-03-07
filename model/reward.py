from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import torch
import tensorflow as tf
import config
from einops.layers.torch import Rearrange

#add config 

#device
#model path 


class reward_model(torch.nn.Module):

    def __init__(self, ):
        super().__init___(config)

        self.config=config

        self.head_hidden_size=config.head_hidden_size

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        
        self.head = torch.nn.Sequential(
                torch.nn.Linear(self.model.config.n_embd, self.head_hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.head_hidden_size, 1),
                Rearrange("... 1 -> ..."),
            )


        for param in self.model.parameters():
            param.require_grad= False

        self.model.to(config.device)   
        self.head.to(config.device)    
 
        

    def forward(
        self, output_sequence: torch.Tensor, output_sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate the sequence of rewards for the given output sequence
        what is the quality of the output sequence tokens?
        Args:
            output_sequence (torch.Tensor): The sequence of tokens to be
                evaluated
            output_sequence_mask (torch.Tensor): Mask for the attention
        Returns:
            torch.Tensor: Rewards for the given output sequence
        """
        output = self.model(
            output_sequence, attention_mask=output_sequence_mask
        )
        # What if the output_sequence is longer than the max context of
        # the model?
        rewards = self.head(output.last_hidden_state)
        if self.config.debug:
            print("RewardModel.forward")
            print("output_sequence.shape", output_sequence.shape)
            print("output_sequence", output_sequence)
            print("reward.shape", rewards.shape)
            print("reward", rewards)
        return rewards

    def parameters(
        self,
    ) -> Iterable[torch.nn.Parameter]:
        """Return the parameters of the reward model"""
        for p in self.model.parameters():
            yield p
        for p in self.head.parameters():
            yield p

    def get_reward(
        self, output_sequence: torch.Tensor, output_sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get the reward for the given output sequence
        Args:
            output_sequence (torch.Tensor): The concatenation of initial input
                and actor output as tokens
            output_sequence_mask (torch.Tensor): Mask for the attention
        """
        rewards = self.forward(output_sequence, output_sequence_mask)
        return rewards[:, -1]

   
    def load(self, path: Optional[str] = None) -> None:
        """Load the model from the path
        Args:
            path (str): path to the model
        """
        if path is None:
            path = self.config.model_folder + "/" + self.config.model + ".pt"
            if os.path.exists(self.config.model_folder) is False:
                os.makedirs(self.config.model_folder)
                print(
                    f"Model folder does not exist. Creating it,"
                    f"and returning without loading the model:\n{path}"
                )
                return
        # load the model and the tokenizer
        if os.path.exists(path) is False:
            print(
                f"Impossible to load the model:\n{path}\n"
                f"The path doesn't exist."
            )
            return
        model_dict = torch.load(path)
        self.model.load_state_dict(model_dict["model"])
        self.head.load_state_dict(model_dict["head"])

    @beartype
    def save(self, path: Optional[str] = None) -> None:
        """Save the model to the path
        Args:
            path (Optional[str], optional): Path to store the model.
                Defaults to None.
        """
        if path is None:
            path = self.config.model_folder + "/" + self.config.model + ".pt"
            if os.path.exists(self.config.model_folder) is False:
                os.makedirs(self.config.model_folder)
        torch.save(
            {"model": self.model.state_dict(), "head": self.head.state_dict()},
            path,
        )


critic_model=reward_model