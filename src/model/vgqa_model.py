from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


class VGQAModel(nn.Module):
    def __init__(
        self,
        encoder_text_name: str,
        encoder_image_name: str,
        num_labels: int,
        intermediate_dim=512,
        intermediate_dim_dense=512,
        dropout=0.5,
    ):
        """
        params pretrained_text_name: name of model used to create text embedding
        params pretrained_image_name: name of model used to create image embedding
        params num_labels: number of classes in the output
        params intermediate_dim:
        params dropout:
        """
        super(VGQAModel, self).__init__()

        self.num_labels = num_labels
        self.dropout = dropout
        self.intermediate_dim = intermediate_dim
        self.intermediate_dim_dense = intermediate_dim_dense

        self.text_encoder = AutoModel.from_pretrained(encoder_text_name)
        self.image_encoder = AutoModel.from_pretrained(encoder_image_name)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # Fusion layer for cross-modal interaction
        self.fusion = nn.Sequential(
            nn.Linear(
                self.text_encoder.config.hidden_size
                + self.image_encoder.config.hidden_size,
                self.intermediate_dim_dense,
            ),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        # Fully-connected classifier
        self.dense = nn.Sequential(
            nn.Linear(intermediate_dim_dense, self.intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(intermediate_dim, self.num_labels)

        self.criterion = nn.functional.binary_cross_entropy_with_logits

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        encoded_text = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        encoded_image = self.image_encoder(
            pixel_values=pixel_values,
            return_dict=True,
        )
        fused_output = self.fusion(
            torch.cat(
                [
                    encoded_text["pooler_output"],
                    encoded_image["pooler_output"],
                ],
                dim=1,
            )
        )
        dense = self.dense(fused_output)
        logits = self.classifier(dense)

        out = {"logits": logits}
        if labels is not None:
            loss = self.criterion(logits, labels.float())
            out["loss"] = loss

        return out
