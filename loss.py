import torch
import torch.nn as nn


def cross_entropy(pred, target):
    """
    Cross entropy loss that accepts soft targets.
    :param pred: predictions for neural network
    :param target: targets, can be soft
    :return: cross entropy loss
    """
    log_softmax = nn.LogSoftmax(dim=0)
    return torch.sum(-target * log_softmax(pred))


def masked_cross_entropy(pred, target, mask):
    """
    Compute masked cross entropy.
    :param pred: attention weights
    :param target: attention target values
    :param mask: attention mask
    :return: masked cross entropy
    """
    cr_ent = 0
    mask = mask.bool()
    # Iterates over the examples in the batch
    for h in range(0, mask.shape[0]):
        # Select the tokens to consider in the cross entropy based on the mask
        cr_ent += cross_entropy(pred[h][mask[h]], target[h][mask[h]])
    # Divide the cross entropy by the number of samples in the batch
    return cr_ent / mask.shape[0]


def compute_loss(outputs, params, attention_mask=None, attention_vals=None, labels=None):
    """
    Compute the final loss.
    :param outputs: predictions
    :param params: parameters' dictionary
    :param attention_mask: attention mask (1 if the token has to be considered, 0 otherwise).
                           Tensor of dimension (batch_size, 128)
    :param attention_vals: attention target values (from the annotators). Tensor of dimension (batch_size, 128)
    :param labels: labels of the texts. Tensor of dimension (batch_size)
    :return: outputs = (loss), logits, (hidden_states), (attentions)
    """

    weights = params['weights']
    num_labels = params['num_classes']
    train_att = params['train_att']
    num_sv_heads = params['num_supervised_heads']
    sv_layer = params['supervised_layer_pos']
    lam = params['att_lambda']
    device = params['device']

    logits = outputs[0]

    # Add hidden layer and attention if possible
    outputs = (logits,) + outputs[1:]

    # If the labels of the samples in the batch are given
    if labels is not None:
        # Compute the loss for the classification of the text
        loss_funct = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
        loss_logits = loss_funct(logits.view(-1, num_labels), labels.view(-1))
        loss = loss_logits

        # If attention weights need to be considered in the loss
        if train_att:
            # Compute the loss for the attention
            loss_att = 0
            # For each head in the multi-headed attention of the last encoder layer
            for i in range(num_sv_heads):
                # Select the attention of head i of the last encoder layer
                # Consider all examples in the batch
                # The attention is attention between the first token (CLS) and all others token
                attention_weights = outputs[1][sv_layer][:, i, 0, :]
                # Scaled the attention loss by the lambda hyperparameter
                loss_att += lam * masked_cross_entropy(attention_weights, attention_vals, attention_mask)
            loss = loss + loss_att
        outputs = (loss,) + outputs
    # Return the outputs = (loss), logits, (hidden_states), (attentions)
    return outputs


def compute_distillation_loss(outputs_teacher, outputs_student, params, attention_mask, labels):
    """
   Compute the knowledge distillation loss which is the cross-entropy on the labels added to the cross-entropy between
   the attention of teacher model and the one of the student model
   :param outputs_teacher: Outputs from the teacher model.
   :param outputs_student: Outputs from the student model.
   :param params: parameters' dictionary
   :param attention_mask: attention mask (1 if the token has to be considered, 0 otherwise).
                           Tensor of dimension (batch_size, 128)
   :param labels: ground truth labels. Tensor of dimension (batch_size)
   :return: distillation loss
   """

    # compute loss of the labels
    criterion = nn.CrossEntropyLoss()
    loss_labels = criterion(outputs_student['logits'], labels)

    # get the attentions of the last layer
    attn_teacher = outputs_teacher[3][-1]
    attn_student = outputs_student.attentions[-1]

    # compute the loss on the attentions
    loss_att = 0
    # only compute the loss on a number of attention heads
    for i in range(params['num_supervised_heads']):
        # Attention head i, all batch samples, attention between the first token (CLS) and all others token
        attention_student = attn_student[:, i, 0, :]
        attention_teacher = attn_teacher[:, i, 0, :]
        loss_att += masked_cross_entropy(attention_student, attention_teacher, attention_mask)

    # compute final distillation loss
    loss = (1 - params['distillation_alpha']) * loss_labels + params['distillation_alpha'] * loss_att

    return loss
