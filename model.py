import torch
import torch.nn as nn
from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertModel


class ECDGDST(BertPreTrainedModel):
    def __init__(self, config, n_op, n_domain, update_id):
        super(ECDGDST, self).__init__(config)
        self.hidden_size = config.hidden_size
        self.domainGuide = DomainGuide(config, n_op, n_domain, update_id)
        self.encoder = Encoder(config)
        self.operationPrediction = OperatePrediction(config.hidden_size, n_op, config.dropout, update_id,)
        self.decoder = ValueGeneration(config, self.encoder.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids,
                state_positions, attention_mask,
                max_value, op_ids=None, max_update=None, teacher=None):
        bert_outputs, sequence_output, pooled_output = self.encoder(input_ids,
                                                                    token_type_ids,
                                                                    attention_mask)
        domain_scores, state_output = self.domainGuide(sequence_output,
                                                       pooled_output,
                                                       state_positions,
                                                       op_ids,
                                                       max_update)
        state_scores, decoder_inputs = self.operationPrediction(state_output,
                                                                input_ids,
                                                                op_ids=op_ids,
                                                                max_update=max_update)
        gen_scores = self.decoder(input_ids,
                                  decoder_inputs,
                                  sequence_output,
                                  pooled_output,
                                  max_value,
                                  teacher)
        return domain_scores, state_scores, gen_scores


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.bert = BertModel(config)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output, pooled_output = bert_outputs[:2]
        return bert_outputs, sequence_output, pooled_output


class DomainGuide(nn.Module):
    def __init__(self, config, n_op, n_domain, update_id):
        super(DomainGuide, self).__init__()
        self.domain_cls = Domain_cls(config.hidden_size, n_domain, config.dropout)
        self.domain_gate = DomainGate(config.hidden_size)  # 添加的域门
        self.n_op = n_op
        self.n_domain = n_domain
        self.update_id = update_id

    def forward(self, sequence_output, pooled_output, state_positions, op_ids=None, max_update=None):
        domain_scores = self.domain_cls(pooled_output)
        state_pos = state_positions[:, :, None].expand(-1, -1, sequence_output.size(-1))
        state_output = torch.gather(sequence_output, 1, state_pos)
        state_output = self.domain_gate(pooled_output, state_output)  # 域门  2*5  2*30*768
        return domain_scores, state_output


class Domain_cls(nn.Module):
    def __init__(self, hidden_size, n_domain, dropout):
        super(Domain_cls, self).__init__()
        self.domain_cls = nn.Linear(hidden_size, n_domain)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pooled_output):
        dropouted_input = self.dropout(pooled_output)
        domain_scores = self.domain_cls(dropouted_input)
        return domain_scores


class DomainGate(nn.Module):
    def __init__(self, hidden_size):
        super(DomainGate, self).__init__()
        self.W_linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_DS = nn.Linear(hidden_size, hidden_size, bias=False)
        self.s_trans = nn.Linear(hidden_size*2, hidden_size, bias=False)

    def forward(self, C_domain, C_slot):
        self.batch_size = C_domain.size(0)
        CI_w = self.W_linear(C_domain)  # B,1,H
        CI_w = torch.unsqueeze(CI_w, 1)
        contact_CI_CS = torch.tanh(C_slot + CI_w)  # B,T,H
        g = self.V(contact_CI_CS)  # B,T,H
        g = g.sum(dim=-1)  # B,T
        C_domain = torch.unsqueeze(C_domain, 1).repeat(1, C_slot.size(1), 1)
        g = torch.unsqueeze(g, -1)
        g = g.repeat(1, 1, C_domain.size(-1))
        S = torch.cat((C_slot, temp), -1)
        S = self.s_trans(S)
        return S  # B , T


class OperatePrediction(nn.Module):
    def __init__(self, hidden_size, n_op, dropout, update_id):
        super(OperatePrediction, self).__init__()
        self.operate_cls = nn.Linear(hidden_size , n_op)
        self.dropout = nn.Dropout(dropout)
        self.update_id = update_id
        self.hidden_size = hidden_size
        self.n_op = n_op

    def forward(self, state_output, input_ids, op_ids=None, max_update=None):
        dropouted_input = self.dropout(state_output)
        state_scores = self.operate_cls(dropouted_input)  # B,J,4
        batch_size = state_scores.size(0)

        if op_ids is None:
            op_ids = state_scores.view(-1, self.n_op).max(-1)[-1].view(batch_size, -1)
        if max_update is None:
            max_update = op_ids.eq(self.update_id).sum(-1).max().item()

        gathered = []
        for b, a in zip(state_output, op_ids.eq(self.update_id)):  # update
            if a.sum().item() != 0:
                v = b.masked_select(a.unsqueeze(-1)).view(1, -1, self.hidden_size)
                n = v.size(1)
                gap = max_update - n
                if gap > 0:
                    zeros = torch.zeros(1, 1 * gap, self.hidden_size, device=input_ids.device)
                    v = torch.cat([v, zeros], 1)
            else:
                v = torch.zeros(1, max_update, self.hidden_size, device=input_ids.device)
            gathered.append(v)
        decoder_inputs = torch.cat(gathered)
        return state_scores, decoder_inputs


class ValueGeneration(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(ValueGeneration, self).__init__()
        self.pad_idx = 0
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.pad_idx)
        self.embed.weight = bert_model_embedding_weights
        self.gru = nn.GRU(config.hidden_size, config.hidden_size, 1, batch_first=True)
        self.w_gen = nn.Linear(config.hidden_size * 3, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

        for n, p in self.gru.named_parameters():
            if 'weight' in n:
                p.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, x, decoder_input, encoder_output, hidden, max_len, teacher=None):
        hidden = hidden.unsqueeze(0)
        mask = x.eq(self.pad_idx)
        batch_size, n_update, _ = decoder_input.size()  # B,J',5 # long
        state_in = decoder_input
        all_point_outputs = torch.zeros(n_update, batch_size, max_len, self.vocab_size).to(x.device)
        result_dict = {}
        for j in range(n_update):
            w = state_in[:, j].unsqueeze(1)  # B,1,D
            slot_value = []
            for k in range(max_len):
                w = self.dropout(w)
                _, hidden = self.gru(w, hidden)  # 1,B,D
                # B,T,D * B,D,1 => B,T
                attn_e = torch.bmm(encoder_output, hidden.permute(1, 2, 0))  # B,T,1
                attn_e = attn_e.squeeze(-1).masked_fill(mask, -1e9)
                attn_history = nn.functional.softmax(attn_e, -1)  # B,T

                # B,D * D,V => B,V
                attn_v = torch.matmul(hidden.squeeze(0), self.embed.weight.transpose(0, 1))  # B,V
                attn_vocab = nn.functional.softmax(attn_v, -1)

                # B,1,T * B,T,D => B,1,D
                context = torch.bmm(attn_history.unsqueeze(1), encoder_output)  # B,1,D

                p_gen = self.sigmoid(self.w_gen(torch.cat([w, hidden.transpose(0, 1), context], -1)))  # B,1
                p_gen = p_gen.squeeze(-1)

                p_context_ptr = torch.zeros_like(attn_vocab).to(x.device)
                p_context_ptr.scatter_add_(1, x, attn_history)  # copy B,V
                p_final = p_gen * attn_vocab + (1 - p_gen) * p_context_ptr  # B,V
                _, w_idx = p_final.max(-1)
                slot_value.append([ww.tolist() for ww in w_idx])
                if teacher is not None:
                    w = self.embed(teacher[:, j, k]).unsqueeze(1)
                else:
                    w = self.embed(w_idx).unsqueeze(1)  # B,1,D
                all_point_outputs[j, :, k, :] = p_final

        return all_point_outputs.transpose(0, 1)
