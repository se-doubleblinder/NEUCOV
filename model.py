from copy import deepcopy
import sys
import torch
import torch.nn as nn
import os
from transformers import RobertaModel


class CodeCoverageMLP(nn.Module):
    def __init__(self, config):
        super(CodeCoverageMLP, self).__init__()
        # First hidden layer
        self.fc1 = nn.Linear(1 * config.hidden_size, config.hidden_size)
        # Second hidden layer
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        # Output layer
        self.fc3 = nn.Linear(config.hidden_size, 1)
        self.forward_activation = torch.nn.GELU()
        # Dropout layer
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None \
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
  
    def forward(self, x):
        # Add first hidden layer
        x = self.forward_activation(self.fc1(x))
        # Add dropout layer
        x = self.dropout(x)
        # Add second hidden layer
        x = self.forward_activation(self.fc2(x))
        # Add dropout layer
        x = self.dropout(x)
        # Add output layer
        x = self.fc3(x)

        outputs = torch.sigmoid(x)
        return outputs

class CodeCoveragePredictionModel(nn.Module):
    def __init__(self, args, config, tokenizer):
        super(CodeCoveragePredictionModel, self).__init__()
        self.max_tokens = args.max_tokens
        self.use_statement_ids = args.use_statement_ids
        self.roberta = RobertaModel.from_pretrained(args.model_key, config=config)
        self.roberta.resize_token_embeddings(len(tokenizer))

        if self.use_statement_ids:
            self.statement_embeddings = nn.Embedding(args.max_tokens, config.hidden_size)

        for param in self.roberta.parameters():
            if args.pretrain:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.coverage_mlp = CodeCoverageMLP(config)
        self.loss_criterion = nn.BCELoss(reduction='mean')

    def forward(self, inputs_ids, inputs_masks, statements_ids, test_input_statements_ids, gold_ids=None):
        try:

            coverage_labels=gold_ids
            device = inputs_ids.device if inputs_ids is not None else 'cpu'
            inputs_embeddings = self.roberta.embeddings.word_embeddings(inputs_ids)
            if self.use_statement_ids: 
                statements_ids[statements_ids == -999] = 511    
                inputs_embeddings += self.statement_embeddings(statements_ids)
            
            roberta_outputs = self.roberta(
                inputs_embeds=inputs_embeddings,
                attention_mask=inputs_masks,
                output_attentions = True,
                output_hidden_states = True,
            )
    
            hidden_states = roberta_outputs.hidden_states

            # Hidden_states has four dimensions: the layer number (for e.g., 13),
            # batch number (for e.g., 8), token number (for e.g., 256),
            # hidden units (for e.g., 768).
    
            # Choice: First Layer / Last layer / Concatenation of last four layers /
            # Sum of all layers.
            outputs_embeddings = hidden_states[-1]
            
            batch_preds, batch_true = [], []
            batch_loss = torch.tensor(0, dtype=torch.float, device=device)
    
            for _id, item_output_embeddings in enumerate(outputs_embeddings):
                statements_embeddings = []

                test_input_statement_ids = test_input_statements_ids[_id][
                    torch.ne(test_input_statements_ids[_id], -999)
                ].tolist()
                
                if self.use_statement_ids: 
                    item_statements_ids =  statements_ids[_id][
                        torch.ne(statements_ids[_id], 511)
                    ].tolist()
                else:
                    item_statements_ids =  statements_ids[_id][
                        torch.ne(statements_ids[_id], -999)
                    ].tolist()

                item_test_input_ids_match = []
                for idx in test_input_statement_ids:
                    item_statements_ids = torch.tensor(item_statements_ids, device=device)
                    idx = torch.tensor(idx, device=device)
                    indices = torch.nonzero(item_statements_ids == idx)
                    if indices.numel() > 0:
                        item_test_input_ids_match += indices.squeeze().tolist()
                item_statements_ids_tensor = torch.tensor(item_statements_ids, device=device)
                
                num_statements_in_item = torch.max(item_statements_ids_tensor).item()

                for sid in range(num_statements_in_item + 1):
                    _statement_ids = (item_statements_ids_tensor == sid).nonzero().squeeze()
                    statement_embedding = torch.mean(item_output_embeddings[_statement_ids], dim=0)
                    # Pooling Strategy: Embeddings of Line number only
                    # statement_embedding = item_output_embeddings[_statement_ids][0]
                    statements_embeddings.append(statement_embedding)
                
                item_preds = self.coverage_mlp(torch.stack(statements_embeddings)).squeeze()
                batch_preds.append(item_preds)

                if coverage_labels is not None:
                    item_true = coverage_labels[_id][coverage_labels[_id] != -999]
                    item_true = item_true[:len(item_preds)]
                    batch_true.append(item_true)
                    v1 = self.loss_criterion(item_preds, item_true.float())
                    batch_loss += v1

            if coverage_labels is None:
                return batch_preds
            else:
                return batch_loss, batch_preds, batch_true
        except Exception as e:
            print(f"An exception of type occurred: {e}")
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
class Seq2Seq(nn.Module):
    def __init__(
            self, encoder, decoder, config, beam_size=None,
            max_length=None, sos_id=None, eos_id=None
        ):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        # self.encoder_key = encoder_key
        self.decoder = decoder
        # self.decoder_key = decoder_key
        self.config = config
        self.register_buffer(
            "bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)
    
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids, target_ids=None):
        if target_ids is None:
            return self.generate(source_ids)

        source_mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(source_ids, attention_mask=source_mask, use_cache=True)
        ids = torch.cat((source_ids, target_ids), -1)
        target_mask = self.bias[:, source_ids.size(-1) :ids.size(-1), :ids.size(-1)].bool()
        target_mask = target_mask & ids[:, None, :].ne(1)
        decoder_output = self.decoder(target_ids, attention_mask=target_mask,
                            past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(decoder_output)

        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])
        outputs = loss, loss * active_loss.sum(), active_loss.sum()
        return outputs
    
    def generate(self, source_ids):
        source_mask = source_ids.ne(1)[:, None, :] * source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(source_ids, attention_mask=source_mask, use_cache=True)
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i: i+1, :, :source_len[i]].repeat(self.beam_size, 1, 1, 1) for x in y] \
                        for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1, :source_len[i]].repeat(self.beam_size, 1)
            for _ in range(self.max_length):
                if beam.done():
                    break
                ids = torch.cat((context_ids, input_ids), -1)
                target_mask = self.bias[:, context_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
                target_mask = target_mask & ids[:, None, :].ne(1)
                out = self.decoder(input_ids, attention_mask=target_mask,
                                    past_key_values=context).last_hidden_state
                hidden_states = out[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p] + \
                                [zero] * (self.max_length - len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))
        preds = torch.cat(preds, 0)
        return preds   


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >=self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished=[]
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i)) 
            unfinished.sort(key=lambda a: -a[0])
            self.finished+=unfinished[:self.size-len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        hyps=[]
        for _,timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j+1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps
    
    def buildTargetTokens(self, preds):
        sentence=[]
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok==self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence