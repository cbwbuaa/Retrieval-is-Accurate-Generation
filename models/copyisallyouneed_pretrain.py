from header import *

class CopyisallyouneedPretrain(nn.Module):

    def __init__(self, **args):
        super(CopyisallyouneedPretrain, self).__init__()
        self.args = args

        # bert-encoder model
        self.phrase_encoder = AutoModel.from_pretrained(
            self.args['phrase_encoder_model'][self.args['lang']]
        )
        self.bert_tokenizer = AutoTokenizer.from_pretrained(
            self.args['phrase_encoder_tokenizer'][self.args['lang']]
        )
        self.bert_tokenizer.add_tokens(['<|endoftext|>', '[PREFIX]'])
        self.prefix_token_id = self.bert_tokenizer.convert_tokens_to_ids('[PREFIX]')
        self.phrase_encoder.resize_token_embeddings(self.phrase_encoder.config.vocab_size+2)
        
        prefix_encoder_hidden_size = 768
        self.s_cls = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(prefix_encoder_hidden_size // 2, 2)
        )
        self.e_cls = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(prefix_encoder_hidden_size // 2, 2)
        )
        # MLP: mapping bert phrase start representations
        self.s_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(prefix_encoder_hidden_size, prefix_encoder_hidden_size // 2)
        )
        # MLP: mapping bert phrase end representations
        self.e_proj = nn.Sequential(
            nn.Dropout(p=args['dropout']),
            nn.Tanh(),
            nn.Linear(prefix_encoder_hidden_size, prefix_encoder_hidden_size // 2)
        )
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, batch):
        ## encode the document with the BERT encoder model
        dids, dids_mask = batch['bert_ids'], batch['bert_mask']
        start_labels, end_labels = batch['start_labels'], batch['end_labels']
        output = self.phrase_encoder(dids, dids_mask, output_hidden_states=True)['hidden_states'][-1]   # [B, S, E]
        
        # collect the phrase start representations and phrase end representations
        s_rep = self.s_proj(output)
        e_rep = self.e_proj(output)    

        s_logits = self.s_cls(s_rep)
        e_logits = self.e_cls(e_rep)

        s_loss = self.gen_loss_fct(s_logits.view(-1, s_logits.shape[-1]), start_labels.view(-1))
        e_loss = self.gen_loss_fct(e_logits.view(-1, e_logits.shape[-1]), end_labels.view(-1))
        
        s_pred = s_logits.argmax(dim=-1)
        e_pred = e_logits.argmax(dim=-1)
        
        s_acc = (s_pred == start_labels)
        s_mask = (start_labels != -1)
        s_valid = s_acc & s_mask
        s_acc = s_valid.sum() / s_mask.sum()

        e_acc = (e_pred == end_labels)
        e_mask = (end_labels != -1)
        e_valid = e_acc & e_mask
        e_acc = e_valid.sum() / e_mask.sum()
        return (
            s_loss,
            e_loss,
            s_acc,
            e_acc
        ) 