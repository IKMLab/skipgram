from data import toxic
import answers
import torch


# Hyperparameters
epoch = 0
max_epochs = 10
emb_dim = 50
num_negs = 5
lr = 0.02
batch_size = 16
m = 5


vocab = toxic.get_vocab()
frequencies = toxic.get_frequencies()
contexts = toxic.get_contexts(m)
pairs = set([])
for center in contexts.keys():
    pairs.update(tuple(zip([center] * len(contexts[center]),
                           list(contexts[center]))))
pairs = list(pairs)
sampler = answers.NegativeSampler(vocab, frequencies, contexts, num_negs)
collate = answers.Collate(sampler)
data_loader = answers.get_data_loader(pairs, batch_size, collate)
model = answers.SkipGram(vocab, emb_dim, num_negs, lr)
if torch.cuda.is_available():
    model.cuda()

while epoch <= max_epochs:
    epoch += 1
    print('Epoch %s' % epoch)
    for step, batch in enumerate(data_loader):
        loss = model.forward(batch)
        model.optimize(loss)
        if step % 100 == 0:
            print('Step %s\t\tLoss %s' % (step, loss.data.cpu().numpy()[0]))
