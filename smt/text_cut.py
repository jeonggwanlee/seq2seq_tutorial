

fread = open('/home/jglee/Research/datasets/LanguageTranslationDataset/europarl/training/europarl-v7.fr-en.en', 'r')

fout = open('europarl-v7.fr-en.en50', 'w')

for _ in range(50):
    line = fread.readline()
    fout.write(line)

fout.close()
fread.close()

fread = open('/home/jglee/Research/datasets/LanguageTranslationDataset/europarl/training/europarl-v7.fr-en.fr', 'r')

fout = open('europarl-v7.fr-en.fr50', 'w')

for _ in range(50):
    line = fread.readline()
    fout.write(line)

fout.close()
fread.close()


