from fastai import *
from fastai.vision import *

sz = 128

path = Path('data')
path_img = path/f'train-{sz}'

fnames = get_image_files(path_img)

pat = r'/([^/]+)_\d+.png$'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=128, test=Path('../test-128'))
data.normalize(imagenet_stats)

learn = ConvLearner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(5)

learn.save('stage-152')

interp = ClassificationInterpretation.from_learner(learn)

#interp.plot_top_losses(9, figsize=(15,11))

learn.load('stage-152')

learn.unfreeze()

learn.fit_one_cycle(10, max_lr=slice(5e-6,5e-3))

learn.save('stage-2')
learn.load('stage-2')

preds = learn.TTA(is_test=True)[0]

top_3 = np.argsort(preds.numpy())[:, ::-1][:, :3]

n, _ = top_3.shape
labels = []
for i in range(n):
    labels.append(' '.join([learn.data.train_ds.ds.classes[idx] for idx in top_3[i]]))
    
#learn.data.test_dl.dl.dataset[0][0]
#labels[0]

key_ids = [path.stem for path in learn.data.test_dl.dl.dataset.x]

os.makedirs(f'subs152', exist_ok=True)

sub = pd.DataFrame({'key_id': key_ids, 'word': labels})

name = 'first_sub_152_simple'

sub.to_csv(f'subs/{name}.csv.gz', index=False, compression='gzip')

pd.read_csv(f'subs/{name}.csv.gz').head()
