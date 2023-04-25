from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

# model = unet()
# model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
# model.fit_generator(myGene,steps_per_epoch=300,epochs=1,callbacks=[model_checkpoint])

# testGene = testGenerator("data/membrane/test")
# results = model.predict_generator(testGene,30,verbose=1)
# saveResult("data/membrane/test",results)

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

# Dont augment:
data_gen_args = dict()

HC18 = trainGenerator(2,'data/HC18/training_set','image','label',data_gen_args,save_to_dir = None)
model = unet()
model_checkpoint = ModelCheckpoint('unet_HC18.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(HC18,steps_per_epoch=30,epochs=2,callbacks=[model_checkpoint])

HC18_test = testGenerator("data/HC18/test_set")
model = unet()
model.load_weights("unet_HC18.hdf5")
results = model.predict_generator(HC18_test,30,verbose=1)
saveResult("data/HC18/test_set/results",results)