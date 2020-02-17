import pandas as pd
import os
import shutil

DATA_DIR = '/home/danm/data/fishml'
data_frame = pd.read_csv(os.path.join(DATA_DIR, 'crop_metadata.csv'))
key_species = pd.read_csv(os.path.join(DATA_DIR, 'key_species.csv'))

print(data_frame.head())
print(key_species)

# Make a key_species list to check against
key_species_list = []

for item in key_species.itertuples(index=False):
    key_species_list.append(f'{item[0]}_{item[1]}_{item[2]}')

# ============================= #
# iterate through each species in the data frame, check if a directory with the same name exists
# and if it does copy the fish image in to that directory.  If it does not, create the directory
# Then copy over the image
# ============================= #


for item in data_frame.itertuples(index=False):  # itertuples is much faster than iterrows
    label = f'{item[2]}_{item[3]}_{item[4]}'  # item["family"]}_{item["genus"]}_{item["species"]
    label_dir = os.path.join(DATA_DIR, label)
    fish_loc = os.path.join(DATA_DIR, 'archive', 'FDFML', 'crops', item[1])

    if label in key_species_list:

        # create the directory
        try:
            os.makedirs(label_dir)
            shutil.copyfile(fish_loc, os.path.join(label_dir, item[1]))
        except:
            # the dir probably exists already so just move it
            try:
                shutil.copyfile(fish_loc, os.path.join(label_dir, item[1]))
            except:
                print(label)


