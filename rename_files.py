import os, unicodecsv as csv
import shutil
# open and store the csv file
IDs = {}
with open('data/HAM10000_metadata.csv','rb') as csvfile:
    print("csv open!")
    timeReader = csv.reader(csvfile, delimiter = ',')
    # build dictionary with associated IDs
    for row in timeReader:
        IDs[row[1]] = row[7]
        #print(IDs[row[1]])
# move files
path = 'data/HAM10000_images_part_2/'
destPath = 'data/renamed_images/'

for old_name in os.listdir(path):
    # ignore files in path which aren't in the csv file
    if old_name[:-4] in IDs:
        print("found an image!")
        # print(old_name[:-4])
        # print("path of old file " + os.path.join(path, old_name))
        # print("path of new file " + os.path.join(destPath, IDs[old_name[:-4]]+ ".jpg"))

        try:
            shutil.copy(os.path.join(path, old_name), os.path.join(destPath, IDs[old_name[:-4]]+ ".jpg" ))
        except:
            print ('File ' + old_name + ' could not be renamed to ' + IDs[old_name] + '!')